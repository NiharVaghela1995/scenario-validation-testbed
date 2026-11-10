"""highD robustness, analyses 2 to 4: tail sensitivity, stability and model checks (docs/HIGHD_ANALYSIS_PLAN.md).

    python experiments/highd_robust_tail.py     # writes results/highd/robust_{tail,stability,checks}.json

Every interval is a percentile bootstrap by recording with all models re-fitted in each replicate. Gaussian, t and mixture models
are the testbed classes on rank marginals (experiments/highd_tail.py). Clayton and Gumbel are bivariate here, in four rotations.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot"), str(ROOT / "experiments")]
import cutin_chain as C  # noqa: E402
import highd_cutins as HC  # noqa: E402
from highd_tail import rank_marginals  # noqa: E402
import series as SE  # noqa: E402

QS, Q0 = (0.05, 0.10, 0.15, 0.20), 0.10
B, SEED, N_MODEL = 400, 2026, 100_000
FULL = ("gaussian", "t", "clayton_best", "clayton_corner", "gumbel_best", "gumbel_corner", "k1", "k2", "k3", "k4", "k5", "k6")
CORE = ("gaussian", "k2")
THETA_C = np.exp(np.linspace(np.log(1e-3), np.log(20), 100))
THETA_G = 1 + np.exp(np.linspace(np.log(5e-4), np.log(19), 100))
QQ_P = np.linspace(0.004, 0.10, 25)
FLIPS = ((0, 0), (0, 1), (1, 0), (1, 1))
RES = ROOT / "results/highd"
DATA = {}


def pseudo(X):
    n = len(X)
    return np.c_[stats.rankdata(X[:, 0]) / (n + 1), stats.rankdata(X[:, 1]) / (n + 1)]


def _log_gumbel(a, b, th):
    x, y = -np.log(a), -np.log(b)
    s = x ** th + y ** th
    A = s ** (1 / th)
    return -A - np.log(a) - np.log(b) + (th - 1) * (np.log(x) + np.log(y)) + (2 / th - 2) * np.log(s) + np.log(A + th - 1)


def fit_arch(U, family):
    """-> {flip: (log-likelihood, theta)} per flip of the two margins. Clayton has its tail at (0, 0), Gumbel at (1, 1)."""
    out = {}
    for f in FLIPS:
        a = 1 - U[:, 0] if f[0] else U[:, 0]
        b = 1 - U[:, 1] if f[1] else U[:, 1]
        grid, fn = (THETA_C, C.ClaytonPair._log_clayton) if family == "clayton" else (THETA_G, _log_gumbel)
        ll = np.array([fn(a, b, t).sum() for t in grid])
        i = int(np.argmax(ll))
        out[f] = (float(ll[i]), float(grid[i]))
    return out


def sample_arch(family, flip, th, n, seed):
    rng = np.random.default_rng(seed)
    if family == "clayton":
        a, w = rng.random(n), rng.random(n)
        b = (a ** -th * (w ** (-th / (1 + th)) - 1) + 1) ** (-1 / th)
    else:
        al = 1 / th
        u, e0 = rng.uniform(-np.pi / 2, np.pi / 2, n), rng.exponential(1, n)
        v = np.sin(al * (u + np.pi / 2)) / np.cos(u) ** (1 / al) * (np.cos(u - al * (u + np.pi / 2)) / e0) ** ((1 - al) / al)
        a, b = np.exp(-(rng.exponential(1, n) / v) ** al), np.exp(-(rng.exponential(1, n) / v) ** al)
    u1, u2 = (1 - a if flip[0] else a), (1 - b if flip[1] else b)
    return np.c_[u1, u2]


CORNER_FLIP = dict(clayton=(0, 1), gumbel=(1, 0))


def model_samples(X, names, seed, n):
    """-> {model: n x 2 sample}, every model fitted on X (rank marginals)."""
    out = {}
    with rank_marginals():
        if "gaussian" in names:
            out["gaussian"] = C.GaussianCopula().fit(X).sample(n, seed)
        if "t" in names:
            out["t"] = C.TCopula().fit(X).sample(n, seed + 1)
        for j, k in enumerate(range(1, 7)):
            if f"k{k}" in names:
                out[f"k{k}"] = C.GMMScores().fit(X, k=k).sample(n, seed + 2 + j)
    if any(m.startswith(("clayton", "gumbel")) for m in names):
        U = pseudo(X)
        for fam in ("clayton", "gumbel"):
            if f"{fam}_best" in names:
                fits = fit_arch(U, fam)
                flip = max(fits, key=lambda f: fits[f][0])
                out[f"{fam}_best"] = sample_arch(fam, flip, fits[flip][1], n, seed + 20)
                corner = CORNER_FLIP[fam]
                out[f"{fam}_corner"] = sample_arch(fam, corner, fits[corner][1], n, seed + 21)
    return out


def ranks(S):
    n = len(S)
    return stats.rankdata(S[:, 0]) / n, stats.rankdata(S[:, 1]) / n


def lams(S, qs):
    g, d = ranks(S)
    return np.array([(g[d > 1 - q] <= q).mean() if (d > 1 - q).any() else np.nan for q in qs])


def corner_counts(S, q):
    g, d = ranks(S)
    hi = d > 1 - q
    return int(hi.sum()), int((g[hi] <= q).sum())


def qq_t(S):
    g, d = ranks(S)
    return np.quantile(np.maximum(g, 1 - d), QQ_P)


def replicate(a):
    key, names, seed, ppc = a
    X, groups = DATA[key]
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    pos = {i: np.flatnonzero(groups == i) for i in ids}
    Y = X[np.concatenate([pos[i] for i in rng.choice(ids, len(ids))])]
    samples = model_samples(Y, names, seed, N_MODEL)
    out = dict(emp=lams(Y, QS), models={m: lams(S, QS) for m, S in samples.items()})
    if ppc:
        n = len(X)
        out["ppc"] = {m: [corner_counts(S[:n], q)[1] for q in (0.10, 0.05)] for m, S in samples.items()}
        out["qq_emp"], out["qq"] = qq_t(Y), {m: qq_t(S) for m, S in samples.items()}
    return out


def ci(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def verdict(d, c, cond, recs):
    if cond < 50 or recs < 5:
        return "not interpretable"
    if d >= 0:
        return "fails"
    return "holds" if c[1] < 0 else "direction only"


def evaluate(key, names, ex, b, ppc=False):
    X, g = DATA[key]
    recs = int(len(np.unique(g)))
    pts = {m: lams(S, QS) for m, S in model_samples(X, names, SEED, 2 * N_MODEL).items()}
    emp = lams(X, QS)
    R = list(ex.map(replicate, [(key, names, SEED + 1 + i, ppc) for i in range(b)], chunksize=4))
    E = np.array([r["emp"] for r in R])
    out = dict(n=len(X), recordings=recs, bootstrap_replicates=b, levels={})
    for i, q in enumerate(QS):
        m_c, k = corner_counts(X, q)
        L = dict(conditioning_events=m_c, joint_extremes=k, empirical=float(emp[i]), ci_empirical=ci(E[:, i]), h1_below_independence=bool(ci(E[:, i])[1] < q), models={})
        for m in names:
            M = np.array([r["models"][m][i] for r in R])
            d, c = float(emp[i] - pts[m][i]), ci(E[:, i] - M)
            L["models"][m] = dict(lam=float(pts[m][i]), ci=ci(M), delta=d, ci_delta=c, verdict=verdict(d, c, m_c, recs))
        out["levels"][f"{q:g}"] = L
    if ppc:
        n = len(X)
        ob = {q: corner_counts(X, q)[1] for q in (0.10, 0.05)}
        out["predictive"] = {}
        for m in names:
            for j, q in enumerate((0.10, 0.05)):
                C_ = np.array([r["ppc"][m][j] for r in R])
                out["predictive"].setdefault(m, {})[f"{q:g}"] = dict(observed=ob[q], mean=float(C_.mean()), interval_95=[float(np.percentile(C_, 2.5)), float(np.percentile(C_, 97.5))],
                                                                     p_le_observed=float((C_ <= ob[q]).mean()), consistent=bool(np.percentile(C_, 2.5) <= ob[q] <= np.percentile(C_, 97.5)))
        Qe = np.array([r["qq_emp"] for r in R])
        out["qq"] = dict(p=np.round(QQ_P, 4).tolist(), data=np.round(qq_t(X), 5).tolist(), data_band=[np.round(np.percentile(Qe, 2.5, axis=0), 5).tolist(), np.round(np.percentile(Qe, 97.5, axis=0), 5).tolist()],
                         models={m: np.round(np.median([r["qq"][m] for r in R], axis=0), 5).tolist() for m in names})
    return out


def loo(a):
    X, g, r = a
    Y = X[g != r]
    S = model_samples(Y, CORE, SEED, N_MODEL)
    return int(r), lams(Y, (Q0,))[0], lams(S["gaussian"], (Q0,))[0], lams(S["k2"], (Q0,))[0]


def pair(d, a, b):
    return d[[a, b]].to_numpy(float), d.rec.to_numpy()


def variants(c, p):
    def filt(s):
        n = int(round(s * 25))
        return c[(c.c_before >= n) & (c.c_after >= n) & (c.f_before >= n) & (c.f_after >= n) & c.f_lane_ok & c.lead_ok & ~c.status.isin(["lane_change", "collision"])
                 & c.gap.between(0, SE.MAX_RANGE, inclusive="right") & (c.v_ego > SE.V_MIN)]
    V = {"stable_0.5s": filt(0.5), "stable_2s": filt(2.0), "follower_track_12s": p[p.f_track_s >= 12],
         "lateral_0.3": p[p.v_lat_cross >= 0.3], "lateral_0.5": p[p.v_lat_cross >= 0.5], "lateral_0.75": p[p.v_lat_cross >= 0.75],
         "no_auxiliary_lanes": p[~p.origin_aux & ~p.target_aux], "only_from_auxiliary": p[p.origin_aux], "leftward": p[p.leftward], "rightward": p[~p.leftward]}
    out = {k: pair(v, "thw", "dv") for k, v in V.items()}
    for key in ("05", "10", "20", "30"):
        g, v, d = c[f"gap0_{key}"], c[f"v_ego0_{key}"], c[f"dv0_{key}"]
        ok = c.primary & g.notna() & (g > 0) & (v > SE.V_MIN)
        q = pd.DataFrame(dict(thw=(g / v)[ok], dv=d[ok], rec=c.rec[ok]))
        out[f"manoeuvre_start_v{key}"] = pair(q, "thw", "dv")
    for name in ("m02", "p02"):
        g, v, d = p[f"gap_{name}"], p[f"v_ego_{name}"], p[f"dv_{name}"]
        ok = g.notna() & (g > 0) & (v > SE.V_MIN)
        q = pd.DataFrame(dict(thw=(g / v)[ok], dv=d[ok], rec=p.rec[ok]))
        out[f"timing_{'minus' if name == 'm02' else 'plus'}_0.2s"] = pair(q, "thw", "dv")
    return out


def strata(p):
    med = float(p.density_veh_km.median())
    S = {f"location_{l}": p[p.location == l] for l in sorted(p.location.unique())}
    S |= {"density_below_median": p[p.density_veh_km <= med], "density_above_median": p[p.density_veh_km > med], "density_at_least_22": p[p.density_veh_km >= 22],
          "follower_car": p[p.follower_class == "Car"], "follower_truck": p[p.follower_class == "Truck"], "cutter_car": p[p.cutter_class == "Car"], "cutter_truck": p[p.cutter_class == "Truck"]}
    for a in ("Car", "Truck"):
        for b in ("Car", "Truck"):
            S[f"cutter_{a.lower()}_follower_{b.lower()}"] = p[(p.cutter_class == a) & (p.follower_class == b)]
    return {k: pair(v, "thw", "dv") for k, v in S.items()}, med


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--b", type=int, default=B)
    a = ap.parse_args(argv)
    c = pd.read_parquet(HC.work_dir() / "candidates.parquet")
    p = c[c.primary].copy()
    DATA.update({"P1": pair(p, "thw", "dv"), "P2": pair(p, "gap", "dv"), "P3": pair(p, "thw", "v_lat_cross")})
    V = variants(c, p)
    S, med = strata(p)
    DATA.update({f"V:{k}": v for k, v in V.items()})
    DATA.update({f"S:{k}": v for k, v in S.items()})
    tail, stab, checks = {}, {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for k in ("P1", "P2", "P3"):
            r = evaluate(k, FULL, ex, a.b, ppc=k == "P1")
            if k == "P1":
                checks["predictive_joint_extremes"], checks["qq"] = r.pop("predictive"), r.pop("qq")
            tail[k] = r
            print(k, r["n"], {q: round(v["empirical"], 4) for q, v in r["levels"].items()}, flush=True)
        tail["variants"] = {}
        for k in V:
            r = evaluate(f"V:{k}", CORE, ex, a.b)
            tail["variants"][k] = r
            print(k, r["n"], r["levels"]["0.1"]["joint_extremes"], flush=True)
        stab["strata"] = {}
        for k in S:
            r = evaluate(f"S:{k}", CORE, ex, a.b)
            stab["strata"][k] = r
            print(k, r["n"], r["levels"]["0.1"]["conditioning_events"], flush=True)
        stab["density_median_veh_km"] = med
        X, g = DATA["P1"]
        L = np.array(list(ex.map(loo, [(X, g, r) for r in np.unique(g)])))
    d_g, d_m = L[:, 1] - L[:, 2], L[:, 1] - L[:, 3]
    stab["leave_one_recording_out"] = dict(recordings=len(L), empirical_range=[float(L[:, 1].min()), float(L[:, 1].max())], gaussian_range=[float(L[:, 2].min()), float(L[:, 2].max())],
                                           mixture_k2_range=[float(L[:, 3].min()), float(L[:, 3].max())], delta_gaussian_range=[float(d_g.min()), float(d_g.max())],
                                           delta_mixture_k2_range=[float(d_m.min()), float(d_m.max())], removals_with_delta_gaussian_ge_0=int((d_g >= 0).sum()), removals_with_delta_mixture_ge_0=int((d_m >= 0).sum()))
    (RES / "robust_tail.json").write_text(json.dumps(tail, indent=1))
    (RES / "robust_stability.json").write_text(json.dumps(stab, indent=1))
    (RES / "robust_checks.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
