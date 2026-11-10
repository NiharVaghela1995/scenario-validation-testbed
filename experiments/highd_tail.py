"""highD pilot, step 3: tail fidelity on real cut-ins. Needs the per-event table from pilot/highd_cutins.py.

    python experiments/highd_tail.py                    # writes results/highd/tail.json (aggregates only)

PRIMARY SPECIFICATION (fixed before any tail result was computed). Pair (THW, dv) at lane crossing, q = 0.10,
lambda(q) = P(THW in its lowest q | dv in its highest q), critical_events.cutin_chain.tail_dependence, all kept cut-ins.
Everything else in `SENSITIVITY` is exploratory: no multiplicity correction is applied to it.

MODELS. cutin_chain.GaussianCopula and cutin_chain.GMMScores (K = 2) fitted to the pair, unchanged. Their lambda is computed
on N_SAMPLE draws. The empirical copula's lambda is the sample value itself. A model's marginals do not affect lambda (ranks),
so the fits use rank marginals instead of the O(n^2) KDE of cutin_chain._Marginal; `kde_check` compares the two on a subsample.

UNCERTAINTY. Bootstrap by recording (recordings are the independent units: events within one share traffic state, site and
time), percentile 95 % intervals, re-fitting every model in every replicate. The event-level bootstrap is reported only to
show how much the clustering matters.
"""
import argparse
import contextlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot")]
import cutin_chain as C  # noqa: E402
import highd_cutins as HC  # noqa: E402

Q, B_PRIMARY, B_SENS, B_IID, N_SAMPLE, SEED = 0.10, 2000, 400, 500, 100_000, 2026
HELDOUT_SHARE = 0.40
KDE_CHECK_N = 3000


class RankMarginal:
    """Empirical CDF with mid-ranks and its inverse: same interface as cutin_chain._Marginal, O(n log n)."""

    def __init__(self, x):
        self.x = np.sort(np.asarray(x, float))
        self.n = len(self.x)

    def cdf(self, v):
        v = np.asarray(v, float)
        u = (np.searchsorted(self.x, v, "left") + np.searchsorted(self.x, v, "right")) / (2 * self.n)
        return np.clip(u, 0.5 / self.n, 1 - 0.5 / self.n)

    def ppf(self, u):
        return np.interp(u, (np.arange(self.n) + 0.5) / self.n, self.x)


@contextlib.contextmanager
def rank_marginals():
    keep = C._Marginal
    C._Marginal = RankMarginal
    try:
        yield
    finally:
        C._Marginal = keep


def lam(thw, dv, q=Q):
    """cutin_chain.tail_dependence on a pair: gap / v_ego is THW when v_ego = 1."""
    return C.tail_dependence(np.c_[np.ones(len(thw)), thw, dv], q)


def corner(thw, dv, q=Q):
    """-> (conditioning events m, joint extremes k), the counts behind lambda(q)."""
    g, d = stats.rankdata(thw) / len(thw), stats.rankdata(dv) / len(dv)
    hi = d > 1 - q
    return int(hi.sum()), int((g[hi] <= q).sum())


def fit(X, k=2):
    return C.GaussianCopula().fit(X), C.GMMScores().fit(X, k=k)


def model_lams(X, seed, q=Q, k=2, n=N_SAMPLE):
    gc, mx = fit(X, k)
    return lam(*gc.sample(n, seed).T, q), lam(*mx.sample(n, seed + 1).T, q), float(np.corrcoef(gc.to_z(X).T)[0, 1])


def _replicate(a):
    X, groups, q, k, cluster, seed = a
    rng = np.random.default_rng(seed)
    if cluster:
        ids = np.unique(groups)
        take = rng.choice(ids, len(ids))
        pos = {i: np.flatnonzero(groups == i) for i in ids}
        idx = np.concatenate([pos[i] for i in take])
    else:
        idx = rng.integers(0, len(X), len(X))
    Y = X[idx]
    with rank_marginals():
        g, m, rho = model_lams(Y, seed, q, k)
    return lam(Y[:, 0], Y[:, 1], q), g, m, rho


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def evaluate(X, groups, ex, q=Q, k=2, b=B_SENS, seed=SEED, cluster=True):
    """Point estimates and paired bootstrap intervals for one population and one q."""
    with rank_marginals():
        g, m, rho = model_lams(X, seed, q, k, n=2 * N_SAMPLE)
    e = lam(X[:, 0], X[:, 1], q)
    mm, kk = corner(X[:, 0], X[:, 1], q)
    out = dict(n=len(X), recordings=int(len(np.unique(groups))), q=q, conditioning_events=mm, joint_extremes=kk,
               independence_value=q, spearman=float(stats.spearmanr(X[:, 0], X[:, 1]).statistic), normal_score_corr=rho,
               empirical=e, gaussian=g, mixture=m)
    jobs = [(X, groups, q, k, cluster, seed + 1 + i) for i in range(b)]
    R = np.array(list(ex.map(_replicate, jobs, chunksize=8)))
    out.update(bootstrap_replicates=b, ci_empirical=ci(R[:, 0]), ci_gaussian=ci(R[:, 1]), ci_mixture=ci(R[:, 2]),
               delta_emp_minus_gaussian=e - g, ci_delta_emp_minus_gaussian=ci(R[:, 0] - R[:, 1]),
               delta_emp_minus_mixture=e - m, ci_delta_emp_minus_mixture=ci(R[:, 0] - R[:, 2]),
               se_empirical=float(R[:, 0].std(ddof=1)), se_delta_emp_minus_gaussian=float((R[:, 0] - R[:, 1]).std(ddof=1)))
    return out


def heldout(X, groups, seed=SEED, q=Q):
    """Fit on 60 % of the recordings, test on the other 40 %: exact binomial test on the corner, copula log-likelihood."""
    ids = np.random.default_rng(seed).permutation(np.unique(groups))
    te_ids = ids[:int(round(HELDOUT_SHARE * len(ids)))]
    te = np.isin(groups, te_ids)
    A, B = X[~te], X[te]
    with rank_marginals():
        gc, mx = fit(A)
        lams = dict(gaussian=lam(*gc.sample(2 * N_SAMPLE, seed).T, q), mixture=lam(*mx.sample(2 * N_SAMPLE, seed + 1).T, q),
                    empirical_train=lam(A[:, 0], A[:, 1], q))
        Zg, Zm = gc.to_z(B), mx.to_z(B)
        ll = dict(gaussian=float(gc.log_copula(Zg).mean()),
                  mixture=float((mx.g.score_samples(Zm) - stats.norm.logpdf(Zm).sum(1)).mean()), independence=0.0)
    m, k = corner(B[:, 0], B[:, 1], q)
    tests = {n: dict(lam=float(v), p_two_sided=float(stats.binomtest(k, m, v).pvalue),
                     p_upper=float(stats.binom.sf(k - 1, m, v))) for n, v in lams.items()}
    return dict(train_events=int((~te).sum()), test_events=int(te.sum()), test_recordings=int(len(te_ids)), conditioning_events=m,
                joint_extremes=k, heldout_lambda=k / m, wilson_95=list(C.wilson(k, m)), models=tests,
                heldout_copula_loglik_per_event=ll, note="binomial tests treat held-out events as independent")


def kde_check(X, seed=SEED, q=Q, n=KDE_CHECK_N):
    """Testbed models with their own KDE marginals against the rank-marginal fits, on one random subsample."""
    Y = X[np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)]
    kde = model_lams(Y, seed, q, n=2 * N_SAMPLE)[:2]
    with rank_marginals():
        rank = model_lams(Y, seed, q, n=2 * N_SAMPLE)[:2]
    return dict(n=len(Y), kde=dict(gaussian=kde[0], mixture=kde[1]), rank=dict(gaussian=rank[0], mixture=rank[1]))


def bic_table(X, ks=(1, 2, 3, 4, 5)):
    from sklearn.mixture import GaussianMixture
    with rank_marginals():
        Z = C.GaussianCopula().fit(X).to_z(X)
    b = {k: float(GaussianMixture(k, covariance_type="full", random_state=0, n_init=5).fit(Z).bic(Z)) for k in ks}
    return dict(bic=b, best_k=min(b, key=b.get))


def severity(ev):
    """Absolute counts of physically tight cut-ins (aggregates, no extreme single values)."""
    closing = ev.ttc.notna()
    n = len(ev)
    c = dict(events=n, closing=int(closing.sum()), share_closing=float(closing.mean()))
    for t in (4.0, 3.0, 2.0, 1.5, 1.0):
        c[f"ttc_lt_{t:g}"] = int((ev.ttc < t).sum())
    for t in (1.0, 0.6, 0.5, 0.3):
        c[f"thw_lt_{t:g}"] = int((ev.thw < t).sum())
    for t in (2.0, 4.0, 8.0):
        c[f"drac_gt_{t:g}"] = int((ev.drac > t).sum())
    c["ttc_lt_2_and_thw_lt_0.5"] = int(((ev.ttc < 2) & (ev.thw < 0.5)).sum())
    c["thw_lt_0.6_and_dv_gt_2"] = int(((ev.thw < 0.6) & (ev.dv > 2)).sum())
    q = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    c["quantiles"] = {v: {str(p): float(ev[v].quantile(p)) for p in q} for v in ("v_ego", "gap", "dv", "thw", "ttc", "drac")}
    c["class"] = {f"{k[0]}->{k[1]}": int(v) for k, v in ev.groupby(["cutter_class", "follower_class"]).size().items()}
    c["by_location"] = {str(k): int(v) for k, v in ev.groupby("location").size().items()}
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--b", type=int, default=B_PRIMARY)
    ap.add_argument("--b-sens", type=int, default=B_SENS)
    ap.add_argument("--out", default=str(ROOT / "results/highd/tail.json"))
    a = ap.parse_args(argv)
    ev = pd.read_parquet(HC.work_dir(a.work) / "cutins.parquet")
    pair = lambda d: (d[["thw", "dv"]].to_numpy(float), d.rec.to_numpy())

    with ProcessPoolExecutor(a.workers) as ex:
        X, g = pair(ev)
        res = dict(spec=dict(pair=["thw", "dv"], at="lane crossing f_c", q=Q, model_samples=2 * N_SAMPLE, seed=SEED),
                   severity=severity(ev), kde_check=kde_check(X), bic=bic_table(X), heldout=heldout(X, g))
        res["primary"] = evaluate(X, g, ex, b=a.b)
        print("primary", {k: res["primary"][k] for k in ("n", "conditioning_events", "joint_extremes", "empirical", "gaussian", "mixture")}, flush=True)
        res["primary_iid"] = evaluate(X, g, ex, b=B_IID, cluster=False)
        res["joint_extremes_by_q"] = {f"{q:g}": dict(zip(("conditioning_events", "joint_extremes"), corner(X[:, 0], X[:, 1], q)))
                                      for q in (0.01, 0.02, 0.05, 0.10, 0.20)}
        valid0 = ev.gap0.notna() & (ev.gap0 > 0) & (ev.v_ego0 > 0.5)
        ev0 = ev[valid0].assign(thw=lambda d: d.gap0 / d.v_ego0, dv=lambda d: d.dv0)
        S = {"pair_at_manoeuvre_start": (*pair(ev0), {}),
             "q_0.05": (X, g, dict(q=0.05)),
             "q_0.20": (X, g, dict(q=0.20)),
             "mixture_k3": (X, g, dict(k=3)),
             "thw_below_3s": (*pair(ev[ev.thw < 3]), {}),
             "closing_only": (*pair(ev[ev.dv > 0]), {}),
             "cars_only": (*pair(ev[(ev.cutter_class == "Car") & (ev.follower_class == "Car")]), {}),
             "location_1": (*pair(ev[ev.location == 1]), {}),
             "other_locations": (*pair(ev[ev.location != 1]), {}),
             "speed_above_60kmh": (*pair(ev[ev.v_ego > 60 / 3.6]), {})}
        res["sensitivity"] = {}
        for name, (x, gg, kw) in S.items():
            res["sensitivity"][name] = evaluate(x, gg, ex, b=a.b_sens, **kw)
            r = res["sensitivity"][name]
            print(name, r["n"], round(r["empirical"], 3), round(r["gaussian"], 3), round(r["mixture"], 3), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
