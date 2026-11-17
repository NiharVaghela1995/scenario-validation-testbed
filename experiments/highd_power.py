"""highD pilot, step 4: with this event count, how reliably can a tail difference be detected?

    python experiments/highd_power.py        # needs results/highd/tail.json and the per-event table; writes results/highd/power.json

TEST. T = lambda_emp(q) - lambda_Gauss(rho_hat), both computed from the same n events, q = 0.10: the Gaussian copula is re-fitted
(rho_hat = correlation of normal scores) in every simulated data set. Reject when T leaves the central 95 % of T under a
Gaussian copula with the rho of the real data at that n (N_NULL draws).

TRUTH. A Gaussian copula with the real rho, perturbed in the corner only, so that the true lambda is lambda_1 and the marginals
stay nearly unchanged: lambda_1 < lambda_G: each corner hit has its THW score redrawn outside the corner with probability
1 - lambda_1 / lambda_G; lambda_1 > lambda_G: each conditioning non-hit is moved into the corner with probability
(lambda_1 - lambda_G) / (1 - lambda_G).

CLUSTERING. Events within a recording are not independent. DEFF = (recording-bootstrap SE / event-bootstrap SE)^2 of the
difference, from tail.json; n_eff = n / DEFF. A direct check draws K recordings with replacement from the real data.

The exact binomial block repeats the testbed's test (experiments/power_heldout.py): corner hits ~ Bin(m, lambda) with the model's
lambda treated as known.
"""
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
import highd_cutins as HC  # noqa: E402

Q, ALPHA, N_NULL, N_SIM, SEED = 0.10, 0.05, 4000, 2000, 2026
NS = (100, 200, 400, 800, 1600, 3200)
KS = (5, 10, 20, 40, 60)
LAM1 = (0.006, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.218)
TAIL = json.loads((ROOT / "results/highd/tail.json").read_text())
P = TAIL["primary"]
RHO = P["normal_score_corr"]
ZQ = stats.norm.ppf(Q)


def lam_gauss_exact(rho, q=Q):
    """P(Z1 <= z_q | Z2 > z_{1-q}) for a bivariate normal with correlation rho."""
    z = stats.norm.ppf(q)
    joint = stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([z, -z])
    return (q - joint) / q


GRID = np.linspace(-0.6, 0.6, 241)
GRID_LAM = np.array([lam_gauss_exact(r) for r in GRID])
LAM_G = float(np.interp(RHO, GRID, GRID_LAM))


def statistic(thw, dv):
    n = len(thw)
    g, d = stats.rankdata(thw) / n, stats.rankdata(dv) / n
    hi = d > 1 - Q
    emp = float((g[hi] <= Q).mean())
    a, b = stats.norm.ppf(stats.rankdata(thw) / (n + 1)), stats.norm.ppf(stats.rankdata(dv) / (n + 1))
    return emp - float(np.interp(np.corrcoef(a, b)[0, 1], GRID, GRID_LAM))


def draw(n, lam1, rng):
    z1 = rng.standard_normal(n)
    z2 = RHO * z1 + np.sqrt(1 - RHO ** 2) * rng.standard_normal(n)
    if lam1 is None:
        return z1, z2
    cond = z2 > -ZQ
    hit = cond & (z1 <= ZQ)
    if lam1 < LAM_G:
        change = hit & (rng.random(n) < 1 - lam1 / LAM_G)
        z1[change] = stats.norm.ppf(rng.uniform(Q, 1, change.sum()))
    else:
        change = cond & ~hit & (rng.random(n) < (lam1 - LAM_G) / (1 - LAM_G))
        z1[change] = stats.norm.ppf(rng.uniform(0, Q, change.sum()))
    return z1, z2


def _cell(a):
    n, lam1, m, seed = a
    rng = np.random.default_rng(seed)
    return np.array([statistic(*draw(n, lam1, rng)) for _ in range(m)])


def exact_binomial(m, lam0, lam1, alpha=ALPHA):
    """Power of the two-sided exact binomial test of H0: lambda = lam0 when the truth is lam1."""
    x = np.arange(m + 1)
    pvals = np.array([stats.binomtest(int(k), m, lam0).pvalue for k in x])
    return float(stats.binom.pmf(x[pvals <= alpha], m, lam1).sum())


def m_for_power(lam0, lam1, target=0.8, top=200_000):
    m = 10
    while m <= top:
        if all(exact_binomial(k, lam0, lam1) >= target for k in (m, m + 1, m + 2, m + 3)):
            return m
        m = int(np.ceil(m * 1.12))
    return None


def crossing(xs, ys, level=0.8):
    for a, b, u, v in zip(xs, xs[1:], ys, ys[1:]):
        if (u - level) * (v - level) <= 0 and u != v:
            return float(a + (level - u) * (b - a) / (v - u))
    return None


def main():
    n_obs = P["n"]
    deff = (P["se_delta_emp_minus_gaussian"] / TAIL["primary_iid"]["se_delta_emp_minus_gaussian"]) ** 2
    n_eff = int(round(n_obs / deff))
    ns = sorted({*NS, n_eff, n_obs})
    ss = np.random.SeedSequence(SEED)
    lam1s = sorted({*LAM1, round(P["empirical"], 6)})
    jobs = [(n, None, N_NULL, int(s.generate_state(1)[0])) for n, s in zip(ns, ss.spawn(len(ns)))]
    cells = [(n, l) for n in ns for l in lam1s]
    jobs += [(n, l, N_SIM, int(s.generate_state(1)[0])) for (n, l), s in zip(cells, ss.spawn(len(cells)))]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_cell, jobs, chunksize=1))
    null = {n: res[i] for i, n in enumerate(ns)}
    band = {n: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for n, v in null.items()}
    sims = {c: res[len(ns) + i] for i, c in enumerate(cells)}
    power = {str(n): {str(l): float(((sims[(n, l)] < band[n][0]) | (sims[(n, l)] > band[n][1])).mean()) for l in lam1s} for n in ns}

    def mde(n):
        p = [power[str(n)][str(l)] for l in lam1s]
        below = [(l, x) for l, x in zip(lam1s, p) if l < LAM_G]
        above = [(l, x) for l, x in zip(lam1s, p) if l > LAM_G]
        lo = crossing([l for l, _ in below][::-1], [x for _, x in below][::-1])
        hi = crossing([l for l, _ in above], [x for _, x in above])
        return dict(lam_lower=lo, lam_upper=hi, delta_lower=None if lo is None else lo - LAM_G, delta_upper=None if hi is None else hi - LAM_G)

    obs = str(round(P["empirical"], 6))
    n80 = crossing(ns, [power[str(n)][obs] for n in ns])

    ev = pd.read_parquet(HC.work_dir() / "cutins.parquet")
    rng = np.random.default_rng(SEED + 99)
    rec_ids = np.unique(ev.rec)
    pos = {r: np.flatnonzero(ev.rec.to_numpy() == r) for r in rec_ids}
    thw, dv = ev.thw.to_numpy(), ev.dv.to_numpy()
    se_scale = {n: float(null[n].std(ddof=1) * np.sqrt(n)) for n in ns}
    scale = float(np.median(list(se_scale.values())))
    cl = {}
    for K in KS:
        hits, sizes = 0, []
        for _ in range(1000):
            idx = np.concatenate([pos[r] for r in rng.choice(rec_ids, K)])
            sizes.append(len(idx))
            se = scale / np.sqrt(len(idx)) * np.sqrt(deff)
            hits += abs(statistic(thw[idx], dv[idx])) > 1.96 * se
        cl[str(K)] = dict(mean_events=float(np.mean(sizes)), share_rejecting=hits / 1000)

    m = P["conditioning_events"]
    binom = {name: dict(lam0=l0, lam1=P["empirical"], power_at_m=exact_binomial(m, l0, P["empirical"]),
                        power_at_m_eff=exact_binomial(int(m / deff), l0, P["empirical"]),
                        m_for_80=m_for_power(l0, P["empirical"]))
             for name, l0 in dict(gaussian=P["gaussian"], mixture=P["mixture"], mixture_k3=TAIL["sensitivity"]["mixture_k3"]["mixture"]).items()}
    for v in binom.values():
        v["n_for_80"] = None if v["m_for_80"] is None else v["m_for_80"] / Q
    zero = TAIL["severity"]["ttc_lt_2"]
    out = dict(spec=dict(q=Q, alpha=ALPHA, n_null=N_NULL, n_sim=N_SIM, seed=SEED), n=n_obs, conditioning_events=m, rho=RHO,
               lam_gauss_exact=LAM_G, deff=deff, n_eff=n_eff, ns=ns, lam1=lam1s, power=power, null_band={str(k): v for k, v in band.items()},
               type1_at_n={str(n): float(np.mean((res[i][:N_NULL // 2] < np.percentile(res[i][N_NULL // 2:], 2.5)) | (res[i][:N_NULL // 2] > np.percentile(res[i][N_NULL // 2:], 97.5)))) for i, n in enumerate(ns)},
               min_detectable_at_n=mde(n_obs), min_detectable_at_n_eff=mde(n_eff), n_for_80_at_observed_lambda=n80,
               recording_resampling=cl, exact_binomial=binom,
               critical_share=dict(events=n_obs, ttc_lt_2=zero, upper_95_one_sided=float(1 - 0.05 ** (1 / n_obs))),
               lambda_ci_relative_width=(P["ci_empirical"][1] - P["ci_empirical"][0]) / P["empirical"])
    (ROOT / "results/highd/power.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("lam_gauss_exact", "deff", "n_eff", "min_detectable_at_n", "min_detectable_at_n_eff", "n_for_80_at_observed_lambda", "recording_resampling", "exact_binomial", "critical_share", "type1_at_n")}, indent=0))


if __name__ == "__main__":
    main()
