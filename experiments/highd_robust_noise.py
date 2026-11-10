"""highD robustness, analysis 5: measurement noise at the crossing frame (docs/HIGHD_ANALYSIS_PLAN.md section 5).

    python experiments/highd_robust_noise.py    # writes results/highd/robust_noise.json

Gaussian noise of sd sigma_pos on each vehicle's position and sd sigma_v on each vehicle's speed. Critical sets use N_MC draws.
For lambda, each of the B replicates draws fresh noise on a recording-bootstrap sample, so its interval carries both sources.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot"), str(ROOT / "experiments")]
import highd_cutins as HC  # noqa: E402
import highd_robust_tail as RT  # noqa: E402

LEVELS = ((0.1, 0.1), (0.3, 0.3), (0.5, 0.5), (0.5, 1.0))
N_MC, B, SEED = 200, 400, 2026
SETS = dict(ttc_lt_4=lambda m: m["ttc"] < 4, ttc_lt_2=lambda m: m["ttc"] < 2, thw_lt_0_6=lambda m: m["thw"] < 0.6, drac_gt_2=lambda m: m["drac"] > 2)
G = {}


def measures(gap, v, vc):
    g, v = np.maximum(gap, 0.01), np.maximum(v, 0.01)
    dv = v - vc
    cl = dv > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(cl, g / dv, np.inf)
    return dict(thw=g / v, ttc=ttc, drac=np.where(cl, dv ** 2 / (2 * g), 0.0), dv=dv)


def noisy(rng, idx, sp, sv):
    n = len(idx)
    return measures(G["gap"][idx] + rng.normal(0, np.sqrt(2) * sp, n), G["v"][idx] + rng.normal(0, sv, n), G["vc"][idx] + rng.normal(0, sv, n))


def replicate(a):
    (sp, sv), seed = a
    rng = np.random.default_rng(seed)
    ids = np.unique(G["rec"])
    take = rng.choice(ids, len(ids))
    idx = np.concatenate([G["pos"][i] for i in take])
    m = noisy(rng, idx, sp, sv)
    X = np.c_[m["thw"], m["dv"]]
    s = RT.model_samples(X, RT.CORE, seed, RT.N_MODEL)
    return float(RT.lams(X, (RT.Q0,))[0]), float(RT.lams(s["gaussian"], (RT.Q0,))[0]), float(RT.lams(s["k2"], (RT.Q0,))[0])


def main():
    c = pd.read_parquet(HC.work_dir() / "candidates.parquet")
    p = c[c.primary]
    G.update(gap=p.gap.to_numpy(), v=p.v_ego.to_numpy(), vc=p.v_cut.to_numpy(), rec=p.rec.to_numpy())
    G["pos"] = {i: np.flatnonzero(G["rec"] == i) for i in np.unique(G["rec"])}
    all_idx = np.arange(len(p))
    base = measures(G["gap"], G["v"], G["vc"])
    base_sets = {k: f(base) for k, f in SETS.items()}
    out = dict(events=len(p), monte_carlo_draws=N_MC, bootstrap_replicates=B, baseline_counts={k: int(v.sum()) for k, v in base_sets.items()}, levels={})
    ss = np.random.SeedSequence(SEED)
    for (sp, sv), s in zip(LEVELS, ss.spawn(len(LEVELS))):
        rng = np.random.default_rng(s)
        draws = {k: [] for k in SETS}
        for _ in range(N_MC):
            m = noisy(rng, all_idx, sp, sv)
            for k, f in SETS.items():
                draws[k].append(f(m))
        L = dict(sigma_pos_m=sp, sigma_v_ms=sv, critical_sets={})
        for k, d in draws.items():
            D = np.array(d)
            cnt = D.sum(1)
            b = base_sets[k]
            inter, un = (D & b).sum(1), (D | b).sum(1)
            jac = None if int(b.sum()) == 0 else inter / np.maximum(un, 1)
            L["critical_sets"][k] = dict(count_mean=float(cnt.mean()), count_interval_95=[float(np.percentile(cnt, 2.5)), float(np.percentile(cnt, 97.5))],
                                         jaccard_mean=None if jac is None else float(jac.mean()), jaccard_interval_95=None if jac is None else [float(np.percentile(jac, 2.5)), float(np.percentile(jac, 97.5))])
        with ProcessPoolExecutor(4) as ex:
            R = np.array(list(ex.map(replicate, [((sp, sv), SEED + 10_000 * LEVELS.index((sp, sv)) + i) for i in range(B)], chunksize=8)))
        e, gm, mm = R.T
        for name, x in (("gaussian", gm), ("k2", mm)):
            d, ci = float((e - x).mean()), RT.ci(e - x)
            L[f"delta_{name}"] = d
            L[f"ci_delta_{name}"] = ci
            L[f"verdict_{name}"] = RT.verdict(d, ci, int(0.1 * len(p)), 60)
            L[f"lam_{name}_mean"] = float(x.mean())
        L["lam_empirical_mean"], L["ci_lam_empirical"] = float(e.mean()), RT.ci(e)
        out["levels"][f"{sp}/{sv}"] = L
        print(sp, sv, round(float(e.mean()), 4), L["verdict_gaussian"], L["verdict_k2"], flush=True)
    (ROOT / "results/highd/robust_noise.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
