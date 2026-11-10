"""highD robustness, analysis 1: the screening methods as classifiers (docs/HIGHD_ANALYSIS_PLAN.md section 1).

    python experiments/highd_robust_screening.py     # writes results/highd/robust_screening.json; per-vehicle arrays stay in $HIGHD_WORK

Vehicle-level unit: the pilot's seeded 10 % control sample. Cut-in-level unit: kept cut-ins with a screenable follower.
The references come from the same signals as the detectors, so this measures agreement with a rule, not independent validity.
"""
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot")]
import detectors as D  # noqa: E402
import fusion as FU  # noqa: E402
import highd_cutins as HC  # noqa: E402
import loaders as L  # noqa: E402
import screening as SC  # noqa: E402
import series as SE  # noqa: E402

METHODS = ("threshold", "pelt", "discord", "union", "gated")
A_GRID = np.exp(np.linspace(np.log(1 / 8), np.log(8), 25))
A_DEFAULT = int(np.argmin(np.abs(A_GRID - 1.0)))
B, SEED, REF_DUR, DRACS = 400, 2026, 0.3, (1.0, 2.0, 3.0)
REFS = ("ttc_lt_4", "thw_lt_0.6", "drac_gt_1", "drac_gt_2", "drac_gt_3")
RES = ROOT / "results/highd"


def discord_intervals(P, t, m, floor):
    """detectors.discords with the matrix profile already computed and the absolute floor as a parameter."""
    if P is None or not len(P):
        return []
    P = np.where(np.isfinite(P), P, np.nan)
    med = np.nanmedian(P)
    mad = np.nanmedian(np.abs(P - med)) * 1.4826
    thr = max(med + D.K_MAD * mad, floor * np.sqrt(m))
    work, out = P.copy(), []
    for _ in range(D.MAX_DISCORDS):
        if np.all(np.isnan(work)):
            break
        i = int(np.nanargmax(work))
        if work[i] <= thr:
            break
        out.append([round(float(t[i]), 2), round(float(t[min(i + m - 1, len(t) - 1)]), 2)])
        work[max(0, i - m):i + m] = np.nan
    return sorted(out)


def sustained(mask, t):
    return bool(D.runs_to_intervals(np.asarray(mask, bool), t, min_dur=REF_DUR, merge_gap=0.0))


def near(iv, x):
    return any(a - HC.MATCH_S <= x <= b + HC.MATCH_S for a, b in iv)


def run(a):
    rec, data = a
    warnings.filterwarnings("ignore", message="A large number of values")
    p = Path(data) / f"{rec:02d}_tracks.csv"
    frames = L.resample(L.load_highd(p))
    span = frames.groupby("vid").t.agg(lambda t: t.max() - t.min())
    egos = sorted(span[span >= SC.MIN_DURATION_S].index)
    control = {e for e, u in zip(egos, np.random.default_rng(rec * 1000 + 7).random(len(egos))) if u < HC.CONTROL_SHARE}
    ev = pd.read_parquet(HC.work_dir() / "cutins.parquet", columns=["rec", "follower", "f_c", "ttc", "thw", "drac"])
    ev = ev[ev.rec == rec]
    t_by = {k: g for k, g in frames.groupby("vid", sort=False)}
    fr_t = frames.t.to_numpy()
    order = np.argsort(fr_t, kind="stable")
    ev_by = {k: g for k, g in ev.groupby("follower")}
    veh, cut = [], []
    for ego in sorted(control | (set(ev_by) & set(egos))):
        g = t_by[ego]
        a0, b0 = g.t.min(), g.t.max()
        lo, hi = np.searchsorted(fr_t[order], [a0 - 0.05, b0 + 0.05])
        ser = SE.compute(frames.iloc[order[lo:hi]], ego)
        x, t = SE.detector_signal(ser), ser.t.to_numpy()
        bk = D.change_points(x)
        seg_edges = [0] + list(bk)
        seg_mean = [float(np.mean(x[s:e])) for s, e in zip(seg_edges[:-1], seg_edges[1:])]
        m = max(4, int(round(D.MP_WINDOW_S / float(np.median(np.diff(t))))))
        P = D.matrix_profile(x, m) if len(x) >= 3 * m else None
        grid = np.zeros((len(A_GRID), len(METHODS)), bool)
        times = ev_by[ego].f_c.to_numpy() / 25.0 if ego in ev_by else np.array([])
        hits = np.zeros((len(times), len(A_GRID), len(METHODS)), bool)
        for i, sc in enumerate(A_GRID):
            th = D.threshold(ser, thr=D.TTC_THR * sc)
            mask = np.zeros(len(x), bool)
            for (s, e), mv in zip(zip(seg_edges[:-1], seg_edges[1:]), seg_mean):
                if mv > D.PELT_LEVEL / sc:
                    mask[s:e] = True
            pe = D.runs_to_intervals(mask, t)
            di = discord_intervals(P, t, m, D.MP_FLOOR_RMS / sc)
            un = sorted(th + pe + di)
            ga = FU.physical_gate(un, ser) if un else []
            sets = (th, pe, di, un, ga)
            grid[i] = [bool(s) for s in sets]
            for j, tc in enumerate(times):
                hits[j, i] = [near(s, tc) for s in sets]
        ok = ser.status != "lane_change"
        ref = dict(ttc_lt_4=sustained(ser.ttc < 4, t), **{"thw_lt_0.6": sustained((ser.thw < 0.6) & ok, t)},
                   **{f"drac_gt_{int(d)}": sustained(ser.drac > d, t) for d in DRACS})
        if ego in control:
            veh.append((rec, grid, ref))
        for j, row in enumerate(ev_by[ego].itertuples() if ego in ev_by else []):
            if a0 + 1 <= row.f_c / 25.0 <= b0 - 1:
                cut.append((rec, hits[j], {"ttc_lt_4": row.ttc < 4, "thw_lt_0.6": row.thw < 0.6, **{f"drac_gt_{int(d)}": row.drac > d for d in DRACS}}))
    return veh, cut


def by_recording(items, ref, recs):
    """-> {rec index: (positives TP[a,m], FP[a,m], FN[a,m], TN[a,m])} arrays of shape R x a x m x 4."""
    idx = {r: i for i, r in enumerate(recs)}
    C = np.zeros((len(recs), len(A_GRID), len(METHODS), 4), np.int64)
    for rec, grid, r in items:
        pos = bool(r[ref])
        k = idx[rec]
        C[k, :, :, 0] += grid & pos
        C[k, :, :, 1] += grid & (not pos)
        C[k, :, :, 2] += (~grid) & pos
        C[k, :, :, 3] += (~grid) & (not pos)
    return C


def rates(c):
    tp, fp, fn, tn = np.moveaxis(c, -1, 0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(precision=tp / (tp + fp), recall=tp / (tp + fn), miss=fn / (tp + fn), fpr=fp / (fp + tn))


def auc(x, y):
    """Area by trapezoid with the (0,0)-(1,1) or end anchors, x sorted; NaN if undefined."""
    o = np.argsort(x, kind="stable")
    return float(np.trapezoid(np.r_[0, y[o], 1], np.r_[0, x[o], 1]))


def pr_auc(r, p):
    ok = np.isfinite(r) & np.isfinite(p)
    if ok.sum() < 2:
        return float("nan")
    o = np.argsort(r[ok], kind="stable")
    return float(np.trapezoid(p[ok][o], r[ok][o]))


def summarise(items, recs, rng):
    out = {}
    for ref in REFS:
        C = by_recording(items, ref, recs)
        T = C.sum(0)
        idx = rng.integers(0, len(recs), (B, len(recs)))
        R = {k: np.nan * np.zeros((B,) + T.shape[:2]) for k in ("precision", "recall", "miss", "fpr")}
        AU, AP = np.zeros((B, len(METHODS))), np.zeros((B, len(METHODS)))
        for b in range(B):
            r = rates(C[idx[b]].sum(0))
            for k in R:
                R[k][b] = r[k]
            AU[b] = [auc(r["fpr"][:, m], np.nan_to_num(r["recall"][:, m])) for m in range(len(METHODS))]
            AP[b] = [pr_auc(r["recall"][:, m], r["precision"][:, m]) for m in range(len(METHODS))]
        r0 = rates(T)
        block = dict(positives=int(T[A_DEFAULT, 0, 0] + T[A_DEFAULT, 0, 2]), negatives=int(T[A_DEFAULT, 0, 1] + T[A_DEFAULT, 0, 3]), methods={})
        for m, name in enumerate(METHODS):
            ci = lambda k: [float(np.nanpercentile(R[k][:, A_DEFAULT, m], 2.5)), float(np.nanpercentile(R[k][:, A_DEFAULT, m], 97.5))]
            block["methods"][name] = dict(confusion=dict(zip(("tp", "fp", "fn", "tn"), map(int, T[A_DEFAULT, m]))),
                                          precision=float(r0["precision"][A_DEFAULT, m]), recall=float(r0["recall"][A_DEFAULT, m]), miss_rate=float(r0["miss"][A_DEFAULT, m]),
                                          ci_precision=ci("precision"), ci_recall=ci("recall"), ci_miss_rate=ci("miss"),
                                          roc_auc=auc(r0["fpr"][:, m], np.nan_to_num(r0["recall"][:, m])), ci_roc_auc=[float(np.nanpercentile(AU[:, m], 2.5)), float(np.nanpercentile(AU[:, m], 97.5))],
                                          pr_auc=pr_auc(r0["recall"][:, m], r0["precision"][:, m]), ci_pr_auc=[float(np.nanpercentile(AP[:, m], 2.5)), float(np.nanpercentile(AP[:, m], 97.5))],
                                          sweep=dict(fpr=np.round(r0["fpr"][:, m], 5).tolist(), recall=np.round(r0["recall"][:, m], 5).tolist(), precision=[None if not np.isfinite(v) else round(float(v), 5) for v in r0["precision"][:, m]]))
        out[ref] = block
    return out


def main():
    data = HC.data_dir()
    recs = HC.recordings(data)
    with ProcessPoolExecutor(3) as ex:
        res = list(ex.map(run, [(r, str(data)) for r in recs]))
    veh = [x for r in res for x in r[0]]
    cut = [x for r in res for x in r[1]]
    rng = np.random.default_rng(SEED)
    out = dict(a_grid=np.round(A_GRID, 4).tolist(), a_default_index=A_DEFAULT, bootstrap_replicates=B, reference_duration_s=REF_DUR, vehicles_control=len(veh), cutins_screened=len(cut),
               vehicle_level=summarise(veh, recs, rng), cutin_level=summarise(cut, recs, rng))
    (RES / "robust_screening.json").write_text(json.dumps(out, indent=1))
    print("vehicles", len(veh), "cutins", len(cut))


if __name__ == "__main__":
    main()
