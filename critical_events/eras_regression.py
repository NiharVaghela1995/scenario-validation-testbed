"""Regression metrics for the later-era methods (fixed seeds, synthetic set). Used by tests/test_regression_eras.py.

    python eras_regression.py --write     # (re)write data/regression_expected_eras.json after an intended change
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eras as R  # noqa: E402
import evaluate as E  # noqa: E402
from synthetic import make_segments  # noqa: E402

EXPECTED = HERE / "data" / "regression_expected_eras.json"


def compute():
    segs = make_segments(n_segments=30, seed=7)
    ids = [s["id"] for s in segs]
    SER = {s["id"]: s["series"] for s in segs}
    GT = {s["id"]: s["gt"] for s in segs}
    perm = np.random.default_rng(16).permutation(ids)
    TUNE, HOLD = sorted(perm[:15]), sorted(perm[15:])
    out = {}
    # autoencoder (lr 1e-3, 60 epochs, seed 0), threshold = 0.95 quantile of tune-window errors
    Wtr = np.concatenate([R.windows(SER[k])[0] for k in TUNE])
    Wva = np.concatenate([R.windows(SER[k])[0] for k in HOLD])
    model, hist = R.train_ae(Wtr, Wva, lr=1e-3, epochs=60, seed=0)
    sc = {k: R.ae_scores(model, SER[k]) for k in ids}
    thr = float(np.quantile(np.concatenate([sc[k][1] for k in TUNE]), 0.95))
    fl = {k: R.score_to_intervals(sc[k][0], SER[k].t.to_numpy(), thr) for k in ids}
    r = E.score(fl, GT)
    out["autoencoder"] = dict(precision=r["precision"], recall=r["recall"], val_mse=hist[-1][1])
    # prediction error
    pe = {k: R.kinematic_pred_error(SER[k]) for k in ids}
    r = E.score({k: R.score_to_intervals(pe[k], SER[k].t.to_numpy(), 3.0) for k in ids}, GT)
    out["kinematic_pred_error_thr3"] = dict(precision=r["precision"], recall=r["recall"])
    ridge = R.fit_ridge([SER[k] for k in TUNE if not GT[k]])
    pr_ = {k: R.ridge_pred_error(ridge, SER[k]) for k in ids}
    r = E.score({k: R.score_to_intervals(pr_[k], SER[k].t.to_numpy(), 2.0) for k in ids}, GT)
    out["ridge_pred_error_thr2"] = dict(precision=r["precision"], recall=r["recall"])
    # queries
    cut = [k for k in ids if "cut_in" in k]
    fl = {k: R.query_intervals(R.query_tight_cut_in(SER[k]), SER[k].t.to_numpy(), 3.0) for k in cut}
    r = E.score(fl, {k: GT[k] for k in cut})
    out["query_tight_cut_in_on_cut_in_logs"] = dict(precision=r["precision"], recall=r["recall"])
    fl = {k: R.query_intervals(R.query_sustained_closing(SER[k]), SER[k].t.to_numpy(), 1.0) for k in ids}
    r = E.score(fl, GT)
    out["query_sustained_closing"] = dict(precision=r["precision"], recall=r["recall"])
    # DDPM (4000 steps, lr 1e-3, seed 0; samples seed 1)
    P = R.car_following_pairs(SER)
    mu, sd = P.mean(0), P.std(0)
    m, _ = R.train_ddpm((P - mu) / sd, steps=4000, lr=1e-3, seed=0)
    S = R.sample_ddpm(m, len(P), seed=1) * sd + mu
    out["ddpm"] = dict(spearman=float(stats.spearmanr(S[:, 0], S[:, 1]).statistic),
                           spearman_real=float(stats.spearmanr(P[:, 0], P[:, 1]).statistic),
                           w1_gap=float(stats.wasserstein_distance(P[:, 1], S[:, 1])))
    return out


TOL = {"precision": 0.05, "recall": 0.05}

if __name__ == "__main__":
    res = compute()
    print(json.dumps(res, indent=1))
    if "--write" in sys.argv:
        EXPECTED.write_text(json.dumps(dict(created="2026-11-03", tolerance=TOL, ddpm_spearman_band=0.10,
                                            ddpm_w1_gap_max=1.5, expected=res), indent=1))
        print("written", EXPECTED)
