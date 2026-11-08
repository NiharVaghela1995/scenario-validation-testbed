"""Regression set for the later-era methods (autoencoder, prediction error, temporal-logic queries, DDPM).

Fails if a detector's precision or recall drops by more than the stored tolerance, or if the DDPM's sample
dependence (Spearman) leaves the stored band or its gap Wasserstein distance exceeds the stored maximum.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "critical_events"
sys.path.insert(0, str(HERE))
import eras_regression as ER  # noqa: E402


def test_era_methods_do_not_regress():
    exp = json.loads(ER.EXPECTED.read_text())
    got = ER.compute()
    tol = exp["tolerance"]
    for k, e in exp["expected"].items():
        if k == "ddpm":
            assert abs(got[k]["spearman"] - e["spearman"]) <= exp["ddpm_spearman_band"], (k, got[k], e)
            assert got[k]["w1_gap"] <= exp["ddpm_w1_gap_max"], (k, got[k])
            continue
        for m in ("precision", "recall"):
            assert got[k][m] >= e[m] - tol[m], (k, m, got[k][m], e[m])
