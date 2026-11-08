"""Critical share alone: how many held-out events it takes to tell the cut-in models apart.

Computed, not cited. Inputs are read from results/generation_cutin.json (truth, accepted-sample shares, tail models).
Test: H0 "held-out events come from the model" (share p0 = model's critical share) against the truth p1 (generator),
one-sample two-sided test at 5 %, power 80 %, NORMAL APPROXIMATION:
    n = [ z_0.975 sqrt(p0 (1 - p0)) + z_0.80 sqrt(p1 (1 - p1)) ]^2 / (p1 - p0)^2
Also: power at n = 160 (normal approximation), the 95 % band of an observed share at n = 160 (normal and exact
binomial), and the tail metric lambda(0.10) on the same held-out set as an exact binomial (16 conditioning events).

    python experiments/power_heldout.py        # writes results/power_heldout.json and .txt
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(os.environ.get("TESTBED_RESULTS", ROOT / "results"))
D = json.loads((HERE / "generation_cutin.json").read_text())
ZA, ZB = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
N_HO = 160
LINES, OUT = [], {"n_heldout": N_HO, "critical_share": {}, "lambda_test": {}}


def say(text=""):
    print(text)
    LINES.append(text)


def n_needed(p0, p1):
    return ((ZA * np.sqrt(p0 * (1 - p0)) + ZB * np.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2


def power(p0, p1, n):
    s0, s1 = np.sqrt(p0 * (1 - p0) / n), np.sqrt(p1 * (1 - p1) / n)
    return stats.norm.sf((p0 + ZA * s0 - p1) / s1) + stats.norm.cdf((p0 - ZA * s0 - p1) / s1)


say("Critical share: held-out events needed to reject the model at 80 % power (two-sided 5 %, normal approximation)")
for g in ("G1", "G2"):
    G = D["robustness"]["generators"][g]
    p1 = G["truth"]["critical_share"]
    h = ZA * np.sqrt(p1 * (1 - p1) / N_HO)
    lo, hi = stats.binom.ppf(0.025, N_HO, p1) / N_HO, stats.binom.ppf(0.975, N_HO, p1) / N_HO
    say(f"\n{g}: truth {p1:.4f}; 95 % band of an observed share at n = {N_HO}: normal [{p1 - h:.3f}, {p1 + h:.3f}], "
        f"exact binomial [{lo:.3f}, {hi:.3f}]")
    models = {k: v["rejection"]["critical_share_accepted"] for k, v in G["models"].items()}
    if g == "G2":
        models.update({k: v["critical_share_valid"] for k, v in D["tail_models_g2"].items()})
    OUT["critical_share"][g] = {"truth": p1, "band_exact": [float(lo), float(hi)], "models": {}}
    for k, p0 in models.items():
        pw, nn = float(power(p0, p1, N_HO)), float(n_needed(p0, p1))
        OUT["critical_share"][g]["models"][k] = {"share": p0, "power_n160": pw, "n_for_80_power": nn}
        say(f"  {k:52s} share {p0:.4f}  power at n=160 {pw:.3f}  n for 80 % power {nn:8.0f}")

say("\nTail metric lambda(0.10) on the G2 held-out set (exact binomial, not an approximation)")
G = D["robustness"]["generators"]["G2"]
m = int(round(0.10 * N_HO))                      # events with dv in its highest 10 % of the held-out set
k = int(round(G["heldout"]["tail_dep_q10"] * m))
say(f"  conditioning events {m}, hits {k} (lambda {k / m:.4f})")
OUT["lambda_test"].update(conditioning=m, hits=k, heldout_lambda=k / m, models={})
lams = {kk: v["tail_dep_q10"] for kk, v in G["models"].items()}
lams.update({kk: v["tail_dep_q10"] for kk, v in D["tail_models_g2"].items()})
for kk, lam in lams.items():
    pv = float(stats.binom.sf(k - 1, m, lam))
    OUT["lambda_test"]["models"][kk] = {"lambda": lam, "p_value": pv}
    say(f"  {kk:52s} lambda {lam:.3f}  P(>= {k} of {m}) = {pv:.2e}")

(HERE / "power_heldout.json").write_text(json.dumps(OUT, indent=1))
(HERE / "power_heldout.txt").write_text("\n".join(LINES) + "\n")
