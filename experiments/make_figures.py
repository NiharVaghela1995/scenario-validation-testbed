"""README figures, built from results/ and data/synthetic/ (nothing is re-simulated).

Every figure carries the label "Illustrative (synthetic)": a designed generator with a known truth, never real traffic.

    python experiments/make_figures.py
"""
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("TESTBED_RESULTS", ROOT / "results"))
OUT = Path(os.environ.get("TESTBED_FIGURES", ROOT / "figures"))
DATA = Path(os.environ.get("TESTBED_DATA", ROOT / "data" / "synthetic"))
OUT.mkdir(parents=True, exist_ok=True)
LABEL = "Illustrative (synthetic)"

INK, INK2, MUTED, GRID, SURF = "#111827", "#4b5563", "#9ca3af", "#e5e7eb", "#ffffff"
BLUE, ORANGE, AQUA, CTX = "#2a78d6", "#eb6834", "#1baf7a", "#c5cad3"
plt.rcParams.update({
    "font.size": 12, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "axes.axisbelow": True, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "xtick.major.size": 0, "ytick.major.size": 0, "axes.titlesize": 12, "axes.titlelocation": "left",
})

d = json.loads((RES / "generation_cutin.json").read_text())
pw = json.loads((RES / "power_heldout.json").read_text())
ROB, TAILM = d["robustness"]["generators"], d["tail_models_g2"]
NAMES = {
    "independent marginals (1-D KDE)": "Independent marginals",
    "multivariate KDE": "Multivariate KDE",
    "Gaussian copula + KDE marginals": "Gaussian copula",
    "t copula + KDE marginals": "t copula",
    "rotated Clayton (gap x dv) + Gaussian rest": "Rotated Clayton",
    "Gaussian mixture on normal scores (GMCM-style, two components)": "GMCM-style mixture",
}
ORDER = list(NAMES)
COL = {k: CTX for k in ORDER}
COL[ORDER[2]], COL[ORDER[5]] = ORANGE, BLUE
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pfmt(p):
    if p >= 0.01:
        return f"{p:.2f}"
    m, e = f"{p:.0e}".split("e")
    return f"{m}·10{str(int(e)).translate(SUP)}"


def model(g, k):
    return ROB[g]["models"].get(k) or (TAILM[k] if g == "G2" else None)


def save(fig, name, label=LABEL):
    fig.text(0.995, 0.005, label, ha="right", va="bottom", fontsize=10, color=INK2, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / f"{name}.png", dpi=130, facecolor=SURF, metadata={"Software": None})
    plt.close(fig)


truth2, held2 = ROB["G2"]["truth"], ROB["G2"]["heldout"]

# 1 cut-in events
rows = list(csv.DictReader((DATA / "cutin_events_g1.csv").open()))
gap, dv = (np.array([float(r[c]) for r in rows]) for c in ("gap", "dv"))
crit = np.array([r["critical"] == "1" for r in rows])
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.scatter(dv[~crit], gap[~crit], s=14, color=CTX, lw=0, label="non-critical")
ax.scatter(dv[crit], gap[crit], s=28, color=ORANGE, lw=0.8, edgecolor=SURF, label="critical (TTC at lane crossing < 2 s)")
ax.set_xlabel("Relative (closing) speed Δv [m/s]")
ax.set_ylabel("Gap at cut-in start [m]")
ax.legend(frameon=False, loc="upper left", fontsize=10.5)
ax.set_title("Critical cut-ins sit in one corner: small gap, high closing speed")
save(fig, "fig1_cutin_events")

# 2 likelihood vs tail on G2
fig, ax = plt.subplots(figsize=(7.4, 4.5))
# (x, y offset in points, horizontal alignment): the two copulas and the two mid-range models nearly coincide
OFFSET = {ORDER[0]: (10, 0, "left"), ORDER[1]: (12, 0, "left"), ORDER[2]: (-10, 0, "right"), ORDER[3]: (0, 60, "center"),
          ORDER[4]: (-4, -60, "left"), ORDER[5]: (-10, 0, "right")}
for k in ORDER:
    v = model("G2", k)
    ax.errorbar(v["tail_dep_q10"], v["heldout_loglik_per_event"], yerr=1.96 * v["heldout_loglik_se"], fmt="o", ms=9,
                color=COL[k] if COL[k] != CTX else MUTED, elinewidth=1.2, mec=SURF, mew=1.5, zorder=3)
    ax.annotate(NAMES[k], xy=(v["tail_dep_q10"], v["heldout_loglik_per_event"]), xytext=OFFSET[k][:2],
                textcoords="offset points", fontsize=10.5, color=INK, va="center", ha=OFFSET[k][2])
ax.axvline(truth2["tail_dep_q10"], color=INK, lw=1.2, ls=(0, (4, 3)))
ax.annotate(f"truth λ = {truth2['tail_dep_q10']:.3f}", xy=(truth2["tail_dep_q10"], 1), xycoords=("data", "axes fraction"),
            xytext=(-6, -8), textcoords="offset points", ha="right", va="top", fontsize=10.5)
ax.set_xlim(-0.065, 0.26)
ax.set_xticks(np.arange(0, 0.26, 0.05))
ax.set_ylim(top=-8.78)
ax.set_xlabel("λ(0.10) in the critical corner (higher = joint tail present)")
ax.set_ylabel("Held-out log-likelihood per event")
ax.set_title("Held-out likelihood does not reveal which model holds the tail (G2)")
save(fig, "fig2_likelihood_vs_tail")

# 3 lambda per model with exact binomial P
LT = pw["lambda_test"]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ys = np.arange(len(ORDER))
lv = [model("G2", k)["tail_dep_q10"] for k in ORDER]
ax.barh(ys, lv, color=[COL[k] for k in ORDER], height=0.58)
for y, k, v in zip(ys, ORDER, lv):
    ax.text(v + 0.006, y, f"{v:.3f}   P = {pfmt(LT['models'][k]['p_value'])}", va="center", fontsize=10.5, color=INK2,
            bbox={"facecolor": SURF, "edgecolor": "none", "pad": 1.5}, zorder=5)
ax.axvline(truth2["tail_dep_q10"], color=INK, lw=1.2, ls=(0, (4, 3)))
ax.text(truth2["tail_dep_q10"] + 0.004, 5.45, f"truth {truth2['tail_dep_q10']:.3f}", fontsize=10.5, va="bottom")
ax.axvline(held2["tail_dep_q10"], color=AQUA, lw=1.6)
ax.text(held2["tail_dep_q10"] + 0.004, 5.45, f"held-out {held2['tail_dep_q10']:.3f}\n({LT['hits']} of {LT['conditioning']} events)", fontsize=10.5, va="bottom")
ax.set_yticks(ys, [NAMES[k] for k in ORDER])
ax.invert_yaxis()
ax.set_xlim(0, 0.45)
ax.grid(axis="y", visible=False)
ax.set_xlabel(f"λ(0.10);  P = exact binomial P(≥ {LT['hits']} of {LT['conditioning']} | model)")
ax.set_title("Only the mixture is not rejected by the tail test on 160 events (G2)")
save(fig, "fig3_lambda_corner")

# 4 power: two panels, no dual axis
P2 = pw["critical_share"]["G2"]["models"]
ks = [k for k in ORDER if k in P2]
ys = np.arange(len(ks))
fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)
vals = [P2[k]["power_n160"] for k in ks]
axs[0].barh(ys, vals, color=[COL[k] for k in ks], height=0.58)
axs[0].axvline(0.8, color=INK, lw=1.2, ls=(0, (4, 3)))
axs[0].text(0.79, len(ks) - 0.55, "80 % target", ha="right", fontsize=10.5, va="bottom")
for y, v in zip(ys, vals):
    axs[0].text(v + 0.012, y, f"{v:.0%}", va="center", fontsize=10.5)
axs[0].set_xlim(0, 1)
axs[0].set_title("Power of a critical-share check at n = 160", fontsize=11.5)
nn = [P2[k]["n_for_80_power"] for k in ks]
axs[1].barh(ys, nn, color=[COL[k] for k in ks], height=0.58)
axs[1].set_xscale("log")
axs[1].axvline(160, color=AQUA, lw=1.6)
for y, v in zip(ys, nn):
    axs[1].text(v * 1.12, y, f"{v:,.0f}", va="center", fontsize=10.5)
axs[1].set_xlim(100, 2e6)
axs[1].set_title("Events for 80 % power (green line: 160)", fontsize=11.5)
axs[0].set_yticks(ys, [NAMES[k] for k in ks])
axs[0].invert_yaxis()
for a in axs:
    a.grid(axis="y", visible=False)
save(fig, "fig4_power")

# 5 importance sampling
IS = d["importance_sampling"]["targets"]
tg = ["TTC_sw < 2 s", "TTC_sw < 1 s"]
x = np.arange(2)
crude_n = [IS[t]["crude_n_for_same_se"] for t in tg]
N_IS = d["importance_sampling"]["n"]
fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.0))
axs[0].bar(x - 0.18, [N_IS, N_IS], 0.34, color=BLUE, label="importance sampling")
axs[0].bar(x + 0.18, crude_n, 0.34, color=CTX, label="crude Monte Carlo, same SE")
for xi, v in zip(x, crude_n):
    axs[0].text(xi + 0.18, v + 120, f"{v:,.0f}", ha="center", fontsize=10.5)
    axs[0].text(xi - 0.18, N_IS + 120, f"{N_IS:,}", ha="center", fontsize=10.5)
axs[0].set_xticks(x, ["P(TTC < 2 s)", "P(TTC < 1 s)"])
axs[0].set_ylabel("Samples for the same standard error")
axs[0].legend(frameon=False, fontsize=10.5, loc="upper left")
axs[0].grid(axis="x", visible=False)
axs[0].set_ylim(0, max(crude_n) * 1.35)
i = IS["TTC_sw < 2 s"]["importance"]
st = [("min", i["w_min"]), ("median", i["w_median"]), ("mean", i["w_mean"]), ("max", i["w_max"])]
axs[1].bar([s for s, _ in st], [v for _, v in st], color=[CTX, CTX, CTX, ORANGE], width=0.55)
axs[1].set_yscale("log")
axs[1].set_ylabel("Importance weight (log scale)")
for j, (_, v) in enumerate(st):
    axs[1].text(j, v * 1.25, f"{v:.3g}", ha="center", fontsize=10.5)
axs[1].set_title(f"ESS {i['ess']:.0f} of {N_IS:,}; largest weight {100 * i['w_max'] / (i['w_mean'] * N_IS):.1f} % of the sum", fontsize=11.5)
axs[1].grid(axis="x", visible=False)
axs[1].set_ylim(1e-3, 200)
save(fig, "fig5_importance_sampling")

# 6 replay spot-check (needs esmini)
runs = d["replay"].get("runs")
if runs:
    it, rt = (np.array([r[c] for r in runs]) for c in ("intended_ttc_sw", "replay_ttc_sw"))
    rep = np.array([r["reproduced"] for r in runs])
    lim = (0, max(it.max(), rt.max()) * 1.08)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(lim, lim, color=GRID, lw=1.2)
    ax.axvline(2.0, color=INK, lw=1, ls=(0, (4, 3)))
    ax.axhline(2.0, color=INK, lw=1, ls=(0, (4, 3)))
    ax.scatter(it[rep], rt[rep], s=70, color=BLUE, edgecolor=SURF, lw=1.5, zorder=3, label=f"class reproduced ({rep.sum()})")
    ax.scatter(it[~rep], rt[~rep], s=90, color=ORANGE, edgecolor=SURF, lw=1.5, zorder=3, marker="D", label=f"class flipped ({(~rep).sum()})")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Intended TTC at lane switch [s]")
    ax.set_ylabel("Replayed TTC in esmini [s]")
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    ax.set_title("Replay spot-check: variants near the 2 s line can change class")
    save(fig, "fig6_replay", label=LABEL + ", simulator replay")
else:
    print("replay figure skipped: no replay runs in results/generation_cutin.json (esmini not found)")
print("figures written to", OUT.name)
