"""highD robustness: figures/highd_robust_fig*.png and docs/HIGHD_ROBUSTNESS.md from results/highd/robust_*.json (needs no highD data).

    python experiments/highd_robust_report.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from highd_figures import BLUE, GREY, INK, LIGHT, MUTED, NOTE, ORANGE  # noqa: E402

RES, FIG = ROOT / "results/highd", ROOT / "figures"
SC, T, ST, CH, NO, CA, PH, PI = (json.loads((RES / f"{n}.json").read_text()) for n in ("robust_screening", "robust_tail", "robust_stability", "robust_checks", "robust_noise", "robust_candidates", "robust_posthoc", "tail"))
GREEN = "#1baf7a"
MC = dict(threshold=MUTED, pelt=BLUE, discord=GREEN, union=ORANGE, gated=INK)
ML = dict(threshold="Threshold", pelt="PELT", discord="Discords", union="Union", gated="Union + gate")
VC = {"holds": "#cfe0f4", "direction only": LIGHT, "fails": "#f7c9b3", "not interpretable": "#ffffff"}
FAM = dict(gaussian="Gaussian", t="t", clayton_best="Clayton (best rotation)", clayton_corner="Clayton (corner)", gumbel_best="Gumbel (best rotation)", gumbel_corner="Gumbel (corner)",
           k1="Mixture K=1", k2="Mixture K=2", k3="K=3", k4="K=4", k5="K=5", k6="K=6")
PAIR = dict(P1="THW × closing speed", P2="gap × closing speed", P3="THW × lateral speed")
MIN_POS_AUC = 10


def stamp(fig):
    fig.text(0.995, 0.006, NOTE, ha="right", va="bottom", fontsize=7.5, color=MUTED)


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, metadata={"Software": None})
    plt.close(fig)


def fig_screen():
    refs = (("ttc_lt_4", "TTC < 4 s"), ("thw_lt_0.6", "THW < 0.6 s"), ("drac_gt_1", "DRAC > 1 m/s²"))
    fig, axs = plt.subplots(2, 3, figsize=(10.4, 6.2))
    for c, (ref, lab) in enumerate(refs):
        b = SC["vehicle_level"][ref]
        for m, col in MC.items():
            s = b["methods"][m]["sweep"]
            x, r = np.array(s["fpr"], float), np.array(s["recall"], float)
            p = np.array([np.nan if v is None else v for v in s["precision"]], float)
            o = np.argsort(x, kind="stable")
            axs[0, c].plot(x[o], r[o], color=col, lw=2, label=ML[m])
            o2 = np.argsort(r, kind="stable")
            axs[1, c].plot(r[o2], p[o2], color=col, lw=2, label=ML[m])
        axs[0, c].set_xscale("symlog", linthresh=0.01)
        axs[0, c].set(title=f"{lab}: {b['positives']} of {b['positives'] + b['negatives']:,}", xlabel="False positive rate", ylim=(-0.02, 1.05))
        axs[1, c].set(xlabel="Recall", ylim=(-0.02, 1.05), xlim=(-0.02, 1.02))
    axs[0, 0].set_ylabel("Recall (ROC)")
    axs[1, 0].set_ylabel("Precision (PR)")
    axs[0, 2].legend(frameon=False, fontsize=8.5, loc="lower right")
    for ax in axs.flat:
        ax.grid(color=LIGHT, lw=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp(fig)
    save(fig, "highd_robust_fig1_screening.png")


def fig_grid():
    ms = list(FAM)
    rows = [(p, q) for p in ("P1", "P2", "P3") for q in ("0.05", "0.1", "0.15", "0.2")]
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    for i, (p, q) in enumerate(rows):
        L = T[p]["levels"][q]
        for j, m in enumerate(ms):
            v = L["models"][m]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=VC[v["verdict"]], ec="white", lw=1.5))
            ax.text(j + 0.5, i + 0.5, f"{v['delta']:+.3f}", ha="center", va="center", fontsize=7.8)
    ax.set_xlim(0, len(ms))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(ms)) + 0.5, [FAM[m] for m in ms], rotation=35, ha="right", fontsize=8.5, color=INK)
    ax.set_yticks(np.arange(len(rows)) + 0.5, [f"{PAIR[p]}, q = {float(q):.2f}" for p, q in rows], fontsize=8.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("λ empirical minus λ model: negative means the model overstates the corner")
    ax.legend(handles=[Patch(fc=VC[k], ec=GREY, label=k) for k in ("holds", "direction only", "fails")], frameon=False, ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.45, -0.3))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp(fig)
    save(fig, "highd_robust_fig2_families.png")


LABEL = {"stable_0.5s": "Stable window 0.5 s", "stable_2s": "Stable window 2 s", "follower_track_12s": "Follower track ≥ 12 s", "lateral_0.3": "Lateral ≥ 0.3 m/s", "lateral_0.5": "Lateral ≥ 0.5 m/s", "lateral_0.75": "Lateral ≥ 0.75 m/s",
         "no_auxiliary_lanes": "No auxiliary lanes", "leftward": "Leftward", "rightward": "Rightward", "manoeuvre_start_v05": "Manoeuvre start, 0.05 m/s", "manoeuvre_start_v10": "Manoeuvre start, 0.10 m/s", "manoeuvre_start_v20": "Manoeuvre start, 0.20 m/s", "manoeuvre_start_v30": "Manoeuvre start, 0.30 m/s",
         "timing_minus_0.2s": "0.2 s before crossing", "timing_plus_0.2s": "0.2 s after crossing", "location_1": "Location 1", "location_5": "Location 5", "density_below_median": "Density ≤ median", "density_above_median": "Density > median", "density_at_least_22": "Density ≥ 22 veh/km",
         "follower_car": "Follower car", "follower_truck": "Follower truck", "cutter_car": "Cutter car", "cutter_truck": "Cutter truck", "cutter_car_follower_car": "Car cuts in, car follows", "cutter_car_follower_truck": "Car cuts in, truck follows", "cutter_truck_follower_truck": "Truck cuts in, truck follows"}


def interp(items):
    return [(k, v) for k, v in items if v["levels"]["0.1"]["conditioning_events"] >= 50 and v["recordings"] >= 5]


def fig_forest():
    rows = interp(list(T["variants"].items())) + interp(list(ST["strata"].items()))
    fig, ax = plt.subplots(figsize=(8.2, 9.2))
    for i, (k, v) in enumerate(rows):
        M = v["levels"]["0.1"]["models"]
        for m, y, col, mk, lab in (("gaussian", i - 0.16, ORANGE, "o", "vs Gaussian copula"), ("k2", i + 0.16, BLUE, "s", "vs mixture K = 2")):
            ax.plot(M[m]["ci_delta"], [y, y], color=col, lw=1.8)
            ax.plot(M[m]["delta"], y, mk, color=col, ms=5.5, mec="white", mew=0.8, label=lab if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [LABEL[k] for k, _ in rows], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Empirical minus model λ(0.10), 95 % recording-bootstrap interval")
    ax.set_title("Pair THW × closing speed; right of zero the model understates")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    stamp(fig)
    save(fig, "highd_robust_fig3_variants_strata.png")


def fig_checks():
    P = CH["predictive_joint_extremes"]
    ms = list(FAM)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    for i, m in enumerate(ms):
        v = P[m]["0.1"]
        a.plot(v["interval_95"], [i, i], color=GREEN if v["consistent"] else ORANGE, lw=2.4)
        a.plot(v["mean"], i, "o", color=INK, ms=4)
    obs = P["gaussian"]["0.1"]["observed"]
    a.axvline(obs, color=INK, ls="--", lw=1)
    a.text(obs + 1.5, -0.45, f"observed {obs}", fontsize=8.5)
    a.set_yticks(range(len(ms)), [FAM[m] for m in ms], fontsize=8.5, color=INK)
    a.invert_yaxis()
    a.set_xlabel("Predicted joint extremes at q = 0.10 (95 %, mean)")
    a.set_title("Predicted counts against observed")
    q = CH["qq"]
    x = np.array(q["data"])
    b.fill_between(x, q["data_band"][0], q["data_band"][1], color=LIGHT, label="Data, recording-bootstrap band")
    b.plot([x.min(), x.max()], [x.min(), x.max()], color=INK, lw=1.2, ls="--", label="Identity")
    for m, col, lab in (("gaussian", ORANGE, "Gaussian"), ("k2", BLUE, "Mixture K = 2"), ("k3", GREEN, "Mixture K = 3")):
        b.plot(x, q["models"][m], color=col, lw=2, label=lab)
    b.set_xlabel("Data quantile of T (rank scale)")
    b.set_ylabel("Model quantile of T")
    b.set_title("Tail QQ, lowest 10 % of T")
    b.legend(frameon=False, fontsize=8.5)
    b.grid(color=LIGHT, lw=0.7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    stamp(fig)
    save(fig, "highd_robust_fig4_model_checks.png")


def fig_noise():
    L = NO["levels"]
    keys = list(L)
    labs = [f"{L[k]['sigma_pos_m']} m, {L[k]['sigma_v_ms']} m/s" for k in keys]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    for s, col, mk, lab in (("ttc_lt_4", INK, "o", "TTC < 4 s"), ("thw_lt_0_6", BLUE, "s", "THW < 0.6 s"), ("drac_gt_2", ORANGE, "D", "DRAC > 2 m/s²")):
        a.plot(range(len(keys)), [L[k]["critical_sets"][s]["jaccard_mean"] for k in keys], color=col, marker=mk, lw=2, label=lab)
    a.axhline(0.5, color=MUTED, ls="--", lw=1)
    a.set_xticks(range(len(keys)), labs, fontsize=8.5)
    a.set(ylim=(0, 1.02), ylabel="Overlap with unperturbed set (Jaccard)", title="Critical sets under noise")
    a.legend(frameon=False, fontsize=9, loc="lower left")
    for i, k in enumerate(keys):
        b.plot(L[k]["ci_delta_gaussian"], [i - 0.13] * 2, color=ORANGE, lw=2)
        b.plot(L[k]["delta_gaussian"], i - 0.13, "o", color=ORANGE, mec="white", label="vs Gaussian copula" if i == 0 else None)
        b.plot(L[k]["ci_delta_k2"], [i + 0.13] * 2, color=BLUE, lw=2)
        b.plot(L[k]["delta_k2"], i + 0.13, "s", color=BLUE, mec="white", label="vs mixture K = 2" if i == 0 else None)
    b.axvline(0, color=MUTED, lw=1)
    b.set_yticks(range(len(keys)), labs, fontsize=8.5, color=INK)
    b.invert_yaxis()
    b.set_xlabel("Empirical minus model λ(0.10)")
    b.set_title("Headline under noise")
    b.legend(frameon=False, fontsize=9, loc="center right")
    for ax in (a, b):
        ax.grid(color=LIGHT, lw=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    stamp(fig)
    save(fig, "highd_robust_fig5_noise.png")


def f3(v):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def f2(v):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def ci(c):
    return f"{c[0]:.3f} to {c[1]:.3f}"


def lvl(key):
    return T[key]["levels"]


P1 = lvl("P1")
q0 = "0.1"
V, S = T["variants"], ST["strata"]
all_v = dict(interp(list(V.items())))
all_s = dict(interp(list(S.items())))
hc = lambda d, m: sum(v["levels"][q0]["models"][m]["verdict"] == "holds" for v in d.values())
fails = lambda d, m: [LABEL[k] for k, v in d.items() if v["levels"][q0]["models"][m]["verdict"] == "fails"]
NV, NSA = len(all_v), len(all_s)
PRE = ("gaussian", "t", "clayton_best", "gumbel_best", "k2")
LOO = ST["leave_one_recording_out"]
PP = CH["predictive_joint_extremes"]
sc_gate = SC["cutin_level"]["ttc_lt_4"]["methods"]["gated"]
sc_thw = SC["cutin_level"]["thw_lt_0.6"]["methods"]["gated"]
scv = SC["vehicle_level"]
frag_g, frag_m = (NV - hc(all_v, "gaussian")) / NV > 0.25, (NV - hc(all_v, "k2")) / NV > 0.25
noise_low = NO["levels"]["0.3/0.3"]
truck, tv = S["follower_truck"]["levels"][q0], PH["by_follower_class"]["Truck"]
car = S["follower_car"]["levels"][q0]
p3 = lvl("P3")[q0]


def verdicts(L):
    return f"Gaussian {L['models']['gaussian']['verdict']}, K = 2 {L['models']['k2']['verdict']}"


def p1_all(m):
    return all(P1[q]["models"][m]["verdict"] == "holds" for q in P1)


def auc_txt(ref, m):
    b = scv[ref]
    return f2(b["methods"][m]["roc_auc"]) if b["positives"] >= MIN_POS_AUC else "n/a"


rows = [
    ("1. Screening, vehicle level (control sample)", f"Against TTC < 4 s ({scv['ttc_lt_4']['positives']} of {SC['vehicles_control']:,} vehicles), union + gate has recall {f2(scv['ttc_lt_4']['methods']['gated']['recall'])} and precision {f2(scv['ttc_lt_4']['methods']['gated']['precision'])}; against THW < 0.6 s recall {f2(scv['thw_lt_0.6']['methods']['gated']['recall'])}, precision {f2(scv['thw_lt_0.6']['methods']['gated']['precision'])}.", "Screening statement holds. Most flags are outside the TTC < 4 s reference."),
    ("1. Screening, cut-in level", f"Union + gate recall {f2(sc_gate['recall'])} at TTC < 4 s, {f3(sc_thw['recall'])} at THW < 0.6 s.", f"Holds under the plan's limits (recall ≥ 0.9 and ≤ 0.5)."),
    ("2. Level q on THW × closing speed", f"Four levels: Gaussian holds in {sum(P1[q]['models']['gaussian']['verdict'] == 'holds' for q in P1)}, K = 2 in {sum(P1[q]['models']['k2']['verdict'] == 'holds' for q in P1)}.", "Yes"),
    ("2. Pair gap × closing speed", f"q = 0.10: λ {f3(lvl('P2')[q0]['empirical'])}, Gaussian {f3(lvl('P2')[q0]['models']['gaussian']['lam'])}, K = 2 {f3(lvl('P2')[q0]['models']['k2']['lam'])}.", f"Yes: {verdicts(lvl('P2')[q0])}."),
    ("2. Pair THW × lateral speed", f"q = 0.10: λ {f3(p3['empirical'])} (above independence), Gaussian {f3(p3['models']['gaussian']['lam'])}, K = 2 {f3(p3['models']['k2']['lam'])}.", f"**No.** {verdicts(p3)}. This corner is populated."),
    ("2. Families on THW × closing speed", "Gaussian, t, Clayton and Gumbel (best rotation) and K = 2: " + ("all overstate at q = 0.10." if all(P1[q0]['models'][m]['verdict'] == 'holds' for m in PRE) else "not all overstate.") + f" K = 3 to 6 do not: Δ {f3(P1[q0]['models']['k3']['delta'])} to {f3(P1[q0]['models']['k6']['delta'])}.", "Yes for the pre-stated families; not for K ≥ 3, as expected."),
    ("2. Cut-in definition variants", f"{NV} interpretable. Gaussian holds in {hc(all_v, 'gaussian')}, K = 2 in {hc(all_v, 'k2')}; K = 2 fails: {', '.join(fails(all_v, 'k2')) or 'none'}.", "Yes, not fragile under the plan's quarter rule." if not (frag_g or frag_m) else "Fragile under the plan's rule."),
    ("3. Leave one recording out", f"Over all {LOO['recordings']} removals, Δ Gaussian {ci(LOO['delta_gaussian_range'])}, Δ K = 2 {ci(LOO['delta_mixture_k2_range'])}.", "Yes" if LOO["removals_with_delta_gaussian_ge_0"] + LOO["removals_with_delta_mixture_ge_0"] == 0 else "No"),
    ("3. Per location", f"Interpretable: 1 ({verdicts(S['location_1']['levels'][q0])}) and 5 ({verdicts(S['location_5']['levels'][q0])}). Locations 2, 3, 4 and 6 have too few recordings or events.", "Yes where interpretable; four of six sites cannot be judged."),
    ("3. Free flow against dense", f"Below median density: {verdicts(S['density_below_median']['levels'][q0])}. Above: {verdicts(S['density_above_median']['levels'][q0])}. At least 22 veh/km: {verdicts(S['density_at_least_22']['levels'][q0])}.", "Yes"),
    ("3. Car against truck", f"Follower car: λ {f3(car['empirical'])}, {verdicts(car)}. Follower truck: λ {f3(truck['empirical'])} (interval {ci(truck['ci_empirical'])}), {verdicts(truck)}.", "**No for truck followers**, a trigger under the plan. See the post hoc note."),
    ("4. Predictive joint extremes", f"Observed {PP['gaussian']['0.1']['observed']} at q = 0.10. Gaussian predicts {PP['gaussian']['0.1']['interval_95'][0]:.0f} to {PP['gaussian']['0.1']['interval_95'][1]:.0f}, K = 2 {PP['k2']['0.1']['interval_95'][0]:.0f} to {PP['k2']['0.1']['interval_95'][1]:.0f}. Consistent models: {', '.join(FAM[m] for m in FAM if PP[m]['0.1']['consistent'])}.", "Yes: neither pre-specified model is consistent with the data."),
    ("5. Measurement noise", f"Up to (0.3 m, 0.3 m/s): {'Gaussian ' + noise_low['verdict_gaussian']}, {'K = 2 ' + noise_low['verdict_k2']}. TTC < 4 s set overlap {f2(NO['levels']['0.1/0.1']['critical_sets']['ttc_lt_4']['jaccard_mean'])} at (0.1, 0.1), {f2(noise_low['critical_sets']['ttc_lt_4']['jaccard_mean'])} at (0.3, 0.3).", "Yes"),
]
TAB = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
DET = "\n".join(f"| {ML[m]} | {auc_txt('ttc_lt_4', m)} | {f2(scv['ttc_lt_4']['methods'][m]['recall'])} | {f2(scv['ttc_lt_4']['methods'][m]['precision'])} | {f2(scv['ttc_lt_4']['methods'][m]['miss_rate'])} | {auc_txt('thw_lt_0.6', m)} | {f2(scv['thw_lt_0.6']['methods'][m]['recall'])} | {f2(scv['thw_lt_0.6']['methods'][m]['precision'])} | {f2(scv['thw_lt_0.6']['methods'][m]['miss_rate'])} |" for m in ML)
CONF = "\n".join(f"| {ML[m]} | " + " | ".join(str(scv[r]['methods'][m]['confusion'][k]) for r in ("ttc_lt_4", "thw_lt_0.6") for k in ("tp", "fp", "fn", "tn")) + " |" for m in ML)
DRAC = ", ".join(f"DRAC > {r[-1]} m/s²: {scv[r]['positives']}" for r in ("drac_gt_1", "drac_gt_2", "drac_gt_3"))
FAM_ROWS = "\n".join(f"| {FAM[m]} | {f3(P1[q0]['models'][m]['lam'])} | {f3(P1[q0]['models'][m]['delta'])} ({ci(P1[q0]['models'][m]['ci_delta'])}) | {P1[q0]['models'][m]['verdict']} | {PP[m]['0.1']['interval_95'][0]:.0f} to {PP[m]['0.1']['interval_95'][1]:.0f} | {'yes' if PP[m]['0.1']['consistent'] else 'no'} |" for m in FAM)
NOISE_ROWS = "\n".join(f"| {L['sigma_pos_m']} m, {L['sigma_v_ms']} m/s | {L['critical_sets']['ttc_lt_4']['count_mean']:.1f} | {f2(L['critical_sets']['ttc_lt_4']['jaccard_mean'])} | {L['critical_sets']['ttc_lt_2']['count_mean']:.1f} | {f2(L['critical_sets']['thw_lt_0_6']['jaccard_mean'])} | {f2(L['critical_sets']['drac_gt_2']['jaccard_mean'])} | {f3(L['lam_empirical_mean'])} | {L['verdict_gaussian']} | {L['verdict_k2']} |" for L in NO["levels"].values())
bc = NO["baseline_counts"]

H1_FAIL = ", ".join(LABEL.get(k, k.replace("_", " ")) for d in (V, S) for k, v in d.items() if not v["levels"][q0]["h1_below_independence"])
H1_P3 = sum(not L["h1_below_independence"] for L in lvl("P3").values())

DOC = f"""# highD pilot: robustness analyses

*Generated by `experiments/highd_robust_report.py` from `results/highd/robust_*.json`. Plan committed before any analysis ran: [`docs/HIGHD_ANALYSIS_PLAN.md`](HIGHD_ANALYSIS_PLAN.md). Data: highD, used under its non-commercial licence; results only, no data shared. Cite Krajewski et al., ITSC 2018, doi:10.1109/ITSC.2018.8569552.*

**Result.** On pooled cut-ins the headline survives the planned stresses of the (THW, closing speed) corner: level, gap in metres, leave-one-out, density and noise, and all definition variants except that the two-component mixture does not overstate it for leftward lane changes. It does not generalise in two ways, and the plan counts both as changing the conclusion. For **truck followers** the corner is not avoided (λ(0.10) = {f3(truck['empirical'])}) and neither model overstates it. For the pair **THW × lateral speed** the corner is populated (λ(0.10) = {f3(p3['empirical'])}) and the two-component mixture understates it. The claim should read: *for the pooled corner of short headway and fast closing, the Gaussian copula and the two-component mixture overstate the tail.*

## Summary

| Analysis | Result | Does the headline hold? |
|---|---|---|
{TAB}

H1 (corner avoided) is checked per configuration in the JSON files. Its interval is not below independence for: {H1_FAIL}, and for {H1_P3} of the 4 levels of THW × lateral speed.

## 1. Screening as a classifier

The references come from the same signals as the detectors, and the gate tests TTC < 4 s or THW < 0.6 s directly, so this measures agreement with a rule and not independent validity. Vehicle-level positives in the control sample of {SC['vehicles_control']:,}: TTC < 4 s: {scv['ttc_lt_4']['positives']}; THW < 0.6 s: {scv['thw_lt_0.6']['positives']}; {DRAC}. The DRAC references have too few positives to read: areas are not shown below 10 positives (a post hoc rule).

| Method | TTC ROC AUC | Recall | Precision | Miss | THW ROC AUC | Recall | Precision | Miss |
|---|---|---|---|---|---|---|---|---|
{DET}

Confusion matrices (vehicle level), TP / FP / FN / TN:

| Method | TTC < 4 TP | FP | FN | TN | THW < 0.6 TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
{CONF}

Intervals for each rate, the cut-in-level tables and all sweeps are in `results/highd/robust_screening.json`. Threshold sweep curves are below, and the discord detector's recall saturates because its relative criterion does not scale with the factor.

![ROC and PR curves for five methods against three references. Aggregated vehicle counts.](../figures/highd_robust_fig1_screening.png)

## 2. Tail sensitivity

Cells show λ empirical minus λ model, so a negative value means the model overstates the corner, coloured by the plan's verdict.

![Grid of differences by pair, level and model family.](../figures/highd_robust_fig2_families.png)

Primary pair, q = 0.10 (empirical λ {f3(P1[q0]['empirical'])}, interval {ci(P1[q0]['ci_empirical'])}, {P1[q0]['joint_extremes']} joint extremes of {P1[q0]['conditioning_events']}):

| Model | λ | Empirical − model (95 %) | Verdict | Predicted joint extremes | Consistent with data |
|---|---|---|---|---|---|
{FAM_ROWS}

Auxiliary lanes under the plan's rule: {CA['recordings_with_auxiliary_lane']} recordings ({', '.join(CA['auxiliary_lanes_by_recording'])}). I did not check these against site maps. {CA['primary_from_aux_lane']} cut-ins start in one, and none end in one, so the only-from-auxiliary variant is not interpretable.

## 3. Stability

![Differences by definition variant and stratum, with 95 % recording-bootstrap intervals. Only interpretable configurations.](../figures/highd_robust_fig3_variants_strata.png)

Leave-one-recording-out: λ empirical ranges {ci(LOO['empirical_range'])}, and no removal gives Δ ≥ 0 for either model. The density median is {ST['density_median_veh_km']:.1f} veh/km.

**Post hoc note on trucks (not in the plan).** λ is rank-based inside each stratum. For truck followers the top decile of closing speed starts at {tv['own_dv_top_decile_starts_ms']:.2f} m/s, which is negative, and only {tv['share_closing'] * 100:.1f} % of them are closing at crossing. So that stratum's corner is short headway with the *least-opening* cut-ins, not fast closing. At the pooled thresholds (THW ≤ {PH['pooled']['thw_lowest_decile_ends_s']:.2f} s, dv > {PH['pooled']['dv_top_decile_starts_ms']:.2f} m/s) truck followers have {tv['pooled_joint_extremes']} joint extremes in {tv['pooled_conditioning_events']} conditioning events, and {PH['by_follower_class']['Car']['pooled_conditioning_events']} of the pooled conditioning events have car followers. The same applies to rightward lane changes ({PH['direction']['rightward']['share_closing'] * 100:.1f} % closing). The stratum failures are therefore mainly a statement about how a within-stratum rank corner is defined. I report them as failures because the plan does, and because the pooled figure is a statement about a car-dominated fast-closing corner.

## 4. Model checks

Predictive counts use the recording-bootstrap fits as an approximate posterior, which is not a formal Bayesian posterior. Neither the Gaussian copula nor the two-component mixture is consistent with the observed count; mixtures with K ≥ 3 are, which shows how flexible they are and is not evidence that any of them is the right model.

![Predicted joint extremes per model, and a tail QQ plot in rank space.](../figures/highd_robust_fig4_model_checks.png)

## 5. Measurement noise

Baseline critical sets: TTC < 4 s {bc['ttc_lt_4']}, TTC < 2 s {bc['ttc_lt_2']}, THW < 0.6 s {bc['thw_lt_0_6']:,}, DRAC > 2 m/s² {bc['drac_gt_2']}.

| Noise (position, speed) | TTC < 4 s (mean count) | Jaccard | TTC < 2 s (mean count) | THW < 0.6 s Jaccard | DRAC > 2 Jaccard | λ emp | Gaussian | K = 2 |
|---|---|---|---|---|---|---|---|---|
{NOISE_ROWS}

![Critical-set overlap and the headline under noise.](../figures/highd_robust_fig5_noise.png)

Small critical sets are sensitive: the TTC < 4 s set has an overlap of {f2(noise_low['critical_sets']['ttc_lt_4']['jaccard_mean'])} at (0.3, 0.3) and {f2(NO['levels']['0.5/1.0']['critical_sets']['ttc_lt_4']['jaccard_mean'])} at the stress level. The tail conclusion is not.

## Deviations from the plan and post hoc items

- Noise: critical sets use {NO['monte_carlo_draws']} draws as planned. λ intervals add {NO['bootstrap_replicates']} replicates that each combine a recording bootstrap with a fresh noise draw, so that "all intervals by recording" holds.
- A stable window of 1 s is the primary and was not re-run. No cut-in ends in an auxiliary lane, so that variant has no events.
- ROC and PR areas are not shown when a reference has fewer than {MIN_POS_AUC} positives (post hoc).
- The truck diagnostic (`experiments/highd_robust_posthoc.py`) is post hoc.
- B = 400 throughout, with no reduction.
- Already seen before the plan: the primary result, q = 0.05 and 0.20, K = 3, and some subsets. These are not confirmatory.

## Limits

No multiple-testing correction: counts of configurations are descriptive. Recordings are correlated within a site. The noise model is independent Gaussian at the crossing frame, with assumed levels. References for screening share signals with the detectors. The auxiliary-lane rule is untested against site maps. Model fits are bivariate.

## Traceability

| Numbers | Result file | Script |
|---|---|---|
| Definition variants, density, auxiliary lanes | `results/highd/robust_candidates.json` and tables outside the repository | `pilot/highd_candidates.py` |
| Confusion matrices, ROC, PR | `results/highd/robust_screening.json` | `experiments/highd_robust_screening.py` |
| Levels, pairs, families, variants | `results/highd/robust_tail.json` | `experiments/highd_robust_tail.py` |
| Leave-one-out, locations, density, classes | `results/highd/robust_stability.json` | `experiments/highd_robust_tail.py` |
| Predictive counts, tail QQ | `results/highd/robust_checks.json` | `experiments/highd_robust_tail.py` |
| Noise | `results/highd/robust_noise.json` | `experiments/highd_robust_noise.py` |
| Post hoc truck diagnostic | `results/highd/robust_posthoc.json` | `experiments/highd_robust_posthoc.py` |
| All tables and figures in this report | The files above | `experiments/highd_robust_report.py` |
"""


def main():
    for fn in (fig_screen, fig_grid, fig_forest, fig_checks, fig_noise):
        fn()
    (ROOT / "docs/HIGHD_ROBUSTNESS.md").write_text(DOC)
    print(len(DOC.split()), "words")


if __name__ == "__main__":
    main()
