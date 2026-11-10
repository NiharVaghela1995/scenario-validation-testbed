"""highD pilot: aggregated figures. Writes results/highd/aggregates.json and figures/highd_fig*.png.

    python experiments/highd_figures.py              # aggregates from the per-event table, then plots
    python experiments/highd_figures.py --plot-only  # plots from the committed JSON files; needs no highD data

Disclosure control. Nothing per-vehicle is plotted: no scatter, no time series, no minimum or maximum. Histograms in physical
units drop every bin with fewer than MIN_CELL events (stored as null), and the axes are cut at the 99th-percentile region. The
dependence map is in rank space (deciles), where a cell count carries no physical value.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/highd"
FIG = ROOT / "figures"
MIN_CELL = 10
NOTE = "Data: highD (fka/ika), non-commercial academic licence; results only, no data shared"
ORANGE, BLUE, INK, GREY, MUTED, LIGHT = "#eb6834", "#2a78d6", "#0f2a4a", "#c3cad3", "#6b7686", "#e6eaef"
HIST = {"thw": ("THW at cut-in [s]", np.arange(0, 5.01, 0.2)), "ttc": ("TTC at cut-in [s], closing cut-ins", np.arange(0, 60.1, 2.5)),
        "drac": ("DRAC at cut-in [m/s²], closing cut-ins", np.arange(0, 1.01, 0.05))}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "font.family": "DejaVu Sans"})


def aggregates():
    import pandas as pd
    sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot")]
    import highd_cutins as HC
    from scipy import stats
    ev = pd.read_parquet(HC.work_dir() / "cutins.parquet")
    out = dict(min_cell=MIN_CELL, histograms={})
    for k, (_, edges) in HIST.items():
        x = ev[k].dropna().to_numpy()
        c, _ = np.histogram(x, edges)
        out["histograms"][k] = dict(edges=[round(float(e), 3) for e in edges], counts=[int(v) if v >= MIN_CELL else None for v in c],
                                    n=int(len(x)), outside_or_suppressed=int(len(x) - sum(v for v in c if v >= MIN_CELL)))
    n = len(ev)
    g = np.minimum(np.ceil(stats.rankdata(ev.thw) / n * 10).astype(int), 10) - 1
    d = np.minimum(np.ceil(stats.rankdata(ev.dv) / n * 10).astype(int), 10) - 1
    grid = np.zeros((10, 10), int)
    np.add.at(grid, (g, d), 1)
    out["decile_counts_thw_by_dv"] = grid.tolist()
    both = ev.ttc.notna() & (ev.hd_ttc > 0)
    out["signal_cross_check"] = dict(gap_vs_dhw_abs_p99_m=float((ev.gap - ev.hd_dhw).abs().quantile(0.99)),
                                     thw_abs_p99_s=float((ev.thw - ev.hd_thw).abs().quantile(0.99)),
                                     ttc_relative_abs_p99=float((ev.ttc[both] / ev.hd_ttc[both] - 1).abs().quantile(0.99)),
                                     closing_agreement=float((ev.ttc.notna() == (ev.hd_ttc > 0)).mean()),
                                     lane_crossing_offset_p99_m=float(ev.cross_offset_m.quantile(0.99)))
    (RES / "aggregates.json").write_text(json.dumps(out))


def stamp(fig):
    fig.text(0.995, 0.008, NOTE, ha="right", va="bottom", fontsize=7.5, color=MUTED)


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, metadata={"Software": None})
    plt.close(fig)


def fig1(E):
    f = E["funnel"]
    labels = [("All lane changes", f["lane_changes"]), ("No follower in target lane", -f["no_follower"]),
              ("Cutter track cut off (±1 s)", -f["cutter_track_short"]), ("Cutter unstable in lane", -f["cutter_lane_unstable"]),
              ("Follower track cut off (±1 s)", -f["follower_track_short"]), ("Follower changing lane", -f["follower_lane_change"]),
              ("Follower status excluded", -f["follower_status"]), ("Follower beyond 150 m", -f["gap_out_of_range"]), ("Kept cut-ins", f["kept"])]
    fig, ax = plt.subplots(figsize=(8, 4.1))
    for i, (t, v) in enumerate(labels):
        ax.barh(i, abs(v), color=INK if v > 0 and i else (GREY if v < 0 else MUTED), height=0.62)
        ax.text(abs(v) + 150, i, f"{abs(v):,}", va="center", fontsize=9)
    ax.set_yticks(range(len(labels)), [t for t, _ in labels], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 15500)
    ax.set_xlabel("Lane changes")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{f['kept']:,} of {f['lane_changes']:,} lane changes are usable cut-ins")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    stamp(fig)
    save(fig, "highd_fig1_funnel.png")


def fig2(A):
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, k in zip(axs, ("thw", "ttc", "drac")):
        h, lab = A["histograms"][k], HIST[k][0]
        e, c = np.array(h["edges"]), h["counts"]
        for i, v in enumerate(c):
            if v is not None:
                ax.bar(e[i], v, width=(e[1] - e[0]) * 0.86, align="edge", color=BLUE)
        ax.set_xlabel(lab, fontsize=9)
        ax.grid(axis="y", color=LIGHT, lw=0.8)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("Cut-ins per bin")
    fig.suptitle(f"Most cut-ins are not closing; bins with fewer than {A['min_cell']} events are not drawn", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    stamp(fig)
    save(fig, "highd_fig2_surrogates.png")


def fig3(A, T):
    g = np.array(A["decile_counts_thw_by_dv"], float)
    exp = g.sum() / 100
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(g / exp, origin="lower", cmap="RdBu_r", norm=LogNorm(vmin=0.2, vmax=5), aspect="equal")
    ax.set_xticks(range(10), [f"{i + 1}" for i in range(10)])
    ax.set_yticks(range(10), [f"{i + 1}" for i in range(10)])
    ax.set_xlabel("Closing speed Δv, decile (10 = fastest closing)")
    ax.set_ylabel("THW, decile (1 = shortest)")
    ax.add_patch(plt.Rectangle((8.5, -0.5), 1, 1, fill=False, ec="white", lw=2))
    P = T["primary"]
    ax.annotate(f"{P['joint_extremes']} joint extremes\nof {P['conditioning_events']} conditioning events", xy=(9, 0.45), xytext=(5.0, 2.5), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK), bbox=dict(boxstyle="round", fc="white", ec=LIGHT))
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.2, 0.5, 1, 2, 5]))
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.2", "0.5", "1", "2", "5"]))
    cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_label("Events relative to independence")
    ax.set_title("Short headway and fast closing almost never coincide")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    stamp(fig)
    save(fig, "highd_fig3_rank_corner.png")


def fig4(T):
    P, K3 = T["primary"], T["sensitivity"]["mixture_k3"]
    rows = [("Empirical copula", P["empirical"], P["ci_empirical"], INK), ("Gaussian copula", P["gaussian"], P["ci_gaussian"], ORANGE),
            ("Mixture K = 2", P["mixture"], P["ci_mixture"], BLUE), ("Mixture K = 3 (exploratory)", K3["mixture"], K3["ci_mixture"], BLUE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    for i, (t, v, c, col) in enumerate(rows):
        a.plot(c, [i, i], color=col, lw=2, solid_capstyle="round")
        a.plot(v, i, "o" if i != 3 else "D", ms=8, color=col, mec="white", mew=1.5)
        a.text(c[1] + 0.004, i, f"{v:.3f}", fontsize=9, va="center")
    a.axvline(P["independence_value"], color=MUTED, ls="--", lw=1)
    a.text(P["independence_value"] - 0.002, 0.0, "independence", fontsize=8.5, color=MUTED, ha="right", va="center")
    a.set_yticks(range(4), [r[0] for r in rows], color=INK)
    a.invert_yaxis()
    a.set_xlim(-0.003, 0.115)
    a.set_xlabel("λ(0.10) with 95 % recording-bootstrap interval")
    a.set_title("Primary result")
    names = [("All (primary)", P), ("Manoeuvre start", T["sensitivity"]["pair_at_manoeuvre_start"]), ("THW < 3 s", T["sensitivity"]["thw_below_3s"]),
             ("Closing only", T["sensitivity"]["closing_only"]), ("Cars only", T["sensitivity"]["cars_only"]), ("Location 1", T["sensitivity"]["location_1"]),
             ("Other locations", T["sensitivity"]["other_locations"]), ("Above 60 km/h", T["sensitivity"]["speed_above_60kmh"])]
    for i, (t, r) in enumerate(names):
        b.plot(r["ci_delta_emp_minus_gaussian"], [i - 0.14] * 2, color=ORANGE, lw=2)
        b.plot(r["delta_emp_minus_gaussian"], i - 0.14, "o", color=ORANGE, ms=7, mec="white", mew=1.2, label="vs Gaussian copula" if i == 0 else None)
        b.plot(r["ci_delta_emp_minus_mixture"], [i + 0.14] * 2, color=BLUE, lw=2)
        b.plot(r["delta_emp_minus_mixture"], i + 0.14, "s", color=BLUE, ms=7, mec="white", mew=1.2, label="vs mixture K = 2" if i == 0 else None)
    b.axvline(0, color=MUTED, lw=1)
    b.set_yticks(range(len(names)), [n for n, _ in names], color=INK)
    b.invert_yaxis()
    b.set_xlabel("Empirical minus model λ, 95 % interval")
    b.set_title("Exploratory subsets")
    b.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2)
    b.grid(axis="x", color=LIGHT, lw=0.8)
    b.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    stamp(fig)
    save(fig, "highd_fig4_lambda.png")


def fig5(W):
    obs = str(round(W["lam1"][[abs(l - 0.006098) < 1e-3 for l in W["lam1"]].index(True) + 1], 6))
    ns = W["ns"]
    lines = [(obs, f"true λ = {float(obs):.3f} (observed)", INK, "o"), ("0.04", "true λ = 0.04", BLUE, "s"), ("0.1", "true λ = 0.10", ORANGE, "D"), ("0.218", "true λ = 0.218", MUTED, "^")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.0))
    for k, t, col, mk in lines:
        y = [W["power"][str(n)][k] for n in ns]
        a.plot(ns, y, color=col, lw=2, marker=mk, ms=6, label=t)
    a.set_xscale("log")
    a.axhline(0.8, color=MUTED, ls="--", lw=1)
    a.axvline(W["n"], color=GREY, lw=1)
    a.text(W["n"] * 0.95, 0.03, f"{W['n']:,} events", ha="right", fontsize=8.5, color=MUTED)
    a.set_xlabel("Cut-ins n (independent)")
    a.set_ylabel(f"Power vs Gaussian copula (λ = {W['lam_gauss_exact']:.3f})")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    a.set_title("Large misfits are visible at about 1,400 events")
    ls = W["lam1"]
    for n, t, col, mk in ((W["n"], f"n = {W['n']:,}", INK, "o"), (W["n_eff"], f"n_eff = {W['n_eff']:,} (clustering)", ORANGE, "s")):
        b.plot(ls, [W["power"][str(n)][str(l)] for l in ls], color=col, lw=2, marker=mk, ms=6, label=t)
    b.axvline(W["lam_gauss_exact"], color=MUTED, ls=":", lw=1)
    b.axhline(0.8, color=MUTED, ls="--", lw=1)
    b.set_xlabel("True λ(0.10)")
    b.set_ylabel("Power")
    b.legend(frameon=False, fontsize=9, loc="center right")
    b.set_title("Differences under about 0.03 stay hidden")
    for ax in (a, b):
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", color=LIGHT, lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    stamp(fig)
    save(fig, "highd_fig5_power.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        aggregates()
    A, E, T, W = (json.loads((RES / f"{n}.json").read_text()) for n in ("aggregates", "extraction", "tail", "power"))
    fig1(E)
    fig2(A)
    fig3(A, T)
    fig4(T)
    fig5(W)


if __name__ == "__main__":
    main()
