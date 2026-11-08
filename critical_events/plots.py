"""Plots (matplotlib, Agg): timeline comparison, penalty sweep, matrix profile, DTW pairs."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def save(fig, path):
    """PNG for the run folder + SVG twin ."""
    from pathlib import Path
    fig.savefig(path, dpi=110, metadata={"Software": None})
    fig.savefig(Path(path).with_suffix(".svg"), metadata={"Creator": None, "Date": None, "Format": None, "Type": None})
    plt.close(fig)


COL = {"label": "#222222", "threshold": "#c0392b", "pelt": "#2c7fb8", "discord": "#e67e22"}


def timeline(segs, flags, labels, path, max_segments=12):
    """One row per segment: 1/TTC trace on top, then bars for labels and each detector's flags."""
    segs = segs[:max_segments]
    fig, axes = plt.subplots(len(segs), 1, figsize=(11, 1.15 * len(segs) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    rows = ["label", "threshold", "pelt", "discord"]
    for ax, s in zip(axes, segs):
        t = s["series"].t.to_numpy()
        x = s["signal"]
        ax.plot(t, x / max(x.max(), 1e-6) * 0.9 + len(rows), color="#888", lw=0.8)
        for r, name in enumerate(rows[::-1]):
            ivs = labels.get(s["id"], []) if name == "label" else flags[name].get(s["id"], [])
            for a, b in ivs:
                ax.broken_barh([(a, max(b - a, 0.2))], (r + 0.15, 0.7), color=COL[name])
        ax.set_yticks(range(len(rows) + 1))
        ax.set_yticklabels([])
        ax.set_ylim(0, len(rows) + 1)
        ax.text(1.002, 0.5, s["id"], transform=ax.transAxes, fontsize=7, va="center")
        ax.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[r]) for r in rows]
    fig.legend(handles, ["label (ground truth)", "threshold TTC<3 s", "PELT", "MP discord"], loc="upper center",
               ncol=4, fontsize=8, frameon=False)
    axes[-1].set_xlabel("t [s]   (grey trace: 1/TTC, scaled per segment)")
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    save(fig, path)


def penalty_sweep(sweeps, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for sid, sw in sweeps.items():
        f, n = zip(*sw)
        ax.plot(f, n, marker="o", ms=3, lw=1, label=sid)
    ax.set_xscale("log")
    ax.set_xlabel("penalty factor  (pen = factor · σ² · log n)")
    ax.set_ylabel("number of change points")
    ax.axvline(3, color="grey", ls=":")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, path)


def matrix_profile_example(seg, P, m, discords, bkps, path):
    t = seg["series"].t.to_numpy()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    a1.plot(t, seg["signal"], color="#444", lw=1)
    for b in bkps[:-1]:
        a1.axvline(t[min(b, len(t) - 1)], color=COL["pelt"], lw=1, ls="--")
    for a, b in seg.get("gt", []):
        a1.axvspan(a, b, color="#999", alpha=0.25)
    a1.set_ylabel("1/TTC [1/s]")
    a1.set_title(f"{seg['id']}: PELT change points (dashed), labelled interval (grey)", fontsize=9)
    a2.plot(t[:len(P)], P, color=COL["discord"], lw=1)
    for a, b in discords:
        a2.axvspan(a, b, color=COL["discord"], alpha=0.25)
    a2.set_ylabel(f"matrix profile (m={m})")
    a2.set_xlabel("t [s]")
    fig.tight_layout()
    save(fig, path)


def dtw_pairs(snips, names, pairs, path):
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 2.6))
    for ax, (i, j, d) in zip(np.atleast_1d(axes), pairs):
        ax.plot(snips[i], lw=1.2, label=names[i])
        ax.plot(snips[j], lw=1.2, label=names[j])
        ax.set_title(f"DTW = {d:.2f}", fontsize=9)
        ax.set_xlabel("sample (0.1 s)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, path)
