"""Labelling helper: plot each segment's series and mark critical intervals.

    python label.py out/esmini                 # GUI: drag on the plot to mark an interval
    python label.py out/esmini --cli           # no GUI: a PNG per segment + type "t0-t1, t0-t1"
    python label.py out/mock --from-gt         # mock: write the synthetic ground truth as labels

GUI keys:  drag = add interval   u = undo last   c = clear segment   n / → = next   p / ← = previous
           s = save              q = save and quit
Labels are saved after every change to <run dir>/labels.json:

    {"version": 1, "labeller": "...", "updated": "...", "definition": "...",
     "segments": {"<seg id>": [[t0, t1], ...], ...}, "done": ["<seg id>", ...]}

A segment with [] in "done" means "looked at, nothing critical" - different from "not labelled yet".
Label definition shown to the labeller (write yours into the JSON with --definition):
    critical = a stretch in which the ego would need a clearly stronger than normal reaction to avoid a
    collision with its lead, from the onset of the situation until it is resolved (closing speed <= 0).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFINITION = ("critical = a stretch in which the ego would need a clearly stronger than normal reaction to "
              "avoid a collision with its lead, from the onset of the situation until it is resolved "
              "(closing speed <= 0)")


def load_segments(run_dir):
    run_dir = Path(run_dir)
    segs = {}
    for f in sorted((run_dir / "series").glob("*.csv")):
        segs[f.stem] = pd.read_csv(f)
    return segs


def load_labels(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 1, "labeller": "", "updated": "", "definition": DEFINITION, "segments": {}, "done": []}


def save_labels(lab, path):
    lab["updated"] = datetime.now().isoformat(timespec="seconds")
    for k in lab["segments"]:
        lab["segments"][k] = sorted([[round(float(a), 1), round(float(b), 1)] for a, b in lab["segments"][k]])
    Path(path).write_text(json.dumps(lab, indent=1))


def parse_intervals(text):
    """'12.5-15, 30-31.2' -> [[12.5, 15.0], [30.0, 31.2]];  '' or 'none' -> []."""
    text = text.strip().lower()
    if text in ("", "none", "-", "0"):
        return []
    out = []
    for part in text.replace(";", ",").split(","):
        a, b = part.strip().split("-")
        a, b = float(a), float(b)
        if b < a:
            a, b = b, a
        out.append([a, b])
    return out


def draw(fig, ser, seg_id, intervals, i, n):
    fig.clf()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    t = ser.t.to_numpy()
    ax1.plot(t, np.clip(ser.ttc, 0, 10), color="#c0392b", lw=1.2, label="TTC (clipped at 10 s)")
    ax1.plot(t, np.clip(ser.thw, 0, 10), color="#2c7fb8", lw=1.0, label="THW")
    ax1.axhline(3, color="grey", ls=":", lw=0.8)
    ax1.set_ylim(0, 10)
    ax1.set_ylabel("s")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.plot(t, ser.inv_ttc.fillna(0), color="#444", lw=1.0)
    ax2.set_ylabel("1/TTC [1/s]")
    ax2.set_xlabel("t [s]")
    for st, col in (("lane_change", "#f1c40f"), ("no_leader", "#bbb"), ("leader_change", "#8e44ad")):
        for tt in t[ser.status.to_numpy() == st]:
            ax2.axvline(tt, color=col, alpha=0.15, lw=2)
    for a, b in intervals:
        for ax in (ax1, ax2):
            ax.axvspan(a, b, color="#e67e22", alpha=0.3)
    fig.suptitle(f"[{i + 1}/{n}] {seg_id}  - drag to mark, u undo, c clear, n/p next/prev, q quit\n"
                 f"marked: {intervals}", fontsize=9)
    fig.canvas.draw_idle()
    return ax1, ax2


def gui(run_dir, out):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector
    segs = load_segments(run_dir)
    ids = list(segs)
    lab = load_labels(out)
    state = {"i": next((k for k, s in enumerate(ids) if s not in lab["done"]), 0)}
    fig = plt.figure(figsize=(12, 6))
    selectors = []

    def show():
        sid = ids[state["i"]]
        ivs = lab["segments"].setdefault(sid, [])
        axes = draw(fig, segs[sid], sid, ivs, state["i"], len(ids))
        selectors.clear()
        for ax in axes:
            selectors.append(SpanSelector(ax, on_select, "horizontal", useblit=True, minspan=0.2,
                                          props=dict(alpha=0.3, facecolor="#e67e22")))

    def on_select(a, b):
        sid = ids[state["i"]]
        lab["segments"].setdefault(sid, []).append([a, b])
        if sid not in lab["done"]:
            lab["done"].append(sid)
        save_labels(lab, out)
        show()

    def on_key(ev):
        sid = ids[state["i"]]
        if ev.key in ("n", "right"):
            if sid not in lab["done"]:
                lab["done"].append(sid)
            save_labels(lab, out)
            state["i"] = min(state["i"] + 1, len(ids) - 1)
        elif ev.key in ("p", "left"):
            state["i"] = max(state["i"] - 1, 0)
        elif ev.key == "u" and lab["segments"].get(sid):
            lab["segments"][sid].pop()
        elif ev.key == "c":
            lab["segments"][sid] = []
        elif ev.key == "s":
            save_labels(lab, out)
        elif ev.key == "q":
            save_labels(lab, out)
            plt.close(fig)
            return
        save_labels(lab, out)
        show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()
    print(f"saved {out}: {sum(len(v) for v in lab['segments'].values())} intervals, "
          f"{len(lab['done'])}/{len(ids)} segments done")


def cli(run_dir, out, inp=input):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    segs = load_segments(run_dir)
    lab = load_labels(out)
    png_dir = Path(run_dir) / "label_png"
    png_dir.mkdir(exist_ok=True)
    for i, (sid, ser) in enumerate(segs.items()):
        if sid in lab["done"]:
            continue
        fig = plt.figure(figsize=(12, 6))
        draw(fig, ser, sid, [], i, len(segs))
        fig.savefig(png_dir / f"{sid}.png", dpi=80, metadata={"Software": None})
        plt.close(fig)
        ans = inp(f"{sid} (open {png_dir / (sid + '.png')}) intervals 't0-t1, ...' / none / q: ")
        if ans.strip().lower() == "q":
            break
        lab["segments"][sid] = parse_intervals(ans)
        lab["done"].append(sid)
        save_labels(lab, out)
    print(f"saved {out}")


def from_gt(run_dir, out):
    gt = json.loads((Path(run_dir) / "ground_truth.json").read_text())
    lab = load_labels(out)
    lab.update(labeller="synthetic ground truth (mock mode)", segments=gt, done=list(gt))
    save_labels(lab, out)
    print(f"saved {out} from ground_truth.json ({sum(len(v) for v in gt.values())} intervals)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", help="labels file (default <run_dir>/labels.json)")
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--from-gt", action="store_true")
    ap.add_argument("--labeller", default="")
    a = ap.parse_args(argv)
    out = a.out or str(Path(a.run_dir) / "labels.json")
    if a.labeller:
        lab = load_labels(out)
        lab["labeller"] = a.labeller
        save_labels(lab, out)
    if a.from_gt:
        from_gt(a.run_dir, out)
    elif a.cli:
        cli(a.run_dir, out)
    else:
        gui(a.run_dir, out)


if __name__ == "__main__":
    sys.exit(main())
