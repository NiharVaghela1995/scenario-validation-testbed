"""Screening on a real-data frames table: signals, three detectors, union and physical gate, per ego vehicle.

    python pilot/screening.py --dataset ngsim --path data/raw/ngsim/us-101.csv --location us-101 --out results/pilot/us-101.csv
    python pilot/screening.py --dataset highd --path data/raw/highd/01_tracks.csv --out results/pilot/highd_01.csv
    python pilot/screening.py --dataset av2 --path data/raw/av2/<log_id> --ego 0 --out results/pilot/av2.csv

Candidates are a queue for human review, not labelled events. Outputs stay in the git-ignored results/pilot/ folder: check
the dataset's licence before publishing anything derived from them.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot")]
import detectors as D  # noqa: E402
import fusion as F  # noqa: E402
import loaders as L  # noqa: E402
from series import compute, detector_signal, status_counts  # noqa: E402

MIN_DURATION_S = 12.0     # the discord window is 4 s; shorter trajectories are not screened


def screen_ego(frames, ego_id):
    """-> (candidate rows, status counts) for one ego vehicle of a 10 Hz frames table."""
    ser = compute(frames, ego_id)
    flags, _ = D.run_all(ser, detector_signal(ser))
    union = F.union_flags({k: {ego_id: v} for k, v in flags.items()}, [ego_id])[ego_id]
    kept = F.physical_gate(union, ser)
    rows = []
    for a, b in kept:
        w = ser[(ser.t >= a) & (ser.t <= b)]
        rows.append(dict(ego=ego_id, t0=a, t1=b, min_ttc=float(w.ttc.min()), min_thw=float(w.thw.min()),
                         max_drac=float(w.drac.max()),
                         detectors="+".join(k for k, v in flags.items() if any(x <= b and y >= a for x, y in v))))
    return rows, status_counts(ser)


def screen(frames, ego_ids=None, max_egos=None):
    """-> (DataFrame of gated candidates, dict with status totals and the number of egos screened)."""
    span = frames.groupby("vid").t.agg(lambda t: t.max() - t.min())
    if ego_ids is None:
        ego_ids = sorted(span[span >= MIN_DURATION_S].index)
    ego_ids = list(ego_ids)[:max_egos]
    rows, totals = [], {s: 0 for s in ("ok", "leader_change", "opening", "no_leader", "lane_change", "collision")}
    for ego in ego_ids:
        r, c = screen_ego(frames, ego)
        rows += r
        for k, v in c.items():
            totals[k] += v
    cols = ["ego", "t0", "t1", "min_ttc", "min_thw", "max_drac", "detectors"]
    return pd.DataFrame(rows, columns=cols), dict(egos=len(ego_ids), status=totals)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("ngsim", "highd", "av2"), required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--location", help="NGSIM location, for example us-101")
    ap.add_argument("--ego", type=int, action="append", help="ego vehicle id (repeatable); default: all long enough")
    ap.add_argument("--max-egos", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.dataset == "ngsim":
        frames = L.load_ngsim(a.path, a.location)
    elif a.dataset == "highd":
        frames = L.resample(L.load_highd(a.path))
    else:
        frames = L.resample(L.load_av2(a.path))
    print("quality:", L.quality_report(frames))
    cand, summary = screen(frames, a.ego, a.max_egos)
    print("screened:", summary, "| candidates:", len(cand))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(out, index=False)


if __name__ == "__main__":
    main()
