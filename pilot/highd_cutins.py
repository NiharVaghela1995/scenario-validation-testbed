"""highD pilot, step 1 and 2: lane-change cut-ins from all recordings, surrogate measures at cut-in, screening.

    python pilot/highd_cutins.py                 # all recordings found in $HIGHD_DATA (default ~/datasets/highD/data)
    python pilot/highd_cutins.py --recordings 1 2 --no-screening

CUT-IN DEFINITION. At frame f_c the cutter C enters target lane L1 (first frame of C with laneId = L1; laneId is highD's
own assignment), and a follower F is behind it in L1. An event is kept only if all of these hold (see FUNNEL):
    follower       C's followingId at f_c is a vehicle F, and F's precedingId at f_c is C (highD's own pointers agree)
    cutter stable  C is in the origin lane for the STABLE_S before f_c and in L1 for the STABLE_S after it
                   (drops double lane changes and manoeuvres cut off by the end of the 420 m field of view)
    follower lane  F is visible and stays in L1 for f_c +- STABLE_S (same window as series.LC_WINDOW)
    new leader     series.compute reports C as F's lead at f_c, and F's status is not "lane_change" or "collision"
    range          0 < gap <= series.MAX_RANGE at f_c
    moving         v_F > series.V_MIN at f_c (THW defined)

MEASURES at f_c come from critical_events/series.py on the native 25 Hz frames of pilot/loaders.py: gap, THW, TTC (NaN when
not closing), DRAC. dv = v_F - v_C (> 0: closing). highD's own dhw/thw/ttc columns are kept only as a cross-check.
MANOEUVRE START t0: walking back from f_c, the last frame at which C's lateral speed toward L1 exceeds V_LAT_START,
capped at T0_MAX_S; NaN if the walk reaches the cap or the start of the track.

SCREENING runs pilot/screening.py's detectors, union and physical gate unchanged, on 10 Hz frames, for vehicles visible
for at least screening.MIN_DURATION_S: every follower of a kept cut-in (catch rate) and a seeded random CONTROL_SHARE of all
such vehicles (population flag rates; about 6 CPU-hours would be needed for all of them).

LICENCE. The per-event table is extracted per-vehicle data. It is written only to $HIGHD_WORK (default
~/.cache/scenario-validation-testbed/highd), never inside this repository; the code refuses a path inside it. Only
aggregated counts go to results/highd/.
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __name__ == "__main__":                 # as a script only: importing this module must not change another process's threads
    for _var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot")]
import detectors as D  # noqa: E402
import fusion as FU  # noqa: E402
import loaders as L  # noqa: E402
import screening as SC  # noqa: E402
import series as SE  # noqa: E402

STABLE_S = SE.LC_WINDOW  # s
V_LAT_START = 0.10       # m/s
T0_MAX_S = 6.0           # s
CONTROL_SHARE = 0.10     # share of screenable vehicles drawn at random, per recording, for population flag rates
MATCH_S = 1.0            # s: a screening interval "catches" a cut-in if it starts or ends within this of f_c, or spans it

FUNNEL = ("lane_changes", "no_follower", "pointer_mismatch", "cutter_track_short", "cutter_lane_unstable",
          "follower_track_short", "follower_lane_change", "lead_mismatch", "follower_status", "gap_out_of_range",
          "too_slow", "kept")
USECOLS = ["frame", "id", "y", "height", "yVelocity", "dhw", "thw", "ttc", "precedingId", "followingId", "laneId"]


def data_dir(arg=None):
    return Path(arg or os.environ.get("HIGHD_DATA", Path.home() / "datasets/highD/data")).expanduser()


def work_dir(arg=None):
    """Directory for per-event tables. Must be outside the repository: extracted per-vehicle data is never stored in it."""
    w = Path(arg or os.environ.get("HIGHD_WORK", Path.home() / ".cache/scenario-validation-testbed/highd")).expanduser().resolve()
    if w == ROOT or ROOT in w.parents:
        raise ValueError(f"{w} is inside the repository; per-vehicle tables must stay outside it")
    w.mkdir(parents=True, exist_ok=True)
    return w


def recordings(data):
    return sorted(int(p.name.split("_")[0]) for p in Path(data).glob("*_tracks.csv"))


def _arrays(df):
    return {k: {c: g[c].to_numpy() for c in g.columns} for k, g in df.groupby("id", sort=False)}


def extract_recording(rec, data):
    """-> (DataFrame with one row per kept cut-in, funnel counts, cross-check counts) for one recording."""
    p = Path(data) / f"{rec:02d}_tracks.csv"
    meta = pd.read_csv(p.with_name(f"{rec:02d}_tracksMeta.csv")).set_index("id")
    info = pd.read_csv(p.with_name(f"{rec:02d}_recordingMeta.csv")).iloc[0]
    hz = float(info.frameRate)
    marks = np.array([float(m) for key in ("upperLaneMarkings", "lowerLaneMarkings") for m in str(info[key]).split(";")])
    tr = _arrays(pd.read_csv(p, usecols=USECOLS).sort_values(["id", "frame"]))
    frames = L.load_highd(p)                       # native 25 Hz, loader unchanged
    pos = _arrays(frames.assign(id=frames.vid).sort_values(["id", "t"]))
    t_all = frames.t.to_numpy()
    n_st = int(round(STABLE_S * hz))
    count = dict.fromkeys(FUNNEL, 0)
    rows, gaps_check = [], dict(meta_lane_changes=int(meta.numLaneChanges.sum()), non_contiguous_tracks=0)
    for c_id in meta.index[meta.numLaneChanges > 0]:
        c = tr[c_id]
        if np.any(np.diff(c["frame"]) != 1):
            gaps_check["non_contiguous_tracks"] += 1
        for i in np.flatnonzero(np.diff(c["laneId"])) + 1:
            count["lane_changes"] += 1
            f_c, l0, l1 = int(c["frame"][i]), int(c["laneId"][i - 1]), int(c["laneId"][i])
            f_id = int(c["followingId"][i])
            if f_id == 0:
                count["no_follower"] += 1
                continue
            fo = tr[f_id]
            j = f_c - int(fo["frame"][0])
            if not (0 <= j < len(fo["frame"])) or int(fo["precedingId"][j]) != c_id:
                count["pointer_mismatch"] += 1
                continue
            if i < n_st or i + n_st > len(c["frame"]):
                count["cutter_track_short"] += 1
                continue
            if np.any(c["laneId"][i - n_st:i] != l0) or np.any(c["laneId"][i:i + n_st] != l1):
                count["cutter_lane_unstable"] += 1
                continue
            if j < n_st or j + n_st >= len(fo["frame"]):
                count["follower_track_short"] += 1
                continue
            if np.any(fo["laneId"][j - n_st:j + n_st + 1] != l1):
                count["follower_lane_change"] += 1
                continue
            t_c = f_c / hz
            lo, hi = np.searchsorted(t_all, [t_c - STABLE_S - 0.2, t_c + STABLE_S + 0.2])
            win = frames.iloc[lo:hi]
            ser = SE.compute(win[(win.vid == f_id) | (win.lane == l1) | (win.vid == c_id)], f_id)
            r = ser.iloc[int(np.argmin(np.abs(ser.t.to_numpy() - t_c)))]
            if r.lead_id != c_id:
                count["gap_out_of_range" if float(fo["dhw"][j]) > SE.MAX_RANGE else "lead_mismatch"] += 1
                continue
            if r.status in ("lane_change", "collision"):
                count["follower_status"] += 1
                continue
            if not (0 < r.gap <= SE.MAX_RANGE):
                count["gap_out_of_range"] += 1
                continue
            if not r.v_ego > SE.V_MIN:
                count["too_slow"] += 1
                continue
            count["kept"] += 1
            row = dict(rec=rec, cutter=int(c_id), follower=f_id, f_c=f_c, location=int(info.locationId),
                       cutter_class=meta.loc[c_id, "class"], follower_class=meta.loc[f_id, "class"],
                       direction=int(meta.loc[c_id, "drivingDirection"]),
                       v_ego=float(r.v_ego), v_cut=float(r.v_lead), gap=float(r.gap), dv=float(r.v_ego - r.v_lead),
                       thw=float(r.thw), ttc=float(r.ttc), drac=float(r.drac), lead_switch=bool(r.lead_switch),
                       hd_dhw=float(fo["dhw"][j]), hd_thw=float(fo["thw"][j]), hd_ttc=float(fo["ttc"][j]),
                       cross_offset_m=float(np.min(np.abs(marks - (c["y"][i] + c["height"][i] / 2)))))
            row.update(_manoeuvre_start(c, pos[c_id], fo, pos[f_id], i, f_c, hz))
            rows.append(row)
    return pd.DataFrame(rows), count, gaps_check, float(info.duration), int(info.numVehicles)


def _manoeuvre_start(c, c_pos, fo, f_pos, i, f_c, hz, v_lat_start=V_LAT_START):
    out = dict(v_lat_cross=float(abs(c["yVelocity"][i])), t_manoeuvre=np.nan, v_lat_mean=np.nan, v_ego0=np.nan, gap0=np.nan,
               dv0=np.nan)
    n_back, n_cap = int(round(hz)), int(round(T0_MAX_S * hz))
    side = np.sign(c["y"][i] - c["y"][i - n_back])
    k = i
    while k > 0 and i - k < n_cap and side * c["yVelocity"][k - 1] > v_lat_start:
        k -= 1
    if k == 0 or i - k >= n_cap:
        return out
    out.update(t_manoeuvre=(i - k) / hz, v_lat_mean=float(abs(c["y"][i] - c["y"][k]) * hz / max(i - k, 1)))
    j0 = f_c - (i - k) - int(fo["frame"][0])
    if j0 < 0:
        return out
    gap0 = float(c_pos["s_rear"][k] - f_pos["s_front"][j0])
    out.update(v_ego0=float(f_pos["v"][j0]), gap0=gap0, dv0=float(f_pos["v"][j0] - c_pos["v"][k]))
    return out


def screen_recording(rec, data, ev):
    """Detectors, union and gate of pilot/screening.py on one recording, matched against its kept cut-ins."""
    p = Path(data) / f"{rec:02d}_tracks.csv"
    frames = L.resample(L.load_highd(p))
    span = frames.groupby("vid").t.agg(lambda t: t.max() - t.min())
    egos = sorted(span[span >= SC.MIN_DURATION_S].index)
    by_vid = {k: g for k, g in frames.groupby("vid", sort=False)}
    t0s = {k: (g.t.min(), g.t.max()) for k, g in by_vid.items()}
    fr_t = frames.t.to_numpy()
    order = np.argsort(fr_t, kind="stable")
    control = {e for e, u in zip(egos, np.random.default_rng(rec * 1000 + 7).random(len(egos))) if u < CONTROL_SHARE}
    ev_by_ego = {k: g.f_c.to_numpy() / 25.0 for k, g in ev.groupby("follower")} if len(ev) else {}
    keys = ("threshold", "pelt", "discord", "union", "gated")
    out = dict(vehicles=int(len(span)), screenable=len(egos), control=len(control), flags={k: 0 for k in keys}, egos_flagged={k: 0 for k in keys},
               gated_overlap_cutin=0, status={s: 0 for s in SE.STATUSES})
    match = []
    for ego in sorted(control | (set(ev_by_ego) & set(egos))):
        a, b = t0s[ego]
        lo, hi = np.searchsorted(fr_t[order], [a - 0.05, b + 0.05])
        sub = frames.iloc[order[lo:hi]]
        ser = SE.compute(sub, ego)
        flags, _ = D.run_all(ser, SE.detector_signal(ser))
        union = FU.union_flags({k: {ego: v} for k, v in flags.items()}, [ego])[ego]
        gated = FU.physical_gate(union, ser)
        times = ev_by_ego.get(ego, np.array([]))
        if ego in control:
            for k, v in dict(flags, union=union, gated=gated).items():
                out["flags"][k] += len(v)
                out["egos_flagged"][k] += int(bool(v))
            for k, v in SE.status_counts(ser).items():
                out["status"][k] += v
            out["gated_overlap_cutin"] += sum(any(x - MATCH_S <= t <= y + MATCH_S for t in times) for x, y in gated)
        for t in times:
            if a + 1 <= t <= b - 1:
                hit = lambda iv: any(x - MATCH_S <= t <= y + MATCH_S for x, y in iv)
                match.append(dict(rec=rec, follower=int(ego), t=float(t), threshold=hit(flags["threshold"]), pelt=hit(flags["pelt"]),
                                  discord=hit(flags["discord"]), union=hit(union), gated=hit(gated)))
    return out, pd.DataFrame(match)


def _worker(args):
    rec, data, do_screen = args
    warnings.filterwarnings("ignore", message="A large number of values")
    t0 = time.time()
    ev, count, chk, dur, nveh = extract_recording(rec, data)
    t1 = time.time()
    scr, match = screen_recording(rec, data, ev) if do_screen else (None, pd.DataFrame())
    return rec, ev, count, chk, dur, nveh, scr, match, t1 - t0, time.time() - t1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--work")
    ap.add_argument("--recordings", type=int, nargs="*")
    ap.add_argument("--no-screening", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results/highd/extraction.json"))
    a = ap.parse_args(argv)
    data, work = data_dir(a.data), work_dir(a.work)
    recs = a.recordings or recordings(data)
    if not recs:
        raise SystemExit(f"no highD recordings under {data}")
    jobs = [(r, str(data), not a.no_screening) for r in recs]
    evs, matches, funnel, per_rec = [], [], dict.fromkeys(FUNNEL, 0), {}
    chk = dict(meta_lane_changes=0, non_contiguous_tracks=0)
    scr_tot = None
    secs = [0.0, 0.0]
    dur_s, n_veh = 0.0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for rec, ev, count, c, dur, nveh, scr, match, t_e, t_s in ex.map(_worker, jobs):
            evs.append(ev)
            matches.append(match)
            for k, v in count.items():
                funnel[k] += v
            for k, v in c.items():
                chk[k] += v
            dur_s += dur
            n_veh += nveh
            secs[0] += t_e
            secs[1] += t_s
            per_rec[f"{rec:02d}"] = count["kept"]
            if scr is not None:
                if scr_tot is None:
                    scr_tot = json.loads(json.dumps(scr))
                else:
                    for k in ("vehicles", "screenable", "control", "gated_overlap_cutin"):
                        scr_tot[k] += scr[k]
                    for g in ("flags", "egos_flagged", "status"):
                        for k, v in scr[g].items():
                            scr_tot[g][k] += v
            print(f"rec {rec:02d}: kept {count['kept']:4d} ({t_e:.1f} s, screening {t_s:.1f} s)", flush=True)
    ev = pd.concat(evs, ignore_index=True)
    ev.to_parquet(work / "cutins.parquet", index=False)
    match = pd.concat(matches, ignore_index=True)
    if len(match):
        match.to_parquet(work / "screening_match.parquet", index=False)
    summary = dict(recordings=len(recs), duration_h=dur_s / 3600, vehicles=n_veh, funnel=funnel, checks=chk,
                   events_per_recording=per_rec, cpu_seconds=dict(extraction=secs[0], screening=secs[1]),
                   parameters=dict(stable_s=STABLE_S, v_lat_start=V_LAT_START, t0_max_s=T0_MAX_S, max_range_m=SE.MAX_RANGE,
                                   v_min=SE.V_MIN, match_s=MATCH_S, control_share=CONTROL_SHARE, min_duration_s=SC.MIN_DURATION_S))
    if scr_tot is not None:
        summary["screening_raw"] = scr_tot
    elif Path(a.out).exists():                                # --no-screening keeps the screening block of an earlier run
        old = json.loads(Path(a.out).read_text())
        if "screening_raw" in old:
            summary["screening_raw"] = old["screening_raw"]
            summary["cpu_seconds"]["screening"] = old["cpu_seconds"]["screening"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("recordings", "vehicles", "funnel", "checks")}, indent=1))


if __name__ == "__main__":
    main()
