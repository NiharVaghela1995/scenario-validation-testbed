"""highD robustness analyses: one row per lane change with a follower, with what the definition variants need.

    python pilot/highd_candidates.py         # writes $HIGHD_WORK/candidates.parquet and results/highd/robust_candidates.json

Unlike pilot/highd_cutins.py this keeps lane changes that fail the primary filters, with the quantities needed to apply each
variant in docs/HIGHD_ANALYSIS_PLAN.md (stable windows, follower track length, lateral speed, auxiliary lanes, direction, timing
offsets, local density). `primary` marks the pilot's filter; the run stops if it does not reproduce the pilot's table exactly.
Per-vehicle output stays outside the repository (highd_cutins.work_dir).
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __name__ == "__main__":
    for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot")]
import highd_cutins as HC  # noqa: E402
import loaders as L  # noqa: E402
import series as SE  # noqa: E402

V_LAT_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)
OFFSET_FRAMES = 5           # 0.2 s at 25 Hz
DENSITY_HALF_M = 150.0
AUX_LAST_SHARE, AUX_MAX_SHARE = 0.10, 0.10


def _run_before(a, i, v):
    """Number of consecutive entries equal to v directly before index i."""
    k = i
    while k > 0 and a[k - 1] == v:
        k -= 1
    return i - k


def _run_from(a, i, v):
    """Number of consecutive entries equal to v starting at index i."""
    k = i
    while k < len(a) and a[k] == v:
        k += 1
    return k - i


def auxiliary_lanes(frames, direction):
    """Lanes in which at most AUX_MAX_SHARE of the vehicles seen are in that lane in the last AUX_LAST_SHARE of the field of view."""
    out, info = set(), {}
    fr = frames.assign(direction=frames.vid.map(direction))
    for d, g in fr.groupby("direction"):
        lo, hi = float(g.s_front.min()), float(g.s_front.max())
        cut = hi - AUX_LAST_SHARE * (hi - lo)
        for lane, h in g.groupby("lane"):
            seen, late = h.vid.nunique(), h[h.s_front >= cut].vid.nunique()
            info[int(lane)] = (seen, late)
            if late <= AUX_MAX_SHARE * seen:
                out.add(int(lane))
    return out, info


def extract(a):
    rec, data = a
    p = Path(data) / f"{rec:02d}_tracks.csv"
    meta = pd.read_csv(p.with_name(f"{rec:02d}_tracksMeta.csv")).set_index("id")
    info = pd.read_csv(p.with_name(f"{rec:02d}_recordingMeta.csv")).iloc[0]
    hz = float(info.frameRate)
    tr = HC._arrays(pd.read_csv(p, usecols=HC.USECOLS).sort_values(["id", "frame"]))
    frames = L.load_highd(p)
    pos = HC._arrays(frames.assign(id=frames.vid).sort_values(["id", "t"]))
    t_all, n_prim = frames.t.to_numpy(), int(round(HC.STABLE_S * hz))
    aux, aux_info = auxiliary_lanes(frames, meta.drivingDirection)
    s_rng = {d: (float(g.s_rear.min()), float(g.s_front.max())) for d, g in frames.assign(d=frames.vid.map(meta.drivingDirection)).groupby("d")}
    rows = []
    for c_id in meta.index[meta.numLaneChanges > 0]:
        c, cp = tr[c_id], pos[c_id]
        for i in np.flatnonzero(np.diff(c["laneId"])) + 1:
            f_c, l0, l1, f_id = int(c["frame"][i]), int(c["laneId"][i - 1]), int(c["laneId"][i]), int(c["followingId"][i])
            if f_id == 0:
                continue
            fo, fp = tr[f_id], pos[f_id]
            j = f_c - int(fo["frame"][0])
            if not (0 <= j < len(fo["frame"])) or int(fo["precedingId"][j]) != c_id:
                continue
            t_c = f_c / hz
            lo, hi = np.searchsorted(t_all, [t_c - HC.STABLE_S - 0.2, t_c + HC.STABLE_S + 0.2])
            win = frames.iloc[lo:hi]
            ser = SE.compute(win[(win.vid == f_id) | (win.lane == l1) | (win.vid == c_id)], f_id)
            r = ser.iloc[int(np.argmin(np.abs(ser.t.to_numpy() - t_c)))]
            d_sign = int(meta.loc[c_id, "drivingDirection"])
            dy = float(np.sign(c["y"][i] - c["y"][max(i - int(round(hz)), 0)]))
            leftward = (d_sign == 2 and dy < 0) or (d_sign == 1 and dy > 0)
            now = win[np.isclose(win.t, t_c)]
            s_c = float(cp["s_front"][i] + cp["s_rear"][i]) / 2
            lo_s, hi_s = s_rng[d_sign]
            a_s, b_s = max(lo_s, s_c - DENSITY_HALF_M), min(hi_s, s_c + DENSITY_HALF_M)
            centre = (now.s_front + now.s_rear) / 2
            count = int(((now.lane == l1) & centre.between(a_s, b_s)).sum())
            row = dict(rec=rec, cutter=int(c_id), follower=f_id, f_c=f_c, location=int(info.locationId), direction=d_sign,
                       cutter_class=meta.loc[c_id, "class"], follower_class=meta.loc[f_id, "class"], origin_lane=l0, target_lane=l1,
                       origin_aux=l0 in aux, target_aux=l1 in aux, leftward=bool(leftward),
                       c_before=_run_before(c["laneId"], i, l0), c_after=_run_from(c["laneId"], i, l1),
                       f_before=_run_before(fo["laneId"], j, l1), f_after=_run_from(fo["laneId"], j + 1, l1) if j + 1 < len(fo["laneId"]) else 0,
                       f_lane_ok=int(fo["laneId"][j]) == l1, f_track_s=len(fo["frame"]) / hz,
                       lead_ok=r.lead_id == c_id, status=str(r.status), hd_dhw=float(fo["dhw"][j]), v_ego=float(r.v_ego), v_cut=float(r.v_lead),
                       gap=float(r.gap), thw=float(r.thw), ttc=float(r.ttc), drac=float(r.drac), v_lat_cross=float(abs(c["yVelocity"][i])),
                       density_veh_km=count / max((b_s - a_s) / 1000.0, 1e-6))
            row["dv"] = row["v_ego"] - row["v_cut"]
            for off, name in ((-OFFSET_FRAMES, "m02"), (OFFSET_FRAMES, "p02")):
                if 0 <= i + off < len(c["frame"]) and 0 <= j + off < len(fo["frame"]):
                    g, v = float(cp["s_rear"][i + off] - fp["s_front"][j + off]), float(fp["v"][j + off])
                    row.update({f"gap_{name}": g, f"v_ego_{name}": v, f"dv_{name}": v - float(cp["v"][i + off])})
                else:
                    row.update({f"gap_{name}": np.nan, f"v_ego_{name}": np.nan, f"dv_{name}": np.nan})
            for th in V_LAT_THRESHOLDS:
                m = HC._manoeuvre_start(c, cp, fo, fp, i, f_c, hz, v_lat_start=th)
                key = f"{int(round(th * 100)):02d}"
                row.update({f"gap0_{key}": m["gap0"], f"v_ego0_{key}": m["v_ego0"], f"dv0_{key}": m["dv0"]})
            rows.append(row)
    return pd.DataFrame(rows), len(aux), {str(k): v for k, v in aux_info.items()}, sorted(aux), int(n_prim)


def primary_filter(c, n=25):
    return ((c.c_before >= n) & (c.c_after >= n) & (c.f_before >= n) & (c.f_after >= n) & c.f_lane_ok & c.lead_ok
            & ~c.status.isin(["lane_change", "collision"]) & c.gap.between(0, SE.MAX_RANGE, inclusive="right") & (c.v_ego > SE.V_MIN))


def main():
    data, work = HC.data_dir(), HC.work_dir()
    recs = HC.recordings(data)
    with ProcessPoolExecutor(3) as ex:
        res = list(ex.map(extract, [(r, str(data)) for r in recs]))
    c = pd.concat([r[0] for r in res], ignore_index=True)
    c["primary"] = primary_filter(c)
    pil = pd.read_parquet(work / "cutins.parquet")
    a = c[c.primary][["rec", "cutter", "f_c"]].sort_values(["rec", "cutter", "f_c"]).reset_index(drop=True)
    b = pil[["rec", "cutter", "f_c"]].sort_values(["rec", "cutter", "f_c"]).reset_index(drop=True)
    assert a.equals(b), f"primary filter gives {len(a)} events, pilot table has {len(b)}"
    k = c[c.primary].merge(pil, on=["rec", "cutter", "f_c"], suffixes=("", "_pilot"))
    for v in ("gap", "dv", "thw"):
        assert np.allclose(k[v], k[v + "_pilot"]), v
    c.to_parquet(work / "candidates.parquet", index=False)
    aux_recs = {f"{r:02d}": x[3] for r, x in zip(recs, res) if x[3]}
    out = dict(candidates=len(c), primary=int(c.primary.sum()), recordings_with_auxiliary_lane=len(aux_recs), auxiliary_lanes_by_recording=aux_recs,
               primary_from_aux_lane=int((c.primary & c.origin_aux).sum()), primary_into_aux_lane=int((c.primary & c.target_aux).sum()),
               primary_leftward=int((c.primary & c.leftward).sum()), reproduces_pilot_table=True)
    (ROOT / "results/highd/robust_candidates.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "auxiliary_lanes_by_recording"}), {k: v for k, v in list(aux_recs.items())[:4]})


if __name__ == "__main__":
    main()
