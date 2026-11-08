"""Loaders from open trajectory datasets to the frames table used by the signals and detectors.

Frames table: one row per (t, vehicle) with columns t [s], vid, lane, s_front [m], s_rear [m], v [m/s]
(s = distance along the road, + in driving direction).

The data themselves are never part of this repository; see data/README.md. Each loader follows the dataset's
published file format. Status: the NGSIM loader was run on a one-minute slice of the public data; the highD and
Argoverse 2 loaders have only been tested on synthetic files in the documented format.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FRAME_COLUMNS = ["t", "vid", "lane", "s_front", "s_rear", "v"]
FT = 0.3048
AV2_VEHICLES = ("REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER",
                "ARTICULATED_BUS", "SCHOOL_BUS", "MOTORCYCLE")


def load_ngsim(path, location=None):
    """NGSIM vehicle trajectories CSV (either the lower-case API columns or the original mixed-case ones) -> frames.

    Local_Y is the front of the vehicle in feet; v_Vel is in ft/s. Vehicle ids repeat between the study periods of a
    location, so one period should be loaded at a time (use the time window of pilot/download_ngsim.py). Known quality
    problems of the raw data (noisy speeds, false collisions) are NOT repaired here; see `quality_report`.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    if "location" in df:
        locations = df.location.unique()
        if location is not None:
            df = df[df.location == location]
        elif len(locations) > 1:
            raise ValueError(f"several locations in the file, choose one of {sorted(locations)}")
    if df.empty:
        raise ValueError(f"no rows for location {location!r}")
    t = (df.global_time - df.global_time.min()) / 1000.0
    front = df.local_y * FT
    out = pd.DataFrame(dict(t=t.round(1), vid=df.vehicle_id.astype(int), lane=df.lane_id.astype(int), s_front=front,
                            s_rear=front - df.v_length * FT, v=df.v_vel * FT))
    return out.drop_duplicates(["t", "vid"]).sort_values(["t", "vid"]).reset_index(drop=True)[FRAME_COLUMNS]


def load_highd(tracks_csv, tracks_meta_csv=None, recording_meta_csv=None):
    """highD XX_tracks.csv (with XX_tracksMeta.csv and XX_recordingMeta.csv next to it) -> frames.

    x is the upper-left corner of the box and `width` is the vehicle length. drivingDirection 1 travels toward smaller
    x, 2 toward larger x, so s is signed per direction; lane ids of the two directions do not overlap.
    """
    tracks_csv = Path(tracks_csv)
    prefix = tracks_csv.name.split("_")[0]
    meta = pd.read_csv(tracks_meta_csv or tracks_csv.with_name(f"{prefix}_tracksMeta.csv"))
    rec = pd.read_csv(recording_meta_csv or tracks_csv.with_name(f"{prefix}_recordingMeta.csv"))
    df = pd.read_csv(tracks_csv).merge(meta[["id", "drivingDirection"]], on="id", validate="m:1")
    left = df.drivingDirection == 1
    front = np.where(left, -df.x, df.x + df.width)
    rear = np.where(left, -(df.x + df.width), df.x)
    out = pd.DataFrame(dict(t=df.frame / float(rec.frameRate.iloc[0]), vid=df.id.astype(int), lane=df.laneId.astype(int),
                            s_front=front, s_rear=rear, v=df.xVelocity.abs()))
    return out.sort_values(["t", "vid"]).reset_index(drop=True)[FRAME_COLUMNS]


def load_av2(log_dir, lane_width=3.5, ego_front_m=3.8, ego_length_m=4.9):
    """One Argoverse 2 Sensor log (annotations.feather + city_SE3_egovehicle.feather) -> frames, ego has vid 0.

    Cuboids are in the ego frame at each timestamp. APPROXIMATIONS: s is the ego's travelled arc length plus the cuboid's
    forward offset (no map, so curvature is ignored); lanes are bands of width `lane_width` around the ego's centre line;
    speeds of other vehicles are finite differences of s. `ego_front_m` (origin to front bumper) and `ego_length_m` are
    assumed values because the dataset guide does not state them.
    """
    log_dir = Path(log_dir)
    ann = pd.read_feather(log_dir / "annotations.feather")
    pose = pd.read_feather(log_dir / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
    stamps = pd.DataFrame(dict(timestamp_ns=np.sort(ann.timestamp_ns.unique())))
    ego = pd.merge_asof(stamps, pose[["timestamp_ns", "tx_m", "ty_m"]], on="timestamp_ns", direction="nearest")
    t = (ego.timestamp_ns - ego.timestamp_ns.iloc[0]).to_numpy() / 1e9
    s_ego = np.r_[0.0, np.cumsum(np.hypot(np.diff(ego.tx_m), np.diff(ego.ty_m)))]
    v_ego = np.gradient(s_ego, t) if len(t) > 1 else np.zeros_like(t)
    ego_tbl = pd.DataFrame(dict(t=t, vid=0, lane=0, s_front=s_ego + ego_front_m, s_rear=s_ego + ego_front_m - ego_length_m, v=v_ego))
    tmap = pd.Series(t, index=ego.timestamp_ns.to_numpy())
    smap = pd.Series(s_ego, index=ego.timestamp_ns.to_numpy())
    a = ann[ann.category.isin(AV2_VEHICLES)].copy()
    a["t"], a["vid"] = a.timestamp_ns.map(tmap), pd.factorize(a.track_uuid, sort=True)[0] + 1
    centre = a.timestamp_ns.map(smap) + a.tx_m
    a["s_front"], a["s_rear"] = centre + a.length_m / 2, centre - a.length_m / 2
    a["lane"] = np.round(a.ty_m / lane_width).astype(int)
    a = a.sort_values(["vid", "t"])
    a["v"] = _track_speed(a)
    frames = pd.concat([ego_tbl, a[FRAME_COLUMNS]], ignore_index=True)
    return frames.sort_values(["t", "vid"]).reset_index(drop=True)[FRAME_COLUMNS]


def _track_speed(a):
    v = pd.Series(0.0, index=a.index)
    for _, g in a.groupby("vid"):
        if len(g) > 1:
            v.loc[g.index] = np.gradient(((g.s_front + g.s_rear) / 2).to_numpy(), g.t.to_numpy())
    return v


def resample(frames, hz=10.0):
    """Linear interpolation of each vehicle onto a common time grid (the detectors assume 10 Hz); lane = last known."""
    step = 1.0 / hz
    parts = []
    for vid, g in frames.groupby("vid"):
        g = g.sort_values("t")
        t0, t1 = np.ceil(g.t.iloc[0] / step - 1e-9) * step, np.floor(g.t.iloc[-1] / step + 1e-9) * step
        if t1 < t0:
            continue
        grid = np.round(np.arange(t0, t1 + step / 2, step), 6)
        idx = np.clip(np.searchsorted(g.t.to_numpy(), grid + 1e-9, side="right") - 1, 0, len(g) - 1)
        parts.append(pd.DataFrame(dict(t=grid, vid=vid, lane=g.lane.to_numpy()[idx],
                                       **{c: np.interp(grid, g.t, g[c]) for c in ("s_front", "s_rear", "v")})))
    return pd.concat(parts, ignore_index=True)[FRAME_COLUMNS]


def quality_report(frames):
    """Simple plausibility counts on a frames table (nothing is changed or removed)."""
    dt = frames.groupby("vid").t.diff().dropna()
    accel = (frames.groupby("vid").v.diff() / frames.groupby("vid").t.diff()).dropna()
    return dict(rows=int(len(frames)), vehicles=int(frames.vid.nunique()), duration_s=float(frames.t.max() - frames.t.min()),
                median_dt_s=float(dt.median()) if len(dt) else float("nan"), negative_speed=int((frames.v < 0).sum()),
                accel_over_8=int((accel.abs() > 8).sum()), time_gaps_over_1s=int((dt > 1.0).sum()))
