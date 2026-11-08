"""Loaders and screening on small synthetic files written in each dataset's published format (no real data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot")]
import loaders as L  # noqa: E402
import screening as SC  # noqa: E402
from series import compute  # noqa: E402

T = np.arange(0, 20, 0.1).round(1)


def ngsim_rows(upper=False):
    """Two vehicles in lane 2, one period; follower closes on leader. Feet and ft/s, as in the dataset."""
    rows = []
    for vid, y0, v in ((1, 0.0, 60.0), (2, 200.0, 40.0)):
        for k, t in enumerate(T):
            rows.append(dict(vehicle_id=vid, global_time=1118846980200 + int(t * 1000), local_y=y0 + v * t, v_length=15.0,
                             v_vel=v, lane_id=2, location="us-101"))
    df = pd.DataFrame(rows)
    if upper:
        df = df.rename(columns=dict(vehicle_id="Vehicle_ID", global_time="Global_Time", local_y="Local_Y", v_length="v_Length",
                                    v_vel="v_Vel", lane_id="Lane_ID", location="Location"))
    return df


@pytest.mark.parametrize("upper", [False, True])
def test_ngsim_units_and_signals(tmp_path, upper):
    p = tmp_path / "ngsim.csv"
    ngsim_rows(upper).to_csv(p, index=False)
    fr = L.load_ngsim(p, "us-101")
    assert list(fr.columns) == L.FRAME_COLUMNS
    a = fr[(fr.vid == 1) & (fr.t == 0)].iloc[0]
    assert a.v == pytest.approx(60 * L.FT) and a.s_front - a.s_rear == pytest.approx(15 * L.FT)
    r = compute(fr, 1).iloc[0]
    assert r.status == "ok"
    assert r.gap == pytest.approx((200 - 15) * L.FT)
    assert r.ttc == pytest.approx((200 - 15) / 20.0)


def test_ngsim_location_must_be_chosen(tmp_path):
    df = ngsim_rows()
    df.loc[df.vehicle_id == 2, "location"] = "i-80"
    p = tmp_path / "ngsim.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="several locations"):
        L.load_ngsim(p)
    assert L.load_ngsim(p, "i-80").vid.unique().tolist() == [2]


def write_highd(d):
    rows = []
    for vid, x0, vx in ((1, 400.0, -30.0), (2, 330.0, -25.0), (3, 10.0, 30.0), (4, 80.0, 25.0)):
        for f in range(0, 500):
            rows.append(dict(frame=f, id=vid, x=x0 + vx * f / 25, width=4.5, xVelocity=vx, laneId=2 if vx < 0 else 5))
    pd.DataFrame(rows).to_csv(d / "01_tracks.csv", index=False)
    pd.DataFrame(dict(id=[1, 2, 3, 4], drivingDirection=[1, 1, 2, 2])).to_csv(d / "01_tracksMeta.csv", index=False)
    pd.DataFrame(dict(id=[1], frameRate=[25])).to_csv(d / "01_recordingMeta.csv", index=False)


def test_highd_both_directions_give_the_same_gap(tmp_path):
    write_highd(tmp_path)
    fr = L.resample(L.load_highd(tmp_path / "01_tracks.csv"))
    assert sorted(np.diff(np.sort(fr.t.unique()))[:3].round(3)) == [0.1, 0.1, 0.1]
    left, right = compute(fr, 1).iloc[0], compute(fr, 3).iloc[0]
    assert left.lead_id == 2 and right.lead_id == 4
    assert left.gap == pytest.approx(70 - 4.5)
    assert right.gap == pytest.approx(70 - 4.5)
    assert left.ttc == pytest.approx((70 - 4.5) / 5) and left.v_ego == pytest.approx(30)


def write_av2(d):
    ts = (1_000_000_000 + np.arange(150) * 100_000_000).astype("int64")
    pd.DataFrame(dict(timestamp_ns=ts, qw=1.0, qx=0.0, qy=0.0, qz=0.0, tx_m=20.0 * ts_seconds(ts), ty_m=0.0, tz_m=0.0)).to_feather(d / "city_SE3_egovehicle.feather")
    rows = []
    for k, t in enumerate(ts):
        rows.append(dict(timestamp_ns=t, track_uuid="aaa", category="REGULAR_VEHICLE", length_m=4.0, width_m=2.0, height_m=1.5, qw=1.0,
                         qx=0.0, qy=0.0, qz=0.0, tx_m=40.0 - 0.5 * k, ty_m=0.2, tz_m=0.0, num_interior_pts=50))
        rows.append(dict(timestamp_ns=t, track_uuid="ped", category="PEDESTRIAN", length_m=0.5, width_m=0.5, height_m=1.7, qw=1.0,
                         qx=0.0, qy=0.0, qz=0.0, tx_m=5.0, ty_m=8.0, tz_m=0.0, num_interior_pts=9))
    pd.DataFrame(rows).to_feather(d / "annotations.feather")


def ts_seconds(ts):
    return (ts - ts[0]) / 1e9


def test_av2_frames_and_relative_speed(tmp_path):
    write_av2(tmp_path)
    fr = L.load_av2(tmp_path, ego_front_m=3.8, ego_length_m=4.9)
    assert set(fr.vid) == {0, 1}                     # pedestrian dropped
    ser = compute(L.resample(fr), 0).iloc[10]
    assert ser.v_ego == pytest.approx(20, abs=1e-6)
    assert ser.v_lead == pytest.approx(15, abs=1e-6)   # closes 0.5 m per 0.1 s
    assert ser.gap == pytest.approx(40 - 5 - 2 - 3.8, abs=1e-6)


def test_screening_flags_a_closing_episode():
    rows = []
    for t in np.arange(0, 60, 0.1).round(1):
        s_lead = 300 + 25 * t if t < 30 else 300 + 25 * 30 + 3 * (t - 30)  # lead drops to 3 m/s
        rows.append((t, 1, 1, 25.0 * t, 25.0 * t - 4.5, 25.0))
        rows.append((t, 2, 1, s_lead, s_lead - 4.5, 25.0 if t < 30 else 3.0))
    fr = pd.DataFrame(rows, columns=L.FRAME_COLUMNS)
    cand, summary = SC.screen(fr, ego_ids=[1])
    assert summary["egos"] == 1 and len(cand) >= 1
    assert cand.min_ttc.min() < 4.0 and (cand.t0 >= 29).all()


def test_quality_report_counts():
    fr = pd.DataFrame([(0.0, 1, 1, 0, -4, 10.0), (0.1, 1, 1, 1, -3, 12.0), (3.0, 1, 1, 2, -2, -1.0)], columns=L.FRAME_COLUMNS)
    q = L.quality_report(fr)
    assert q["rows"] == 3 and q["negative_speed"] == 1 and q["accel_over_8"] == 1 and q["time_gaps_over_1s"] == 1
