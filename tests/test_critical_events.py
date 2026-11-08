"""Unit tests (mock only; the esmini test skips if esmini is missing)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "critical_events"))

import detectors as D  # noqa: E402
import evaluate as E  # noqa: E402
import label  # noqa: E402
import similarity as S  # noqa: E402
import tail as T  # noqa: E402
from series import compute, detector_signal  # noqa: E402


def frames(rows):
    f = pd.DataFrame(rows, columns=["t", "vid", "lane", "s_front", "v"])
    f["s_rear"] = f.s_front - 4.5
    return f


def test_ttc_thw_drac_and_edge_cases():
    rows = []
    for k, t in enumerate(np.arange(0, 3, 0.1).round(1)):
        rows.append((t, 0, 1, 0.0, 20.0))                       # ego
        if t < 1.0:
            rows.append((t, 1, 1, 24.5, 10.0))                  # closing: gap 20 m, dv 10 -> TTC 2 s
        elif t < 2.0:
            rows.append((t, 1, 1, 24.5, 25.0))                  # opening
        # t >= 2: no leader
    ser = compute(frames(rows), 0)
    r0 = ser.iloc[0]
    assert r0.status == "ok" and r0.ttc == pytest.approx(2.0) and r0.thw == pytest.approx(1.0)
    assert r0.drac == pytest.approx(10 ** 2 / 40) and r0.inv_ttc == pytest.approx(0.5)
    assert (ser[(ser.t >= 1.0) & (ser.t < 2.0)].status == "opening").all()
    assert ser[(ser.t >= 1.0) & (ser.t < 2.0)].ttc.isna().all()
    assert (ser[ser.t >= 2.0].status == "no_leader").all()


def test_collision_leader_change_and_lane_change():
    rows = [(0.0, 0, 1, 0.0, 20.0), (0.0, 1, 1, 30.0, 20.0),
            (0.1, 0, 1, 2.0, 20.0), (0.1, 1, 1, 30.0, 20.0), (0.1, 2, 1, 8.0, 15.0),     # cut-in
            (0.2, 0, 1, 4.0, 20.0), (0.2, 2, 1, 6.0, 15.0)]                               # overlap
    ser = compute(frames(rows), 0)
    assert ser.status.tolist()[1:] == ["leader_change", "collision"]
    assert bool(ser.lead_switch.iloc[1])
    rows = [(round(t, 1), 0, 1 if t < 1.5 else 2, t * 20, 20.0) for t in np.arange(0, 3, 0.1)]
    ser = compute(frames(rows), 0)
    lc = ser[ser.status == "lane_change"]
    assert len(lc) >= 19 and lc.inv_ttc.isna().all()
    assert np.isfinite(detector_signal(ser)).all()


def test_threshold_runs_merge_and_drop():
    t = np.arange(0, 10, 0.1)
    ttc = np.full_like(t, 10.0)
    ttc[20:30] = 2.0
    ttc[33:40] = 2.5           # 0.3 s gap -> merged with the first run
    ttc[70:72] = 1.0           # 0.1 s long -> dropped
    ser = pd.DataFrame(dict(t=t, ttc=ttc))
    assert D.threshold(ser) == [[2.0, 3.9]]


def test_pelt_finds_step_and_penalty_matters():
    rng = np.random.default_rng(0)
    x = np.r_[np.zeros(200), np.full(60, 0.5), np.zeros(200)] + rng.normal(0, 0.02, 460)
    bk = D.change_points(x)
    assert any(abs(b - 200) <= 3 for b in bk) and any(abs(b - 260) <= 3 for b in bk)
    sweep = dict(D.pelt_penalty_sweep(x))
    assert sweep[0.1] >= sweep[1000]


def test_discord_finds_planted_anomaly_but_not_twins():
    rng = np.random.default_rng(1)
    t = np.arange(0, 60, 0.1)
    x = rng.normal(0, 0.005, len(t))
    bump = 0.4 * np.hanning(30)
    x1 = x.copy()
    x1[300:330] += bump
    fl, P, m = D.discords(x1, t)
    assert any(a <= 31 and b >= 29 for a, b in fl)
    x2 = x1.copy()
    x2[100:130] += bump                                       # identical twin: each is the other's neighbour
    fl2, _, _ = D.discords(x2, t)
    assert not any(a <= 31 and b >= 29 for a, b in fl2)


def test_interval_rule():
    assert E.matches([10.0, 11.0], [11.8, 15.0])                 # within the 1 s tolerance
    assert not E.matches([10.0, 10.5], [12.0, 15.0])
    assert not E.matches([10.0, 11.0], [11.8, 15.0], min_iou=0.3)
    r = E.score({"a": [[1, 2], [50, 51]]}, {"a": [[1.5, 3.0], [20, 25]]})
    assert (r["precision"], r["recall"]) == (0.5, 0.5)


def test_dtw_absorbs_speed_difference():
    s = np.sin(np.linspace(0, np.pi, 40)) ** 2
    slow = np.sin(np.linspace(0, np.pi, 52)) ** 2
    shifted = np.r_[np.zeros(4), s[:-4]]
    assert S.dtw(s, slow, band=0.3) < S.euclid_resampled(s, shifted)
    assert S.dtw(s, s) == 0.0


def test_pot_declustering():
    t = np.arange(0, 100, 0.1)
    y = np.zeros_like(t)
    y[100:110] = 1.0
    y[105] = 2.0
    y[600] = 1.5
    assert sorted(T.decluster(t, y, 0.5)) == [1.0, 1.5]


def test_label_parse_and_roundtrip(tmp_path):
    assert label.parse_intervals("12.5-15, 31.2-30") == [[12.5, 15.0], [30.0, 31.2]]
    assert label.parse_intervals("none") == []
    f = tmp_path / "labels.json"
    lab = label.load_labels(f)
    lab["segments"]["s1"] = [[3.04, 5.0]]
    lab["done"] = ["s1"]
    label.save_labels(lab, f)
    assert json.loads(f.read_text())["segments"]["s1"] == [[3.0, 5.0]]


def test_mock_run_end_to_end(tmp_path, monkeypatch):
    import run_critical_events as R
    log = R.main(["--mock"])
    out = Path(R.HERE) / "out" / "mock"
    for name in ("results.md", "results.json", "timeline.png", "penalty_sweep.png", "dtw_pairs.png",
                 "dtw_pairs.json", "tail.json", "run_log.md", "ground_truth.json", "flags.json"):
        assert (out / name).exists(), name
    assert 20 <= log["segments"] <= 40
    rows = json.loads((out / "results.json").read_text())
    assert {r["detector"] for r in rows} == {"threshold", "pelt", "discord"}
    assert len(json.loads((out / "dtw_pairs.json").read_text())["closest"]) == 3
    # labelling helper in CLI mode on the mock output ("none" for every segment, then quit)
    answers = iter(["1-2"] + ["none"] * 100)
    label.cli(out, tmp_path / "labels.json", inp=lambda prompt: next(answers))
    lab = json.loads((tmp_path / "labels.json").read_text())
    assert len(lab["done"]) == log["segments"] and sum(len(v) for v in lab["segments"].values()) == 1


def test_esmini_one_run(tmp_path):
    import esmini_logs
    if not esmini_logs.available():
        pytest.skip("esmini not found (ESMINI_BIN)")
    p = dict(esmini_logs.plan(4)[1])            # one hard_brake run
    rc, csv = esmini_logs.run(esmini_logs.build(p, tmp_path))
    assert rc == 0
    ser = compute(esmini_logs.parse_csv(csv), 0)
    assert len(ser) >= 590 and ser.ttc.notna().any()
