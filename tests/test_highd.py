"""highD pilot checks. Skipped automatically unless the licensed data is present ($HIGHD_DATA, default ~/datasets/highD/data)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot")]
import highd_cutins as HC  # noqa: E402

DATA = HC.data_dir()
pytestmark = pytest.mark.skipif(not any(DATA.glob("*_tracks.csv")), reason="highD data not available")
PER_VEHICLE_KEYS = {"cutter", "follower", "f_c", "hd_dhw", "hd_thw", "hd_ttc", "t0", "t1", "ego", "x", "y"}


@pytest.fixture(scope="module")
def rec1():
    return HC.extract_recording(1, DATA)


def test_funnel_adds_up(rec1):
    ev, count, chk, _, _ = rec1
    assert count["lane_changes"] == chk["meta_lane_changes"]
    assert sum(v for k, v in count.items() if k != "lane_changes") == count["lane_changes"]
    assert len(ev) == count["kept"] > 0
    assert chk["non_contiguous_tracks"] == 0


def test_signals_agree_with_highd_columns(rec1):
    ev = rec1[0]
    assert (ev.gap - ev.hd_dhw).abs().quantile(0.99) < 0.05
    assert (ev.thw - ev.hd_thw).abs().quantile(0.99) < 0.05
    closing = ev.ttc.notna()
    assert (closing == (ev.hd_ttc > 0)).mean() > 0.98
    both = closing & (ev.hd_ttc > 0)
    assert ((ev.ttc[both] / ev.hd_ttc[both] - 1).abs()).quantile(0.95) < 0.05


def test_cutin_definition_holds(rec1):
    ev = rec1[0]
    assert ev.gap.between(0, 150, inclusive="right").all() and (ev.v_ego > 0.5).all()
    assert ev.cross_offset_m.quantile(0.99) < 0.2      # laneId changes where the centre crosses a marking


def test_work_dir_refuses_the_repository():
    with pytest.raises(ValueError, match="inside the repository"):
        HC.work_dir(ROOT / "results/highd")


def test_no_per_vehicle_data_in_committed_results():
    def walk(o):
        if isinstance(o, dict):
            assert not PER_VEHICLE_KEYS & set(o), set(o) & PER_VEHICLE_KEYS
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            assert len(o) < 100
            for v in o:
                walk(v)
    files = list((ROOT / "results/highd").glob("*.json"))
    assert files
    for p in files:
        walk(json.loads(p.read_text()))


def test_no_raw_data_tracked():
    names = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    bad = [n for n in names if n.endswith(("_tracks.csv", "_tracksMeta.csv", "_recordingMeta.csv", "_highway.png", ".parquet"))]
    assert not bad, bad
