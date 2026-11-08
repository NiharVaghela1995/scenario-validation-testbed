"""Regression set: re-run all detectors on the fixed synthetic segments and compare with stored numbers.

Fails if any detector's precision or recall drops more than `tolerance` below the value stored in
critical_events/data/regression_expected.json, or if the number of flags changes by more than 20 %. After an
intended improvement, regenerate the file and record the change (date, reason) in the commit message.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "critical_events"
sys.path.insert(0, str(HERE))

import detectors as D  # noqa: E402
import evaluate as E  # noqa: E402
from series import detector_signal  # noqa: E402
from synthetic import make_segments  # noqa: E402


def test_detectors_do_not_regress():
    exp = json.loads((HERE / "data" / "regression_expected.json").read_text())
    segs = make_segments(n_segments=30, seed=7)
    labels = {s["id"]: s["gt"] for s in segs}
    flags = {d: {} for d in ("threshold", "pelt", "discord")}
    for s in segs:
        r, _ = D.run_all(s["series"], detector_signal(s["series"]))
        for d in flags:
            flags[d][s["id"]] = r[d]
    tol = exp["tolerance"]
    for d, e in exp["expected"].items():
        got = E.score(flags[d], labels)
        assert got["precision"] >= e["precision"] - tol, (d, "precision", got["precision"], e["precision"])
        assert got["recall"] >= e["recall"] - tol, (d, "recall", got["recall"], e["recall"])
        assert abs(got["flags"] - e["flags"]) <= 0.2 * e["flags"], (d, "flags", got["flags"], e["flags"])
