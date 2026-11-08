"""Compare freshly computed results with the published numbers in tests/data/published_numbers.json.

    python experiments/check_numbers.py [RESULTS_DIR]     # exit code 1 on any mismatch
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "tests" / "data" / "published_numbers.json"


def load_published():
    return json.loads(PUBLISHED.read_text())["numbers"]


def lookup(results_dir, entry):
    """Value at entry['path'] inside entry['file'] of results_dir."""
    node = json.loads((Path(results_dir) / entry["file"]).read_text())
    for key in entry["path"]:
        node = node[key]
    return node


def replay_ran(results_dir):
    return bool(json.loads((Path(results_dir) / "generation_cutin.json").read_text())["replay"].get("runs"))


def within(entry, got):
    want = entry["value"]
    tol = entry.get("tol_abs", 0.0) + entry.get("tol_rel", 0.0) * abs(want)
    return math.isclose(float(got), float(want), abs_tol=tol, rel_tol=0.0)


def check(results_dir):
    """-> list of (entry, got or None, status), status in ok / MISMATCH / skipped."""
    rows, has_replay = [], replay_ran(results_dir)
    for e in load_published():
        if e.get("needs_esmini") and not has_replay:
            rows.append((e, None, "skipped"))
            continue
        got = lookup(results_dir, e)
        rows.append((e, got, "ok" if within(e, got) else "MISMATCH"))
    return rows


def main():
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("TESTBED_RESULTS", ROOT / "results"))
    rows = check(res)
    print(f"{'number':58s} {'published':>12s} {'computed':>12s}  status")
    for e, got, st in rows:
        print(f"{e['label']:58s} {e['value']:12.5g} {'-' if got is None else format(got, '12.5g'):>12s}  {st}")
    n_bad = sum(st == "MISMATCH" for *_, st in rows)
    print(f"\n{sum(st == 'ok' for *_, st in rows)} ok, {n_bad} mismatched, {sum(st == 'skipped' for *_, st in rows)} skipped")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
