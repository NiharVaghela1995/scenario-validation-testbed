"""Union of the three detectors and the physical gate on the synthetic screening run (precision and recall).

Needs the output of `python critical_events/run_critical_events.py --mock`. Writes results/fusion_gate.json.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("TESTBED_RESULTS", ROOT / "results"))
RUN = Path(os.environ.get("TESTBED_SCREENING_OUT", RES / "screening")) / "mock"
sys.path.insert(0, str(ROOT / "critical_events"))
import evaluate as E  # noqa: E402
import fusion as F  # noqa: E402

ids = sorted(p.stem for p in (RUN / "series").glob("*.csv"))
series = {k: pd.read_csv(RUN / "series" / f"{k}.csv") for k in ids}
truth = json.loads((RUN / "ground_truth.json").read_text())
flags = json.loads((RUN / "flags.json").read_text())


def scored(fl):
    r = E.score({k: fl.get(k, []) for k in ids}, {k: truth[k] for k in ids})
    return dict(precision=r["precision"], recall=r["recall"], flags=r["flags"])


out = {d: scored(flags[d]) for d in flags}
union = F.union_flags(flags, ids)
out["union"] = scored(union)
out["union_plus_gate"] = scored({k: F.physical_gate(union[k], series[k]) for k in ids})
out = dict(data="synthetic test bed only", gate=f"TTC < {F.GATE_TTC_S} s or THW < {F.GATE_THW_S} s", scores=out)
(RES / "fusion_gate.json").write_text(json.dumps(out, indent=1))
for k, v in out["scores"].items():
    print(f"{k:18s} precision {v['precision']:.3f}  recall {v['recall']:.3f}  flags {v['flags']}")
