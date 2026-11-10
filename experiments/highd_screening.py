"""highD pilot, step 2: aggregate the screening results of pilot/highd_cutins.py. Writes results/highd/screening.json.

    python experiments/highd_screening.py

Catch rate: share of kept cut-ins whose follower has a flagged interval within +-1 s of the cut-in, by TTC at cut-in. A cut-in
is not a label for criticality: the catch rate says which cut-ins the detectors react to, not whether they are right.
Flag rates: the seeded random control sample of screenable vehicles.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot")]
import cutin_chain as C  # noqa: E402
import highd_cutins as HC  # noqa: E402

DET = ("threshold", "pelt", "discord", "union", "gated")
RES = ROOT / "results/highd"


def main():
    work = HC.work_dir()
    ev = pd.read_parquet(work / "cutins.parquet").assign(t=lambda d: d.f_c / 25.0)
    match = pd.read_parquet(work / "screening_match.parquet")
    df = match.merge(ev[["rec", "follower", "t", "ttc", "thw"]], on=["rec", "follower", "t"], validate="m:1")
    assert len(df) == len(match)
    strata = {"all": df.index == df.index, "closing": df.ttc.notna(), "not_closing": df.ttc.isna(), "ttc_below_4s": df.ttc < 4,
              "ttc_4_to_8s": (df.ttc >= 4) & (df.ttc < 8), "ttc_8s_or_more": df.ttc >= 8, "thw_below_0.6s": df.thw < 0.6}
    rate = {}
    for s, m in strata.items():
        n = int(m.sum())
        rate[s] = dict(events=n)
        for d in DET:
            k = int(df.loc[m, d].sum())
            rate[s][d] = dict(caught=k, share=k / n if n else None, wilson_95=list(C.wilson(k, n)) if n else None)
    ex = json.loads((RES / "extraction.json").read_text())
    raw = ex["screening_raw"]
    ctrl = raw["control"]
    flag = {d: dict(flags=raw["flags"][d], vehicles_flagged=raw["egos_flagged"][d], share_of_vehicles=raw["egos_flagged"][d] / ctrl,
                    wilson_95=list(C.wilson(raw["egos_flagged"][d], ctrl))) for d in DET}
    out = dict(cutins_screened=len(df), cutins_not_screened=len(ev) - len(df), vehicles=raw["vehicles"], screenable_vehicles=raw["screenable"],
               control_vehicles=ctrl, catch_rate=rate, control_flag_rates=flag, gate_removed_share=1 - raw["flags"]["gated"] / raw["flags"]["union"],
               gated_flags_overlapping_a_cutin=raw["gated_overlap_cutin"], gated_flags=raw["flags"]["gated"],
               share_gated_overlapping_cutin=raw["gated_overlap_cutin"] / raw["flags"]["gated"], status_control=raw["status"])
    (RES / "screening.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("status_control",)}, indent=0)[:3500])


if __name__ == "__main__":
    main()
