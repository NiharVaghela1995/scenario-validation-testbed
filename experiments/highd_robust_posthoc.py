"""highD robustness: POST HOC diagnostics, added after the planned analyses showed truck followers break the headline.

    python experiments/highd_robust_posthoc.py     # writes results/highd/robust_posthoc.json

Not in docs/HIGHD_ANALYSIS_PLAN.md. Asks whether the truck-follower result comes from stratum-specific rank thresholds: lambda uses
each population's own quantiles, and a truck follower's top closing-speed decile can be a much lower speed than the pooled one.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "critical_events"), str(ROOT / "pilot")]
import highd_cutins as HC  # noqa: E402

Q = 0.10


def main():
    c = pd.read_parquet(HC.work_dir() / "candidates.parquet")
    p = c[c.primary]
    pool_dv, pool_thw = float(p.dv.quantile(1 - Q)), float(p.thw.quantile(Q))
    out = dict(q=Q, pooled=dict(dv_top_decile_starts_ms=pool_dv, thw_lowest_decile_ends_s=pool_thw), by_follower_class={}, direction={})
    for k, g in p.groupby("follower_class"):
        dv, thw = float(g.dv.quantile(1 - Q)), float(g.thw.quantile(Q))
        cond = g[g.dv > pool_dv]
        hits = cond[cond.thw <= pool_thw]
        own = g[g.dv > dv]
        out["by_follower_class"][k] = dict(events=len(g), median_dv_ms=float(g.dv.median()), share_closing=float((g.dv > 0).mean()), own_dv_top_decile_starts_ms=dv, own_thw_lowest_decile_ends_s=thw,
                                           pooled_conditioning_events=len(cond), pooled_joint_extremes=len(hits), lambda_at_pooled_thresholds=len(hits) / len(cond) if len(cond) else None,
                                           own_conditioning_events=len(own), own_joint_extremes=int((own.thw <= thw).sum()))
    for k, g in p.groupby("leftward"):
        out["direction"]["leftward" if k else "rightward"] = dict(events=len(g), median_dv_ms=float(g.dv.median()), share_closing=float((g.dv > 0).mean()))
    (ROOT / "results/highd/robust_posthoc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
