"""Fusion of detector flags: union of intervals, then a physical gate that keeps an interval only if it is tight enough."""

GATE_TTC_S = 4.0     # keep an interval if its minimum TTC is below this ...
GATE_THW_S = 0.6     # ... or its minimum time headway is below this


def union_flags(flags, ids):
    """flags: {detector: {segment id: [[t0, t1], ...]}} -> {segment id: sorted intervals of all detectors}."""
    return {k: sorted(sum((flags[d].get(k, []) for d in flags), [])) for k in ids}


def physical_gate(intervals, series, ttc_s=GATE_TTC_S, thw_s=GATE_THW_S):
    """Keep the intervals [t0, t1] in which `series` reaches min TTC < ttc_s or min THW < thw_s."""
    keep = []
    for a, b in intervals:
        w = series[(series.t >= a) & (series.t <= b)]
        if (w.ttc.min() < ttc_s) or (w.thw.min() < thw_s):
            keep.append([a, b])
    return keep
