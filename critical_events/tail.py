"""Tail ranking and peaks-over-threshold.

Ranking: every flagged event gets its minimum TTC (and max DRAC); sort ascending by min TTC.

Peaks over threshold (POT): take the per-timestep criticality y = 1/TTC (bigger = worse), keep the
exceedances y - u over a high threshold u, DECLUSTER them (runs method: exceedances closer than R_DECLUSTER
seconds belong to one cluster; keep each cluster's maximum - consecutive samples of one event are not
independent), and fit a generalised Pareto distribution GPD(c, scale) with location fixed at 0
(scipy.stats.genpareto.fit(..., floc=0)). With c < 0 the fitted tail has a finite end point u - scale/c;
"P(1/TTC > 1/TTC_crash)" is then an extrapolation - meaningful only with enough independent exceedances
(rule of thumb here: >= 30; fewer -> the fit is reported but flagged as unreliable).
"""
import numpy as np
from scipy.stats import genpareto

R_DECLUSTER = 5.0
MIN_EXCEEDANCES = 30


def rank_events(events, series_by_seg):
    rows = []
    for ev in events:
        s = series_by_seg[ev["seg"]]
        w = s[(s.t >= ev["t0"]) & (s.t <= ev["t1"])]
        ttc = w.ttc.dropna()
        rows.append(dict(ev, min_ttc=float(ttc.min()) if len(ttc) else float("inf"),
                         max_drac=float(w.drac.max()) if w.drac.notna().any() else 0.0))
    return sorted(rows, key=lambda r: r["min_ttc"])


def decluster(t, y, u, r=R_DECLUSTER):
    """Cluster maxima of exceedances y > u (runs method with run length r seconds)."""
    idx = np.flatnonzero(y > u)
    if not len(idx):
        return np.array([])
    peaks, cur = [], [idx[0]]
    for i in idx[1:]:
        if t[i] - t[cur[-1]] > r:
            peaks.append(max(cur, key=lambda j: y[j]))
            cur = [i]
        else:
            cur.append(i)
    peaks.append(max(cur, key=lambda j: y[j]))
    return y[peaks] - u


def pot(signals_by_seg, series_by_seg, u_quantile=0.98):
    ys = np.concatenate([signals_by_seg[k] for k in signals_by_seg])
    u = float(np.quantile(ys, u_quantile))
    exc_all = np.concatenate([decluster(series_by_seg[k].t.to_numpy(), signals_by_seg[k], u)
                              for k in signals_by_seg])
    n_raw = int((ys > u).sum())
    res = dict(u=u, u_ttc_s=(1 / u if u > 0 else float("inf")), n_raw_exceedances=n_raw,
               n_declustered=int(len(exc_all)), reliable=bool(len(exc_all) >= MIN_EXCEEDANCES))
    if len(exc_all) >= 5:
        c, _, scale = genpareto.fit(exc_all, floc=0)
        res.update(shape_c=float(c), scale=float(scale))
        if c < 0:
            res["endpoint_inv_ttc"] = float(u - scale / c)
    return res
