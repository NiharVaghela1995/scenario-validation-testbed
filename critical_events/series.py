"""Surrogate safety series per timestep for one ego vehicle.

Input: a long table `frames` with one row per (t, vehicle):
    t [s], vid, lane, s_front [m], s_rear [m], v [m/s]      (s = distance along the road, + in driving direction)
plus the id of the ego vehicle. Output: one row per timestep of the ego with

    lead_id, gap [m] (ego front bumper -> lead rear bumper), v_ego, v_lead,
    ttc [s], thw [s], drac [m/s^2], inv_ttc [1/s], status

Definitions (UN R157 TTC; Westhofen et al. review):
    TTC  = gap / (v_ego - v_lead)          only if v_ego > v_lead (closing), else undefined
    THW  = gap / v_ego                     only if v_ego > V_MIN
    DRAC = (v_ego - v_lead)^2 / (2 gap)    only if closing (lead assumed to keep its speed)
    1/TTC = (v_ego - v_lead) / gap         0 when not closing; finite, continuous, "bigger = worse"

Edge cases -> status (TTC is NaN for every status except "ok"):
    no_leader      nobody ahead in the ego lane (within MAX_RANGE)
    opening        v_ego <= v_lead: no collision course. Westhofen et al. define TTC = infinity here; the
                   code stores NaN + this status so that infinity never enters a mean or a plot
    lane_change    the ego's lane id changed within +-LC_WINDOW s: lead assignment is ambiguous
    collision      gap <= 0 (bounding boxes overlap longitudinally)
    leader_change  the lead id differs from the previous step (cut-in / cut-out); TTC is computed if
                   closing, but the series jumps here - the flag (and column lead_switch) marks it.
                   A cut-in can leave a tiny gap that is OPENING (TTC undefined, THW ~ 0.1 s)
THW and gap stay defined while opening; inv_ttc is set to 0 for opening/no_leader and to NaN for
lane_change, and is filled for the detectors by `detector_signal()`.
"""
import numpy as np
import pandas as pd

V_MIN = 0.5          # m/s: below this THW is undefined (standstill)
MAX_RANGE = 150.0    # m: a vehicle further ahead is not treated as a lead
LC_WINDOW = 1.0      # s: lead assignment ambiguous this long around an ego lane change
INV_TTC_CAP = 5.0    # 1/s: cap for the detector signal (TTC 0.2 s); collisions are set to the cap

STATUSES = ["ok", "leader_change", "opening", "no_leader", "lane_change", "collision"]


def compute(frames, ego_id):
    """Per-timestep TTC, THW, DRAC, 1/TTC and status for `ego_id` (see module docstring)."""
    f = frames.sort_values(["t", "vid"])
    ego = f[f.vid == ego_id].drop_duplicates("t").reset_index(drop=True)
    oth = f[f.vid != ego_id]
    o_t, o_vid, o_lane = oth.t.to_numpy(), oth.vid.to_numpy(), oth.lane.to_numpy()
    o_front, o_rear, o_v = oth.s_front.to_numpy(), oth.s_rear.to_numpy(), oth.v.to_numpy()
    order = np.argsort(o_t, kind="stable")
    bounds = np.searchsorted(o_t[order], ego.t.to_numpy(), side="left"), \
        np.searchsorted(o_t[order], ego.t.to_numpy(), side="right")
    e_t, e_lane, e_front, e_v = (ego[c].to_numpy() for c in ("t", "lane", "s_front", "v"))
    lc_times = e_t[np.r_[False, np.diff(e_lane) != 0]]
    rows, prev_lead = [], None
    for i in range(len(ego)):
        row = dict(t=e_t[i], lead_id=np.nan, gap=np.nan, v_ego=e_v[i], v_lead=np.nan,
                   ttc=np.nan, thw=np.nan, drac=np.nan, inv_ttc=0.0, status="no_leader", lead_switch=False)
        idx = order[bounds[0][i]:bounds[1][i]]
        m = (o_lane[idx] == e_lane[i]) & (o_front[idx] > e_front[i]) & (o_rear[idx] - e_front[i] < MAX_RANGE)
        idx = idx[m]
        lead = None
        if len(idx):
            j = idx[np.argmin(o_rear[idx])]
            lead = o_vid[j]
            gap, closing = o_rear[j] - e_front[i], e_v[i] - o_v[j]
            row.update(lead_id=lead, gap=gap, v_lead=o_v[j])
            if e_v[i] > V_MIN:
                row["thw"] = max(gap, 0.0) / e_v[i]
            if gap <= 0:
                row.update(status="collision", ttc=0.0, inv_ttc=INV_TTC_CAP)
            elif closing > 0:
                row.update(ttc=gap / closing, drac=closing ** 2 / (2 * gap),
                           inv_ttc=min(closing / gap, INV_TTC_CAP), status="ok")
            else:
                row["status"] = "opening"
            if prev_lead is not None and lead != prev_lead:
                row["lead_switch"] = True
                if row["status"] in ("ok", "opening"):
                    row["status"] = "leader_change"
        prev_lead = lead
        if len(lc_times) and np.min(np.abs(lc_times - e_t[i])) <= LC_WINDOW:
            row.update(status="lane_change", ttc=np.nan, drac=np.nan, inv_ttc=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def detector_signal(ser):
    """Finite criticality signal for the detectors: 1/TTC, gaps (lane changes) linearly interpolated."""
    x = ser.inv_ttc.astype(float).interpolate(limit_direction="both").fillna(0.0)
    return np.clip(x.to_numpy(), 0.0, INV_TTC_CAP)


def status_counts(ser):
    c = ser.status.value_counts()
    return {s: int(c.get(s, 0)) for s in STATUSES}


def min_ttc(ser, t0=None, t1=None):
    s = ser if t0 is None else ser[(ser.t >= t0) & (ser.t <= t1)]
    v = s.ttc.dropna()
    return float(v.min()) if len(v) else float("inf")
