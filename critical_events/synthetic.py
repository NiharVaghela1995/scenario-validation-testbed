"""Mock mode - small synthetic highway logs with known critical intervals.

Every segment is 60 s at 10 Hz on a one-way road with lanes 1 and 2. The ego follows its lead with the
Intelligent Driver Model (IDM, Treiber et al. 2000) with a 0.8 s reaction delay or, in "inattentive" phases, keeps its speed until
TTC < 1.5 s and then brakes. Events are planted at known times:

    hard_brake  lead brakes at `decel` m/s^2 for `dur` s                         (sudden onset)
    cut_in      a car enters the ego lane `gap` m ahead, `dv` m/s slower       (sudden, leader change)
    drift       lead slows at 0.25 m/s^2 while the ego is inattentive           (gradual onset)
    ego_lc      ego moves to the other lane (no lead there for a while)         (TTC undefined)
    mild_brake  lead brakes gently - NOT critical by design (a distractor)

Ground truth: an event is critical by design (`critical=True`) for hard_brake with decel >= 5, cut_in with
gap <= 15 m and dv >= 4 m/s, and every drift. Its interval runs from the event onset to the end of the
closing phase (the earliest instant past the gap minimum with v_ego <= v_lead), measured in the simulation -
not from a TTC threshold, so no detector is favoured by construction. These are design rules
for a test bed; real labels come from `label.py`.
"""
import numpy as np
import pandas as pd

DT, T_END = 0.1, 60.0
L_CAR = 4.5
IDM = dict(v0=33.0, T=1.2, s0=2.0, a=1.5, b=2.0)
MAX_DECEL = 9.0
REACTION_S = 0.8     # ego perception-reaction delay (design value)


def idm_acc(v, gap, dv, v0=None):
    p = dict(IDM)
    v0 = v0 or p["v0"]
    s_star = p["s0"] + v * p["T"] + v * dv / (2 * np.sqrt(p["a"] * p["b"]))
    return p["a"] * (1 - (v / v0) ** 4 - (max(s_star, 0) / max(gap, 0.1)) ** 2)


def lead_speed_profile(t, v_init, events, rng):
    """Lead speed over time from the planted events (piecewise accelerations) + mild random variation."""
    acc = 0.15 * np.sin(2 * np.pi * t / rng.uniform(15, 30) + rng.uniform(0, 6))
    for ev in events:
        k = (t >= ev["t0"]) & (t < ev["t0"] + ev.get("dur", 0))
        if ev["kind"] in ("hard_brake", "mild_brake"):
            acc[k] = -ev["decel"]
            k2 = (t >= ev["t0"] + ev["dur"] + 2) & (t < ev["t0"] + ev["dur"] + 2 + ev["dur"] * ev["decel"] / 1.5)
            acc[k2] = 1.5                                   # recover the lost speed afterwards
        if ev["kind"] == "drift":
            acc[k] = -0.25
    v = np.maximum(v_init + np.cumsum(acc) * DT, 0.0)
    return v


def simulate(seg_id, events, rng, v_init=27.0, gap_init=45.0):
    t = np.arange(0, T_END, DT)
    n = len(t)
    v_lead = lead_speed_profile(t, v_init, events, rng)
    s_lead = 200.0 + np.cumsum(v_lead) * DT
    s_ego, v_ego = np.zeros(n), np.zeros(n)
    lane_ego = np.ones(n, dtype=int)
    s_ego[0], v_ego[0] = s_lead[0] - L_CAR - gap_init, v_init
    cut = next((e for e in events if e["kind"] == "cut_in"), None)
    lc = next((e for e in events if e["kind"] == "ego_lc"), None)
    drift = next((e for e in events if e["kind"] == "drift"), None)
    cutter_s = cutter_v = None
    braking_late = False
    pending = [0.0] * int(round(REACTION_S / DT))           # commands not yet applied (reaction delay)
    rows = []
    for i in range(n):
        ti = t[i]
        if lc and ti >= lc["t0"]:
            lane_ego[i] = 2 if ti < lc["t0"] + lc["dur"] else 1
        # current lead seen by the ego
        cand = []
        if lane_ego[i] == 1:
            cand.append(("lead", s_lead[i], v_lead[i]))
        if cut and ti >= cut["t0"]:
            if cutter_s is None:
                cutter_s, cutter_v = s_ego[i] + cut["gap"] + L_CAR, max(v_ego[i] - cut["dv"], 5.0)
            if lane_ego[i] == 1:
                cand.append(("cutter", cutter_s, cutter_v))
        if i < n - 1:
            ahead = [c for c in cand if c[1] - L_CAR > s_ego[i]]
            if ahead:
                _, sl, vl = min(ahead, key=lambda c: c[1])
                gap = sl - L_CAR - s_ego[i]
                inattentive = drift and drift["t0"] <= ti < drift["t0"] + drift["dur"] + 8
                if inattentive and not braking_late:
                    closing = v_ego[i] - vl
                    if closing > 0 and gap / closing < 1.5:
                        braking_late = True
                    a = 0.0
                if braking_late:
                    a = -6.0 if v_ego[i] > vl - 0.5 else 0.0
                    if v_ego[i] <= vl - 0.5:
                        braking_late = False
                        drift["dur"] = -100                  # attentive again
                elif not inattentive:
                    a = idm_acc(v_ego[i], gap, v_ego[i] - vl)
            else:
                a = idm_acc(v_ego[i], 1e3, 0.0)
            pending.append(float(np.clip(a, -MAX_DECEL, 2.5)))
            a = pending.pop(0)
            v_ego[i + 1] = max(v_ego[i] + a * DT, 0.0)
            s_ego[i + 1] = s_ego[i] + v_ego[i] * DT + 0.5 * a * DT ** 2
            if cutter_s is not None:
                cutter_s += cutter_v * DT
        rows.append((ti, 0, lane_ego[i], s_ego[i], v_ego[i]))
        rows.append((ti, 1, 1, s_lead[i], v_lead[i]))
        if cutter_s is not None:
            rows.append((ti, 2, 1, cutter_s, cutter_v))
        elif cut:
            rows.append((ti, 2, 2, s_ego[i] + 30.0, v_ego[i]))   # cutter waits in lane 2 before the cut-in
    fr = pd.DataFrame(rows, columns=["t", "vid", "lane", "s_front", "v"])
    noise = rng.normal(0, 0.05, len(fr))                        # position sensor noise
    fr["s_front"] = fr.s_front + noise
    fr["v"] = np.maximum(fr.v + rng.normal(0, 0.05, len(fr)), 0)
    fr["s_rear"] = fr.s_front - L_CAR
    fr["t"] = fr.t.round(1)
    return fr


def is_critical(ev):
    k = ev["kind"]
    return (k == "hard_brake" and ev["decel"] >= 5) or (k == "cut_in" and ev["gap"] <= 15 and ev["dv"] >= 4) \
        or k == "drift"


def gt_interval(ser, ev):
    """Onset -> end of the closing phase, measured on the simulated (noise-free enough) series."""
    t0 = ev["t0"]
    after = ser[ser.t >= t0]
    if not len(after):
        return None
    tmin = after.loc[after.gap.idxmin(), "t"] if after.gap.notna().any() else t0 + 3
    late = after[(after.t > tmin) & ((after.v_ego <= after.v_lead) | after.v_lead.isna())]
    t1 = float(late.t.iloc[0]) if len(late) else float(after.t.iloc[-1])
    return [round(float(t0), 1), round(max(t1, t0 + 2.0), 1)]


def plan(n_segments=30, seed=7):
    """Event plan for each segment (deterministic for a seed)."""
    rng = np.random.default_rng(seed)
    kinds = ["hard_brake", "cut_in", "drift", "mild_brake", "ego_lc", "none"]
    plans = []
    for k in range(n_segments):
        kind = kinds[k % len(kinds)]
        evs = []
        t0 = float(np.round(rng.uniform(15, 35), 1))
        if kind == "hard_brake":
            evs.append(dict(kind=kind, t0=t0, decel=float(rng.choice([3.5, 6.0, 7.0, 8.0])),
                            dur=float(np.round(rng.uniform(1.5, 2.5), 1))))
            if k % 4 == 0:                                       # twin event: the same brake again later
                evs.append(dict(evs[0], t0=t0 + 20))
        elif kind == "cut_in":
            evs.append(dict(kind=kind, t0=t0, gap=float(rng.choice([8.0, 12.0, 15.0, 25.0])),
                            dv=float(rng.choice([3.0, 5.0, 7.0]))))
        elif kind == "drift":
            evs.append(dict(kind=kind, t0=t0, dur=20.0))
        elif kind == "mild_brake":
            evs.append(dict(kind=kind, t0=t0, decel=1.5, dur=2.0))
        elif kind == "ego_lc":
            evs.append(dict(kind=kind, t0=t0, dur=6.0))
        plans.append(dict(id=f"syn_{k:02d}_{kind}", events=evs, seed=int(rng.integers(1 << 30))))
    return plans


def make_segments(n_segments=30, seed=7):
    """List of segments: {id, frames, ego_id, events, gt (critical intervals)}."""
    from series import compute
    segs = []
    for p in plan(n_segments, seed):
        rng = np.random.default_rng(p["seed"])
        events = [dict(e) for e in p["events"]]
        fr = simulate(p["id"], [dict(e) for e in events], rng)
        ser = compute(fr, 0)
        gt = []
        for ev in events:
            ev["critical"] = bool(is_critical(ev))
            if ev["critical"]:
                iv = gt_interval(ser, ev)
                if iv:
                    gt.append(iv)
        segs.append(dict(id=p["id"], frames=fr, ego_id=0, events=events, gt=gt, series=ser))
    return segs
