"""Simulator data source - trajectories generated in esmini from four scenario families.

Families:
    follow       lead varies its speed gently; ego on esmini's ACCController (timeGap 1.0 s)
    hard_brake   lead brakes at `decel` m/s^2 to `v_end`; ego on ACC
    cut_in       a car `gap` m ahead in lane -2 changes into the ego lane at t0 (2 s) and slows by `dv`; ego on ACC
    late_driver  lead slows gently (`drift` 0.2-0.5 m/s^2); ego has NO controller and brakes at `brake`
                 m/s^2 only when TTC < `ttc_react` s (OpenSCENARIO TimeToCollisionCondition) - gradual onset
Every run: 60 s, fixed time step 0.1 s, headless, --csv_logger. The run's own event times give "design"
intervals (same rule as the mock: onset -> end of closing phase, for parameters that are critical by
design); hand labels from label.py are the ground truth for evaluation.

esmini is found via ESMINI_BIN, then on PATH; without it this mode
is skipped with a message.
"""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ESMINI = Path(os.environ.get("ESMINI_BIN") or shutil.which("esmini") or "esmini-not-found")
DURATION, DT = 60.0, 0.1


def available():
    return ESMINI.exists()


def _car(xosc, name):
    return xosc.Vehicle(name, xosc.VehicleCategory.car, xosc.BoundingBox(1.8, 4.5, 1.5, 1.4, 0, 0.75),
                        xosc.Axle(0.5, 0.6, 1.6, 2.8, 0.3), xosc.Axle(0, 0.6, 1.6, 0, 0.3), 69, 10, 10)


def _speed_event(xosc, name, v, rate, t):
    ev = xosc.Event(name, xosc.Priority.override)
    ev.add_action(name + "A", xosc.AbsoluteSpeedAction(
        v, xosc.TransitionDynamics(xosc.DynamicsShapes.linear, xosc.DynamicsDimension.rate, rate)))
    ev.add_trigger(xosc.ValueTrigger(name + "T", 0, xosc.ConditionEdge.rising,
                                     xosc.SimulationTimeCondition(t, xosc.Rule.greaterThan)))
    return ev


def build(p, outdir):
    """Write road + scenario for one parameter dict p; returns the .xosc path."""
    from scenariogeneration import xodr, xosc
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    road = outdir / "road.xodr"
    if not road.exists():
        r = xodr.create_road(xodr.Line(3000), id=0, left_lanes=0, right_lanes=2, lane_width=3.5)
        odr = xodr.OpenDrive("m16_road")
        odr.add_road(r)
        odr.adjust_roads_and_lanes()
        odr.write_xml(str(road))
    ent = xosc.Entities()
    if p["kind"] == "late_driver":
        ent.add_scenario_object("Ego", _car(xosc, "ego"))
    else:
        props = xosc.Properties()
        props.add_property("esminiController", "ACCController")
        props.add_property("timeGap", "1.0")
        props.add_property("setSpeed", str(p["v0"]))
        ent.add_scenario_object("Ego", _car(xosc, "ego"), xosc.Controller("ACC", props))
    ent.add_scenario_object("Lead", _car(xosc, "lead"))
    init = xosc.Init()
    step = xosc.TransitionDynamics(xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 0)

    def place(name, s, lane, v):
        init.add_init_action(name, xosc.TeleportAction(xosc.LanePosition(s, 0, lane, 0)))
        init.add_init_action(name, xosc.AbsoluteSpeedAction(v, step))

    v0 = p["v0"]
    place("Ego", 50, -1, v0)
    # ACC families start at the ACC equilibrium (4.5 m car + 1.0 s * v0 + 3 m) so the ego does not first close up
    place("Lead", 50 + (45 if p["kind"] == "late_driver" else 4.5 + 1.0 * v0 + 3), -1, v0)
    if p["kind"] != "late_driver":
        init.add_init_action("Ego", xosc.ActivateControllerAction(False, True))
    sb_stop = xosc.ValueTrigger("Stop", 0, xosc.ConditionEdge.rising,
                                xosc.SimulationTimeCondition(DURATION, xosc.Rule.greaterThan), "stop")
    sb = xosc.StoryBoard(init, sb_stop)
    lead_man = xosc.Maneuver("LeadMan")
    if p["kind"] != "late_driver":                  # gentle background speed variation of the lead
        lead_man.add_event(_speed_event(xosc, "Var1", v0 - p["var_dv"], 0.5, 8))
        lead_man.add_event(_speed_event(xosc, "Var2", v0, 0.5, p["var2_t"]))
    if p["kind"] == "hard_brake":
        lead_man.add_event(_speed_event(xosc, "Brake", p["v_end"], p["decel"], p["t0"]))
        lead_man.add_event(_speed_event(xosc, "Recover", v0 - 2, 1.5, p["t0"] + 8))
    if p["kind"] == "late_driver":
        lead_man.add_event(_speed_event(xosc, "Drift", v0 - 12, p["drift"], p["t0"]))
        ego_man = xosc.Maneuver("EgoMan")
        ev = xosc.Event("LateBrake", xosc.Priority.override)
        ev.add_action("LateBrakeA", xosc.AbsoluteSpeedAction(
            v0 - 14, xosc.TransitionDynamics(xosc.DynamicsShapes.linear, xosc.DynamicsDimension.rate,
                                             p["brake"])))
        ev.add_trigger(xosc.EntityTrigger("TTC", 0, xosc.ConditionEdge.rising, xosc.TimeToCollisionCondition(
            p["ttc_react"], xosc.Rule.lessThan, entity="Lead", freespace=True), "Ego"))
        ego_man.add_event(ev)
        sb.add_maneuver(ego_man, "Ego")
    sb.add_maneuver(lead_man, "Lead")
    if p["kind"] == "cut_in":
        ent.add_scenario_object("Cutter", _car(xosc, "cutter"))
        place("Cutter", 50 + 4.5 + p["gap"], -2, v0)   # right lane, `gap` m (free space) ahead of the ego
        cman = xosc.Maneuver("CutMan")
        cman.add_event(_speed_event(xosc, "CutVar1", v0 - p["var_dv"], 0.5, 8))
        ev = xosc.Event("CutIn", xosc.Priority.parallel)
        ev.add_action("CutInLC", xosc.AbsoluteLaneChangeAction(
            -1, xosc.TransitionDynamics(xosc.DynamicsShapes.sinusoidal, xosc.DynamicsDimension.time, 2.0)))
        ev.add_trigger(xosc.ValueTrigger("CutInT", 0, xosc.ConditionEdge.rising,
                                         xosc.SimulationTimeCondition(p["t0"], xosc.Rule.greaterThan)))
        ev2 = xosc.Event("CutSlow", xosc.Priority.parallel)   # the cutter falls back towards the ego
        ev2.add_action("CutSlowA", xosc.AbsoluteSpeedAction(
            v0 - p["var_dv"] - p["dv"], xosc.TransitionDynamics(xosc.DynamicsShapes.linear,
                                                               xosc.DynamicsDimension.rate, 2)))
        ev2.add_trigger(xosc.ValueTrigger("CutSlowT", 0, xosc.ConditionEdge.rising,
                                          xosc.SimulationTimeCondition(p["t0"], xosc.Rule.greaterThan)))
        cman.add_event(ev2)
        cman.add_event(ev)
        sb.add_maneuver(cman, "Cutter")
    path = outdir / f"{p['id']}.xosc"
    xosc.Scenario(p["id"], "scenario_validation_testbed", xosc.ParameterDeclarations(), ent, sb,
                  xosc.RoadNetwork(road.name), xosc.Catalog(), osc_minor_version=2).write_xml(str(path))
    return path


def run(xosc_path, timeout=120):
    cwd = Path(xosc_path).parent
    csv = cwd / (Path(xosc_path).stem + ".csv")
    pr = subprocess.run([str(ESMINI), "--headless", "--osc", Path(xosc_path).name, "--fixed_timestep", str(DT),
                         "--csv_logger", csv.name, "--collision", "--disable_stdout",
                         "--logfile_path", Path(xosc_path).stem + ".log"],
                        cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return pr.returncode, csv


def parse_csv(csv):
    """esmini --csv_logger file -> frames (t, vid, lane, s_front, s_rear, v) + collision flag."""
    lines = Path(csv).read_text().splitlines()
    h = next(i for i, l in enumerate(lines) if l.startswith("Index"))
    head = [c.strip() for c in lines[h].split(",")]
    per = next(i for i, c in enumerate(head[3:], 3) if c.startswith("#2")) - 2
    rows = []
    for l in lines[h + 1:]:
        c = [x.strip() for x in l.split(",")]
        if len(c) < 2 + per:
            continue
        t = float(c[1])
        for k in range((len(c) - 2) // per):
            b = c[2 + k * per: 2 + (k + 1) * per]
            if not b[0]:
                continue
            # block: name, id, speed, wheel angle, wheel rot, bb_x, bb_y, bb_z, bb_len, ... , s (20), t, lane (22)
            vid, v, bbx, blen = int(b[1]), float(b[2]), float(b[5]), float(b[8])
            s, lane = float(b[20]), int(float(b[22]))
            rows.append((round(t, 2), vid, lane, s + bbx + blen / 2, s + bbx - blen / 2, v, b[30]))
    fr = pd.DataFrame(rows, columns=["t", "vid", "lane", "s_front", "s_rear", "v", "collision_ids"])
    return fr


def plan(n=24, seed=16):
    rng = np.random.default_rng(seed)
    kinds = ["follow", "hard_brake", "cut_in", "late_driver"]
    ps = []
    for k in range(n):
        kind = kinds[k % 4]
        p = dict(id=f"esm_{k:02d}_{kind}", kind=kind, v0=float(rng.choice([22.0, 25.0, 28.0])),
                 t0=float(np.round(rng.uniform(18, 35), 1)), var2_t=float(np.round(rng.uniform(45, 55), 1)),
                 var_dv=float(rng.choice([1.0, 2.0, 3.0, 4.0])))
        if kind == "hard_brake":
            p.update(decel=float(rng.choice([3.0, 5.0, 7.0, 9.0])), v_end=float(rng.choice([5.0, 10.0])))
        if kind == "late_driver":
            p.update(drift=float(rng.choice([0.2, 0.3, 0.5])), ttc_react=float(rng.choice([1.0, 1.5, 2.0])),
                     brake=float(rng.choice([4.0, 6.0, 8.0])))
        if kind == "cut_in":
            p.update(gap=float(rng.choice([6.0, 10.0, 15.0])), dv=float(rng.choice([3.0, 6.0])))
        ps.append(p)
    return ps


def is_critical(p):
    k = p["kind"]
    return (k == "hard_brake" and p["decel"] >= 5) or (k == "cut_in" and p["gap"] <= 10 and p["dv"] >= 6) \
        or k == "late_driver"


def make_segments(workdir, n=24, seed=16):
    from series import compute
    from synthetic import gt_interval
    segs = []
    for p in plan(n, seed):
        x = build(p, workdir)
        rc, csv = run(x)
        if rc != 0 or not csv.exists():
            print(f"esmini failed for {p['id']} (exit {rc})")
            continue
        fr = parse_csv(csv)
        ser = compute(fr, 0)
        design = []
        if is_critical(p):
            onset = p["t0"]
            if p["kind"] == "cut_in":                       # onset = the moment the cutter enters the lane
                ch = ser[(ser.lead_id == 2)]
                onset = float(ch.t.iloc[0]) if len(ch) else p["t0"]
            iv = gt_interval(ser, dict(t0=onset))
            if iv:
                design.append(iv)
        segs.append(dict(id=p["id"], frames=fr, ego_id=0, events=[dict(p, critical=is_critical(p))],
                         gt=design, series=ser, collision=bool((fr.collision_ids.astype(str).str.len() > 0).any())))
    return segs
