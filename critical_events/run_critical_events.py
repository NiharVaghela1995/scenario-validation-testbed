"""Screening runner: series -> three detectors -> labels -> precision/recall -> DTW -> tail.

    python run_critical_events.py --mock      # 30 synthetic segments, synthetic ground truth (~40 s incl. numba JIT)
    python run_critical_events.py --esmini    # 24 esmini runs; evaluated against out/esmini/labels.json if you
                                              # labelled them (label.py), else against the design intervals
Outputs in out/<mode>/: series/*.csv, ground_truth.json, flags.json, results.md/.json, plots *.png,
dtw_pairs.json, tail.json, run_log.md, run_log.json.
"""
import argparse
import os
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import detectors as D  # noqa: E402
import evaluate as E  # noqa: E402
import plots  # noqa: E402
import similarity as S  # noqa: E402
import tail as T  # noqa: E402
from series import detector_signal, status_counts  # noqa: E402

DETECTORS = ["threshold", "pelt", "discord"]


def merge_flags(lists, gap=1.0):
    ivs = sorted([list(x) for l in lists for x in l])
    out = []
    for a, b in ivs:
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def load(mode, out):
    if mode == "mock":
        from synthetic import make_segments
        return make_segments()
    import esmini_logs
    if not esmini_logs.available():
        sys.exit(f"esmini not found at {esmini_logs.ESMINI} - set ESMINI_BIN or use --mock")
    return esmini_logs.make_segments(out / "esmini_runs")


def labels_for(mode, out, segs):
    design = {s["id"]: s["gt"] for s in segs}
    if mode == "mock":
        return design, "synthetic ground truth (design rule, synthetic.py)"
    f = out / "labels.json"
    if f.exists():
        lab = json.loads(f.read_text())
        if set(lab.get("done", [])) >= set(design):
            return {k: lab["segments"].get(k, []) for k in design}, f"hand labels ({f.name}, {lab.get('labeller', '')})"
        print(f"labels.json covers {len(lab.get('done', []))}/{len(design)} segments - using design intervals")
    return design, "design intervals (esmini event times; run label.py for hand labels)"


def disagreement(segs, flags, labels):
    """Per labelled interval: which detectors found it; per false flag: which event kind it sits in."""
    found, false_by_kind = [], {d: {} for d in DETECTORS}
    for s in segs:
        kinds = "+".join(sorted({e["kind"] for e in s["events"]})) or "none"
        for lab in labels.get(s["id"], []):
            found.append(dict(seg=s["id"], label=lab, **{d: any(E.matches(f, lab) for f in flags[d].get(s["id"], []))
                                                         for d in DETECTORS}))
        for d in DETECTORS:
            for f in flags[d].get(s["id"], []):
                if not any(E.matches(f, l) for l in labels.get(s["id"], [])):
                    false_by_kind[d][kinds] = false_by_kind[d].get(kinds, 0) + 1
    return found, false_by_kind


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--esmini", action="store_true")
    a = ap.parse_args(argv)
    mode = "mock" if a.mock else "esmini"
    out = Path(os.environ.get("TESTBED_SCREENING_OUT", HERE / "out")) / mode
    (out / "series").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log = dict(mode=mode, started=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
               versions={p: version(p) for p in ("numpy", "scipy", "pandas", "ruptures", "stumpy", "numba",
                                                 "matplotlib")})

    segs = load(mode, out)
    for s in segs:
        s["signal"] = detector_signal(s["series"])
        s["series"].to_csv(out / "series" / f"{s['id']}.csv", index=False)
    (out / "ground_truth.json").write_text(json.dumps({s["id"]: s["gt"] for s in segs}, indent=1))
    (out / "events.json").write_text(json.dumps({s["id"]: s["events"] for s in segs}, indent=1, default=float))
    labels, label_source = labels_for(mode, out, segs)
    log.update(segments=len(segs), samples=int(sum(len(s["series"]) for s in segs)), label_source=label_source,
               labelled_intervals=int(sum(len(v) for v in labels.values())),
               status_counts={k: int(sum(status_counts(s["series"])[k] for s in segs))
                              for k in status_counts(segs[0]["series"])},
               t_load_s=round(time.time() - t_start, 1))

    # detectors
    t0 = time.time()
    flags = {d: {} for d in DETECTORS}
    extra = {}
    for s in segs:
        r, ex = D.run_all(s["series"], s["signal"])
        extra[s["id"]] = ex
        for d in DETECTORS:
            flags[d][s["id"]] = r[d]
    (out / "flags.json").write_text(json.dumps(flags, indent=1))
    log["t_detectors_s"] = round(time.time() - t0, 1)
    log["detector_settings"] = dict(ttc_thr_s=D.TTC_THR, pelt_model="l2", pelt_min_size=D.PELT_MIN_SIZE,
                                    pelt_pen="%.1f*sigma^2*log(n)" % D.PELT_PEN_FACTOR, pelt_level=D.PELT_LEVEL,
                                    mp_window_s=D.MP_WINDOW_S, mp_normalize=False, discord_k_mad=D.K_MAD,
                                    max_discords=D.MAX_DISCORDS)

    # evaluation
    rows = E.table(flags, labels)
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    md = E.to_markdown(rows)
    (out / "results.md").write_text(f"Labels: {label_source}\n\nRule: {E.__doc__.strip()}\n\n" + md)
    found, false_by_kind = disagreement(segs, flags, labels)
    log["detected_by"] = {d: int(sum(f[d] for f in found)) for d in DETECTORS}
    log["detected_by_none"] = [f"{f['seg']} {f['label']}" for f in found if not any(f[d] for d in DETECTORS)]
    log["detected_only_by"] = {d: [f"{f['seg']} {f['label']}" for f in found
                                   if f[d] and not any(f[o] for o in DETECTORS if o != d)] for d in DETECTORS}
    log["false_flags_by_event_kind"] = false_by_kind

    # plots
    order = sorted(segs, key=lambda s: (-len(labels.get(s["id"], [])), s["id"]))
    show = [s for s in segs if labels.get(s["id"])][:8] + [s for s in segs if not labels.get(s["id"])][:4]
    plots.timeline(show, flags, labels, out / "timeline.png")
    sweep_ids = [s["id"] for s in order[:3]]
    sweeps = {sid: D.pelt_penalty_sweep(next(s for s in segs if s["id"] == sid)["signal"]) for sid in sweep_ids}
    plots.penalty_sweep(sweeps, out / "penalty_sweep.png")
    log["penalty_sweep"] = sweeps
    ex_seg = next((s for s in segs if any(e["kind"] in ("drift", "late_driver") for e in s["events"])), segs[0])
    ex = extra[ex_seg["id"]]
    plots.matrix_profile_example(ex_seg, ex["mp"], ex["m"], flags["discord"][ex_seg["id"]], ex["bkps"],
                                 out / "mp_pelt_example.png")

    # DTW over the union of all flags
    t0 = time.time()
    events = [dict(seg=s["id"], t0=a, t1=b) for s in segs
              for a, b in merge_flags([flags[d].get(s["id"], []) for d in DETECTORS])]
    by_id = {s["id"]: s for s in segs}
    snips = S.snippets(events, {k: v["series"] for k, v in by_id.items()}, {k: v["signal"] for k, v in by_id.items()})
    keep = [k for k, sn in enumerate(snips) if len(sn) and sn.max() >= S.MIN_PEAK]   # ignore noise-only flags
    log["dtw_dropped_noise_flags"] = len(events) - len(keep)
    events, snips = [events[k] for k in keep], [snips[k] for k in keep]
    names = [f"{e['seg']}@{e['t0']:.0f}s" for e in events]
    Dm = S.distance_matrix(snips)
    pairs = S.closest_pairs(Dm, 3)
    cl = S.clusters(Dm, 4)
    eu = [S.euclid_resampled(snips[i], snips[j]) for i, j, _ in pairs]
    plots.dtw_pairs(snips, names, pairs, out / "dtw_pairs.png")
    (out / "dtw_pairs.json").write_text(json.dumps(dict(
        n_events=len(events), band=S.BAND,
        closest=[dict(a=names[i], b=names[j], dtw=round(d, 3), euclid_resampled=round(e, 3))
                 for (i, j, d), e in zip(pairs, eu)],
        clusters={int(c): [names[k] for k in np.flatnonzero(cl == c)] for c in np.unique(cl)}), indent=1))
    log["t_dtw_s"] = round(time.time() - t0, 1)
    log["dtw_events"] = len(events)

    # tail
    ranked = T.rank_events(events, {k: v["series"] for k, v in by_id.items()})
    potres = T.pot({k: v["signal"] for k, v in by_id.items()}, {k: v["series"] for k, v in by_id.items()})
    (out / "tail.json").write_text(json.dumps(dict(top10=ranked[:10], pot=potres), indent=1, default=float))
    log["tail_top5"] = [f"{r['seg']}@{r['t0']:.1f}s minTTC={r['min_ttc']:.2f}" for r in ranked[:5]]
    log["pot"] = potres
    log["runtime_s"] = round(time.time() - t_start, 1)
    (out / "run_log.json").write_text(json.dumps(log, indent=1, default=str))

    lines = [f"# Run log - {mode}", "", f"- started {log['started']}, runtime {log['runtime_s']} s "
             f"(load {log['t_load_s']} s, detectors {log['t_detectors_s']} s incl. numba JIT, DTW {log['t_dtw_s']} s)",
             f"- Python {log['python']}; " + ", ".join(f"{k} {v}" for k, v in log["versions"].items()),
             f"- {log['segments']} segments, {log['samples']} samples; labels: {label_source}; "
             f"{log['labelled_intervals']} labelled intervals",
             f"- timestep status counts: {log['status_counts']}", "", "## Results", "", md, "",
             f"Detected by: {log['detected_by']}; by none: {log['detected_by_none']}",
             f"Only by one detector: {log['detected_only_by']}",
             f"False flags by event kind: {false_by_kind}", "",
             f"## DTW: {len(events)} flagged events with peak 1/TTC >= {S.MIN_PEAK} (union; "
             f"{log['dtw_dropped_noise_flags']} noise-only flags dropped), closest pairs", "",
             *[f"- {names[i]} ~ {names[j]}: DTW {d:.2f} (resampled Euclidean {e:.2f})" for (i, j, d), e in zip(pairs, eu)],
             "", "## Tail", "", *[f"- {x}" for x in log["tail_top5"]], f"- POT: {potres}", "",
             f"## PELT penalty sweep (factor, #change points)", "", *[f"- {k}: {v}" for k, v in sweeps.items()]]
    (out / "run_log.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return log


if __name__ == "__main__":
    main()
