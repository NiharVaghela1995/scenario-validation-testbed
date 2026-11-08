"""Precision and recall of flagged intervals against labelled intervals.

Interval-overlap rule (stated once, used everywhere):
    Pad every labelled interval by TOL = 1.0 s on both sides. A flagged interval is a TRUE POSITIVE if it
    touches or overlaps at least one padded labelled interval. A labelled interval is DETECTED if
    at least one flagged interval overlaps it (padded). Then
        precision = true-positive flags / all flags
        recall    = detected labels / all labels
    Several flags on one label all count as true positives (fragmentation is reported separately as
    flags per detected label). The stricter variant (`min_iou=0.3`) additionally requires
    IoU >= 0.3 between the flag and the unpadded label.
"""
TOL = 1.0


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a, b):
    inter = overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def matches(flag, label, tol=TOL, min_iou=0.0):
    padded = [label[0] - tol, label[1] + tol]
    hit = flag[0] <= padded[1] and flag[1] >= padded[0]          # closed intervals touch or overlap
    return hit and (min_iou == 0.0 or iou(flag, label) >= min_iou)


def score(flags_by_seg, labels_by_seg, tol=TOL, min_iou=0.0):
    tp_flags = n_flags = det = n_labels = 0
    for seg, labels in labels_by_seg.items():
        flags = flags_by_seg.get(seg, [])
        n_flags += len(flags)
        n_labels += len(labels)
        tp_flags += sum(any(matches(f, l, tol, min_iou) for l in labels) for f in flags)
        det += sum(any(matches(f, l, tol, min_iou) for f in flags) for l in labels)
    p = tp_flags / n_flags if n_flags else float("nan")
    r = det / n_labels if n_labels else float("nan")
    f1 = 2 * p * r / (p + r) if n_flags and n_labels and (p + r) > 0 else float("nan")
    return dict(flags=n_flags, tp_flags=tp_flags, labels=n_labels, detected=det,
                precision=p, recall=r, f1=f1, flags_per_detected=(tp_flags / det if det else float("nan")))


def table(results_by_detector, labels_by_seg):
    rows = []
    for det, flags in results_by_detector.items():
        for rule, kw in (("any overlap ±1 s", {}), ("IoU ≥ 0.3", {"min_iou": 0.3})):
            rows.append(dict(detector=det, rule=rule, **score(flags, labels_by_seg, **kw)))
    return rows


def to_markdown(rows):
    head = "| Detector | Rule | Flags | TP flags | Labels | Detected | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|---|\n"
    f = lambda v: "–" if v != v else f"{v:.2f}"
    return head + "".join(
        f"| {r['detector']} | {r['rule']} | {r['flags']} | {r['tp_flags']} | {r['labels']} | {r['detected']} | "
        f"{f(r['precision'])} | {f(r['recall'])} | {f(r['f1'])} |\n" for r in rows)
