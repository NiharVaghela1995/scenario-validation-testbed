"""DTW between flagged events, clustering and the closest pairs.

DTW (Sakoe & Chiba 1978) aligns two sequences by a monotone warping path and sums the local costs along it.
Cost: O(n * m) time for the full matrix; the Sakoe-Chiba band |i - j * n/m| <= w restricts the path to a
diagonal band and cuts the cost to O(n * w). A narrow band also stops pathological warpings
(Ratanamahatana & Keogh). Plain numpy here, so there is no extra dependency.

Event snippet: 1/TTC from PRE s before to POST s after the minimum TTC of a flagged interval (union of all
detectors' flags, merged), so snippets of different events have different lengths only if they hit the log
boundary; their *shapes* differ in speed, which is what DTW absorbs.
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

PRE, POST = 4.0, 4.0
BAND = 0.1          # Sakoe-Chiba half-width as a share of the longer sequence
MIN_PEAK = 0.1      # 1/s: snippets that never reach TTC <= 10 s are noise, not events


def dtw(a, b, band=BAND):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, m = len(a), len(b)
    w = max(int(np.ceil(band * max(n, m))), abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        c = i * m / n
        lo, hi = max(1, int(c - w)), min(m, int(c + w) + 1)
        for j in range(lo, hi + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def euclid_resampled(a, b):
    n = max(len(a), len(b))
    ra = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a)), a)
    rb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b)), b)
    return float(np.linalg.norm(ra - rb))


def snippets(events, series_by_seg, signal_by_seg):
    """events: list of {seg, t0, t1, ...} -> list of 1-D arrays around each event's min TTC."""
    out = []
    for ev in events:
        ser, x = series_by_seg[ev["seg"]], signal_by_seg[ev["seg"]]
        t = ser.t.to_numpy()
        k = (t >= ev["t0"]) & (t <= ev["t1"])
        i_peak = np.flatnonzero(k)[np.argmax(x[k])] if k.any() else int(np.searchsorted(t, ev["t0"]))
        dt = float(np.median(np.diff(t)))
        lo, hi = max(0, i_peak - int(PRE / dt)), min(len(t), i_peak + int(POST / dt) + 1)
        out.append(x[lo:hi])
    return out


def distance_matrix(snips, band=BAND):
    n = len(snips)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw(snips[i], snips[j], band)
    return D


def closest_pairs(D, k=3):
    iu = np.triu_indices(len(D), 1)
    order = np.argsort(D[iu])[:k]
    return [(int(iu[0][o]), int(iu[1][o]), float(D[iu][o])) for o in order]


def clusters(D, n_clusters=4):
    if len(D) < 2:
        return np.ones(len(D), int)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=min(n_clusters, len(D)), criterion="maxclust")
