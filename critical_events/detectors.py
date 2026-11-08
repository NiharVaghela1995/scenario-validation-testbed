"""Three detectors that turn a series into flagged intervals [t0, t1] (seconds).

(a) threshold   TTC < TTC_THR s (default 3.0 s: an ASSUMPTION; UN R157 uses 2 s for cut-ins,
                the literature uses 1.5-4 s). Runs shorter than MIN_DUR are
                dropped, runs closer than MERGE_GAP are merged.
(b) pelt        ruptures.Pelt(model="l2", min_size, jump=1) on the 1/TTC signal; predict(pen=...).
                A segment between two change points is flagged if its mean 1/TTC exceeds
                LEVEL (1/s, i.e. the segment's mean TTC is below 1/LEVEL) - a change point alone says
                "something changed", not "it got dangerous".
                Default penalty: 3 * sigma^2 * log(n), sigma^2 estimated robustly from first differences
                (MAD); `pelt_penalty_sweep` shows how the result moves with the penalty.
(c) discord     stumpy.stump(signal, m, normalize=False) = non-normalised matrix profile (Euclidean
                distance to the nearest non-trivial neighbour subsequence). Discords = local maxima of the
                profile above max(median + K_MAD * MAD, MP_FLOOR_RMS * sqrt(m)), at most MAX_DISCORDS per
                log, each excluding +-m. Without the absolute floor a quiet log still has "discords": the
                most unusual stretch of a boring log is still boring.
                The flagged interval is the discord subsequence [i, i + m).
"""
import numpy as np
import ruptures as rpt

TTC_THR = 3.0
MIN_DUR, MERGE_GAP = 0.3, 0.5
PELT_MIN_SIZE = 10            # samples (1 s at 10 Hz)
PELT_LEVEL = 0.2              # 1/s  (segment mean TTC below 5 s)
PELT_PEN_FACTOR = 3.0
MP_WINDOW_S = 4.0
K_MAD, MAX_DISCORDS = 5.0, 3
MP_FLOOR_RMS = 0.05          # 1/s: a discord must differ by at least this RMS 1/TTC from its neighbour


def runs_to_intervals(mask, t, min_dur=MIN_DUR, merge_gap=MERGE_GAP):
    """Boolean mask over samples -> list of [t0, t1] intervals (merged, short ones dropped)."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    iv = [[float(t[s]), float(t[e])] for s, e in zip(starts, ends)]
    merged = [iv[0]]
    for a, b in iv[1:]:
        if a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [[round(a, 2), round(b, 2)] for a, b in merged if b - a >= min_dur]


# (a) ------------------------------------------------------------------------------------------
def threshold(ser, thr=TTC_THR):
    ttc = ser.ttc.to_numpy(float)
    return runs_to_intervals(np.nan_to_num(ttc, nan=np.inf) < thr, ser.t.to_numpy())


# (b) ------------------------------------------------------------------------------------------
def robust_sigma2(x):
    d = np.diff(x)
    mad = np.median(np.abs(d - np.median(d)))
    return max((1.4826 * mad) ** 2 / 2, 1e-4)      # var of x from var of differences


def default_penalty(x, factor=PELT_PEN_FACTOR):
    return factor * robust_sigma2(x) * np.log(len(x))


def change_points(x, pen=None, min_size=PELT_MIN_SIZE, model="l2"):
    """Breakpoint indices (ruptures convention: the last one is len(x))."""
    pen = default_penalty(x) if pen is None else pen
    return rpt.Pelt(model=model, min_size=min_size, jump=1).fit(np.asarray(x, float)).predict(pen=pen)


def pelt(x, t, pen=None, level=PELT_LEVEL, min_size=PELT_MIN_SIZE):
    bkps = change_points(x, pen, min_size)
    mask = np.zeros(len(x), bool)
    start = 0
    for b in bkps:
        if np.mean(x[start:b]) > level:
            mask[start:b] = True
        start = b
    return runs_to_intervals(mask, t), bkps


def pelt_penalty_sweep(x, factors=(0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000)):
    """Number of change points for penalties factor * sigma^2 * log(n)."""
    base = robust_sigma2(x) * np.log(len(x))
    return [(f, len(change_points(x, f * base)) - 1) for f in factors]


# (c) ------------------------------------------------------------------------------------------
def matrix_profile(x, m):
    import stumpy
    mp = stumpy.stump(np.asarray(x, float), m, normalize=False)
    return mp[:, 0].astype(float)


def discords(x, t, window_s=MP_WINDOW_S, dt=None, k_mad=K_MAD, max_k=MAX_DISCORDS):
    dt = dt or float(np.median(np.diff(t)))
    m = max(4, int(round(window_s / dt)))
    if len(x) < 3 * m:
        return [], np.array([]), m
    P = matrix_profile(x, m)
    P = np.where(np.isfinite(P), P, np.nan)
    med = np.nanmedian(P)
    mad = np.nanmedian(np.abs(P - med)) * 1.4826
    thr = max(med + k_mad * mad, MP_FLOOR_RMS * np.sqrt(m))   # relative AND absolute (noise-free logs: MAD ~ 0)
    P_work = P.copy()
    out = []
    for _ in range(max_k):
        if np.all(np.isnan(P_work)):
            break
        i = int(np.nanargmax(P_work))
        if P_work[i] <= thr:
            break
        out.append([round(float(t[i]), 2), round(float(t[min(i + m - 1, len(t) - 1)]), 2)])
        P_work[max(0, i - m):i + m] = np.nan                  # exclusion zone
    return sorted(out), P, m


def run_all(ser, signal):
    t = ser.t.to_numpy()
    th = threshold(ser)
    pe, bkps = pelt(signal, t)
    di, P, m = discords(signal, t)
    return {"threshold": th, "pelt": pe, "discord": di}, {"bkps": bkps, "mp": P, "m": m}
