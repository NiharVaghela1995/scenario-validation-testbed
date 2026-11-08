"""The generation chain on one event type (cut-ins), synthetic test bed only.

Event parameters, taken at t0, the instant the cutting-in vehicle begins to move across:
    v_ego  [m/s]  ego speed
    gap    [m]    free space between the ego's front and the cutter's rear (longitudinal)
    dv     [m/s]  relative speed v_ego - v_cutter (> 0: the cutter is slower, the gap closes)
    v_lat  [m/s]  the cutter's lateral speed; lane change duration T_lc = LANE_W / v_lat
Half-way through the manoeuvre, at T_lc / 2 (sinusoidal lateral path), the centre of the cutting-in vehicle enters the ego lane. Assuming constant
speeds until then, the gap at that moment is gap_sw = gap - dv * T_lc / 2 and the criticality measure is
TTC_sw = gap_sw / dv (dv > 0). An event is CRITICAL if TTC_sw < TTC_CRIT (2 s, the UN R157 cut-in line).

generate_events: a seeded, deliberately non-Gaussian generator (conditional structure, rejection of physically
impossible events), standing in for mined cut-ins. It is NOT one of the fitted model families, so no model is right
by construction.
"""
import numpy as np

PARAMS = ["v_ego", "gap", "dv", "v_lat"]
UNITS = {"v_ego": "m/s", "gap": "m", "dv": "m/s", "v_lat": "m/s"}
LANE_W = 3.5          # m
TTC_CRIT = 2.0        # s
A_MAX = 8.0           # m/s^2: required deceleration above this is beyond a passenger car's braking
V_LAT_MAX = 3.0       # m/s


def derived(X):
    """X (n x 4, PARAMS order) -> dict of T_lc, gap_sw, ttc_sw (inf if not closing), drac_sw, v_cut."""
    X = np.atleast_2d(X)
    v, g, dv, vl = X.T
    t_lc = LANE_W / np.maximum(vl, 1e-6)
    g_sw = g - dv * t_lc / 2
    closing = dv > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(closing & (g_sw > 0), g_sw / dv, np.inf)
        drac = np.where(closing & (g_sw > 0), dv ** 2 / (2 * g_sw), 0.0)
    return dict(t_lc=t_lc, gap_sw=g_sw, ttc_sw=ttc, drac_sw=drac, v_cut=v - dv)


def validity(X):
    """Physical-validity rules per sample -> dict rule -> boolean array (True = rule satisfied), plus 'all'."""
    X = np.atleast_2d(X)
    v, g, dv, vl = X.T
    d = derived(X)
    rules = {
        "gap > 0": g > 0,
        "0 < v_ego <= 45 m/s": (v > 0) & (v <= 45),
        "v_cutter > 0": d["v_cut"] > 0,
        "0 < v_lat <= 3 m/s": (vl > 0) & (vl <= V_LAT_MAX),
        "gap at lane crossing > 0": d["gap_sw"] > 0,
        "required decel <= 8 m/s^2": d["drac_sw"] <= A_MAX,
    }
    rules["all"] = np.logical_and.reduce(list(rules.values()))
    return rules


def critical(X):
    return derived(X)["ttc_sw"] < TTC_CRIT


def generate_events(n=400, seed=2026):
    """n physically valid cut-ins with correlated parameters (rejection sampling; seeded)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        m = 4 * n
        v = 12.0 + 30.0 * rng.beta(5, 4, m)                                   # 12..42 m/s, mode ~ 29
        dv = rng.normal(1.5 + 0.15 * (v - 28.0), 3.0)                         # faster ego -> larger closing
        log_tau = rng.normal(np.log(0.7) + 0.05 * dv, 0.4)                    # larger closing -> larger accepted time gap
        gap = v * np.exp(log_tau)
        v_lat = np.exp(rng.normal(np.log(0.9) - 0.4 * (log_tau - np.log(0.7)), 0.2))   # tight gaps -> brisk lane change
        X = np.c_[v, gap, dv, v_lat]
        out += list(X[validity(X)["all"]])
    return np.asarray(out[:n])


# ------------------------------------------------------------------------------------------------ fitted models
from scipy import stats  # noqa: E402


class _Marginal:
    """1-D Gaussian KDE (Scott's rule) with an analytic CDF and a grid-based inverse CDF."""

    def __init__(self, x):
        self.k = stats.gaussian_kde(x)
        self.x = np.asarray(x, float)
        self.h = float(np.sqrt(self.k.covariance[0, 0]))
        self.grid = np.linspace(self.x.min() - 6 * self.h, self.x.max() + 6 * self.h, 4001)
        self.cdf_grid = self.cdf(self.grid)

    def logpdf(self, v):
        return np.log(np.maximum(self.k(v), 1e-300))

    def cdf(self, v):
        v = np.asarray(v, float)
        return stats.norm.cdf((v[:, None] - self.x[None, :]) / self.h).mean(1)

    def ppf(self, u):
        return np.interp(u, self.cdf_grid, self.grid)

    def sample(self, n, rng):
        return self.k.resample(n, seed=rng)[0]


class Independent:
    """Product of 1-D KDE marginals: the 'independent ranges' baseline with data-fitted marginals."""
    name = "independent marginals (1-D KDE)"

    def fit(self, X):
        self.m = [_Marginal(X[:, j]) for j in range(X.shape[1])]
        return self

    def logpdf(self, X):
        return sum(m.logpdf(X[:, j]) for j, m in enumerate(self.m))

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        return np.c_[tuple(m.sample(n, rng) for m in self.m)]

    def info(self):
        return dict(bandwidths=[m.h for m in self.m])


class KDE:
    """Multivariate Gaussian KDE on standardised parameters (Scott's rule, full covariance)."""
    name = "multivariate KDE"

    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.k = stats.gaussian_kde(((X - self.mu) / self.sd).T)
        return self

    def logpdf(self, X):
        return self.k.logpdf(((X - self.mu) / self.sd).T) - np.log(self.sd).sum()

    def sample(self, n, seed):
        return self.k.resample(n, seed=np.random.default_rng(seed)).T * self.sd + self.mu

    def info(self):
        return dict(scott_factor=float(self.k.factor))


class GaussianCopula:
    """Gaussian copula on normal scores of 1-D KDE marginals (the same marginals as Independent)."""
    name = "Gaussian copula + KDE marginals"

    def fit(self, X):
        self.m = [_Marginal(X[:, j]) for j in range(X.shape[1])]
        Z = self.to_z(X)
        self.R = np.corrcoef(Z.T)
        self.Ri = np.linalg.inv(self.R)
        self.logdet = float(np.linalg.slogdet(self.R)[1])
        return self

    def to_z(self, X):
        U = np.c_[tuple(m.cdf(X[:, j]) for j, m in enumerate(self.m))]
        return stats.norm.ppf(np.clip(U, 1e-9, 1 - 1e-9))

    def from_z(self, Z):
        U = stats.norm.cdf(Z)
        return np.c_[tuple(m.ppf(U[:, j]) for j, m in enumerate(self.m))]

    def log_copula(self, Z):
        return -0.5 * self.logdet - 0.5 * np.einsum("ij,jk,ik->i", Z, self.Ri - np.eye(len(self.R)), Z)

    def logpdf(self, X):
        return sum(m.logpdf(X[:, j]) for j, m in enumerate(self.m)) + self.log_copula(self.to_z(X))

    def sample_z(self, n, seed, mean=None):
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(np.zeros(len(self.R)) if mean is None else mean, self.R, size=n)

    def sample(self, n, seed):
        return self.from_z(self.sample_z(n, seed))

    def info(self):
        return dict(bandwidths=[m.h for m in self.m], R=np.round(self.R, 3).tolist())


MODELS = (Independent, KDE, GaussianCopula)


# ------------------------------------------------------------------------------------------------ importance sampling
def ttc_valid(X):
    """TTC at lane crossing for physically valid samples, inf for invalid ones (an invalid sample is no event)."""
    t = derived(X)["ttc_sw"].copy()
    t[~validity(X)["all"]] = np.inf
    return t


def ce_mean_shift(cop, ttc_max, n_pilot=2000, elite=0.1, seed=11):
    """One cross-entropy step in the copula's normal-score space: mean z of the pilot samples with the smallest
    TTC at lane crossing (the elite quantile, or all samples below ttc_max if there are more of them)."""
    Z = cop.sample_z(n_pilot, seed)
    t = ttc_valid(cop.from_z(Z))
    k = max(int(elite * n_pilot), int((t < ttc_max).sum()))
    return Z[np.argsort(t)[:k]].mean(0)


def is_estimate(cop, mu, ttc_max, n, seed):
    """Importance sampling of P(valid and TTC_sw < ttc_max) under the copula: proposal N(mu, R) in z-space,
    weight w = phi_R(z) / phi_R(z - mu) = exp(-mu' R^-1 z + mu' R^-1 mu / 2) (the marginal transforms cancel)."""
    Z = cop.sample_z(n, seed, mean=mu)
    a = cop.Ri @ mu
    w = np.exp(-Z @ a + 0.5 * mu @ a)
    hit = ttc_valid(cop.from_z(Z)) < ttc_max
    f = w * hit
    return dict(estimate=float(f.mean()), se=float(f.std(ddof=1) / np.sqrt(n)), ess=float(w.sum() ** 2 / (w ** 2).sum()),
                ess_hits=float(f.sum() ** 2 / (f ** 2).sum()) if hit.any() else 0.0, hit_share=float(hit.mean()),
                w_mean=float(w.mean()), w_min=float(w.min()), w_median=float(np.median(w)), w_max=float(w.max()),
                w_hits_median=float(np.median(w[hit])) if hit.any() else None), Z, w, hit


def crude_estimate(cop, ttc_max, n, seed):
    hit = ttc_valid(cop.sample(n, seed)) < ttc_max
    p = float(hit.mean())
    return dict(estimate=p, se=float(np.sqrt(max(p * (1 - p), 1e-300) / n)), hits=int(hit.sum()))


# ------------------------------------------------------------------------------------------------ replay in esmini
T0_LC, REPLAY_S = 2.0, 15.0


def build_cutin_xosc(p, outdir, name, acc=False):
    """One cut-in variant as OpenSCENARIO (scenariogeneration): ego in lane -1 at constant v_ego (acc=False, open loop)
    or on esmini's ACCController (acc=True; timeGap 1 s, setSpeed v_ego), cutter in lane -2 placed so that its free gap equals `gap` at t0 = 2 s, then a sinusoidal lane
    change of duration LANE_W / v_lat; both keep their speeds otherwise. 15 s, no other traffic."""
    from pathlib import Path
    from scenariogeneration import xodr, xosc
    import esmini_logs as EL
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    road = outdir / "road.xodr"
    if not road.exists():
        r = xodr.create_road(xodr.Line(3000), id=0, left_lanes=0, right_lanes=2, lane_width=LANE_W)
        odr = xodr.OpenDrive("cutin_road")
        odr.add_road(r)
        odr.adjust_roads_and_lanes()
        odr.write_xml(str(road))
    v, g, dv, vl = (float(p[k]) for k in PARAMS)
    ent = xosc.Entities()
    if acc:
        props = xosc.Properties()
        props.add_property("esminiController", "ACCController")
        props.add_property("timeGap", "1.0")
        props.add_property("setSpeed", str(v))
        ent.add_scenario_object("Ego", EL._car(xosc, "ego"), xosc.Controller("ACC", props))
    else:
        ent.add_scenario_object("Ego", EL._car(xosc, "ego"))
    ent.add_scenario_object("Cutter", EL._car(xosc, "cutter"))
    init = xosc.Init()
    step = xosc.TransitionDynamics(xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 0)
    for nm, s, lane, sp in (("Ego", 50.0, -1, v), ("Cutter", 50.0 + 4.5 + g + dv * T0_LC, -2, v - dv)):
        init.add_init_action(nm, xosc.TeleportAction(xosc.LanePosition(s, 0, lane, 0)))
        init.add_init_action(nm, xosc.AbsoluteSpeedAction(sp, step))
    if acc:
        init.add_init_action("Ego", xosc.ActivateControllerAction(False, True))
    sb = xosc.StoryBoard(init, xosc.ValueTrigger("Stop", 0, xosc.ConditionEdge.rising,
                                                 xosc.SimulationTimeCondition(REPLAY_S, xosc.Rule.greaterThan), "stop"))
    man = xosc.Maneuver("CutMan")
    ev = xosc.Event("CutIn", xosc.Priority.parallel)
    ev.add_action("CutInLC", xosc.AbsoluteLaneChangeAction(
        -1, xosc.TransitionDynamics(xosc.DynamicsShapes.sinusoidal, xosc.DynamicsDimension.time, LANE_W / vl)))
    ev.add_trigger(xosc.ValueTrigger("CutInT", 0, xosc.ConditionEdge.rising,
                                     xosc.SimulationTimeCondition(T0_LC, xosc.Rule.greaterThan)))
    man.add_event(ev)
    sb.add_maneuver(man, "Cutter")
    path = outdir / f"{name}.xosc"
    xosc.Scenario(name, "scenario_validation_testbed_cutin", xosc.ParameterDeclarations(), ent, sb, xosc.RoadNetwork(road.name),
                  xosc.Catalog(), osc_minor_version=2).write_xml(str(path))
    return path


def _run(p, outdir, name, acc):
    import esmini_logs as EL
    import series as SE
    code, csv = EL.run(build_cutin_xosc(p, outdir, name, acc), timeout=60)
    fr = EL.parse_csv(csv)
    collision = bool(fr.collision_ids.astype(str).str.strip().replace({"nan": ""}).ne("").any())
    ser = SE.compute(fr.drop(columns="collision_ids"), ego_id=0)
    return code, ser, collision


def replay(p, outdir, name, window_s=4.0):
    """Replay one variant in esmini twice.
    Open loop, with the ego speed held fixed to match the criticality definition: TTC at the moment the cutter becomes the ego's
    lead (its reference point enters the lane). Reproduced = (replayed TTC_sw < TTC_CRIT) == (intended TTC_sw < TTC_CRIT).
    Closed loop (ego on esmini's ACCController): the system's response within window_s after the switch - min gap,
    min THW, min TTC, ego peak deceleration, collision. Not part of the reproduction check."""
    X = np.array([[p[k] for k in PARAMS]])
    dd = derived(X)
    out = dict(name=name, **{k: float(p[k]) for k in PARAMS}, intended_ttc_sw=float(dd["ttc_sw"][0]),
               intended_gap_sw=float(dd["gap_sw"][0]), t_switch_planned=T0_LC + LANE_W / p["v_lat"] / 2)
    code, ser, col = _run(p, outdir, name + "_open", acc=False)
    sw = ser[ser.lead_id == 1]                # first step with the cutter (vid 1) as lead; the ego had no lead before
    out.update(esmini_exit_open=code, switched=bool(len(sw)))
    if len(sw):
        r = sw.iloc[0]
        out.update(t_switch=float(r.t), replay_gap_sw=float(r.gap),
                   replay_ttc_sw=float(r.ttc) if np.isfinite(r.ttc) else float("inf"))
        out["intended_critical"] = out["intended_ttc_sw"] < TTC_CRIT
        out["replay_critical"] = out["replay_ttc_sw"] < TTC_CRIT
        out["reproduced"] = out["intended_critical"] == out["replay_critical"]
    code, ser, col = _run(p, outdir, name + "_acc", acc=True)
    sw = ser[ser.lead_id == 1]
    if len(sw):
        t_sw = float(sw.t.iloc[0])
        win = ser[(ser.t >= t_sw) & (ser.t <= t_sw + window_s)]
        a = np.gradient(ser.v_ego.to_numpy(), 0.1)
        ttc = win.ttc.dropna()
        out["acc"] = dict(esmini_exit=code, min_gap=float(win.gap.min()), min_thw=float(win.thw.min()),
                          min_ttc=float(ttc.min()) if len(ttc) else float("inf"), peak_decel=float(-a.min()),
                          collision=col, decel_over_limit=bool(-a.min() > A_MAX))
    return out


# ------------------------------------------------------------------------------------------------ robustness: second generator
GENERATOR_DEPENDENCE = {
    "G1": "Essentially Gaussian: given v_ego, (dv, log time gap, log v_lat) is a linear-Gaussian chain (dv linear in v_ego; "
          "log(gap/v_ego) linear in dv; log v_lat linear in log(gap/v_ego); normal noise). Departures from normality come from three places: a Beta "
          "draw for ego speed, multiplying time gap by v_ego to get gap, and discarding impossible events. No tail dependence.",
    "G2": "G1 body (92 %) plus an 8 % 'aggressive' component (before rejection): dv ~ N(6.5, 1.0), log(gap/v_ego) = log 0.38 "
          "- 0.08 (dv - 6.5) + N(0, 0.12), log v_lat ~ N(log 1.7, 0.15): small time gaps coincide with fast closing only in "
          "this tail component (the body links larger closing speed to LARGER gaps), a joint tail no Gaussian dependence has.",
}


def generate_events_tail(n=400, seed=2026, share=0.08):
    """Generator G2: G1 plus a tail component in which tight gaps come with high closing speed."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        m = 4 * n
        v = 12.0 + 30.0 * rng.beta(5, 4, m)
        agg = rng.random(m) < share
        dv = np.where(agg, rng.normal(6.5, 1.0, m), rng.normal(1.5 + 0.15 * (v - 28.0), 3.0, m))
        log_tau = np.where(agg, np.log(0.38) - 0.08 * (dv - 6.5) + rng.normal(0, 0.12, m),
                           rng.normal(np.log(0.7) + 0.05 * dv, 0.4, m))
        gap = v * np.exp(log_tau)
        v_lat = np.where(agg, np.exp(rng.normal(np.log(1.7), 0.15, m)),
                         np.exp(rng.normal(np.log(0.9) - 0.4 * (log_tau - np.log(0.7)), 0.2, m)))
        X = np.c_[v, gap, dv, v_lat]
        out += list(X[validity(X)["all"]])
    return np.asarray(out[:n])


class TCopula(GaussianCopula):
    """Student-t copula on the same 1-D KDE marginals: R from Kendall's tau (R_ij = sin(pi tau_ij / 2)), degrees of
    freedom nu by maximum likelihood over a grid on the training data. Symmetric tail dependence for finite nu."""
    name = "t copula + KDE marginals"
    NU_GRID = (2, 3, 4, 6, 8, 12, 20, 30, 50)

    def fit(self, X):
        self.m = [_Marginal(X[:, j]) for j in range(X.shape[1])]
        d = X.shape[1]
        R = np.eye(d)
        for i in range(d):
            for j in range(i + 1, d):
                R[i, j] = R[j, i] = np.sin(np.pi * stats.kendalltau(X[:, i], X[:, j]).statistic / 2)
        w, V = np.linalg.eigh(R)                                    # keep R positive definite
        R = V @ np.diag(np.maximum(w, 1e-6)) @ V.T
        dd = np.sqrt(np.diag(R))
        self.R = R / np.outer(dd, dd)
        self.Ri = np.linalg.inv(self.R)
        self.logdet = float(np.linalg.slogdet(self.R)[1])
        U = np.clip(np.c_[tuple(m.cdf(X[:, j]) for j, m in enumerate(self.m))], 1e-9, 1 - 1e-9)
        self.ll_by_nu = {nu: float(self._log_c(U, nu).sum()) for nu in self.NU_GRID}
        self.nu = max(self.ll_by_nu, key=self.ll_by_nu.get)
        return self

    def _log_c(self, U, nu):
        d = U.shape[1]
        T = stats.t.ppf(U, nu)
        q = np.einsum("ij,jk,ik->i", T, self.Ri, T)
        from scipy.special import gammaln
        log_mvt = (gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi) - 0.5 * self.logdet
                   - (nu + d) / 2 * np.log1p(q / nu))
        return log_mvt - stats.t.logpdf(T, nu).sum(1)

    def logpdf(self, X):
        U = np.clip(np.c_[tuple(m.cdf(X[:, j]) for j, m in enumerate(self.m))], 1e-9, 1 - 1e-9)
        return sum(m.logpdf(X[:, j]) for j, m in enumerate(self.m)) + self._log_c(U, self.nu)

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        Z = rng.multivariate_normal(np.zeros(len(self.R)), self.R, size=n)
        W = rng.chisquare(self.nu, n) / self.nu
        U = stats.t.cdf(Z / np.sqrt(W)[:, None], self.nu)
        return np.c_[tuple(m.ppf(U[:, j]) for j, m in enumerate(self.m))]

    def info(self):
        return dict(bandwidths=[m.h for m in self.m], R=np.round(self.R, 3).tolist(), nu=self.nu,
                    train_copula_loglik_by_nu={str(k): round(v, 2) for k, v in self.ll_by_nu.items()})


def tail_dependence(X, q=0.10):
    """Empirical tail dependence for the dangerous corner: P(gap/v_ego in its lowest q | dv in its highest q),
    ranks within X. Independent variables give about q; a Gaussian copula with this rank correlation gives far less."""
    g = stats.rankdata(X[:, 1] / X[:, 0]) / len(X)
    d = stats.rankdata(X[:, 2]) / len(X)
    hi = d > 1 - q
    return float((g[hi] <= q).mean()) if hi.any() else float("nan")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def reject_invalid(S):
    """The hand-over filter. Keep only physically valid samples; returns (accepted samples, acceptance rate).
    The accepted samples follow the fitted model truncated to the valid region (density / acceptance rate)."""
    ok = validity(S)["all"]
    return S[ok], float(ok.mean())


# ------------------------------------------------------------------------------------------------ tail models
class ClaytonPair(GaussianCopula):
    """Pair-copula construction: (gap, dv) by a Clayton copula rotated to the critical corner, i.e. Clayton on
    (u_gap, 1 - u_dv) whose lower tail is small gap x high dv; (v_ego, v_lat) given (gap, dv) by the Gaussian copula's
    conditional c4(u)/c2(u_gap, u_dv). Valid density; theta by maximum likelihood on the training data."""
    name = "rotated Clayton (gap x dv) + Gaussian rest"

    @staticmethod
    def _log_clayton(a, b, th):
        return (np.log1p(th) - (th + 1) * (np.log(a) + np.log(b))
                - (2 + 1 / th) * np.log(a ** -th + b ** -th - 1))

    def fit(self, X):
        super().fit(X)
        U = np.clip(np.c_[tuple(m.cdf(X[:, j]) for j, m in enumerate(self.m))], 1e-9, 1 - 1e-9)
        a, b = U[:, 1], 1 - U[:, 2]
        grid = np.exp(np.linspace(np.log(1e-3), np.log(20), 200))
        ll = [self._log_clayton(a, b, t).sum() for t in grid]
        self.theta = float(grid[int(np.argmax(ll))])
        self.R2 = self.R[np.ix_([1, 2], [1, 2])]
        return self

    def logpdf(self, X):
        U = np.clip(np.c_[tuple(m.cdf(X[:, j]) for j, m in enumerate(self.m))], 1e-9, 1 - 1e-9)
        Z = stats.norm.ppf(U)
        Z2 = Z[:, [1, 2]]
        R2i = np.linalg.inv(self.R2)
        log_c2 = -0.5 * np.linalg.slogdet(self.R2)[1] - 0.5 * np.einsum("ij,jk,ik->i", Z2, R2i - np.eye(2), Z2)
        return (sum(m.logpdf(X[:, j]) for j, m in enumerate(self.m)) + self.log_copula(Z) - log_c2
                + self._log_clayton(U[:, 1], 1 - U[:, 2], self.theta))

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        th = self.theta
        a = rng.random(n)                                       # Clayton by conditional inversion
        w = rng.random(n)
        b = (a ** -th * (w ** (-th / (1 + th)) - 1) + 1) ** (-1 / th)
        ug, udv = np.clip(a, 1e-9, 1 - 1e-9), np.clip(1 - b, 1e-9, 1 - 1e-9)
        z2 = np.c_[stats.norm.ppf(ug), stats.norm.ppf(udv)]
        o, k = [0, 3], [1, 2]                                   # Gaussian conditional of (v_ego, v_lat) given (gap, dv)
        S12 = self.R[np.ix_(o, k)]
        A = S12 @ np.linalg.inv(self.R2)
        cov = self.R[np.ix_(o, o)] - A @ S12.T
        zo = z2 @ A.T + rng.multivariate_normal(np.zeros(2), cov, size=n)
        Z = np.c_[zo[:, 0], z2[:, 0], z2[:, 1], zo[:, 1]]
        return self.from_z(Z)

    def info(self):
        return dict(theta=self.theta, lower_tail_dependence=2 ** (-1 / self.theta))


class GMMScores(GaussianCopula):
    """GMCM-style: a K-component Gaussian mixture (EM, scikit-learn, seeded) on the normal scores of the 1-D KDE
    marginals; density g(z) * prod f_i(x_i) / prod phi(z_i) (change of variables, valid). Not the full GMCM estimator
    of Reichenbaecher et al. (there the mixture's own marginals define the scores); an approximation."""
    name = "Gaussian mixture on normal scores (GMCM-style, two components)"

    def fit(self, X, k=2):
        from sklearn.mixture import GaussianMixture
        self.m = [_Marginal(X[:, j]) for j in range(X.shape[1])]
        Z = self.to_z(X)
        self.g = GaussianMixture(k, covariance_type="full", random_state=0, n_init=5).fit(Z)
        self.R = np.corrcoef(Z.T)
        return self

    def logpdf(self, X):
        Z = self.to_z(X)
        return (sum(m.logpdf(X[:, j]) for j, m in enumerate(self.m)) + self.g.score_samples(Z)
                - stats.norm.logpdf(Z).sum(1))

    def sample(self, n, seed):
        self.g.random_state = seed
        Z, _ = self.g.sample(n)
        return self.from_z(Z[np.random.default_rng(seed).permutation(n)])

    def info(self):
        return dict(weights=np.round(self.g.weights_, 3).tolist(), means=np.round(self.g.means_, 2).tolist())
