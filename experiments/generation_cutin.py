"""End-to-end generation chain on cut-ins, synthetic test bed only.

Generate events -> train / held-out split -> fit six models -> held-out comparison -> importance sampling toward the
critical region -> replay in esmini (if available) -> robustness check on G2, the generator that has a joint tail.
Library: critical_events/cutin_chain.py. Every number goes to results/generation_cutin.json.

    python experiments/generation_cutin.py > results/generation_cutin.txt
"""
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(os.environ.get("TESTBED_RESULTS", ROOT / "results"))
FIG = Path(os.environ.get("TESTBED_FIGURES", ROOT / "figures"))
DATA = Path(os.environ.get("TESTBED_DATA", ROOT / "data" / "synthetic"))
for _d in (HERE, FIG, DATA):
    _d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "critical_events"))
import cutin_chain as C  # noqa: E402

N_EVENTS, SEED, TRAIN_SHARE = 400, 2026, 0.6
OUT = dict(python=platform.python_version(), data="synthetic test bed only",
           params=C.PARAMS, units=C.UNITS, criticality=f"TTC at lane crossing < {C.TTC_CRIT} s", )


def spearman(X):
    return stats.spearmanr(X).statistic


# ------------------------------------------------------------------------------------------------ 
X = C.generate_events(N_EVENTS, SEED)
perm = np.random.default_rng(SEED + 1).permutation(len(X))
n_tr = int(TRAIN_SHARE * len(X))
TR, HO = X[perm[:n_tr]], X[perm[n_tr:]]
d = C.derived(X)
OUT["events"] = dict(
    n=len(X), seed=SEED, n_train=len(TR), n_heldout=len(HO), split="random by event, seed 2027",
    generator="v_ego = 12 + 30 Beta(5,4); dv ~ N(1.5 + 0.15 (v_ego - 28), 3.0); log(gap/v_ego) ~ N(log 0.7 + 0.05 dv, 0.4); "
              "log v_lat ~ N(log 0.9 - 0.4 (log(gap/v_ego) - log 0.7), 0.2); physically impossible events rejected",
    quantiles={p: dict(zip(["p1", "p25", "p50", "p75", "p99"], np.percentile(X[:, i], [1, 25, 50, 75, 99]).round(2).tolist()))
               for i, p in enumerate(C.PARAMS)},
    spearman=np.round(spearman(X), 3).tolist(),
    critical_share=float(C.critical(X).mean()), critical_n=int(C.critical(X).sum()),
    critical_share_train=float(C.critical(TR).mean()), critical_share_heldout=float(C.critical(HO).mean()),
    closing_share=float((X[:, 2] > 0).mean()), ttc_sw_min=float(d["ttc_sw"].min()),
    drac_sw_max=float(d["drac_sw"].max()), all_valid=bool(C.validity(X)["all"].all()))
with open(DATA / "cutin_events_g1.csv", "w") as f:
    f.write("split," + ",".join(C.PARAMS) + ",gap_sw,ttc_sw,critical\n")
    for k, i in enumerate(perm):
        f.write(("train" if k < n_tr else "heldout") + "," + ",".join(f"{x:.4f}" for x in X[i])
                + f",{d['gap_sw'][i]:.4f},{d['ttc_sw'][i]:.4f},{int(C.critical(X[i:i+1])[0])}\n")
ev = OUT["events"]
print(f"# generation_cutin\n\n## Events: n {ev['n']} (train {ev['n_train']}, held-out {ev['n_heldout']}); "
      f"critical {ev['critical_n']} ({ev['critical_share']:.3f}); closing {ev['closing_share']:.2f}; all valid {ev['all_valid']}")
for p, q in ev["quantiles"].items():
    print(f"  {p:6s} {q}")
print("  Spearman\n", np.array(ev["spearman"]))

# ------------------------------------------------------------------------------------------------ 
FITTED = {}
OUT["fit"] = dict(train_n=len(TR), gmcm="GMCM-style approximation (Gaussian mixture on normal scores) run on G2 only; the full GMCM estimator was not run", models={})
print("\n## Fits on the train split")
for M in C.MODELS:
    m = M().fit(TR)
    FITTED[m.name] = m
    ll = m.logpdf(TR)
    OUT["fit"]["models"][m.name] = dict(train_loglik_per_event=float(ll.mean()), **m.info())
    print(f"  {m.name:34s} train log-lik/event {ll.mean():8.3f}  {m.info() if M is not C.GaussianCopula else ''}")
print("  copula correlation of normal scores R\n", np.array(FITTED[C.GaussianCopula.name].info()["R"]))

# ------------------------------------------------------------------------------------------------ 
N_SAMPLES = 10000
iu = np.triu_indices(len(C.PARAMS), 1)
rho_ho = spearman(HO)


def compare(S):
    return dict(w1={p: float(stats.wasserstein_distance(HO[:, j], S[:, j])) for j, p in enumerate(C.PARAMS)},
                spearman_err_mean=float(np.abs(spearman(S)[iu] - rho_ho[iu]).mean()),
                spearman_err_max=float(np.abs(spearman(S)[iu] - rho_ho[iu]).max()))


SAMPLES = {}
OUT["heldout_comparison"] = dict(n_heldout=len(HO), n_samples=N_SAMPLES, sample_seed=7, spearman_heldout=np.round(rho_ho, 3).tolist(),
                         reference_train_vs_heldout=compare(TR), models={})
print(f"\n## Held-out comparison ({len(HO)} held-out events, {N_SAMPLES} samples per model)")
print("  reference (train events vs held-out):", {k: (np.round(list(v.values()), 3).tolist() if isinstance(v, dict) else round(v, 3))
                                                   for k, v in OUT["heldout_comparison"]["reference_train_vs_heldout"].items()})
for name, m in FITTED.items():
    S = m.sample(N_SAMPLES, seed=7)
    SAMPLES[name] = S
    ll = m.logpdf(HO)
    val = C.validity(S)
    r = dict(heldout_loglik_per_event=float(ll.mean()), heldout_loglik_se=float(ll.std(ddof=1) / np.sqrt(len(ll))),
             **compare(S), validity={k: float(v.mean()) for k, v in val.items()},
             critical_share_samples=float(C.critical(S[val["all"]]).mean()))
    OUT["heldout_comparison"]["models"][name] = r
    print(f"  {name:34s} LL {r['heldout_loglik_per_event']:7.3f} ± {r['heldout_loglik_se']:.3f} | W1 "
          + " ".join(f"{p} {r['w1'][p]:.3f}" for p in C.PARAMS)
          + f" | rho err mean {r['spearman_err_mean']:.3f} max {r['spearman_err_max']:.3f} | valid {r['validity']['all']:.4f}"
          + f" | critical {r['critical_share_samples']:.3f}")
    print("     invalid by rule:", {k: round(1 - v, 4) for k, v in r["validity"].items() if v < 1})
M_ = OUT["heldout_comparison"]["models"]
win = dict(loglik=max(M_, key=lambda k: M_[k]["heldout_loglik_per_event"]),
           spearman=min(M_, key=lambda k: M_[k]["spearman_err_mean"]),
           validity=max(M_, key=lambda k: M_[k]["validity"]["all"]),
           w1_mean_rank=min(M_, key=lambda k: np.mean([sorted(M_[j]["w1"][p] for j in M_).index(M_[k]["w1"][p]) for p in C.PARAMS])))
OUT["heldout_comparison"]["best_by_metric"] = win
print("  best by metric:", win)
LLHO = {n: m.logpdf(HO) for n, m in FITTED.items()}
cop, kde, ind = C.GaussianCopula.name, C.KDE.name, C.Independent.name
OUT["heldout_comparison"]["paired_loglik_diff"] = {}
for a, b in ((cop, kde), (cop, ind), (kde, ind)):
    dlt = LLHO[a] - LLHO[b]
    se = float(dlt.std(ddof=1) / np.sqrt(len(dlt)))
    OUT["heldout_comparison"]["paired_loglik_diff"][f"{a} - {b}"] = dict(mean=float(dlt.mean()), se=se, z=float(dlt.mean() / se),
                                                                  share_events_better=float((dlt > 0).mean()))
    print(f"  paired LL diff {a} - {b}: {dlt.mean():.3f} ± {se:.3f} (z {dlt.mean() / se:.1f}; better on {(dlt > 0).mean():.0%} of events)")

# ------------------------------------------------------------------------------------------------ 
COP = FITTED[C.GaussianCopula.name]
N_IS, N_REF = 1000, 1_000_000
OUT["importance_sampling"] = dict(model=C.GaussianCopula.name, n=N_IS, n_reference=N_REF,
                                     proposal="N(mu, R) in normal-score space; mu = mean z of the 10 % pilot samples (n 2000) with the smallest TTC at lane crossing (one cross-entropy step)",
                                     targets={})
print(f"\n## Importance sampling under the Gaussian copula (n = {N_IS} per estimator; reference n = {N_REF:,})")
IS_KEEP = {}
for ttc_max in (2.0, 1.0):
    mu = C.ce_mean_shift(COP, ttc_max)
    ref = C.crude_estimate(COP, ttc_max, N_REF, seed=99)
    crude = C.crude_estimate(COP, ttc_max, N_IS, seed=21)
    isr, Z, w, hit = C.is_estimate(COP, mu, ttc_max, N_IS, seed=21)
    IS_KEEP[ttc_max] = (mu, Z, w, hit)
    vrf = (crude["se"] / isr["se"]) ** 2 if isr["se"] > 0 else None
    n_equiv = (ref["estimate"] * (1 - ref["estimate"]) / isr["se"] ** 2) if isr["se"] > 0 else None
    OUT["importance_sampling"]["targets"][f"TTC_sw < {ttc_max:g} s"] = dict(
        mu_z=dict(zip(C.PARAMS, np.round(mu, 3).tolist())), reference=ref, crude=crude, importance=isr,
        variance_reduction=vrf, crude_n_for_same_se=n_equiv)
    print(f"  target TTC_sw < {ttc_max:g} s: reference p = {ref['estimate']:.5f} ± {ref['se']:.5f}")
    print(f"    shift mu_z {dict(zip(C.PARAMS, np.round(mu, 2).tolist()))}")
    print(f"    crude MC  : {crude['estimate']:.5f} ± {crude['se']:.5f} ({crude['hits']} hits)")
    print(f"    IS        : {isr['estimate']:.5f} ± {isr['se']:.5f}; hits {isr['hit_share']:.1%}; ESS {isr['ess']:.0f} of {N_IS} "
          f"(hits-only ESS {isr['ess_hits']:.0f}); w mean {isr['w_mean']:.3f}, median {isr['w_median']:.4f}, max {isr['w_max']:.2f}")
    print(f"    variance reduction x{vrf:.1f}: crude MC would need about {n_equiv:,.0f} samples for the same SE")

# ------------------------------------------------------------------------------------------------ 
import esmini_logs as EL  # noqa: E402
OUT["replay"] = dict(simulator="esmini", available=EL.available())
print("\n## Replay in esmini")
if EL.available():
    mu, Z, w, hit = IS_KEEP[2.0]
    Xp = COP.from_z(Z)
    ok = C.validity(Xp)["all"]
    t = C.derived(Xp)["ttc_sw"]
    rng = np.random.default_rng(31)
    crit_idx = rng.choice(np.flatnonzero(ok & (t < C.TTC_CRIT)), 5, replace=False)
    non_idx = rng.choice(np.flatnonzero(ok & (t >= C.TTC_CRIT) & np.isfinite(t)), 5, replace=False)
    rows = []
    for k, i in enumerate(np.r_[crit_idx, non_idx]):
        p = dict(zip(C.PARAMS, Xp[i]))
        r = C.replay(p, HERE / "replay_cutin", f"cutin_{k:02d}")
        rows.append(r)
        a = r.get("acc", {})
        print(f"  {r['name']}: v {p['v_ego']:.1f} gap {p['gap']:.1f} dv {p['dv']:.2f} vlat {p['v_lat']:.2f} | intended TTC_sw {r['intended_ttc_sw']:.2f} s"
              f" | replay {r.get('replay_ttc_sw', float('nan')):.2f} s (switch {r.get('t_switch', float('nan')):.1f} vs {r['t_switch_planned']:.2f} s)"
              f" | reproduced {r.get('reproduced')} | ACC: min gap {a.get('min_gap', float('nan')):.1f} m, min THW {a.get('min_thw', float('nan')):.2f} s,"
              f" peak decel {a.get('peak_decel', float('nan')):.1f}, collision {a.get('collision')}")
    rep = [r for r in rows if r.get("switched")]
    OUT["replay"].update(
        n=len(rows), selection="5 intended-critical + 5 non-critical valid variants from the importance-sampling proposal (seed 31)",
        criterion=f"open loop (ego constant speed): replayed TTC at lead switch < {C.TTC_CRIT} s iff intended",
        reproduced=int(sum(r["reproduced"] for r in rep)), switched=len(rep),
        ttc_abs_err_median=float(np.median([abs(r["replay_ttc_sw"] - r["intended_ttc_sw"]) for r in rep])),
        switch_delay_median=float(np.median([r["t_switch"] - r["t_switch_planned"] for r in rep])),
        acc_collisions=int(sum(r.get("acc", {}).get("collision", False) for r in rows)),
        acc_decel_over_limit=int(sum(r.get("acc", {}).get("decel_over_limit", False) for r in rows)),
        runs=rows, files="results/replay_cutin/ (xosc, csv, log per run; regenerated, not tracked)")
    rp = OUT["replay"]
    print(f"  reproduced {rp['reproduced']} of {rp['n']}; median |TTC error| {rp['ttc_abs_err_median']:.2f} s; median switch delay "
          f"{rp['switch_delay_median']:.2f} s; ACC runs: collisions {rp['acc_collisions']}, decel > 8 m/s^2 in {rp['acc_decel_over_limit']}")
else:
    OUT["replay"]["status"] = "planned (esmini not found)"
    print("  esmini not available: replay planned")

# ------------------------------------------------------------------------------------------------ 
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {C.Independent.name: "#2a78d6", C.KDE.name: "#eb6834", C.GaussianCopula.name: "#1baf7a"}
SHORT = {C.Independent.name: "independent", C.KDE.name: "multivariate KDE", C.GaussianCopula.name: "Gaussian copula"}
GREY, INK = "#9a9a94", "#3b4652"
LAB = {"v_ego": "ego speed [m/s]", "gap": "cut-in gap [m]", "dv": "relative speed [m/s]", "v_lat": "lateral speed [m/s]"}
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.edgecolor": "#b9c1c9", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
TAG = "Illustrative (synthetic)"


def save(fig, name):
    fig.text(0.995, 0.005, TAG, ha="right", va="bottom", fontsize=8, color=INK, fontweight="bold")
    fig.savefig(FIG / f"{name}.png", dpi=110, metadata={"Software": None})
    plt.close(fig)


# marginals: held-out histogram vs sample densities
fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.0))
for j, p in enumerate(C.PARAMS):
    lo, hi = np.percentile(np.r_[HO[:, j], TR[:, j]], [0.5, 99.5])
    pad = 0.15 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 300)
    ax[j].hist(HO[:, j], bins=18, range=(lo - pad, hi + pad), density=True, color=GREY, alpha=0.45, edgecolor="white", lw=0.6,
               label="held-out events")
    for name, S in SAMPLES.items():
        ax[j].plot(grid, stats.gaussian_kde(S[:, j])(grid), color=COL[name], lw=1.6, label=SHORT[name],
                   ls=(0, (4, 2)) if name == C.Independent.name else "-", zorder=3 if name == C.Independent.name else 2)
    ax[j].set_xlabel(LAB[p])
    ax[j].set_yticks([])
ax[0].set_ylabel("density")
ax[0].legend(fontsize=6, frameon=False, loc="upper left")
fig.suptitle("Cut-in parameters: held-out events vs 10,000 samples per model (the independent and copula models use identical KDE marginals)",
             fontsize=8, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
save(fig, "gen_cutin_marginals")

# pair plot: rows = models, columns = the six parameter pairs
pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
fig, ax = plt.subplots(3, 6, figsize=(7.6, 4.0))
sub = np.random.default_rng(5).choice(N_SAMPLES, 800, replace=False)
for r, (name, S) in enumerate(SAMPLES.items()):
    for c, (i, j) in enumerate(pairs):
        a = ax[r, c]
        a.scatter(S[sub, i], S[sub, j], s=1.5, color=COL[name], alpha=0.35, lw=0)
        a.scatter(HO[:, i], HO[:, j], s=2.5, color="#2b2b28", alpha=0.7, lw=0)
        lo_i, hi_i = np.percentile(HO[:, i], [0, 100]); lo_j, hi_j = np.percentile(HO[:, j], [0, 100])
        a.set_xlim(lo_i - 0.15 * (hi_i - lo_i), hi_i + 0.15 * (hi_i - lo_i))
        a.set_ylim(lo_j - 0.15 * (hi_j - lo_j), hi_j + 0.15 * (hi_j - lo_j))
        a.tick_params(labelsize=5, length=2)
        if r == 2:
            a.set_xlabel(C.PARAMS[i], fontsize=6.5)
        a.set_ylabel(C.PARAMS[j], fontsize=6.5, labelpad=1)
        if c == 0:
            a.set_title(SHORT[name], loc="left", fontsize=7, color=INK, x=0.0)
fig.suptitle("Pairwise view: 800 samples per model in colour, held-out events in dark", fontsize=8, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.02, 1, 0.95), h_pad=0.6, w_pad=0.3)
save(fig, "gen_cutin_pairs")

# validity: share of samples violating each rule, per model
rules = [k for k in next(iter(OUT["heldout_comparison"]["models"].values()))["validity"] if k != "all"]
rules = [k for k in rules if any(1 - OUT["heldout_comparison"]["models"][n]["validity"][k] > 0 for n in SAMPLES)] + ["all"]
fig, ax = plt.subplots(figsize=(7.6, 2.1))
h = 0.26
for m_i, name in enumerate(SAMPLES):
    v = [100 * (1 - OUT["heldout_comparison"]["models"][name]["validity"][k]) for k in rules]
    y = np.arange(len(rules)) + (m_i - 1) * h
    ax.barh(y, v, height=h * 0.9, color=COL[name], label=SHORT[name])
    ax.text(v[-1] + 0.1, y[-1], f"{v[-1]:.1f} %", va="center", fontsize=6, color=INK)
ax.set_yticks(np.arange(len(rules)), [("any rule (invalid sample)" if k == "all" else k) for k in rules])
ax.invert_yaxis()
ax.set_xlabel("share of 10,000 samples violating the rule [%]")
ax.grid(axis="x", color="#e3e8ec", lw=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=6, frameon=False, loc="upper right")
ax.set_title("Physical validity of sampled cut-ins (held-out events: all valid by construction)", loc="left", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
save(fig, "gen_cutin_validity")
OUT["figures"] = [f"figures/gen_cutin_{n}.png" for n in ("marginals", "pairs", "validity")]
print("\n## Figures:", OUT["figures"])

# ------------------------------------------------------------------------------------------------ robustness: generator with tail dependence
N_TRUTH = 200_000
ALL_MODELS = (C.Independent, C.KDE, C.GaussianCopula, C.TCopula)


def run_chain(gen, label):
    """Generate, split, fit and compare for one generator with all four models; truth = a large fresh sample of the same generator."""
    Xg = gen(N_EVENTS, SEED)
    pg = np.random.default_rng(SEED + 1).permutation(len(Xg))
    tr, ho = Xg[pg[:n_tr]], Xg[pg[n_tr:]]
    truth = gen(N_TRUTH, 7)
    res = dict(dependence=C.GENERATOR_DEPENDENCE[label], n=len(Xg), n_train=len(tr), n_heldout=len(ho),
               truth=dict(n=N_TRUTH, critical_share=float(C.critical(truth).mean()), tail_dep_q10=C.tail_dependence(truth, 0.10),
                          tail_dep_q05=C.tail_dependence(truth, 0.05), spearman=np.round(spearman(truth), 3).tolist()),
               heldout=dict(critical=int(C.critical(ho).sum()), tail_dep_q10=C.tail_dependence(ho, 0.10)), models={})
    lls = {}
    for M in ALL_MODELS:
        m = M().fit(tr)
        S = m.sample(N_SAMPLES, seed=7)
        ok = C.validity(S)["all"]
        ll = m.logpdf(ho)
        lls[m.name] = ll
        rho_h = spearman(ho)
        res["models"][m.name] = dict(
            heldout_loglik_per_event=float(ll.mean()), heldout_loglik_se=float(ll.std(ddof=1) / np.sqrt(len(ll))),
            train_loglik_per_event=float(m.logpdf(tr).mean()),
            w1={p: float(stats.wasserstein_distance(ho[:, j], S[:, j])) for j, p in enumerate(C.PARAMS)},
            spearman_err_mean=float(np.abs(spearman(S)[iu] - rho_h[iu]).mean()),
            spearman_err_vs_truth=float(np.abs(spearman(S)[iu] - spearman(truth)[iu]).mean()),
            valid_share=float(ok.mean()), critical_share_valid=float(C.critical(S[ok]).mean()),
            tail_dep_q10=C.tail_dependence(S[ok], 0.10), tail_dep_q05=C.tail_dependence(S[ok], 0.05),
            **({"nu": m.nu} if isinstance(m, C.TCopula) else {}))
    g = C.GaussianCopula.name
    res["paired_vs_gaussian_copula"] = {n: dict(mean=float((lls[n] - lls[g]).mean()), se=float((lls[n] - lls[g]).std(ddof=1) / np.sqrt(len(ho))))
                                        for n in lls if n != g}
    res["best_heldout_loglik"] = max(res["models"], key=lambda k: res["models"][k]["heldout_loglik_per_event"])
    return res, (tr, ho, truth)


OUT["robustness"] = dict(tail_dependence_definition="P(time gap in its lowest q | relative speed in its highest q), ranks within the sample",
                            generators={})
CHAINS = {}
print("\n## Robustness: G1 (as above) and G2 (tail component), four models incl. t copula")
for label, gen in (("G1", C.generate_events), ("G2", C.generate_events_tail)):
    res, CHAINS[label] = run_chain(gen, label)
    if label == "G2":
        with open(DATA / "cutin_events_g2.csv", "w") as f:
            f.write("split," + ",".join(C.PARAMS) + "\n")
            for name_, arr in (("train", CHAINS[label][0]), ("heldout", CHAINS[label][1])):
                for row in arr:
                    f.write(name_ + "," + ",".join(f"{x:.4f}" for x in row) + "\n")
    OUT["robustness"]["generators"][label] = res
    print(f"  {label}: {res['dependence']}")
    print(f"    truth (n {N_TRUTH:,}): critical {res['truth']['critical_share']:.4f}; tail dep q0.10 {res['truth']['tail_dep_q10']:.3f}, q0.05 {res['truth']['tail_dep_q05']:.3f}"
          f" | held-out: critical {res['heldout']['critical']}/{res['n_heldout']}, tail dep q0.10 {res['heldout']['tail_dep_q10']:.3f}")
    for name, r in res["models"].items():
        pv = res["paired_vs_gaussian_copula"].get(name)
        print(f"    {name:34s} LL {r['heldout_loglik_per_event']:7.3f}" + (f" (vs Gauss cop {pv['mean']:+.3f} ± {pv['se']:.3f})" if pv else " (reference)        ")
              + f" | rho err {r['spearman_err_mean']:.3f} | valid {r['valid_share']:.3f} | critical {r['critical_share_valid']:.3f}"
              + f" | tail dep q0.10 {r['tail_dep_q10']:.3f} q0.05 {r['tail_dep_q05']:.3f}" + (f" | nu {r['nu']}" if "nu" in r else ""))
    print(f"    best held-out log-lik: {res['best_heldout_loglik']}")

# ------------------------------------------------------------------------------------------------ critical share with 95 % intervals
print("\n## Critical share (TTC at lane crossing < 2 s), 95 % Wilson intervals")
for label, res in OUT["robustness"]["generators"].items():
    k, n = res["heldout"]["critical"], res["n_heldout"]
    lo, hi = C.wilson(k, n)
    res["heldout"].update(critical_share=k / n, critical_ci95=[lo, hi])
    tru = res["truth"]["critical_share"]
    print(f"  {label}: held-out {k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]; truth {tru:.4f}")
    for name, r in res["models"].items():
        tr_, ho_, _ = CHAINS[label]
        nv = int(round(r["valid_share"] * N_SAMPLES))
        kk = int(round(r["critical_share_valid"] * nv))
        mlo, mhi = C.wilson(kk, nv)
        r["critical_ci95"] = [mlo, mhi]
        r["critical_vs_truth"] = "over" if mlo > tru else ("under" if mhi < tru else "consistent")
        r["critical_within_heldout_ci"] = bool(lo <= r["critical_share_valid"] <= hi)
        print(f"    {name:34s} {r['critical_share_valid']:.3f} [{mlo:.3f}, {mhi:.3f}] -> vs truth: {r['critical_vs_truth']}; "
              f"inside held-out interval: {r['critical_within_heldout_ci']}")

# ------------------------------------------------------------------------------------------------ rejection before hand-over
print("\n## Reject physically invalid samples before hand-over (accepted samples only)")
for label, res in OUT["robustness"]["generators"].items():
    tr_, ho_, truth_ = CHAINS[label]
    for M in ALL_MODELS:
        m = M().fit(tr_)
        S = m.sample(N_SAMPLES, seed=7)
        A, acc = C.reject_invalid(S)            # the hand-over filter
        ok = C.validity(S)["all"]
        r = res["models"][m.name]
        r["rejection"] = dict(
            acceptance_rate=acc, n_accepted=int(ok.sum()),
            w1_all={p: float(stats.wasserstein_distance(ho_[:, j], S[:, j])) for j, p in enumerate(C.PARAMS)},
            w1_accepted={p: float(stats.wasserstein_distance(ho_[:, j], A[:, j])) for j, p in enumerate(C.PARAMS)},
            critical_share_of_all_samples=float((C.critical(S) & ok).mean()),   # invalid samples count as non-events
            critical_share_accepted=float(C.critical(A).mean()),
            heldout_loglik_truncated=float(r["heldout_loglik_per_event"] - np.log(acc)))
        rr = r["rejection"]
        print(f"  {label} {m.name:34s} accept {acc:.3f} | W1 all -> accepted: "
              + ", ".join(f"{p} {rr['w1_all'][p]:.2f}->{rr['w1_accepted'][p]:.2f}" for p in C.PARAMS)
              + f" | critical (accepted) {rr['critical_share_accepted']:.3f} | LL truncated {rr['heldout_loglik_truncated']:.3f}")
# the main chain numbers on accepted samples, G1
OUT["heldout_comparison"]["accepted_only"] = {n: OUT["robustness"]["generators"]["G1"]["models"][n]["rejection"] for n in OUT["heldout_comparison"]["models"]}

# ------------------------------------------------------------------------------------------------ tail models on G2
print("\n## Tail models on G2 (same metrics as the robustness table)")
res = OUT["robustness"]["generators"]["G2"]
tr_, ho_, truth_ = CHAINS["G2"]
g = C.GaussianCopula.name
ll_g = C.GaussianCopula().fit(tr_).logpdf(ho_)
OUT["tail_models_g2"] = {}
for M in (C.ClaytonPair, C.GMMScores):
    m = M().fit(tr_)
    S = m.sample(N_SAMPLES, seed=7)
    A, acc = C.reject_invalid(S)
    ll = m.logpdf(ho_)
    dl = ll - ll_g
    nv = len(A)
    kk = int(C.critical(A).sum())
    lo, hi = C.wilson(kk, nv)
    tru = res["truth"]["critical_share"]
    r = dict(heldout_loglik_per_event=float(ll.mean()), heldout_loglik_se=float(ll.std(ddof=1) / np.sqrt(len(ll))),
             paired_vs_gaussian_copula=dict(mean=float(dl.mean()), se=float(dl.std(ddof=1) / np.sqrt(len(dl)))),
             train_loglik_per_event=float(m.logpdf(tr_).mean()),
             critical_share_valid=kk / nv, critical_ci95=[lo, hi],
             critical_vs_truth="over" if lo > tru else ("under" if hi < tru else "consistent"),
             tail_dep_q10=C.tail_dependence(A, 0.10), tail_dep_q05=C.tail_dependence(A, 0.05), acceptance_rate=acc,
             w1_accepted={p: float(stats.wasserstein_distance(ho_[:, j], A[:, j])) for j, p in enumerate(C.PARAMS)}, **m.info())
    OUT["tail_models_g2"][m.name] = r
    print(f"  {m.name:52s} LL {r['heldout_loglik_per_event']:.3f} (vs Gauss cop {dl.mean():+.3f} ± {r['paired_vs_gaussian_copula']['se']:.3f})"
          f" | critical {kk / nv:.3f} [{lo:.3f}, {hi:.3f}] {r['critical_vs_truth']} (truth {tru:.3f}) | tail dep q0.10 {r['tail_dep_q10']:.3f}"
          f" (truth {res['truth']['tail_dep_q10']:.3f}) | kept {acc:.3f} | {m.info()}")

(HERE / "generation_cutin.json").write_text(json.dumps(OUT, indent=1, default=float))
