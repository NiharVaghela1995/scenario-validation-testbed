# highD pilot: robustness analysis plan

Written and committed before any analysis below was run. Anything added afterwards is labelled **post hoc** in the report, and any deviation from this plan is listed there.

**Already seen before this plan** (so not confirmatory, whatever they show here): the primary result in `docs/HIGHD_PILOT.md`, q = 0.05 and 0.20 on the primary pair, mixture K = 3, BIC for K = 1 to 5, and the subsets reported there (manoeuvre start, THW below 3 s, closing only, cars only, location 1 against the others, above 60 km/h).

## Headline under test

At the primary specification (pair THW and closing speed dv at lane crossing, q = 0.10, λ(q) = P(THW in lowest q | dv in highest q), 8,191 cut-ins):

- **H1.** Real cut-ins avoid the corner: the 95 % interval of λ_emp lies below independence (q).
- **H2.** The Gaussian copula overstates it: the 95 % interval of Δ_G = λ_emp − λ_Gaussian lies below 0.
- **H3.** The two-component mixture overstates it: the 95 % interval of Δ_M = λ_emp − λ_mixture(K = 2) lies below 0.

**Verdict per configuration**, always stated separately for H2 and H3:

| Verdict | Rule |
|---|---|
| holds | 95 % interval of Δ entirely below 0 |
| direction only | Δ < 0 but the interval includes 0 |
| fails | Δ ≥ 0 |
| not interpretable | fewer than 50 conditioning events, or fewer than 5 recordings |

All intervals are percentile intervals from a bootstrap by recording, with every model re-fitted in each replicate. B = 400 unless stated, seed 2026. If a run is infeasible on 4 CPU cores within about 3 hours, B is reduced and recorded as a deviation. There is no multiple-testing correction, so counts of configurations that hold are descriptive.

## 1. Screening as a classifier

**Units.** (a) Primary: vehicles in the seeded 10 % random control sample of the pilot, which estimates population rates. (b) Secondary: the kept cut-ins with a screenable follower, using reference values at the crossing frame and a flag within ±1 s.

**Reference definitions** for (a), held for at least 0.3 s (the detectors' `MIN_DUR`) at any time in the track, on the 10 Hz series, excluding `lane_change` samples: TTC < 4 s; THW < 0.6 s; DRAC > D for D ∈ {1, 2, 3} m/s², with D = 2 as the main value (an assumption, not a standard). For (b) the same thresholds apply at the crossing frame.

**Methods.** Threshold, PELT, discords, union, union + gate, unchanged at their defaults.

**Outputs.** Confusion matrix, precision, recall and miss rate per method and reference, with recording-bootstrap intervals. Sweep: one sensitivity factor a on a log grid of 25 values from 1/8 to 8 scaling the threshold's TTC limit (3 s × a), PELT's level (0.2 / a) and the discord floor (0.05 / a). The physical gate stays at its defaults. ROC and PR curves and their areas per method and reference.

**Known circularity, stated in advance.** The references are computed from the same TTC and THW signals the detectors use, and the gate tests TTC < 4 s or THW < 0.6 s directly. This measures agreement with a rule, not validity against independent labels. Union + gate against a TTC or THW reference is close to definitional.

**Would change the conclusion.** The pilot says union + gate flags all 18 cut-ins with TTC below 4 s but few with short headway. That statement changes if cut-in-level recall at TTC < 4 s falls below 0.9, or recall at THW < 0.6 s exceeds 0.5. A method's precision below 0.5 at default would mean its flags are mostly outside the reference, and would be reported as such.

## 2. Tail sensitivity

**Levels.** u = q ∈ {0.05, 0.10, 0.15, 0.20}.

**Pairs** at lane crossing, with the corner declared here: P1 (THW low, dv high), the primary; P2 (gap in metres low, dv high); P3 (THW low, lateral speed at crossing high).

**Definition variants**, on P1 at q = 0.10:
- Track length: required stable window around crossing 0.5 s, 1 s (primary), 2 s; follower visible for at least 12 s.
- Lateral: minimum lateral speed at crossing 0.3, 0.5 or 0.75 m/s. Manoeuvre start threshold 0.05, 0.10, 0.20, 0.30 m/s, on the pair (THW, dv) at manoeuvre start.
- Merge lanes: highD has no merge annotation. A lane in a recording counts as auxiliary if at most 10 % of the vehicles seen in it are in it within the last 10 % of the field of view in the driving direction. Variants: excluding cut-ins from or into auxiliary lanes, and only cut-ins from them. If no auxiliary lane is found, this is reported as not applicable.
- Direction: leftward and rightward lane changes separately.
- Timing: evaluating the pair 0.2 s before and 0.2 s after the crossing frame.

**Families** on P1 to P3 at all four levels: Gaussian copula, t copula, Clayton and Gumbel (each in four rotations, rotation chosen by likelihood in each replicate, and also the rotation aimed at the corner), mixture K = 1 to 6 (`GMMScores`), empirical copula. Definition variants use Gaussian, K = 2 and empirical.

**Would change the conclusion.** H2 or H3 is called fragile if its verdict is not "holds" in more than a quarter of the P1 configurations at q = 0.10 across definition variants, or if it fails at any q on P1. "Models overstate the tail" is stated as general only if every family with a pre-stated form (Gaussian, t, Clayton, Gumbel, K = 2) overstates it. If some do not, the claim is restricted to those that do. K ≥ 3 matching the corner is expected from the pilot and would not change H3, which concerns K = 2.

## 3. Stability

- Leave-one-recording-out, all 60: λ_emp, λ_Gaussian, λ_mixture(K = 2), with range and the number of recordings whose removal gives Δ ≥ 0.
- Per highD location (six).
- Free flow against dense. Density is vehicles in the target lane within ±150 m of the cut-in at crossing (visible length only), split at its median over kept cut-ins; also a dense stratum of at least 22 vehicles/km.
- Car against truck: follower class, cutter class, and the four combinations.

**Would change the conclusion.** Any leave-one-out giving Δ ≥ 0, or H2 or H3 failing in an interpretable stratum. A stratum with fewer than 50 conditioning events or fewer than 5 recordings is reported but not used. Locations 2 to 6 are likely to be small, so location results may be largely not interpretable.

## 4. Model checks

**Predictive counts of joint extremes.** For each family: the same recording-bootstrap fits act as an approximate posterior. Each fit simulates n events, and joint extremes are counted at q = 0.10 and 0.05. Report the predictive distribution against the observed count and the predictive tail probability. This is a bootstrap approximation, not a formal Bayesian posterior.

**Tail QQ.** T = max(rank THW, 1 − rank dv) measures distance to the corner. QQ of the lowest 10 % of T, data against each model, with a recording-bootstrap band. Rank space only.

**Would change the conclusion.** A model is inconsistent with the data if the observed count falls outside the central 95 % of its predictive distribution. H2 or H3 is contradicted if the Gaussian or the K = 2 mixture is consistent at q = 0.10.

## 5. Measurement noise

Independent Gaussian noise on position (each vehicle) and speed (each vehicle) at the crossing frame, 200 Monte Carlo draws per level, (σ_pos m, σ_v m/s) ∈ {(0.1, 0.1), (0.3, 0.3), (0.5, 0.5), (0.5, 1.0)}. Gap, dv, THW, TTC and DRAC are recomputed. Reported: size of the critical sets (TTC < 4, TTC < 2, THW < 0.6, DRAC > 2), their overlap with the unperturbed sets (Jaccard), and λ_emp, λ_Gaussian, λ_mixture(K = 2) under noise.

**Limits stated in advance.** Noise applies at the crossing frame only, with no trajectory re-filtering. The levels are assumptions chosen to bracket and exceed typical errors of drone-derived tracks.

**Would change the conclusion.** H2 or H3 fails at any level up to (0.3, 0.3). The critical set is called unstable if mean Jaccard for TTC < 4 s is below 0.5 at (0.1, 0.1).

## Reporting

- `docs/HIGHD_ROBUSTNESS.md`, generated from `results/highd/robust_*.json`: one summary table (analysis → result → does the headline hold?), every number traced to a script.
- Aggregated plots only: no per-vehicle trajectories, scatter or extreme values.
- Per-event tables stay outside the repository.
- Deck slides change only where the headline changes, under the rules above.
