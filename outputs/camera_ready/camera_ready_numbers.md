# Camera-ready numbers: final (ICAART 2027 #388)

Generated 2026-11-17.
- Source: `camera_ready_numbers.json`, built by `run_cr_final_numbers.py`. The rows below are generated from it by `run_cr_numbers_table.py`, which also writes them, with their flags, into the JSON under `table`. No value is typed by hand.
- Figures are not regenerated (as instructed).

## Rule applied

- **Data:** corrected logging (o_t with a_{t−1}). Evaluation episodes 30000–30099 (100 per task; 46,700 sites with E available).
- **Refit on corrected probe-fit episodes 10000–10099, with the paper's recipe.** This covers everything the paper learned from the training log:
  - the probe v: StandardScaler + logistic regression, C = 1, lbfgs, random_state = 0; label 1[KL_t > calibration median];
  - the PCA for the geometry statistics;
  - the SAEs (3 runs per task, retrained);
  - the training-log KL median, replaced by the calibration median (episodes 20000–20049).
- **Unchanged:** γ = 0.95 / 0.70 / 0.90 (0.95 for the partial-observation cartpole models), EMARec α = 0.02 / 0.5 / 0.02, each section's C_t definition, and the canonical 50-direction null with common random numbers.
  - Fig. 2c uses the Section-6 C_t on the same sites as Fig. 2a.
- **Readout = probe probability σ(v^T h_t + b).** The logit v^T h_t + b is stored next to every readout number in the JSON and shown in the Note column.
  - For interventions, the logit is affine in v^T h, so the logit z equals the projection z recorded by the causal runs.
- **CIs:** trajectory bootstrap, 1,000 resamples.
- **Causal numbers** come from the refit-probe runs in `A_raw/`: steering on the 13 models, noisy steering, Task G, and the partial-observation ablation. They were computed with this rule before the instruction; nothing about them changes. The SAE numbers come from `B_sae_corrected_refit.json`, rerun today with the refit v.

## Numbers that change sign or cross their threshold, relative to the published value

**Section 8 / Table 6: steering persistence (central claim).**
- The k=0 dissociation holds on all 13 models:
  - readout z: cartpole +10.6 ± 1.5, reacher +11.1 ± 0.8, pendulum +11.4 ± 4.7
  - E: max |z| = 0.59; noisy max |z| = 0.37
- **The readout effect at k=10 no longer clears z = 2 on 8 of 13 models.**

  | Task | Published k=10 z | Final k=10 z, per model |
  |---|---|---|
  | pendulum | +11.4 ± 0.6 | −0.0, −0.1, +0.1, −0.2 (4/4 below 2) |
  | reacher | +5.9 ± 0.7 | +2.0, +0.3, −0.0, +2.6 (3/4 below 2) |
  | cartpole | +12.2 ± 1.4 | +5.2, +9.1, +4.7, +0.8, +11.0 (1/5 below 2) |

- The 13-model readout z range is therefore −0.2..+17.6 (published +5.2..+19.2).
- The logit gives the same picture: k=10 z per model is −0.2..+0.1 (pendulum) and +0.0..+2.6 (reacher).
- So "the change persists ten steps later" holds on cartpole (4 of 5 models) only.

**Section 9 / Tables 8–9: atoms** (refit SAEs, paper's selection rule, 3 runs per task):
- **Cartpole:** no selected atom moves the readout. Readout z is −0.08, +0.38 and −1.28 (published #156: −5.0). The E effect holds (+7.04, +3.27, +7.11).
- **Reacher:** no selected atom is KL-independent; |r(act, KL)| is 0.11–0.24 (published #612: 0.02–0.06). The E effect holds and is larger (+6.58, +33.98, +18.30). The readout is not moved (+1.20, +1.30, −0.07).
- **Pendulum:** the selected atom moves E in 2 of 3 runs (+18.95, +0.95, +4.89; published #310: |z| < 1).
- **Summary:** "ablating some selected atoms changes imagination error" holds and is stronger, in 8 of 9 runs. The readout-moving atom and the KL-independent atom do not reappear.

**Table 3 and the Section 5.4 sign paragraph:**
- **Reacher within-bin r(Rec, C_t):** −0.09 → **+0.29**, a sign flip.
- **Reacher KL-matched AUROC:** 0.57–0.75 → 0.48 (0.41–0.57); 2 of 4 models fall below 0.5.
- **Cartpole KL-matched AUROC:** 0.71 → 0.61 on the primary, still above 0.5; 0.43–0.65 across models, with 2 of 5 models below 0.5.

**Section 5.3 geometry:**
- On 1 of 4 pendulum models (the primary), v's top-10-PC share is 6.17%, above the random mean of 3.9%.
- The other 12 of 13 models stay below the random 5th percentile.

**Section 6.3 / Fig. 2c:**
- **Pendulum's external scramble z: −11.7 → +21.0, a sign flip.** With the Section-6 C_t on the Fig. 2a sites (Decision 2), real-order C_t beats the scrambled one on pendulum too, so "on pendulum the order effect reverses" no longer holds.

**Section 6.1:**
- Reacher's r(C_t, E_K) no longer decays with horizon: +0.48 / +0.51 / +0.52 / +0.52 (published "decays").

**Section 7 / Table 5:**
- **AUROC:** partial > full on **2 of 5** pairs (0.972 / 0.966 on average; published 0.871 / 0.951).
- **R²(readout, C_t):** partial > full on 4 of 5 (0.816 / 0.841).
- **Incremental value smaller under partial observation:** 3 of 5 pairs.
- So "the signal strengthens under partial observation on every pair" fails.

**Section 5.2:**
- Pendulum: C_t explains more of the readout than KL on 3 of 4 models (on the primary: +0.53).
- **Probability vs logit (stored, not a threshold change):** with the probability readout, C_t is ahead of KL on 5/5 cartpole, 3/4 reacher and 3/4 pendulum models. With the logit it is ahead on only **1/5, 0/4 and 1/4**.

  The probability's saturation does part of the work in "C_t clearly ahead of current KL".
- **Task G v-ablation:** the probability null is **degenerate** (flagged). Random-direction ablations swing the saturated probability (null sd 0.38), so the probability z is −1.5. The logit z is **−27.7** (published ≈ −22).

**Holds (no sign or threshold change):**
- Table 4 ΔR², all CIs > 0:

  | Task | ΔR² [95% CI] |
  |---|---|
  | cartpole | +0.0168 [+0.0079, +0.0288] |
  | reacher | +0.0053 [+0.0015, +0.0084] |
  | pendulum | +0.0022 [+0.0009, +0.0039] |

- Horizon correlations positive on every task.
- The head-to-head with EMARec, CI > 0 on all three tasks.
- Fig. 1d scramble z: +22.8 / +36.3 / +2.4 (pendulum barely above 2).
- Fig. 2c on cartpole and reacher.
- The steering E null.
- The §8.2 nulls.
- The geometry on 12 of 13 models.
- §10.3.
- SAE diffuseness.

## All numbers

| § | Quantity | Published | Final | Note | Flag |
|---|---|---|---|---|---|
| 5.1 | cartpole probe AUROC, clean | 0.872 | 0.983 |  |  |
| 5.1 | cartpole probe AUROC, σ=0.1 noise | 0.807 | 0.832 |  |  |
| 5.1 | KL-matched groups: KL (nats), low- vs high-Rec | 22.9 vs 23.9 | 20.9 vs 21.9 |  |  |
| 5.1 | KL-matched groups: Rec, low vs high | 0.052 vs 0.471 | 0.054 vs 0.384 |  |  |
| 5.1 | cartpole KL-matched AUROC (clean+noisy / clean only) | 0.71 / 0.71 | 0.61 / 0.58 |  |  |
| 5.1 | cartpole within-bin r(Rec, C_t) | +0.39 | +0.23 |  |  |
| 5.2/Fig1b | cartpole R²(readout, C_t), primary | 0.80 | 0.839 [0.803, 0.868] | logit 0.687 |  |
| 5.2/Fig1b | cartpole R²(readout, KL_t), primary; C_t − KL | 0.52; C_t clearly ahead | 0.595 [0.548, 0.646]; +0.244 (C_t ahead on 5/5 models) | logit: R²(KL) 0.687, C_t − KL +0.000, C_t ahead on 1/5 models |  |
| 5.2/Fig1b | reacher R²(readout, C_t), primary | — | 0.503 [0.465, 0.539] | logit 0.475 |  |
| 5.2/Fig1b | reacher R²(readout, KL_t), primary; C_t − KL | —; about the same | 0.474 [0.447, 0.506]; +0.028 (C_t ahead on 3/4 models) | logit: R²(KL) 0.608, C_t − KL -0.133, C_t ahead on 0/4 models |  |
| 5.2/Fig1b | pendulum R²(readout, C_t), primary | — | 0.891 [0.868, 0.911] | logit 0.712 |  |
| 5.2/Fig1b | pendulum R²(readout, KL_t), primary; C_t − KL | —; C_t clearly ahead | 0.360 [0.319, 0.397]; +0.532 (C_t ahead on 3/4 models) | logit: R²(KL) 0.608, C_t − KL +0.104, C_t ahead on 1/4 models | C_t not ahead on 1/4 models |
| 5.2/Fig1d | cartpole scramble z, ridge h→C_t (scrambled R²) | +18.7 (0.69) | +22.8 (0.23; real 0.82) |  |  |
| 5.2/Fig1d | reacher scramble z, ridge h→C_t (scrambled R²) | +39.0 (0.14) | +36.3 (0.02; real 0.53) |  |  |
| 5.2/Fig1d | pendulum scramble z, ridge h→C_t (scrambled R²) | +51.4 (0.45) | +2.4 (0.94; real 0.94) |  |  |
| 5.2 | cartpole v-ablation, readout z (Task-G sites) | z ≈ −22, 100th pct | -1.5 (pct 0) | logit z -27.7. Probability null DEGENERATE: random-direction ablations swing the saturated probability (null sd 0.38, v's change -0.86); the logit null is not | /z/ < 2 (probability; null degenerate) |
| 5.3/T3 | cartpole share of ‖v‖² in top-10 PCs: primary (range over models); random mean / 5th pct | 0.5%; random 3.9% | 0.62% (0.58–1.61%); random 3.9% / 1.5% |  |  |
| 5.3/T3 | reacher share of ‖v‖² in top-10 PCs: primary (range over models); random mean / 5th pct | 0.2%; random 3.9% | 1.27% (0.71–1.27%); random 3.9% / 1.4% |  |  |
| 5.3/T3 | pendulum share of ‖v‖² in top-10 PCs: primary (range over models); random mean / 5th pct | 1.1–1.5%; random 3.9% | 6.17% (0.56–6.17%); random 3.9% / 1.7% |  | above random mean on 1/4 models |
| 5.3 | cartpole share in top-50 PCs; random | 9%; 19.5% | 7.4%; 19.6% |  |  |
| 8.3 | cartpole share of v outside top-50; random | 91%; ≈80% | 93%; 80% |  |  |
| Fig1c (old stat.) | cartpole mean angle of v to top-50 PCs | ≈88° | 84.8° (random 87.1°) |  |  |
| Fig1c (old stat.) | reacher mean angle of v to top-50 PCs | ≈88° | 84.2° (random 87.1°) |  |  |
| Fig1c (old stat.) | pendulum mean angle of v to top-50 PCs | ≈88° | 87.0° (random 87.1°) |  |  |
| T3 | cartpole R²(readout, C_t): primary; mean (range) over models | 0.80 | 0.84; 0.83 (0.81–0.85) | logit mean 0.66 (0.57–0.69) |  |
| T3 | reacher R²(readout, C_t): primary; mean (range) over models | 0.26 (mean of 4) | 0.50; 0.52 (0.50–0.55) | logit mean 0.47 (0.44–0.48) |  |
| T3 | pendulum R²(readout, C_t): primary; mean (range) over models | 0.86 (mean of 4) | 0.89; 0.77 (0.58–0.89) | logit mean 0.60 (0.50–0.71) |  |
| T3 | cartpole KL-matched AUROC: primary (range over models) | 0.71 | 0.61 (0.43–0.65) |  | other side of 0.5 on 2/5 models |
| T3 | reacher KL-matched AUROC: primary (range over models) | 0.57–0.75 | 0.48 (0.41–0.57) |  | other side of 0.5 on 2/4 models |
| T3 | pendulum KL-matched AUROC: primary (range over models) | 0.32–0.50 | 0.44 (0.35–0.45) |  |  |
| T3 | cartpole within-bin r(Rec, C_t), primary | +0.39 | +0.23 |  |  |
| T3 | reacher within-bin r(Rec, C_t), primary | -0.09 | +0.29 |  | sign flip |
| T3 | pendulum within-bin r(Rec, C_t), primary | -0.12 | -0.07 |  |  |
| 5.4 | pendulum r(KL_t, C_t) | 0.66 | 0.62 [0.59, 0.65] |  |  |
| 5.4 | pendulum AUROC with groups relabelled by C_t | 0.53 | 0.31 |  |  |
| 4.2 | γ / EMARec α | 0.95, 0.70, 0.90 / 0.02, 0.5, 0.02 | unchanged (fixed constants) |  |  |
| T4/Fig2b | cartpole ΔR² of C_t, K=10 | +0.0280 [+0.0154, +0.0433] | +0.0168 [+0.0079, +0.0288] | 46700 sites / 100 episodes; controls kl, rec, emarec, ens |  |
| 6.2 | cartpole baseline R² of controls | ≈2/3 | 0.702 [0.631, 0.764] |  |  |
| 6.1/Fig2a | cartpole r(C_t, E_K), K=1/5/10/20 | all > 0, ≈0.6, flat | +0.63 / +0.65 / +0.64 / +0.60 | K=10 CI [+0.60, +0.68] |  |
| 6.3 | cartpole ΔR² adding C_t to EMARec alone | +0.177 | +0.1664 [+0.1319, +0.1999] | reverse (EMARec beyond C_t) +0.1659 |  |
| 6.3/Fig2c | cartpole external scramble z (Section-6 C_t, same sites as Fig 2a) | +19.6 | +25.1 | real r +0.638 vs scrambled +0.117 |  |
| T4/Fig2b | reacher ΔR² of C_t, K=10 | +0.0182 [+0.0083, +0.0309] | +0.0053 [+0.0015, +0.0084] | 46700 sites / 100 episodes; controls kl, rec, emarec |  |
| 6.2 | reacher baseline R² of controls | ≈1/8 | 0.591 [0.514, 0.699] |  |  |
| 6.1/Fig2a | reacher r(C_t, E_K), K=1/5/10/20 | all > 0, decays | +0.48 / +0.51 / +0.52 / +0.52 | K=10 CI [+0.50, +0.56] | no longer decays (flat) |
| 6.3 | reacher ΔR² adding C_t to EMARec alone | +0.049 | +0.0999 [+0.0416, +0.1156] | reverse (EMARec beyond C_t) +0.2636 |  |
| 6.3/Fig2c | reacher external scramble z (Section-6 C_t, same sites as Fig 2a) | +21.5 | +54.6 | real r +0.524 vs scrambled -0.003 |  |
| T4/Fig2b | pendulum ΔR² of C_t, K=10 | +0.0019 [+0.0001, +0.0051] | +0.0022 [+0.0009, +0.0039] | 46700 sites / 100 episodes; controls kl, rec, emarec |  |
| 6.2 | pendulum baseline R² of controls | ≈2/5 | 0.492 [0.468, 0.517] |  |  |
| 6.1/Fig2a | pendulum r(C_t, E_K), K=1/5/10/20 | all > 0, decays; ≈0.17 at K=10 | +0.28 / +0.25 / +0.19 / +0.11 | K=10 CI [+0.17, +0.21] |  |
| 6.3 | pendulum ΔR² adding C_t to EMARec alone | +0.013 | +0.0485 [+0.0398, +0.0563] | reverse (EMARec beyond C_t) +0.0909 |  |
| 6.3/Fig2c | pendulum external scramble z (Section-6 C_t, same sites as Fig 2a) | -11.7 | +21.0 | real r +0.192 vs scrambled +0.010 | SIGN FLIP |
| 10.3 | cartpole reward-inflating sites; C_t drift vs other | 162/6000; C_t 7.5 vs 8.9 | 789/46700; 8.39 vs 8.96 |  |  |
| 10.3 | reacher reward-inflating sites; C_t drift vs other | 91/6000; C_t 1.2 vs 1.7 | 695/46700; 0.58 vs 1.64 |  |  |
| T5 | partial obs, probe AUROC: full / partial (5 pairs) | 0.871±0.028 / 0.951±0.013 | 0.972±0.015 / 0.966±0.009; partial > full on 2/5 |  | "every pair" fails (2/5) |
| T5 | partial obs, R²(readout, C_t): full / partial (5 pairs) | 0.761±0.035 / 0.824±0.013 | 0.816±0.039 / 0.841±0.031; partial > full on 4/5 | logit 0.655 / 0.718; partial > full on 5/5 | "every pair" fails (4/5) |
| T5 | partial obs, ΔR² of C_t for E (beyond KL, Rec): full / partial (5 pairs) | 0.062±0.027 / 0.009±0.003 | 0.029±0.039 / 0.011±0.005; partial < full on 3/5 |  | direction holds on 3/5 |
| T6 | cartpole steering z: readout k=0 / k=10 / E (n=5) | +12.9±1.6 / +12.2±1.4 / +0.2±0.4 | +10.6 ± 1.5 / +6.2 ± 4.0 / -0.1 ± 0.4 | k=10 per model: +5.2, +9.1, +4.7, +0.8, +11.0; logit k=0 / k=10: +16.6 ± 1.9 / +7.1 ± 5.0 (k=10 per model: +5.3, +10.1, +4.7, +1.4, +14.2) | k=10 readout z < 2 on 1/5 |
| T6 | reacher steering z: readout k=0 / k=10 / E (n=4) | +13.3±1.4 / +5.9±0.7 / +0.3±0.4 | +11.1 ± 0.8 / +1.2 ± 1.3 / -0.1 ± 0.2 | k=10 per model: +2.0, +0.3, -0.0, +2.6; logit k=0 / k=10: +16.7 ± 0.2 / +1.2 ± 1.2 (k=10 per model: +1.9, +0.4, +0.0, +2.6) | k=10 readout z < 2 on 3/4 |
| T6 | pendulum steering z: readout k=0 / k=10 / E (n=4) | +15.6±3.0 / +11.4±0.6 / +0.6±0.8 | +11.4 ± 4.7 / -0.1 ± 0.1 / +0.2 ± 0.3 | k=10 per model: -0.0, -0.1, +0.1, -0.2; logit k=0 / k=10: +17.4 ± 2.4 / -0.0 ± 0.1 (k=10 per model: -0.0, +0.1, +0.0, -0.2) | k=10 readout z < 2 on 4/4 |
| 8.1 | readout z range over 13 models (k=0 and k=10) | +5.2..+19.2 | -0.2..+17.6 | logit -0.2..+19.7 | minimum < 2 |
| 8.1 | max /E z/ over 13 models | 1.48 | 0.59 |  |  |
| 8.2 | steering E null under σ=0.1 noise, max /z/ (3 primary models) | 1.43 | 0.37 |  |  |
| 8.2 | v-ablation E z, full observation (5 models) | −0.37±0.36 | -0.15±0.51 | max /z/ 0.71 |  |
| 8.2 | v-ablation E z, partial observation (5 models) | +0.13±0.22 | -0.38±0.37 | max /z/ 0.78 |  |
| T7 | share of v: top-1 / top-4 / top-10 atoms (3 tasks × 3 SAEs) | 0.3–0.5% / 0.9–1.2% / ≤5% | 0.24–0.46% / 0.91–1.42% / 2.10–2.94% |  |  |
| 9.1 | random-vector reconstruction cosine | ≈0.97 | 1.000 |  |  |
| T8/T9 | cartpole SAE run 0: atom #1879 — r(act,KL) / r(act,E); ablation z E / readout; cos(atom, v) | #156: +0.499 / positive; +10.30 / -5.0 | +0.260 / +0.508; +7.04 / -0.08; +0.009 | readout logit z +0.10; probability null sd 0.378 | readout effect no longer clears /z/ > 2 |
| T8/T9 | cartpole SAE run 1: atom #1039 — r(act,KL) / r(act,E); ablation z E / readout; cos(atom, v) | (new run) | -0.517 / -0.512; +3.27 / +0.38; +0.024 | readout logit z +0.43; probability null sd 0.378 | readout effect no longer clears /z/ > 2 |
| T8/T9 | cartpole SAE run 2: atom #1741 — r(act,KL) / r(act,E); ablation z E / readout; cos(atom, v) | (new run) | +0.518 / +0.529; +7.11 / -1.28; +0.002 | readout logit z -0.05; probability null sd 0.378 | readout effect no longer clears /z/ > 2 |
| T8/T9 | reacher SAE run 0: atom #1586 — r(act,KL) / r(act,E); ablation z E / readout; cos(atom, v) | #612: 0.02–0.06 / +0.21–0.29; +2.67 / n.s. | -0.188 / -0.294; +6.58 / +1.20; -0.080 | readout logit z +3.10; probability null sd 0.257 | not KL-independent |
| T8/T9 | reacher SAE run 1: atom #1837 — r(act,KL) / r(act,E); ablation z E / readout; cos(atom, v) | (new run) | +0.112 / +0.210; +33.98 / +1.30; -0.079 | readout logit z +9.50; probability null sd 0.257 | not KL-independent |
| T8/T9 | reacher SAE run 2: atom #887 — r(act,KL) / r(act,E); ablation z E / readout; cos(atom, v) | (new run) | +0.240 / +0.209; +18.30 / -0.07; -0.015 | readout logit z -0.30; probability null sd 0.257 | not KL-independent |
| T8/T9 | pendulum SAE run 0: atom #747 — r(act,KL) / r(act,E); ablation z E / readout; cos(atom, v) | #310: — / —; +0.00 / +0.0 | +0.530 / +0.644; +18.95 / +0.99; -0.007 | readout logit z +0.10; probability null sd 0.401 | E effect now clears z > 2 |
| T8/T9 | pendulum SAE run 1: atom #1821 — r(act,KL) / r(act,E); ablation z E / readout; cos(atom, v) | (new run) | -0.543 / -0.647; +0.95 / -1.33; +0.031 | readout logit z -2.56; probability null sd 0.401 |  |
| T8/T9 | pendulum SAE run 2: atom #1754 — r(act,KL) / r(act,E); ablation z E / readout; cos(atom, v) | (new run) | -0.497 / -0.547; +4.89 / +1.16; -0.024 | readout logit z +2.32; probability null sd 0.401 | E effect now clears z > 2 |
