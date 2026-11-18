# Item 1 — Corrected-logging re-check of the published numbers (ICAART 2027 #388)

Run 2026-11-16/17 on the local M4, CPU. Code:
- `src/camera_ready/{core,causal}.py`
- `run_cr_A_{log,analysis,causal,sae,report}.py`
- `tests/test_camera_ready.py`

Machine-readable outputs:
- `A_consistency.json`
- `A_report_paperprobe/A_table_autogen.md` (103 rows)
- `A_report_frozenprobe/A_table_autogen.md` (103 rows)

## 0. Bottom line

1. **One published finding changes in kind because of the logging fix: reacher SAE atom #612.**
   - Its ablation effect on imagination error is E z = **+1.61** with corrected logging, against **+2.67** published. That is below the paper's z > 2 threshold.
   - Running the *same code* with the old action pairing gives +2.41, so the change is caused by the logging fix.
   - Its held-out KL-independence also weakens: r(act, KL) is +0.13 (published 0.02–0.06), and r(act, E) is +0.15 (published +0.21–0.29).
   - The paper's selection rule no longer picks #612 (reselected per SAE run: #1586, #616, #1344).
2. **Reacher's descriptive statistics change substantially, also because of the logging fix.** The old-pairing control reproduces the published reacher values each time.
   - within-bin r(Rec, C_t): −0.09 → **+0.29**, a sign flip
   - KL-matched AUROC: 0.57–0.75 → 0.48 (0.41–0.57); some models fall below 0.5
   - baseline R² of {KL, Rec, EMARec} for E: ≈1/8 → **0.59**
   - head-to-head C_t over EMARec: +0.049 → +0.100
   - r(C_t, E_10): 0.26 → 0.52

   The old pairing degraded reacher's reconstruction badly (mean Rec 5.09 vs 0.027 corrected), which inflated how much was left for C_t to explain.
3. **Everything else holds in kind:**
   - Table 4 ΔR² excludes 0 on all three tasks.
   - The steering dissociation holds on all 13 models, with readout k=10 z between +5.5 and +21.0 and max |E z| = 1.27.
   - The §8.2 ablation and noise nulls hold.
   - Atom #156 still raises E and still moves the readout.
   - Atom #310 is still null.
   - Pendulum's reversed external scramble holds (−11.6, published −11.7).
   - The low-variance geometry holds.
   - C_t is still lower at reward-inflating sites.
4. **Correction to my earlier flag 1.** Three of the four "changes" I reported were caused by a protocol change I introduced, not by the logging fix:
   - pendulum and reacher steering k=10
   - the #156 readout effect

   That protocol change was refitting v on frozen-model episodes instead of using the paper's probe. Section 2 explains. Only #612 was a logging effect.
5. **Two published claims are not reproducible on any re-loggable data, with either pairing:**
   - Section 5.2's R²(readout, C_t) = 0.80 / 0.26 / 0.86. These were computed on held-out *training-log* states, collected online during training from evolving model snapshots; those states cannot be re-logged.
   - Table 5's "on every pair". It holds for AUROC (5/5) but fails for R² on 1 of 5 pairs. This row is not attributed: the old-pairing control was not run for the partial-observation models.

   Details are in Sections 3–4.

## 1. What was run

- **Corrected logging:**
  - o_t is paired with a_{t−1} (zeros at t=0), and imagination from site t feeds a_t exactly once.
  - `test_action_convention` passes for both model classes (11 tests pass in total).
  - Byte-reproducible across processes (sha256 check).
  - Episodes are identical to the sprint cache.
- **Episodes:** 100 evaluation episodes per task, 46,700 sites (paper: 6,000 subsampled). A further 50 calibration and 100 probe-fit episodes per task are used only where a frozen-model probe or a calibration choice is needed.
- **Models:**
  - all 13 standard models (cartpole: primary + seeds 1–4; reacher and pendulum: primary + 3 replicates each); cartpole `seed_0` is extra, reported separately and not aggregated;
  - the 10 partial-observation models (5 pairs);
  - SAEs: the paper's 3 published dictionaries, plus 3 retrained on corrected states with the paper's recipe.
- **Paper definitions kept:**
  - C_t: per-trajectory median for Section 6/Table 4, γ = 0.95 / 0.70 / 0.90; a global threshold for Fig. 2c / Phase 2.
  - E_{t,10}.
  - The 50-direction null (canonical file).
  - Steering λ ∈ {−2..2}σ_v, with z of the slope.
  - Ablation h − (h·d)d.
  - Task G's C_t-weighted site rule for §5.2.
  - Atom selection: top-20-variance atoms, best |r(E)| on half A, evaluated on half B.
  - The drift-site rule for §10.3.
  - Geometry in standardized h space, as the paper did (`run_second_env.py:171`).
- **Two probe variants on corrected states**, because the result depends on which is used (Section 2):
  - **Paper probe (main comparison):** v fit exactly as the paper did, on 60% of each model's training log. This isolates the logging fix.
  - **Frozen-model probe:** v refit on 100 probe-fit episodes from the frozen model, label 1[KL > calibration median].
- **Attribution control:** the identical code with the old pairing (o_t with a_t), primary models, both probe variants (`control_oldpairing/`). Differences between control and corrected are due to the logging fix alone. Differences between control and published are due to my re-implementation's protocol (site sets, sample size, CRN, evaluation episodes vs training log).
- **Deviations from the paper's procedure:**
  - v and every null direction use the same sites with common random numbers. The paper ran v on 500 sites and the null on 200 different sites, without CRN.
  - The null is the canonical 50-direction file. The paper used RNG seeds 6060 (steering) and 3033 (atoms).
  - EMARec α is selected on calibration episodes by the paper's rule (0.05 / 0.5 / 0.2 corrected, against published 0.02 / 0.5 / 0.02).

## 2. Readout definition (item 2)

**Every published readout number used the probe probability** (`predict_proba`):
- Section 5.2, Fig. 1a/b and Table 3: R²(readout, C_t)
- Table 5: R²
- Table 6: readout at k=0 and k=10
- the §5.2 Task G ablation
- Tables 8 and 9: atom readout effects
- the §8.2 ablation readout

The projection v^T h_t appears in the old code only inside σ_v and inside the ablation operator. AUROCs (probe and KL-matched) are identical under both definitions, because the probability is monotone in the projection. Both definitions are reported below wherever the paper used the probability. From now on the projection is used everywhere.

**The probe-fitting set matters more than the definition.** On corrected states:

| Steering readout z at k=10 | paper probe | frozen-model probe |
|---|---|---|
| cartpole | +13.1 ± 2.2 | +6.2 ± 4.0 |
| reacher | +6.5 ± 1.2 | +1.2 ± 1.3 |
| pendulum | +12.6 ± 5.6 | −0.1 ± 0.1 |

So the persistence of the steering effect depends on which probe v is. That is worth one sentence in the camera-ready. The same holds under the old pairing: control k=10 with the frozen probe gives +7.4 / +0.5 / −0.1.

## 3. Side-by-side: published vs corrected

Columns:
- **Corr. (paper probe):** corrected logging with the paper's probe, the main comparison.
- **Corr. (frozen probe):** corrected logging with v refit on frozen-model episodes.
- **Control:** old pairing, same code, primary models. It is shown where it decides the attribution, given as paper-probe value / frozen-probe value.

Flags:
- **⚠ logging:** the change is due to the logging fix, because the control reproduces the published value.
- **⚠ protocol:** the change is also present under the old pairing.
- **✗:** changes in kind (sign, or the significance threshold is crossed).

### 3.1 Section 5.1 (cartpole)

| Quantity | Published | Corr. (paper probe) | Corr. (frozen probe) | Control | Flag |
|---|---|---|---|---|---|
| probe AUROC, clean | 0.872 | 0.907 | 0.983 | 0.884 / 0.940 | — |
| probe AUROC, σ=0.1 | 0.807 | 0.797 | 0.832 | — | — |
| KL-matched groups, KL (nats) | 22.9 vs 23.9 | 20.9 vs 21.9 | same | — | — |
| KL-matched groups, Rec | 0.052 vs 0.471 | 0.054 vs 0.384 | same | — | — |
| KL-matched AUROC (pooled) | 0.71 | 0.65 | 0.61 | 0.67 / 0.61 | lower, still > 0.5 |
| KL-matched AUROC (clean only) | 0.71 | 0.67 | 0.58 | — | lower, still > 0.5 |
| within-bin r(Rec, C_t) | +0.39 | +0.21 | +0.18 | +0.33 / +0.33 | lower, same sign |

### 3.2 Section 5.2 / Fig. 1b,d

| Quantity | Published | Corrected | Control | Flag |
|---|---|---|---|---|
| R²(readout, C_t) cartpole / reacher / pendulum (prob.) | 0.80 / 0.26 (mean of 4) / 0.86 (mean of 4) | frozen probe: 0.86 / 0.53 / 0.87 (means over models; primary 0.86 / 0.50 / 0.97); paper probe: 0.38 / 0.02 / 0.13 (primary) | paper probe: 0.42 / 0.20 / 0.14; frozen probe: 0.83 / 0.39 / 0.95 | **⚠ protocol:** published values were computed on training-log states and are not reproducible on evaluation episodes with either pairing. With a probe fit on the same kind of states (frozen probe), the finding holds and is stronger. |
| R²(projection, C_t) | not published | frozen probe 0.66 / 0.47 / 0.69; paper probe 0.55 / 0.02 / 0.74 | — | — |
| R²(readout, KL) vs R²(readout, C_t), frozen probe | C_t clearly ahead on cartpole and pendulum, about equal on reacher | +0.26 / +0.03 / +0.61 (C_t − KL) | — | holds |
| ridge h→C_t scramble z (paper γ) | +18.7 / +39.0 / +51.4 | +22.8 / +36.3 / **+2.4** | +24.9 / +58.0 / +9.1 | pendulum much weaker but still > 2. Mostly ⚠ protocol (+9.1 even under the old pairing); the logging fix lowers it further. |
| scrambled R² | 0.69 / 0.14 / 0.45 | 0.23 / 0.02 / 0.94 | 0.22 / 0.07 / 0.93 | ⚠ protocol |
| Task G v-ablation readout z | ≈ −22 (100th pct) | paper probe: prob −84.7, proj −14.3 | −45.0 | holds. With the frozen probe the probability null is degenerate (saturated probe: prob z −1.5); proj z is −27.7. |

### 3.3 Section 5.3 and Table 3 geometry (standardized space, as published)

| Share of ‖w‖² in top-10 PCs | Published | Corr. (paper probe), primary (range) | Corr. (frozen probe) | Random |
|---|---|---|---|---|
| cartpole | 0.5% | 0.83% (0.72–1.17) | 0.62% | 3.9% |
| reacher | 0.2% | 0.17% (0.17–0.32) | 1.27% | 3.8% |
| pendulum | 1.1–1.5% | 1.49% (0.86–1.49) | **6.17%**, above random | 4.0% |

Other geometry rows, paper probe:
- cartpole top-50: 8.9% (published 9%; random 19.5%)
- share of v outside the top-50: 91% (published 91%; random 81%)
- mean angle to the top-50 PCs: 88.4° / 88.5° / 88.2° (published ≈88°; random 87.1°)

**Verdict: holds with the paper's probe.** With a probe refit on frozen-model states, pendulum's v is *not* low-variance (6.2%, above the random mean). The geometry claim is therefore probe-dependent on pendulum.

### 3.4 Table 3 rows 2–3 and the Section 5.4 text

| Quantity | Published | Corr. (paper probe) | Corr. (frozen probe) | Control | Flag |
|---|---|---|---|---|---|
| KL-matched AUROC, reacher, range over models | 0.57–0.75 | 0.56 (0.50–0.56) | 0.48 (0.41–0.57) | 0.64 / 0.58 | **⚠ logging:** range no longer holds; crosses 0.5 for some models |
| KL-matched AUROC, pendulum, range over models | 0.32–0.50 | 0.41 (0.41–0.53) | 0.44 (0.35–0.45) | 0.42 / 0.44 | ranges overlap; one model above 0.5 with the paper probe |
| within-bin r(Rec, C_t), reacher | −0.09 | **+0.31** | **+0.29** | −0.06 / −0.06 | **✗ sign flip, ⚠ logging** |
| within-bin r(Rec, C_t), pendulum | −0.12 | −0.14 | −0.02 | −0.21 / −0.11 | holds |
| pendulum r(KL, C_t) | 0.66 | 0.61 | 0.60 | — | holds |
| pendulum AUROC relabelled by C_t | 0.53 | 0.35 | 0.49 | — | near or below chance either way |

Consequence for the text: the paper's sign argument ("+ on cartpole, − on pendulum, reacher unexplained: −0.09 but AUROC > 0.5") changes. Reacher's within-bin r is now positive.

### 3.5 Section 6 / Table 4 / Fig. 2 (no probe involved, primary models)

C_t uses the paper's definition: per-trajectory median, paper γ, controls {KL, Rec, EMARec} (+ ensemble on cartpole). Trajectory bootstrap, 1,000 resamples.

| Quantity | Published | Corrected | Control | Flag |
|---|---|---|---|---|
| ΔR² cartpole | +0.0280 [+0.0154, +0.0433] | **+0.0179 [+0.0087, +0.0302]** | +0.0262 | holds (CI > 0) |
| ΔR² reacher | +0.0182 [+0.0083, +0.0309] | **+0.0053 [+0.0015, +0.0084]** | +0.0088 | holds, smaller (partly ⚠ logging) |
| ΔR² pendulum | +0.0019 [+0.0001, +0.0051] | **+0.0047 [+0.0028, +0.0070]** | +0.0130 | holds |
| ΔR² with calibration median and calibration γ (protocol-corrected) | — | +0.0427 [+0.0284, +0.0596] / +0.0105 [+0.0058, +0.0140] / +0.0213 [+0.0156, +0.0261] | — | all CIs > 0 |
| baseline R² of controls | ≈2/3 / ≈1/8 / ≈2/5 | 0.70 / **0.59** / 0.48 | 0.68 / 0.12 / 0.44 | reacher ⚠ logging |
| r(C_t, E_K), K = 1/5/10/20, cartpole | all > 0, ≈0.6, flat | +0.63 / +0.65 / +0.64 / +0.60 | — | holds |
| r(C_t, E_K), K = 1/5/10/20, reacher | all > 0, decays | +0.48 / +0.51 / +0.52 / +0.52 | — | all > 0, but **no longer decays** (flat) |
| r(C_t, E_K), K = 1/5/10/20, pendulum | all > 0, decays (≈0.17 at K=10) | +0.28 / +0.25 / +0.19 / +0.11 | — | holds |
| head-to-head: C_t added to EMARec | +0.177 / +0.049 / +0.013 | +0.157 / **+0.100** / **+0.0007** [+0.0001, +0.0017] | +0.146 / +0.031 / +0.000 | cartpole holds; reacher doubles (⚠ logging); pendulum ~0, CI still > 0 |
| external scramble z (Fig. 2c definition) | +19.6 / +21.5 / −11.7 | +43.0 / +58.7 / **−11.6** | — | holds, including pendulum's reversal |

### 3.6 Section 7 / Table 5 (partial observation, 5 pairs)

Not attributed: the old-pairing control was not run for these models.

| Quantity | Published full / partial | Corr. (paper probe) | Corr. (frozen probe) | Flag |
|---|---|---|---|---|
| probe AUROC | 0.871±0.028 / 0.951±0.013 | 0.905±0.025 / 0.934±0.025; partial > full on **5/5** | 0.972 / 0.966; **2/5** | paper probe holds; frozen probe fails |
| R²(readout, C_t), probability | 0.761±0.035 / 0.824±0.013 | 0.570±0.136 / 0.735±0.131; **4/5** | 0.847 / 0.861; **3/5** | **✗ "on every pair" fails** (one pair) |
| R²(projection, C_t) | not published | 0.585 / 0.701; 4/5 | 0.632 / 0.715; **5/5** | — |
| ΔR² of C_t for E (beyond KL, Rec) | 0.062±0.027 / 0.009±0.003 | 0.050±0.069 / 0.013±0.007; full > partial on 4/5 | same | direction holds on 4/5 pairs; full-model variance is large |

### 3.7 Section 8 / Table 6 (steering, the paper's 13 models)

| Task (n) | Published k=0 / k=10 / E | Corr. paper probe, prob. | Corr. paper probe, proj. | Corr. frozen probe, prob. | Flag |
|---|---|---|---|---|---|
| cartpole (5) | +12.9±1.6 / +12.2±1.4 / +0.2±0.4 | +13.7±2.5 / +13.1±2.2 / +0.2±0.9 | +16.5±1.6 / +14.6±1.7 / +0.2±0.9 | +10.6 / +6.2 / −0.1 | holds |
| reacher (4) | +13.3±1.4 / +5.9±0.7 / +0.3±0.4 | +13.2±0.5 / +6.5±1.2 / +0.1±0.5 | +15.4±1.2 / +6.5±1.2 / +0.1±0.5 | +11.1 / **+1.2** / −0.1 | holds with the paper probe; k=10 is probe-dependent |
| pendulum (4) | +15.6±3.0 / +11.4±0.6 / +0.6±0.8 | +15.9±5.6 / +12.6±5.6 / +0.4±0.8 | +16.7±1.8 / +12.0±3.0 / +0.4±0.8 | +11.4 / **−0.1** / +0.2 | holds with the paper probe; k=10 is probe-dependent |

Summary rows:
- Readout z range across all models, k=0 and k=10: published +5.2..+19.2; corrected **+5.5..+24.2** (probability, paper probe) and +5.5..+18.9 (projection).
- Max |E z|: published 1.48; corrected **1.27**. The E null holds on every model under both probes.
- Extra model, cartpole `seed_0` (not one of the 13): k=0 +12.2, k=10 +11.6, E +0.9.

### 3.8 Section 8.2

| Quantity | Published | Corrected | Flag |
|---|---|---|---|
| v-ablation E z, full obs (5 models) | −0.37±0.36 | −0.58±0.24 (paper probe); −0.15±0.51 (frozen probe) | holds |
| v-ablation E z, partial obs (5 models) | +0.13±0.22 | −0.41±0.29; −0.38±0.37 | holds |
| noisy steering, max \|E z\| (3 primary models) | 1.43 | 0.53; 0.37 | holds |

### 3.9 Section 9 / Tables 7–9 (primary models, published SAE dictionaries)

| Quantity | Published | Corr. (paper probe) | Control (old pairing, paper probe) | Flag |
|---|---|---|---|---|
| share of v: top-1 / top-4 / top-10 | 0.3–0.5% / 0.9–1.2% / ≤5% | 0.24–0.53% / 0.88–1.40% / 2.0–2.7% | — | slightly outside the published ranges at both ends (as already noted in September); "diffuse" holds |
| #156 cartpole: ablation E z / readout z (prob., proj.) | +10.3 / −5.0 | +7.76 / **−9.70**, −5.91 | +7.61 / −7.23 | holds |
| #156: r(act, KL), r(act, E) held-out | +0.499 / positive | +0.555 / +0.511; reselected on run 0 | — | holds |
| **#612 reacher: ablation E z / readout z** | **+2.67** / n.s. | **+1.61** / −0.55 | **+2.41** / −0.49 | **✗ ⚠ logging: E effect no longer clears z > 2** |
| #612: r(act, KL), r(act, E) held-out | 0.02–0.06 / +0.21–0.29 | +0.13 / +0.15; **not reselected** | — | KL-independence weakens |
| reacher atom reselected per SAE run: r(KL), r(E) | 0.02–0.06 / +0.21–0.29 | −0.22..+0.31 / −0.28..+0.21 (#1586, #616, #1344) | — | the "replicates across SAE runs" claim fails |
| #310 pendulum: ablation E z / readout z | \|z\| < 1 both | +0.33 / +0.12 | +0.49 / +0.14 | holds |

**SAEs retrained on corrected states with the paper's recipe** (variance explained 0.998, 0.3% dead atoms), atoms reselected by the paper's rule:
- **cartpole #1879:** E z +7.04, readout z −0.25. The E effect holds; the readout effect does not appear.
- **reacher #1586:** E z **+6.58**, readout z +0.6. This is an E-only atom, the #612 *pattern*, but with r(act, KL) not near 0 across runs.
- **pendulum #747:** E z **+18.95**, readout z −0.4. A pendulum atom now *does* move E. The paper's "third atom has no effect" changes under a retrained SAE.
- Shares of v: top-1 0.22–0.29%, top-4 0.83–0.88%, top-10 ≤2.1%. "Diffuse" holds.

### 3.10 Section 10.3 (C_t at reward-inflating sites)

| Task | Published | Corrected | Flag |
|---|---|---|---|
| cartpole | 162 / 6,000 sites; C_t 7.5 vs 8.9 | 789 / 46,700; 8.39 vs 8.96 | holds (C_t lower) |
| reacher | 91 / 6,000; 1.2 vs 1.7 | 695 / 46,700; 0.58 vs 1.64 | holds |

Section 10.2 (the Kalman simulation) uses no world model and is unaffected.

## 4. Changes in kind (the stop-rule list)

| # | Item | Cause | Severity |
|---|---|---|---|
| 1 | **#612 ablation E z +2.67 → +1.61**, below z > 2; KL-independence weakens; not reselected; no replication across SAE runs | logging fix (control +2.41) | Changes one of the paper's four findings: the "two of three selected atoms move E" statement in Sections 9, 10, 12 and the abstract. With retrained SAEs, an E-moving reacher atom exists (#1586, z +6.6), but it is a different atom. |
| 2 | Reacher within-bin r(Rec, C_t): −0.09 → +0.29 (sign flip); KL-matched AUROC range 0.57–0.75 → 0.41–0.57 | logging fix | Changes the Table 3 row and the sign-relation paragraph after Table 3. |
| 3 | Table 5: "signal strengthens on every pair" holds for AUROC (5/5) but R² fails on 1/5 pairs | not attributed | Wording only ("4 of 5 pairs" for R²). |
| 4 | Section 5.2 R²(readout, C_t) values 0.80 / 0.26 / 0.86 | protocol: training-log evaluation, not re-loggable | Qualitative claim holds (frozen probe: 0.86 / 0.50 / 0.97; C_t ahead of KL on cartpole and pendulum). Numbers must change. |
| 5 | Steering k=10 persistence and pendulum geometry depend on which probe v is | protocol (probe-fitting set) | Holds as published with the paper's probe. Fails on reacher and pendulum with a probe refit on frozen-model states. A sensitivity worth stating. |

Not changed in kind:
- Table 4 (all CIs > 0)
- the steering dissociation (13/13 models; E null everywhere)
- the §8.2 nulls
- #156 and #310
- Fig. 2c, including pendulum's reversal
- the geometry, with the paper's probe
- Section 10.3
- the Kalman simulation

## 5. Anything that looks wrong

- **Probe saturation.** Probes refit on corrected frozen-model states are near-saturated (cartpole AUROC 0.98, pendulum 0.999, σ_v = 0.0026). The probability readout is then degenerate under random-direction ablations (null sd 0.38). Use the projection for any frozen-probe analysis; the projection is used from now on anyway.
- **Pendulum ridge-scramble z.** Published +51.4; corrected +2.4. My protocol reproduces only +9.1 even with the old pairing, so the published magnitude depends on the original in-sample, training-log-median protocol.
- **Unattributed rows.** The old-pairing control covers only the primary models, so the partial-observation rows (Table 5) are not attributed. That control would take about 15 minutes if you want it.
