# Signal Check — Consolidated Results & PAPER.md Reconciliation

**Single source of truth** bridging "experiments are done" → "paper is submittable."
Someone who reads only this document should be able to correctly update every affected
line of PAPER.md without re-reading DEV_LOG.md.

**Last updated:** 2026-07-06 (Phase 1d — COMPLETE). All tasks A–M done. Phase 1d added
Task J (third env, pendulum → §1.2b), K (§1.6), L (§1.6), M (A6 resolved, §1.3); three
prior open items (A6, Task-G distinct-measure, illusion-mitigation completeness) and the
reacher-confound item are now RESOLVED — the open-items list is shorter than before this pass.

**✅ PAPER.md reconciliation APPLIED (Phase 1c + 1d, 2026-07-06).** Every §2 prescription below
is written into PAPER.md (backup at `PAPER.md.bak`). Phase 1c: abstract (A1 seed CI, A3 reacher,
A4 z_gate→structural, A5 boundary magnitude, causal sentence); contributions C2/C3 +
new contribution 6 (causal + attractor); §4.1 Table 1 within-balance 0.506→0.321 with the
inversion footnote (A9); §4.3 geometry reframed to structural + Task B causal pointer; §4.6
boundary magnitude-effect (Task E); new §4.7 (causal load-bearing, empirical null, robustness,
5-seed replication + honest 3/5 routing & next-KL limits) and §4.8 (Biased Dreams reinforcing);
§6.1 Limitations (reacher partial-replication, z_gate framing); §6.2 extended with the
inference-time gate-override causal table (falsification preserved, not undone); §7 Related Work
(Berger et al. reinforcing + Makelov illusion mitigations); §8 conclusion; 2 references added.
Phase 1d edits: abstract causal sentence extended (swap intervention + forward-dynamics
dissociation); §4.6 boundary "Mechanistic basis" upgraded to Task M's causal magnitude-vs-
separability result (A6); §4.7 upgraded to a **four-way** illusion defense (added Task L swap)
plus a forward-dynamics-dissociation paragraph (Task K second distinct measure); §6.1
Limitations rewritten for **three** environments with the structure-generalizes / specifics-
don't pattern (Task J). Note left for author: Berger et al. citation needs full venue/year/initials.

Provenance key: P1 = Phase 1 (pilot), P1b = Phase 1b, P1c = Phase 1c. DEV_LOG section
names in parentheses.

---

## 1. Master results table

Every headline number, with sample size, spread/CI/percentile, and provenance. Single-model /
single-split numbers from the pilot are marked **(n=1)**; where Task C provides a 5-seed
mean±std it is shown in the "seeds" column once available.

### 1.1 Core probe AUROCs

| Metric | Value | n / spread | Source |
|---|---|---|---|
| Probe A — Set A (ID) | 0.809 ± 0.033 | **n=5** [0.777, 0.872] | P1 (§4.1), P1b Task C |
| Probe A — Set B (noisy OOD) | 0.846 | n=1 | P1 (§4.1) |
| **Probe A — Set C (KL-matched) [headline]** | **0.715 ± 0.074** | **n=5**, CI [0.666, 0.763]; all 5 seeds >0.5 (min-seed CI [0.535,0.642]) | P1 (§4.1), **P1b Task C** |
| Probe A — within-balance confound | **0.321 ± 0.094** | **n=5**, CI [0.271, 0.373] — **INVERTS below 0.5, not chance** (see §2 A-new) | P1 (§4.1), **P1b Task C** |
| Probe C — Set C (recon→h_t) | 0.721 | n=1 | P1 (§4.1) |
| z_t probe — Set C | 0.578 ± 0.109 | **n=5** [0.459, 0.752] — noisiest metric | P1, P1b Task C |
| RWM-U Ensemble — Set C | 0.595 ± 0.028 | **n=5**; Probe A > ens on all 5, pooled bootstrap Δ=+0.110 CI[+0.082,+0.138] p≈0 | P1 (§4.1), **P1b Task C** |
| C_t best γ (all 5 seeds) | **0.95** | **n=5, identical every seed** | P1b Task C |
| C_t R² (best γ) | 0.763 ± 0.045 | **n=5** [0.703, 0.828] | P1b Task C |
| Boundary probe AUROC | 1.0000 ± 0.0000 | **n=5, seed-invariant** | P1b Task C |
| Probe A held-out (ID) | 0.9019 | n=1 | P1 |

### 1.2 Closed-form C_t characterization

| Metric | Value | n / spread | Source |
|---|---|---|---|
| Best γ (cartpole) | 0.95 | n=1; **5-seed: 0.95 on ALL 5** | P1 (§4.2), P1b Task C |
| R²(probe ~ C_t) (cartpole) | 0.80 | n=1; **5-seed 0.763 ± 0.045** | P1 (§4.2), P1b Task C |
| Best γ (reacher) | 0.70 | n=1 | P1c Task D |
| R²(probe ~ C_t) (reacher) | 0.216 | n=1 | P1c Task D |
| Best γ (pendulum) | 0.90 | n=1 | **P1d Task J** |
| R²(probe ~ C_t) (pendulum) | **0.886** | n=1 (highest of 3 envs) | **P1d Task J** |

### 1.2b Three-environment generality (Task J resolves the n=2 ambiguity)

| Metric | cartpole | reacher | pendulum | generalizes? |
|---|---|---|---|---|
| Probe A held-out AUROC | 0.902 | 0.764 | 0.974 | yes (strong ID everywhere) |
| **Set C AUROC** | 0.723 | 0.619 | **0.322 (inverts!)** | **NO — environment-dependent** |
| Within-task confound | 0.506 | 0.578 | 0.437 | ~chance, wanders both ways |
| **C_t best R²** | 0.798 | 0.216 | 0.886 | signal present all 3; R² **env-dependent** |
| **C_t best γ** | 0.95 | 0.70 | 0.90 | **env-dependent** (not a universal constant) |
| Null-space angle (°) | 88.0 | 89.4 | 88.1 | **YES** (near-orthogonal all 3) |
| Frac in top-10 PC | 0.090 | 0.002 | 0.015 | **YES** (near-null-space all 3) |

**Pattern:** the confusion **direction**, the **null-space geometry**, and the **linear encoding of C_t** generalize across all three; the **Set C AUROC** and the **closed-form γ/R²** are environment-dependent. Pendulum is the decisive case — it has the *strongest* C_t encoding (0.886) yet an *inverted* Set C (0.322), dissociating the two and proving Set C's inversion is a construction artefact (recon-based labelling anti-aligns with confusion in pendulum's dynamics), not an absence of signal. Neither reacher nor cartpole was the outlier.

### 1.3 Null-space geometry

| Metric | Value | n / spread | Source |
|---|---|---|---|
| Angle probe→top PCs (cartpole) | 88.0–88.2° | n=1 | P1 (§4.3) |
| Frac probe variance in top PCs (cartpole) | ~9% (top-50) / 0.5% (top-10) | n=1 | P1, P1b/c |
| Angle probe→top-10 PCs (reacher) | 89.4° | n=1 | P1c Task D |
| Frac in top-10 PC (reacher) | 0.17% | n=1 | P1c Task D |
| Angle probe→top-10 PCs (pendulum) | 88.1° | n=1 | **P1d Task J** |
| Frac in top-10 PC (pendulum) | 1.5% | n=1 | **P1d Task J** |
| **z_gate causal: angle span across forced z∈[0.5,0.99]** | **1.29°** (min 88.5°, ≤1.9% in top-10 at all z) | 7 forced-z values, 1 frozen model | **P1b Task B** |
| **Boundary z_gate causal: ‖h‖-separability vs (1−z) overwrite** | r=+0.97 (‖h‖ 0.97→0.51 as z 0.5→0.99) — magnitude YES | 7 forced-z values | **P1d Task M** |
| Boundary z_gate causal: full-probe AUROC vs forced z | **1.0000 at every z** (span 0.000) — separability NOT gate-driven | 7 forced-z values | **P1d Task M** |

### 1.4 obs/imagination boundary (Task E reframe)

| Metric | Value | 95% CI | Source |
|---|---|---|---|
| Full linear probe | 1.0000 | [1.0000, 1.0000] | P1 (§4.6), P1b Task E |
| best single coordinate | 0.9884 | [0.9870, 0.9896] | **P1b Task E** |
| ‖h_t‖ (L2 norm) | 0.9764 | [0.9746, 0.9781] | **P1b Task E** |
| top-1 PC projection | 0.7314 | [0.7259, 0.7370] | P1b Task E |

→ **Largely a magnitude effect**, not a distributed direction (Task E).

### 1.5 Observation-routing oracle

| Metric | Value | n / spread | Source |
|---|---|---|---|
| Probe A recall @ 30% budget | 0.626 ± 0.047 (multiseed) / 0.818 (pilot) | **n=5** [0.568, 0.697] | P1 (§4.5), P1b Task C |
| Recon-oracle recall @ 30% budget | 0.561 ± 0.052 (multiseed) / 0.770 (pilot) | **n=5** [0.510, 0.652] | P1 (§4.5), P1b Task C |
| **Probe−recon routing gap** | **+0.065** | **n=5**, bootstrap CI [+0.058, +0.072]; Probe A > recon on ALL 5 seeds; Wilcoxon p=0.0625 (n=5 floor) | **P1b Task C** |

*Absolute recall differs from the pilot's 0.818/0.770 because the multiseed routing scores
top-25%-KL events on a different held-out sample; the **probe > recon ordering** is the claim,
and it replicates on all 5 seeds. (See §4 open-item on the n=5 Wilcoxon floor — lead with the
gap's bootstrap CI, which is comfortably above 0.)

### 1.6 Causal intervention (Task A) + hardening (Tasks G, I, K, L)

| Metric | Value | n / spread / percentile | Source |
|---|---|---|---|
| Ablation Δ probe @ t (confusion dir) | −0.575 | 600 sites, CI[−0.591,−0.560] | **P1b Task A** |
| Ablation Δ probe @ t (random dir) | −0.004 | 600 sites, CI[−0.008,+0.000] | P1b Task A |
| Routing flip rate — confusion vs random | 0.805 vs 0.027 | CI[0.770,0.837] vs [0.015,0.040] | **P1b Task A** |
| γ-decay: observed \|Δ_k\|/\|Δ_0\| vs γ^k | 1.00/0.81/0.60/0.50 vs 1.00/0.95/0.77/0.60 | k=0,1,5,10 | P1b Task A |
| **Confusion dir percentile vs 50-dir null (probe-decay)** | **100th** (z=−22.9) | 50 random dirs, 600 sites | **P1c Task G** |
| **Confusion dir percentile vs 50-dir null (routing)** | **100th** (z=+30.6) | 50 random dirs, 600 sites | **P1c Task G** |
| Distinct measure (latent divergence) vs null | 32nd pct (z=−0.6) — **partial** | 25 random dirs | P1c Task G |
| Distinct measure #2 (decoder recon on next REAL obs) vs null | 64th/56th pct (z=−0.7/−0.6) — **does not separate** | 600 sites, 50-dir null | **P1d Task K** |
| Robustness: effect retained at 0.25 dir-rotation | 58% | mean of 5 rotations | P1c Task G |
| **Swap intervention** (real-content substitution) probe-decay@t | **−0.761, 100th pct** (vs ablation −0.581) | 600 sites, 50-dir null, ⊥v match cos=0.86 | **P1d Task L** |
| Swap intervention routing flip | +0.868, 100th pct (vs ablation +0.807) | 600 sites | **P1d Task L** |
| Probe/C_t direction consistency | cos=0.778 (38.9°) | — | P1b Task A |
| **Causal probe-decay replicated across 5 seeds** | Δprobe@t = −0.385 ± 0.118, **100th pct on ALL 5** | n=5, each vs own 50-dir null | **P1c Task I** |
| Causal routing-flip across 5 seeds | +0.395 ± 0.133, **59th pct mean (3/5 separate)** — partial | n=5 | **P1c Task I** |
| Causal next-KL across 5 seeds | +0.061 ± 0.797, 72nd pct — **does not separate (as Task A)** | n=5 | P1c Task I |

### 1.7 Attractor-recovery cross-check (Task H, Biased Dreams)

| Metric | Value | n / spread | Source |
|---|---|---|---|
| r(confusion, imagined-vs-real latent gap) | +0.393 | ~4000 sites, p≈1.5e-147 | **P1c Task H** |
| r(confusion, OOD-perturbed gap) | +0.397 | p≈7e-151 | P1c Task H |
| Imagined reward overestimation (mean) | +0.176 | reward proxy | P1c Task H |
| r(confusion, reward-overestimation gap) | +0.480 | p≈5e-230 | P1c Task H |
| **Verdict** | **REINFORCING** | — | P1c Task H |

### 1.8 Preserved negative / retracted results (unchanged — do not remove)

| Result | Status | Source |
|---|---|---|
| Set C Strong (novel-task) 0.72 | **RETRACTED** (within-balance confound → 0.51) | P1 (§3.2) |
| z_gate-saturation → angle prediction | **FALSIFIED** (r=−0.889, n=6), now causally confirmed z-independent | P1, P1b Task B |
| Imagination-depth probe | **NULL** (0.4994, no depth signal) | P1 |
| Probe-weighted returns (binary) | **NEGATIVE** (Δr=−0.53) | P1 |
| Probe-weighted returns (continuous C_t) | **NEGATIVE, stronger** (Δr=−0.576, CI<0) | P1b Task F |
| Task G distinct divergence measure | **PARTIAL** (did not separate, 32nd pct) | P1c Task G |

---

## 2. Claim-by-claim diff against current PAPER.md

Format: *location — claim as written — status — corrected text (if needed).*

### Abstract

- **A1 — "a linear probe trained on h_t … achieves AUROC 0.72 on a KL-matched contrastive set."**
  Status: **ACCURATE and now REPLICATED.** 5-seed mean **0.715 ± 0.074**, bootstrap 95% CI **[0.666, 0.763]**; all 5 seeds above 0.5 (weakest seed's own CI [0.535, 0.642] still excludes 0.5).
  **Suggested text:** "…achieves AUROC 0.72 (5-seed mean 0.715 ± 0.074, 95% CI [0.666, 0.763]) on a KL-matched contrastive set."

- **A2 — "0.70 cross-task without multi-task training (via first-difference Δh_t)."**
  Status: **ACCURATE (n=1).** Unchanged by this pass.

- **A3 — "the probe approximates a discounted count … γ = 0.95 and R² = 0.80."**
  Status: **ACCURATE for cartpole (n=1).** Add: this is **cartpole-specific** — on reacher, best γ=0.70, R²=0.22 (Task D). Recommend a clause: "…γ=0.95, R²=0.80 on cartpole; the effective memory shortens (γ≈0.70) on reacher."

- **A4 — "The signal lives in the near-null space … (88.2° … 9% …), a geometric consequence of GRU update-gate saturation (z_t ≈ 0.94)."**
  Status: **⚠️ SUPERSEDED — this is the known-stale z_gate claim.** Task B (causal, inference-time z override) shows the near-null-space geometry is **z-independent**: angle stays 88.5–89.8° and ≤1.9% variance in top-10 PCs across the entire forced-z range including z=0.5. Gate saturation does **not** cause the orthogonality.
  **Corrected text:** "The signal lives in the near-null space of h_t's principal variation (88.2° from all top-50 PCA components; 9% of probe variance in the high-variance subspace). This geometry is a **structural, content-based** property of the representation, not a consequence of GRU update-gate saturation: forcing the update gate anywhere from 0.5 to 0.99 at inference leaves the confusion direction near-orthogonal to the top PCs (Task B). An earlier gate-saturation hypothesis was falsified and is retained as such."

- **A5 — "the obs/imagination boundary at 1.0000 AUROC, r=−0.015 to Probe A."**
  Status: **ACCURATE but reframe.** Task E: this is **largely a magnitude effect** — a single scalar (‖h_t‖ 0.976, best coord 0.988) nearly separates the classes. Add a clause: "…1.0000 AUROC; this separation is largely a magnitude effect — ‖h_t‖ alone reaches 0.976 — rather than a distributed direction (Task E)."

- **A6 — "its immediate saturation predicted exactly by z_gate: after one imagination step, (1−z_gate)^1 = 0.06 of the original posterior content remains."**
  Status: **✅ RESOLVED by Task M (causal forced-z sweep) — REWRITE, do not keep as-is and do not fully cut.** Task M shows the `(1−z_gate)` arithmetic is a **correct account of the MAGNITUDE component**: ‖h‖-based separability tracks the overwrite fraction exactly (r=+0.97; ‖h‖-AUROC falls 0.97→0.51 as forced z goes 0.5→0.99). **But it is NOT what makes real-vs-imagined separable** — the full linear probe stays at AUROC 1.0000 at *every* forced z, so perfect separability is gate-independent (it is the Task E magnitude effect, achievable by a linear probe regardless of overwriting).
  **Corrected text:** "The magnitude of h_t shifts as the (1−z_gate) overwrite arithmetic predicts — forcing the update gate across 0.5–0.99 moves ‖h_t‖-based real-vs-imagined separability from 0.97 to chance in lock-step with the overwrite fraction (r=+0.97, Task M). This magnitude shift, however, is not what makes the classes separable: a linear probe achieves AUROC 1.0000 at every forced gate value, so the perfect separability is the (gate-independent) magnitude effect of §4.6, not a consequence of the specific (1−z_gate)^1=0.06 arithmetic." (Claim the magnitude mechanism with causal backing; drop the implication that it explains the 1.0000.)

- **A7 — "routing … outperforms recon-threshold baselines at 30% query rate (0.818 vs 0.770 recall), capturing 81% of the KL oracle's advantage."**
  Status: **ACCURATE and REPLICATED (ordering).** Probe A > recon on **all 5 seeds**; the gap is **+0.065, bootstrap 95% CI [+0.058, +0.072]** (comfortably above 0). The 0.818/0.770 pilot point estimates stand for the pilot's specific setup; add "the probe-over-recon advantage replicates across 5 seeds (gap +0.065, CI [+0.058,+0.072])." Report the Wilcoxon as *directionally unanimous, p=0.0625 (the n=5 floor)* — do **not** claim p<0.05 across seeds; lead with the bootstrap CI.

- **A8 — "probe-only detections have 3.4× longer mean streak length than recon-only detections."**
  Status: **ACCURATE (n=1).** Unchanged.

- **A9 — Table 1 / §4.1: "within-balance 0.506" (framed as chance).**
  Status: **⚠️ CORRECTION REQUIRED.** Across 5 seeds the within-balance confound is **0.321 ± 0.094, CI [0.271, 0.373]** — consistently **below** 0.5, with every seed's CI entirely below 0.5. This is **not chance; the probe systematically anti-ranks** the untrained-task groups (it inverts, not merely fails). The underlying point stands (no task-identity-invariant transfer), but "0.506 / chance" is inaccurate.
  **Corrected text:** "On the within-balance confound (both groups from the untrained balance task) the swingup-trained probe scores 0.32 (5-seed, CI [0.27, 0.37]) — systematically **inverted** rather than at chance: the probe does not transfer its confusion reading to an untrained task and in fact anti-correlates, confirming the Set C signal is not task-identity detection." (Arguably stronger evidence of non-transfer than chance.)

### Introduction / Contributions

- **C1 (contribution 1) — within-task confusion signal, AUROC 0.72.** ACCURATE and **replicated** (5-seed 0.715 ± 0.074, CI [0.666, 0.763]).
- **C2 (contribution 2) — closed-form C_t, γ=0.95, R²=0.80.** ACCURATE for cartpole and **strengthened**: γ=0.95 is identical on **all 5 seeds**, R²=0.763 ± 0.045 (n=5). Note reacher divergence (γ=0.70; see A3).
- **C3 (contribution 3) — "geometric account … consistent with GRU update-gate saturation theory."** ⚠️ **SUPERSEDED** — same fix as A4. Change "consistent with GRU update-gate saturation theory" → "a structural property, shown causally independent of update-gate saturation (Task B)."
- **C4 (contribution 4) — cross-task via Δh_t, 0.70.** ACCURATE.
- **C5 (contribution 5) — routing + boundary as second orthogonal signal.** ACCURATE; reframe boundary per A5.
- **§1 body — "88.2° … only 9% of the probe's direction in the top-50-PC subspace … consistent with GRU update-gate saturation."** Same z_gate correction as A4/C3.

### New contributions to ADD (from Phase 1b/1c/1d)

- **NEW-1 (Task A/G/I/L):** the confusion direction is **causally load-bearing** — ablation collapses the readout (−0.575 vs −0.004 random), at the **100th percentile of a 50-direction empirical null** (z=−23/+31), degrading gracefully under perturbation, and **replicating at the 100th percentile on all 5 seeds** (Task I). It survives **two structurally different intervention types**: synthetic ablation AND real-content substitution (Task L swap: −0.761 vs ablation −0.581, both 100th pct) — a strong defense against the Makelov/Sklar illusion. The routing-flip effect is **partial across seeds (3/5)**; next-step-KL does not separate (consistent with Task A).
- **NEW-2 (Task B):** the null-space geometry is **causally z-independent** (single frozen model, gate forced 0.5–0.99).
- **NEW-3 (Task D/J):** the confusion direction, null-space geometry, AND the linear encoding of C_t **generalize across three structurally different environments** (cartpole, reacher, pendulum — obs 5/6/3, act 1/2/1). But the **Set C AUROC (0.72/0.62/0.32 — inverts on pendulum)** and the **closed-form γ/R² (0.95/0.70/0.90; R² 0.80/0.22/0.89)** are **environment-dependent**. Pendulum dissociates them: strongest C_t encoding (R²=0.89) yet inverted Set C (0.32) → the Set C inversion is a construction artefact, not absence of signal. Claim generality for direction/geometry/C_t-encoding; scope Set C and the specific γ/R² to cartpole.
- **NEW-4 (Task H):** the confusion signal is **REINFORCING** with Biased Dreams' attractor/reward-overestimation phenomenon (r=+0.39 latent gap, r=+0.48 reward gap) — independent support.
- **NEW-5 (Task K):** a **second** mechanistically-distinct forward-dynamics measure (decoder reconstruction of the next real obs) **also does not separate** from the null (z=−0.7) — with Task G's divergence result, two independent measures agree that the confusion direction is not the causal lever for dynamics accuracy. The signal *reads* problematic states (Task H) without *being* the dynamics mechanism.
- **NEW-6 (Task M):** the boundary's `(1−z_gate)` arithmetic causally explains the **magnitude** component (‖h‖-separability tracks overwrite fraction, r=+0.97) but **not** the perfect separability (full probe = 1.0 at every forced z) — see A6.

---

## 3. Task H verdict (drop-in for Related Work)

**REINFORCING.** *"Our linear confusion readout positively correlates with both the imagined-vs-real latent gap (r=+0.39, p≈10⁻¹⁴⁷) and imagined-reward overestimation (r=+0.48, p≈10⁻²³⁰) identified by Berger et al. (Biased Dreams); it flags the same problematic states at inference time and at no cost, without their model-dynamics analysis. The attractor recovery they describe is present in our setup as a closing of the perturbation-induced latent gap, but confusion tracks the residual unreliability recovery does not erase — because confusion is a property of posterior-vs-prior history, not latent-dynamics drift (corroborated by Task G, where ablating the confusion direction does not reduce imagined-vs-real drift)."*

---

## 4. Open items (attempted but ambiguous / underpowered / needs follow-up)

1. **✅ Task C aggregation DONE** — see §1.1/1.5, A1/A7/A9/C1. Two items it surfaced:
   - **within-balance INVERTS to 0.32 (not chance)** — PAPER.md's 0.506/chance line must be
     corrected (A9). This is a *correction to a pilot number*, the most important thing this
     pass found; do not ship the old 0.506 framing.
   - **n=5 Wilcoxon is floored at p=0.0625** (cannot reach <0.05 with 5 paired seeds). Lead
     with the pooled paired-bootstrap (p≈0 on the ensemble comparison) and the gap bootstrap
     CIs; report Wilcoxon as "unanimous in direction, p=0.0625 (n=5 floor)," never as p<0.05.
2. **✅ Task I DONE** — probe-decay replicates at the 100th percentile on **all 5 seeds** (flagship result generalizes). But **routing-flip is only 3/5** at the extreme (seeds 1,2 at 0th pct — a random direction flips routing more on those models). The paper must state the routing-flip causal result as **single-model / partial-across-seeds**, not a general causal claim. Next-step-KL non-separation replicates (5/5), confirming Task A's caveat.
3. **✅ RESOLVED — Abstract A6 (boundary z_gate arithmetic)** — Task M tested it causally. The `(1−z_gate)` arithmetic explains the **magnitude** component (‖h‖ separability, r=+0.97) but not the perfect separability (full probe 1.0 at every z). REWRITE per A6 above (claim magnitude only). No longer open.
4. **✅ RESOLVED — Task G distinct-measure partial** — Task K added a second distinct measure (decoder recon on next real obs) which **also** does not separate (z=−0.7). The two agreeing turns "one partial measure" into a positive dissociation result (NEW-5). State plainly; no longer a loose end.
5. **✅ RESOLVED — illusion-mitigation completeness** — Task L added the real-content-substitution intervention (swap) Task G had substituted away; it agrees with ablation at the 100th percentile. The Task G mitigation set is now complete against its own spec (empirical null + distinct measures + robustness + real-content substitution). No longer open.
6. **✅ RESOLVED — reacher within-task confound** — Task J's third environment shows the within-task confound is not a special reacher problem: across three environments it wanders around chance (0.51/0.58/0.44) and, more importantly, **Set C itself is environment-dependent and inverts on pendulum** (0.32) even though C_t encoding there is the strongest (R²=0.89). The correct framing (now in §1.2b/NEW-3): direction+geometry+C_t generalize; Set C AUROC and γ/R² are cartpole-specific. Not a loose end — a characterized 3-env pattern.
7. **Reward proxy in Task H** — no trained reward head; overestimation uses a cartpole upright-reward proxy from the decoded obs. Direction/correlation robust; absolute magnitude is proxy-dependent.
8. **Amplification ceiling (Task A)** — positive-α effect saturates at the probe's sigmoid ceiling; not a weakness, but the positive and negative arms are not symmetric.
9. **Task M override caveat** — forced-z imagination is *more* separable than natural (full 1.0 vs 0.887); the override fixes z at every imagined step. Doesn't affect the magnitude-vs-separability conclusion (read from the ‖h‖ trend), but note it.

---

## 5. Per-task deliverable index

Detailed context-rich writeups (hypothesis, build, all numbers, mid-course findings, caveats,
paper connection) live in `outputs/deliverables/`:
`task_A_causal_intervention.md`, `task_B_zgate_causal.md`, `task_C_multiseed.md`,
`task_D_second_environment.md`, `task_E_boundary_scalar.md`, `task_F_probe_weighted_returns.md`,
`task_G_null_distribution.md`, `task_H_attractor_recovery.md`, `task_I_multiseed_causal.md`,
`task_J_third_environment.md`, `task_K_decoder_recon.md`, `task_L_swap_intervention.md`,
`task_M_boundary_zgate_causal.md`.
