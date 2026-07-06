# Signal Check — Consolidated Results & PAPER.md Reconciliation

**Single source of truth** bridging "experiments are done" → "paper is submittable."
Someone who reads only this document should be able to correctly update every affected
line of PAPER.md without re-reading DEV_LOG.md.

**Last updated:** 2026-07-06 (Phase 1c). **Status of pending items:** Task C aggregation
and Task I are marked ⏳ and filled in when the 5-seed job completes; everything else is final.

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
| Probe A — Set A (ID) | 0.863 | n=1 · seeds ⏳ | P1 (§4.1) |
| Probe A — Set B (noisy OOD) | 0.846 | n=1 · seeds ⏳ | P1 (§4.1) |
| **Probe A — Set C (KL-matched) [headline]** | **0.723** | n=1; 5-seed ⏳ (seed0 0.761, CI[0.712,0.805]) | P1 (§4.1), P1b Task C |
| Probe A — within-balance confound | 0.506 | n=1; 5-seed ⏳ (must straddle 0.5) | P1 (§4.1), P1b Task C |
| Probe C — Set C (recon→h_t) | 0.721 | n=1 | P1 (§4.1) |
| z_t probe — Set C | ~0.72 | n=1; 5-seed ⏳ | P1, P1b Task C |
| RWM-U Ensemble — Set C | 0.744 | n=1; 5-seed paired test ⏳ | P1 (§4.1), P1b Task C |
| Probe A held-out (ID) | 0.9019 | n=1 | P1 |

### 1.2 Closed-form C_t characterization

| Metric | Value | n / spread | Source |
|---|---|---|---|
| Best γ (cartpole) | 0.95 | n=1; 5-seed ⏳ (seed0 0.95) | P1 (§4.2) |
| R²(probe ~ C_t) (cartpole) | 0.80 | n=1; 5-seed ⏳ (seed0 0.76) | P1 (§4.2) |
| Best γ (reacher) | 0.70 | n=1 | P1c Task D |
| R²(probe ~ C_t) (reacher) | 0.216 | n=1 | P1c Task D |

### 1.3 Null-space geometry

| Metric | Value | n / spread | Source |
|---|---|---|---|
| Angle probe→top PCs (cartpole) | 88.0–88.2° | n=1 | P1 (§4.3) |
| Frac probe variance in top PCs (cartpole) | ~9% (top-50) / 0.5% (top-10) | n=1 | P1, P1b/c |
| Angle probe→top-10 PCs (reacher) | 89.4° | n=1 | P1c Task D |
| Frac in top-10 PC (reacher) | 0.17% | n=1 | P1c Task D |
| **z_gate causal: angle span across forced z∈[0.5,0.99]** | **1.29°** (min 88.5°, ≤1.9% in top-10 at all z) | 7 forced-z values, 1 frozen model | **P1b Task B** |

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
| Probe A recall @ 30% budget | 0.818 | n=1; 5-seed ⏳ (seed0 0.604*) | P1 (§4.5), P1b Task C |
| Recon-oracle recall @ 30% budget | 0.770 | n=1; 5-seed ⏳ (seed0 0.558*) | P1 (§4.5), P1b Task C |
| Probe−recon routing gap | +0.048 | 5-seed bootstrap CI + paired test ⏳ | P1b Task C |

*seed0's absolute recall differs from the pilot because the multiseed routing uses top-25%-KL
events on a different held-out sample; the **probe > recon ordering** is what replicates and is
the claim. Confirm across all 5 seeds in the aggregation.

### 1.6 Causal intervention (Task A) + hardening (Tasks G, I)

| Metric | Value | n / spread / percentile | Source |
|---|---|---|---|
| Ablation Δ probe @ t (confusion dir) | −0.575 | 600 sites, CI[−0.591,−0.560] | **P1b Task A** |
| Ablation Δ probe @ t (random dir) | −0.004 | 600 sites, CI[−0.008,+0.000] | P1b Task A |
| Routing flip rate — confusion vs random | 0.805 vs 0.027 | CI[0.770,0.837] vs [0.015,0.040] | **P1b Task A** |
| γ-decay: observed \|Δ_k\|/\|Δ_0\| vs γ^k | 1.00/0.81/0.60/0.50 vs 1.00/0.95/0.77/0.60 | k=0,1,5,10 | P1b Task A |
| **Confusion dir percentile vs 50-dir null (probe-decay)** | **100th** (z=−22.9) | 50 random dirs, 600 sites | **P1c Task G** |
| **Confusion dir percentile vs 50-dir null (routing)** | **100th** (z=+30.6) | 50 random dirs, 600 sites | **P1c Task G** |
| Distinct measure (latent divergence) vs null | 32nd pct (z=−0.6) — **partial** | 25 random dirs | P1c Task G |
| Robustness: effect retained at 0.25 dir-rotation | 58% | mean of 5 rotations | P1c Task G |
| Probe/C_t direction consistency | cos=0.778 (38.9°) | — | P1b Task A |
| **Causal effect replicated across 5 seeds** | ⏳ | mean±std + per-seed null pct | **P1c Task I** |

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
  Status: **ACCURATE (n=1); strengthen with seeds ⏳.** Add the 5-seed mean±std and the bootstrap 95% CI once Task C aggregation lands. Seed-0 alone gives 0.761 [0.712, 0.805].

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
  Status: **⚠️ RE-EXAMINE.** This is a *different* z_gate claim (about the boundary, not the confusion geometry). It is not directly tested by Task B (which tests the confusion-direction geometry). Keep, but flag in Open Items — given A4's correction, any z_gate-mechanism language deserves scrutiny. The boundary being a magnitude effect (A5) is the more defensible framing.

- **A7 — "routing … outperforms recon-threshold baselines at 30% query rate (0.818 vs 0.770 recall), capturing 81% of the KL oracle's advantage."**
  Status: **ACCURATE (n=1); needs seed CI + paired test ⏳** (Task C provides both). The ordering replicates on seed 0; confirm the gap's bootstrap CI and Wilcoxon across 5 seeds.

- **A8 — "probe-only detections have 3.4× longer mean streak length than recon-only detections."**
  Status: **ACCURATE (n=1).** Unchanged.

### Introduction / Contributions

- **C1 (contribution 1) — within-task confusion signal, AUROC 0.72.** ACCURATE; add seeds ⏳.
- **C2 (contribution 2) — closed-form C_t, γ=0.95, R²=0.80.** ACCURATE for cartpole; note reacher divergence (see A3).
- **C3 (contribution 3) — "geometric account … consistent with GRU update-gate saturation theory."** ⚠️ **SUPERSEDED** — same fix as A4. Change "consistent with GRU update-gate saturation theory" → "a structural property, shown causally independent of update-gate saturation (Task B)."
- **C4 (contribution 4) — cross-task via Δh_t, 0.70.** ACCURATE.
- **C5 (contribution 5) — routing + boundary as second orthogonal signal.** ACCURATE; reframe boundary per A5.
- **§1 body — "88.2° … only 9% of the probe's direction in the top-50-PC subspace … consistent with GRU update-gate saturation."** Same z_gate correction as A4/C3.

### New contributions to ADD (from Phase 1b/1c)

- **NEW-1 (Task A/G/I):** the confusion direction is **causally load-bearing** — ablation collapses the readout (−0.575 vs −0.004 random) and flips 80% of routing decisions; at the **100th percentile of a 50-direction empirical null** (z=−23/+31), degrading gracefully under perturbation (Makelov/Sklar illusion checks passed). [+ 5-seed replication ⏳ Task I]
- **NEW-2 (Task B):** the null-space geometry is **causally z-independent** (single frozen model, gate forced 0.5–0.99).
- **NEW-3 (Task D):** the signal and the null-space geometry **replicate on reacher** (different dynamics, obs 6/act 2); the closed-form C_t is cartpole-specific.
- **NEW-4 (Task H):** the confusion signal is **REINFORCING** with Biased Dreams' attractor/reward-overestimation phenomenon (r=+0.39 latent gap, r=+0.48 reward gap) — independent support.

---

## 3. Task H verdict (drop-in for Related Work)

**REINFORCING.** *"Our linear confusion readout positively correlates with both the imagined-vs-real latent gap (r=+0.39, p≈10⁻¹⁴⁷) and imagined-reward overestimation (r=+0.48, p≈10⁻²³⁰) identified by Berger et al. (Biased Dreams); it flags the same problematic states at inference time and at no cost, without their model-dynamics analysis. The attractor recovery they describe is present in our setup as a closing of the perturbation-induced latent gap, but confusion tracks the residual unreliability recovery does not erase — because confusion is a property of posterior-vs-prior history, not latent-dynamics drift (corroborated by Task G, where ablating the confusion direction does not reduce imagined-vs-real drift)."*

---

## 4. Open items (attempted but ambiguous / underpowered / needs follow-up)

1. **⏳ Task C aggregation** — 5-seed mean±std, bootstrap CIs (Set C, within-balance, routing gap), and paired tests (Probe A vs ensemble; Probe A vs recon oracle). Job running; fill §1.1/1.5 and A1/A7/C1 when done. **Blocker for finalizing the abstract's headline CIs.**
2. **⏳ Task I** — causal effect across 5 seeds; fill §1.6 NEW-1 and confirm the next-step-KL non-separation replicates.
3. **Abstract A6 (boundary z_gate `(1−z_gate)` arithmetic)** — not directly tested; re-examine given the A4 z_gate correction. Lower confidence than the rest of the boundary story; the magnitude-effect framing (A5/Task E) is safer.
4. **Task G distinct-measure partial** — the imagined-vs-real divergence measure did **not** separate from its null (32nd pct). Honest partial, reconciled with Task H, but worth a sentence in the paper so a reviewer doesn't read it as a hidden failure.
5. **reacher within-task confound = 0.578** (vs cartpole 0.506) — the second-env signal is present but **less cleanly decoupled from task identity**. State this in Limitations; do not claim reacher is as clean as cartpole.
6. **Reward proxy in Task H** — no trained reward head; overestimation uses a cartpole upright-reward proxy from the decoded obs. Direction/correlation robust; absolute magnitude is proxy-dependent.
7. **Amplification ceiling (Task A)** — positive-α effect saturates at the probe's sigmoid ceiling; not a weakness, but the positive and negative arms are not symmetric.

---

## 5. Per-task deliverable index

Detailed context-rich writeups (hypothesis, build, all numbers, mid-course findings, caveats,
paper connection) live in `outputs/deliverables/`:
`task_A_causal_intervention.md`, `task_B_zgate_causal.md`, `task_C_multiseed.md` ⏳,
`task_D_second_environment.md`, `task_E_boundary_scalar.md`, `task_F_probe_weighted_returns.md`,
`task_G_null_distribution.md`, `task_H_attractor_recovery.md`, `task_I_multiseed_causal.md` ⏳.
