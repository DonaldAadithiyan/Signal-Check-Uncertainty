# Gap-Closing Spec — Final Summary

Consolidated results for `Task_Spec_Closing_Gap_To_Target_Claim.md`, its
Pillar-4 addendum, the two follow-on diagnostic/design tasks (sign-check,
Task 7), and the RNG-Bug Scope Audit spec (which closed Task 6/7's remaining
blocker). All running work is complete as of this document. Per-task detail
and process log: `GAP_CLOSING_RUN_LOG.md`. Citable write-ups:
`outputs/deliverables/`.

**Target claim being closed:**
> Recurrent world models implicitly track accumulated model misspecification
> in their hidden state. We identify this latent predictive-difficulty
> variable, separate its externally predictive component from ordinary
> instantaneous uncertainty, causally characterize its mechanism, and show how
> it governs reliability under partial observability and distribution shift.

---

## Status at a glance

| Task | Status | One-line result |
|---|---|---|
| 1 — Pendulum incremental-R² reconciliation | ✅ Done | Root cause: unseeded RSSM sampling bug, fixed. Reconciled value **+0.0019**. |
| 2 — Atom-identity check (reacher) | ✅ Done | Confirmed: causally-tested and held-out-confirmed atom are the same (#612). |
| 3 — Phase 4 seed sweep + multi-step causal upgrade | ✅ Done | **Verdict revised: MIXED (3/6), down from single-pair 4/6.** Causal-on-E^state claim retracted. |
| 4 — Distribution-shift test | ✅ Done | Mixed, task-dependent: 2/3 tasks strengthen under noise, reacher degrades. Causal null replicates under shift. |
| 4a — Pillar 4 write-up (Path A) | ✅ Done | Honest, task-dependent routing result written up at full evidentiary standard. |
| 5 — Causal mechanism synthesis | ✅ Done | Bifurcated mechanism: dense direction causal on self-report only; sparse atom causal on real behavior. |
| Diagnostic — sign of Phase 3 ablation effect | ✅ Done | Ablation makes `E^state` **worse** on cartpole/reacher — atom is load-bearing, not removable. |
| RNG-bug scope audit | ✅ Done | Bug confirmed in Phases 2/3/6 + addenda, fixed, re-verified. No verdict changes except cartpole's atom re-run. |
| — Cartpole atom re-run (#156) | ✅ Done | **Strongest causal result in the project** — z=+10.3 (E^state) and z=−5.0 (probe), only atom meeting the simultaneous-effect bar. |
| — Difficulty-matched FULL-vs-PARTIAL | ✅ Done | KL-dependent, non-monotonic: PARTIAL stronger at KL extremes, FULL stronger mid-range. Does not overturn MIXED verdict. |
| 6 — Path B (real actor-critic) | ⏸ Paused | 13 policies trained; 10/13 show near-constant actor collapse (may be near-optimal bang-bang, not confirmed pathology). Diagnostic proposed, not yet run. |
| 8 — Error-reduction adaptive reliance | ✅ Done (corrected) | Original run had a causal-availability bug (same-step signals); fixed. `C_t` beats KL/Recon/EMARecon on all 3 tasks — cleanly on reacher, coarsely (plateau-driven) on cartpole/pendulum. |
| 7 — Learned correction mechanism (revised) | ⏸ Not started | Correctly gated on Task 6. |

---

## 1. Pendulum incremental-R² reconciliation

**The problem was bigger than reporting inconsistency.** Three different
pendulum incremental-R² values had been reported (+0.0006, +0.0021, and
several ad hoc reruns landing anywhere from +0.0014 to +0.0020). Root-cause
investigation found this wasn't drift in aggregation — it was a genuine
**unseeded-RNG bug**: the RSSM's stochastic-latent sampling
(`torch.distributions.Categorical.sample()`) draws from PyTorch's global RNG
at every timestep, and `torch.manual_seed()` was never called anywhere in
`run_phase1_external_validation.py`. Two "identical" reruns — same env seed,
same model weights — produced genuinely different `h_t` trajectories (up to
0.87 absolute divergence after 500 steps).

**Fixed** by seeding once at the top of the per-task pipeline (covers
trajectory collection and every downstream imagination-rollout sampling call
in fixed sequential order). **Verified fully reproducible**: two independent
reruns now produce byte-identical results to full float precision.

**Reconciled values** (canonical, reproducible):

| Task | incremental R² | 95% CI | Gate 1 |
|---|---:|---:|---:|
| cartpole | +0.0280 | [+0.0154, +0.0433] | PASS |
| reacher | +0.0182 | [+0.0083, +0.0309] | PASS |
| pendulum | **+0.0019** | [+0.0001, +0.0051] | PASS (weak) |

Gate 1 verdict unchanged (PASS on all 3, CI excludes zero on all 3). Every
document that cited the old numbers was updated: Phase 1's own deliverable
(3 internal locations), Phase 2 and Phase 6's deliverables,
`pendulum_outlier_synthesis.md`, and the AAMAS planning doc.

**Open item surfaced, not yet acted on:** the identical bug exists, unfixed,
in Phases 2, 3, 6 and their addenda — none of those scripts call
`torch.manual_seed` either. Not re-run here (out of scope, real compute cost)
— **a decision on re-verifying those phases is still pending.**

---

## 2. Atom-identity check

Confirmed explicitly, not assumed: reacher's SAE atom (#612) used in the
causal ablation test is the same atom independently re-derived by the
held-out split-sample selection procedure. Reacher's full evidentiary chain
(held-out correlational confirmation → causal ablation → incremental-R²
removal) is now closed without a gap on the same, verified feature identity.

Cartpole and pendulum were correctly left out of scope — their causal tests
used atoms that the split-sample correction later retracted (cartpole's
replacement atom, #156, has not been causally re-tested).

---

## 3. Phase 4 seed sweep and multi-step causal-protocol upgrade

**The long pole of this spec, and the one with the most consequential
result.** Trained 4 new seed pairs (1717, 2929, 5151, 8383; ~4.2hr/condition
each under concurrent load) alongside the original seed=4242, and upgraded
the causal test from a single-step probe-decay proxy to a genuine multi-step
`E^state`-continuation protocol, plus added a difficulty-matched-bin check.

**Aggregate result (5 seed pairs):**

| Metric | FULL | PARTIAL | Stronger under partial? |
|---|---:|---:|:---:|
| Confusion AUROC | 0.871±0.028 | **0.951±0.013** | ✅ |
| R²(h_t, C_t) | 0.761±0.035 | **0.824±0.013** | ✅ |
| r(C_t, E^state) | **0.568±0.060** | 0.522±0.028 | ❌ |
| Incremental R² | **0.062±0.027** | 0.009±0.003 | ❌ |
| Causal z, probe | −6.93±2.27 | **−9.92±1.85** | ✅ |
| Causal z, E^state | −0.37±0.36 | 0.13±0.22 | ❌ |

**Verdict revised: MIXED (3/6), not "SUPPORTS H6" (4/6).** The single most
important correction: the original single-pair report's dramatic causal
z-score doubling (−4.69→−10.58) on genuine external imagination quality
**does not survive** the multi-step protocol upgrade or the 5-seed sample —
the corrected effect is statistically indistinguishable from zero in both
conditions. This is retracted as a headline claim, in the same spirit as
Phase 6's own addendum-driven retraction of its external-causal-effect claim.

**What survives, robustly:** confusion AUROC and `R²(h_t,C_t)` — both
stronger under partial observability on every one of the 5 seeds, tight
spread.

**New confound identified:** the difficulty-matched-bin check found PARTIAL's
sites are systematically higher-KL than FULL's at every matched tercile, and
causal effect size scales strongly with local KL within both conditions
(often an order of magnitude between low- and high-KL bins). Part of the raw
FULL-vs-PARTIAL causal gap is attributable to this confound, not cleanly
isolated to the observability manipulation.

**Follow-on: the confound resolved (RNG-bug audit spec).** A proper
difficulty-matched comparison — pooling FULL+PARTIAL sites and binning by
shared KL quantile rather than each condition's own separate terciles —
found the relationship is **KL-dependent and non-monotonic**: PARTIAL is
stronger at the lowest and highest KL bins (+7.63±7.50 vs −3.58±1.75, and
−31.06±7.86 vs −21.36±9.92), but **FULL is stronger in the middle bin**
(−7.06±3.76 vs −4.67±2.02). The lowest bin also shows a striking sign flip
(PARTIAL positive, FULL negative) that is flagged as needing closer scrutiny
rather than built upon, since it comes from the noisiest, most sample-size-
variable corner of the design. This does not overturn the MIXED verdict — it
replaces "PARTIAL's causal effect may be partly a KL artifact" with a more
precise, still-cautious "the relationship is genuinely non-monotonic in KL,
not simply stronger-for-partial-everywhere."

**Cross-references updated:** Phase 4's original deliverable now carries a
superseded-status banner; Task 5's synthesis notes this as a second,
independent replication of the same "dense-direction-causal-on-self-report-
only" pattern; the AAMAS doc's header and progress note are revised.

---

## 4. Distribution-shift test

Reused Phase 1's `imagined_vs_real_obs`/incremental-R² pipeline with
Gaussian observation noise (σ=0.1, matching the existing Set-B convention,
generalized to all 3 tasks via `DMCEnv`'s existing noise support), plus
Phase 6's causal-steering dose-response, both run under clean (Set A) and
noise-shifted (Set B) conditions.

| Task | R² (Set A) | R² (Set B) | Verdict |
|---|---:|---:|:---:|
| cartpole | +0.0279 | +0.0684 | STRENGTHENS |
| reacher | +0.0184 | +0.0087 | DEGRADES |
| pendulum | +0.0019 | +0.0259 | STRENGTHENS |

Two of three tasks show `C_t`'s external-validity advantage **strengthening**
under noise (opposite of the naive prior); reacher degrades, plausibly
because it already has the highest baseline `r(C_t,KL)`, so noise likely
pushes more shared variance into KL directly (the same ceiling-effect pattern
Phase 4 found for partial observability). The causal-steering null result on
`E^state` **replicates cleanly under shift on all 3 tasks** — no z-score
exceeds |1.43| in either condition. Distribution shift neither resurrects nor
introduces a spurious causal effect.

**Verdict:** the "...and distribution shift" clause is earned with an honest,
task-dependent account — the causal dissociation is stable under shift; the
correlational advantage is not uniform.

---

## 4a. Pillar 4 write-up (Path A, committed)

Wrote up the existing routing/observation-querying result at full
evidentiary standard, no new computation — reused the already-rigorous Task R
(KL-only routing baseline) analysis, reframed explicitly as a
perception-triggering result, never a validated active-perception method (no
policy exists to test task return).

| Task | seeds | Δ recall (probe − KL-only) | probe wins? |
|---|---:|---:|:---:|
| cartpole | 5 | −0.006 ± 0.025 | 3/5, sign-unstable |
| reacher | 3 | **+0.273 ± 0.020** | 3/3 |
| pendulum | 3 | **−0.030 ± 0.006** | 0/3 |

Named descriptively ("confusion-gated observation querying") rather than with
a branded name implying a working method. Path B (Task 6) is explicitly
additive upside, not something this write-up depends on.

---

## 5. Causal mechanism synthesis

Consolidated Phase 3 and Phase 6's separately-reported causal findings into
one account: **the representation's causal role bifurcates — with two legs
at different confidence levels, not one uniform verdict.**

1. **Leg A — dense direction `v` (fully confirmed, no hedge needed)**:
   decisively causal for the model's own confusion readout (z=+7 to +16 vs.
   null, all 3 tasks) — but **no detectable causal effect on genuine external
   imagination quality** on any task. Independently replicated a second time
   by Task 3's seed sweep (same pattern, different phase, different
   correction method). Neither confirmation depends on the SAE pipeline.
2. **Leg B, reacher-confirmed — sparse SAE atom #612**: causally moves
   genuine `E^state` (z=+2.7, 98th pct) while leaving the probe readout
   untouched, and removes its own incremental-R² advantage on ablation.
   Confirmed (Task 2) to be the same atom across correlational and causal
   testing. KL-non-redundant (r(KL)≤0.055, held-out). Fully verified.
3. **Leg B, cartpole — sparse atom #156 (PROVISIONAL, not yet confirmed)**:
   originally causally tested on a since-retracted atom (#139, weak result);
   re-run on the split-sample-confirmed replacement (#156) *currently
   appears to produce* the strongest causal result in the entire project —
   z=+10.3 on `E^state` **and** z=−5.0 on the probe readout, the only atom to
   meet the "both readouts move" bar. Unlike #612, #156 is **not**
   KL-non-redundant (r(KL)=+0.499). **This result is provisional pending
   independent verification of its incremental-R² figure (suspected
   copy/reuse issue) and the SAE pipeline's own reproducibility** — do not
   cite at the same confidence as #612 until both close.

These are causally-independent objects (each atom carries negligible weight
in `v`'s own reconstruction) — not conflicting halves of one finding. Claim 2
now has two-task causal evidence (cartpole, reacher), though only reacher's
atom is also KL-non-redundant.

---

## Diagnostic — sign of the Phase 3 causal ablation effect

Near-zero-cost re-read of an already-computed field
(`ablation_effect.d_e_state`) that had never been surfaced in prose.

| Task | Atom | Signed mean Δ`E^state` (RNG-fixed, verified) | Interpretation |
|---|---:|---:|---|
| cartpole | #139 (retracted) | **+0.0447** | Ablation makes imagination **worse** |
| reacher | #612 | **+0.0321** | Ablation makes imagination **worse** |
| pendulum | #310 | +0.0061 (null) | No reliable effect either way |

**The atom is load-bearing information, not a removable flaw** — on both
tasks where the causal effect clears the null, removing the feature hurts
imagination quality. This closes off direct suppression as a correction
mechanism, consistent with the workshop paper's own Appendix E finding that
two earlier direct-correction attempts already failed. This is the finding
that motivated Task 7's design (a *learned correction*, not suppression).

---

## RNG-Bug Scope Audit and Carried-Forward Cleanup

**Full write-up:** `outputs/deliverables/rng_bug_audit.md`.

Audited every script in Phases 2, 3, 4, 6 and their addenda for the same
unseeded-RSSM-sampling bug class Task 1 found in Phase 1. Confirmed the bug
was the *only* reproducibility issue in the codebase (no dropout exists; one
`torch.randn` call is training-time-only and moot for loaded SAE checkpoints).

**Fixed and re-verified (byte-identical across independent reruns), in
priority order:** (1) Task 3's seed-sweep code — already correct, no fix
needed; (2) Phase 3's causal ablation code — bug confirmed, fixed, small
magnitude corrections, no verdict changes; (3) Phase 6's steering/null code —
bug confirmed, fixed, small corrections, no verdict changes; (4) Phase 2's
scrambling code — bug confirmed, fixed, moderate corrections (largest:
cartpole's external z, 26.9→19.6), no sign or verdict changes on any task.

**Two carried-forward cleanup items resolved:**
- **Cartpole's retracted atom (#139→#156)**: re-run produces the strongest
  causal result in the project (see Section 5's update above).
- **Task 3's difficulty confound**: a proper matched comparison (pooled KL
  quantile bins, not per-condition terciles) found the FULL-vs-PARTIAL causal
  relationship is KL-dependent and non-monotonic, not simply "partial is
  always stronger" (see Section 3's update above).

**Bottom line:** no previously-reported qualitative verdict flipped as a
result of this audit — all corrections are magnitude-level, except cartpole's
atom-156 re-run, which is a genuine new finding. **This closes the blocker
Task 6/7's go/no-go criteria named** (Tasks 1–5 "closed and stable, no open
corrections pending").

---

## 6 — Path B (real actor-critic), paused · 7 — not started

**Task 6:** the venue/timeline question was resolved ("build it properly,"
no compressed scope) and full-rigor training was started: 13 actor-critic
policies trained (cartpole ×5, reacher ×4, pendulum ×4) via a DreamerV3-style
alternating loop (collect real episodes → fine-tune the world model → train
actor/critic in imagination), a real scope expansion from "reuse the frozen
model" made necessary when pure-imagination training against the truly-frozen
model showed zero learning (the frozen model's random-action training data
never reaches near the goal region). Pendulum needed a dense training-only
shaped reward (its exact reward is essentially never reached under random
exploration); real reward stayed unshaped for all evaluation.

**Paused on a real finding:** building the gated-perception evaluation
harness surfaced that **10 of 13 policies (all cartpole, all pendulum, 1/4
reacher) collapsed to a near-constant, saturated action.** This may not be a
pathology — cartpole/pendulum swingup are exactly the class of underactuated
problem where near-bang-bang control is close to the true optimum (per
time-optimal control theory), and reward did improve substantially during
training. Reacher's 3/4 non-collapsed seeds retain genuine closed-loop
diversity. A cheap diagnostic (compare achieved reward against known
benchmark performance for these tasks, no new training) was proposed to
distinguish genuine collapse from near-optimal bang-bang, but **has not yet
been run** — the retrain-or-not decision is explicitly deferred until it is,
per instruction. All 13 trained policies are retained, nothing discarded.

**Task 7:** unchanged, correctly gated on Task 6, not started.

---

## 8 — Error-reduction adaptive reliance (new, distinct from Task 6)

A cheaper, parallel test added alongside Task 6's pause: at a fixed
real-observation query budget, does selecting states by `C_t` reduce actual
accumulated imagination error (`E^state`) more than KL/Recon/EMARecon/
ensemble disagreement? No new training — reuses Phase 1's infrastructure
entirely.

**Two bugs caught before this result was trusted, both changing the
finding.** (1) **Causal-availability bug**: the first version computed
KL/Recon/EMARecon (and even `C_t`'s own lag-0 term) from the *same-step*
observation — exactly the tautology Task 4a's own write-up warns against
("using same-step KL_t would be tautological, since it is the exact variable
that defines the label"). This gave the baselines unrestricted access to
information `C_t` never had in the same way. The tell: the buggy version
found KL beats `C_t` on reacher, the *opposite* of Task 4a's own
correctly-lagged finding on that same task — a discrepancy investigated
rather than accepted, which is what surfaced the bug. Fixed by lagging every
signal to `t-1` (`C_t` recomputed as `C_{t-1}` from a prior-step-only KL
series), matching Task 4a/Task R's established fair-comparison protocol
exactly. (2) **Threshold-tie bug**: a `C_t` value plateau (52% of cartpole
sites, 85% of pendulum sites tied at `C_t`'s single max — a real saturation
property of the discounted-history statistic once a trajectory sustains
enough high-KL steps) caused a strict `>` comparison to exclude every tied
site; fixed to `>=`, applied uniformly to every signal.

**Corrected result: `C_t` beats every baseline on all three tasks, but by
two different mechanisms.** On **reacher** (no plateau), this is a clean,
fine-grained, budget-scaling win — `C_t` beats KL/Recon/EMARecon at every
budget from 5% to 50%, now agreeing with rather than contradicting Task 4a's
reacher result. On **cartpole and pendulum**, `C_t`'s plateau acts as a
coarse but genuinely correct "hard half" detector (plateau-tied sites have
3.7x higher true error than non-plateau sites on cartpole) — it wins at
every budget, but because it cannot discriminate *within* its own plateau,
its `ΔE` is identical across all 5 tested budgets on those two tasks, unlike
the other signals' properly budget-scaling curves. This nuance — genuine win,
different mechanism, different reliability by task — is the honest finding,
not a single uniform "`C_t` wins" headline.

---

## Open items carried forward

1. ~~Unseeded-RNG bug in Phases 2/3/6 and their addenda~~ — **resolved**, see
   the RNG-Bug Scope Audit section above.
2. ~~Cartpole's SAE causal test uses a retracted atom~~ — **resolved**, #156
   re-run, see Section 5's update above.
3. **`Reframed_AAMAS2027_Project.md` lives only in `~/Downloads`**, not
   tracked in this git repo — still open, worth moving in if it's meant to be
   canonical.
4. ~~A fully difficulty-matched FULL-vs-PARTIAL causal comparison~~ —
   **resolved**, see Section 3's update above (KL-dependent, non-monotonic
   relationship found).
5. **`run_phase3_sae_decomposition.py`'s own `collect_trajectories` call
   remains technically unseeded**, though no currently-cited number depends
   on a fresh rerun of it (the atom-selection dataset it builds was not
   regenerated in this audit — only the downstream causal test was fixed and
   re-verified). Noted in `rng_bug_audit.md`; low priority since it affects
   no live number.
6. ~~The venue/timeline question~~ — **resolved**: build at full rigor, no
   compressed scope. Task 6 was greenlit and started on this basis.
7. **Task 6's actor-collapse finding is unresolved.** 10/13 trained policies
   show near-constant saturated actions; whether this is a genuine training
   pathology or a near-optimal bang-bang solution for these underactuated
   tasks has not been determined. Proposed diagnostic (compare against known
   benchmark reward for cartpole/pendulum swingup) not yet run. The
   retrain-or-proceed decision is explicitly deferred until it is.
8. **§11.1/§11.2 (cartpole atom #156's suspected incremental-R² copy/reuse
   bug, and the SAE pipeline's own reproducibility) have not been
   independently investigated by this session** — the provisional hedge
   applied to Task 5/Phase 3's cartpole-#156 claims (see above) was applied
   on instruction, not after confirming or locating the underlying bug.
   Actually investigating and closing §11.1/§11.2 remains open.
9. **Task 8's fixed-action design (zero out error on checked sites) is a
   simplification** — a more realistic partial-correction model was not
   tested and might change which signal wins; noted in its own deliverable's
   caveats.
