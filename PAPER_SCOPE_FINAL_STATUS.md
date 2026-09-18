# Four-Pillar Paper — Final Closing Status

**Paper scope, locked:** Emergence + External validity + Mechanism + Causal
dissociation, plus Phase 5 (filtering theory) and Task 4 (distribution
shift) as minor supporting results. **Reliability-aware use (Pillar 4, Task
6/7/8) is explicitly out of scope** for this paper — future work, not a
failed thread; the underlying results are sound but that thread is
unfinished and doesn't gate the four pillars that are done.

**Standing rule, unchanged:** every honest negative or mixed result within
the four kept pillars stays exactly as reported — pendulum-outlier pattern,
§6.4 drift-detection miss, Phase 6's retracted external-causal-effect claim,
every MIXED-not-clean verdict. None of it is softened by this closing pass.

---

## Status: all three remaining verification items CLOSED

### Item 1 — Cartpole atom #156's pre-ablation incremental-R² number

**CLOSED.** Independently re-derived from raw JSON
(`outputs/phase3_sae/phase3b_causal_results_atom156.json`,
`outputs/phase3_sae/phase3b_causal_results.json`), not from any summary
table. **Finding: no copy/reuse bug exists in the data.** Cartpole #156 =
0.003814... (+0.0038); reacher #612 = 0.005392... (+0.0054) — genuinely
different values. The apparent match was a **prose transcription error**:
three sentences (two in `phase3_sae_decomposition.md`, one in
`task_5_causal_mechanism_synthesis.md`) had misquoted reacher's own value as
"+0.0038 → +0.0004" instead of its correct "+0.0054 → ~0.0000" — which
already matched Phase 3's own §9.7 table correctly. All three instances
fixed. Cartpole #156's own number stands, independently verified, unchanged.

### Item 2 — SAE-decomposition script reproducibility

**CLOSED.** Checked in full:
- `load_pooled_h()` (the SAE's own 300K-vector training data) reads
  pre-saved `training_states.npz` files from disk with **no RSSM sampling at
  all** — already fully deterministic before this check, unaffected by the
  RNG-bug class fixed elsewhere in the project.
- `collect_eval_trajectories_and_targets()` (feeds §9.3/§9.4's atom
  selection) **did** have the unseeded-RSSM-sampling bug — fixed identically
  to every other instance in the RNG-bug audit (`torch.manual_seed` once at
  the top). `run_addendum_phase3_hardening.py` imports this function
  directly, so the fix propagates automatically.
- Verified byte-reproducible: two calls (same process and separate process
  invocations) produced identical `h`/`e_state`/`ct` arrays
  (`np.array_equal`, exact).
- SAE training itself (`train_topk_sae`, same seed): two full 40-epoch runs
  produced **byte-identical decoder weights** (max abs diff = 0.0) and
  identical loss curves.
- **Atom-index stability under a same-seed rerun — the deeper question —
  confirmed directly**: two independently-trained (same-seed) SAEs produced
  the identical ordered top-10 atom list for cartpole's confusion direction
  (`[185, 1871, 643, 1531, 1805, 1812, 1301, 137, 29, 825]`, exact match).
  This settles the correct scope for the stability question — same-seed
  reruns reproduce atom identity exactly; the project's separate, correct
  statement that atom identity is not expected to align *across different
  seeds* is an unrelated claim about a deliberate design choice (different
  random dictionary initializations), already handled via decoder-column
  cosine matching for cross-seed comparison.

**No re-anchoring to a feature-identity method was needed** — atom indices
#612, #156, #310 (and the retracted #139) are all confirmed stable,
reproducible identities.

### Item 3 — Phase 6 seed extension

**CLOSED.** Extended `run_addendum_phase6_hardening.run_task_hardening`
(already RNG-bug-fixed, independently verified reproducible) — reused
completely unmodified — to the project's standard multi-seed counts: 5
cartpole, 4 reacher/pendulum, reusing existing multiseed base-model
checkpoints. No new training.

**Aggregate result — both headline findings robustly replicate:**

| Task | n seeds | probe_k0 z (mean±std) | probe_k10 z (mean±std) | `E^state` z (mean±std) |
|---|---:|---:|---:|---:|
| cartpole | 5 | +12.87 ± 1.64 | +12.15 ± 1.37 | +0.22 ± 0.37 |
| reacher | 4 | +13.34 ± 1.41 | +5.88 ± 0.66 | +0.32 ± 0.40 |
| pendulum | 4 | +15.56 ± 2.96 | +11.36 ± 0.59 | +0.62 ± 0.80 |

Every one of the 13 individual seed-task pairs' probe z-scores falls in
[+5.2, +19.2] (decisive on every seed); every individual `E^state` z-score
stays below 1.5 (range: −0.46 to +1.48), none approaching significance. The
single-seed caveat on Phase 6's two headline findings is now closed.

---

## What this means for the paper

**Causal dissociation (the pillar these three items feed) is now fully
closed, end to end:**
- **Leg A** (dense direction `v` causal on self-report, null on external
  quality) — established at full confidence from the start, independently
  confirmed twice (Phase 6 direct steering; Task 3's POMDP seed sweep),
  neither dependent on the SAE pipeline. Now further hardened by Item 3's
  multi-seed extension.
- **Leg B** (sparse atoms causal on external quality, mirror-image pattern)
  — established on reacher (#612) from the start; **now established on
  cartpole (#156) at the same confidence level**, following Items 1 and 2's
  closure. Cartpole #156 remains the project's only atom meeting the
  simultaneous probe-and-external causal bar; correlationally KL-redundant
  (unlike reacher's KL-non-redundant #612) — a real, stated scoping
  distinction, not a hedge.

**No further verification work is pending for the four-pillar paper's
claims.** All items from every prior closing pass (RNG-bug audit,
carried-forward cleanup, this final three-item list) are resolved.

---

## Explicitly out of scope, removed from active tracking

- Pillar 4 (confusion-gated observation querying)
- Task 6 (Path B, real actor-critic) — including the actor-collapse
  diagnostic, now moot for this paper. 13 trained policies retained,
  untouched, available if this thread becomes its own future paper.
- Task 7 (learned correction mechanism) — never started, stays that way.
- Task 8 (error-reduction adaptive reliance) — including its pending
  tie-breaking/budget-matching robustness questions and the Task 4a/Task 8
  pendulum reconciliation, neither of which needs further chasing since
  Task 8 doesn't ship in this paper.

None of the above needs any further action for this paper. All code and
results from this thread remain in the repository, untouched, for a future
paper if this direction is picked up again.
