## 3. Paper claims: status under the corrected pipeline

This section is hand-written against `paper/icaart_main.tex` and is fixed *before* any real sprint numbers exist. Legend:
- **RERUN → key**: the number is replaced by the corrected-pipeline value at that `results.json` key (shown in section 2a).
- **REMOVE**: there is no corrected rerun in this sprint, so the claim and its number come out.
- **UNAFFECTED**: no world-model state logging is involved.

Old and new numbers are never shown side by side as if they were comparable.

### Abstract / Introduction
| Claim | Status |
|---|---|
| "low-variance direction … nearly orthogonal to its dominant principal components" | RERUN → `p1.*.geometry`. The wording must follow the projection fraction vs. the random baseline (W3). "Nearly orthogonal" survives only if w's fraction is below the random 5th percentile. |
| "predicts … beyond KL, Rec and smoothed Rec, bootstrap-significant gains on all three tasks" | RERUN → `p2.*.table3`. The control set now also includes EMAKL. The claim holds only for tasks whose CI excludes 0. |
| "becomes internally stronger when the model must filter unobserved state" | **REMOVE.** The POMDP scripts (`run_phase4_pomdp_analysis.py:81-83`, `run_phase4_seed_sweep_analysis.py:101-103`) log states with the W1 pairing. They are not in this sprint. |
| "distributed across many dictionary atoms, yet contains identifiable sparse components" | P5 only. Without a P5 rerun: **REMOVE** both. The "distributed" claim may stay only if the SAE decomposition itself is rerun on corrected states. |
| "steering … changes the internal readout on every seed while leaving external quality unchanged" | RERUN → `p3.*` (Table 5). "On every seed" needs `--models all`; otherwise it becomes "on each task's primary model". |
| "ablating specific sparse components changes external predictive quality" | P5 only, otherwise **REMOVE**. |
| Contribution 3 (mechanistic structure) and the sparse half of contribution 4 | P5 only, otherwise **REMOVE**. |

### §3 Setup
| Claim | Status |
|---|---|
| Eq. 1 `h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})` | Now **true of every analysis**, because the pipeline enforces it (W1 test). The old analyses did not follow it. |
| "five models … cartpole, four each reacher and pendulum" | Unchanged. Say which models enter each analysis. |
| γ "best fit by h_t (0.95/0.70/0.90; memories 13/3/10)" | RERUN → `p1.*.variants.C.gamma`. The rule is now: max R²(readout, C_t) over {0.5, 0.7, 0.8, 0.9, 0.95} **on calibration episodes**. Recompute the effective memories. The median is now the calibration median. |
| "100 held-out real trajectories per task (6,000 sites per task)" | RERUN → `p2.*.n_sites`: all sites t ∈ [12, 478] of 100 evaluation episodes, about 46,700 per task, with no subsampling. |
| Statistical protocol: "Every stochastic step … is seeded, and these pipelines reproduce their outputs exactly" | Rewrite for the new pipeline: explicit generators, episode-level splits, calibration/evaluation separation. That sentence is true of `src/rerun` only. |
| "Candidate SAE atoms are selected and evaluated on disjoint halves" | P5 only. |

### §4 Emergence
| Claim | Status |
|---|---|
| AUROC 0.872 (clean) / 0.807 (σ=0.1 noise) | RERUN → `p1.*.probe_eval_auroc` and `p1.*.table2.auroc_noisy`. |
| KL-matched contrast: 22.9 vs 23.9 nats, 0.052 vs 0.471 Rec, AUROC 0.714 / 0.712 | RERUN → `p1.*.table2.kl_matched_pooled` and `kl_matched_clean` (the group means are there too). |
| R² ≈ 0.80 (C_t) vs 0.52 (KL), cartpole | RERUN → `p1.cartpole.primary.variants.C.r2`, `r2_readout_kl`. **Add the C_t^past number and the partial R² beyond the current flag.** These are the review's main request. |
| Scramble z +18.7 / +39.0 / +51.4 (Fig. 1d, "how well h_t explains C_t") | RERUN → `p1.*.ridge_scramble.C.z`. Now the current flag is held fixed. Also report `ridge_scramble.C_past`. |
| "ablating it … 100th percentile … z ≈ −22" | RERUN → `p3.*.nulls.null50.ablate_r0` (ablation z vs the 50-direction null, now on the corrected pipeline). The old Task G value was never seeded (W6). "Substituting another held-out state's projection" was not rerun: **REMOVE** that clause. |
| Geometry "about 88° … cos 88° ≈ 3.5% of its norm" | **Wrong as written.** 88° was the mean of per-PC angles, and a random 256-d vector scores 87.1° on that measure. Replace with `p1.*.geometry.standardized_top50.{frac, random_mean, random_p5, percentile_vs_random}` and the subspace angle arccos(√frac). |
| Table 2 | RERUN → section 2a Table 2 (all rows). The angle row is replaced by the projection fraction. |
| "KL-matched direction follows the sign of within-bin r(Rec, C_t)" | RERUN → check the sign agreement in Table 2 before keeping the sentence. |

### §5 External
| Claim | Status |
|---|---|
| "C_t positively correlated with E^state at every K on every task; r ≈ 0.6 on cartpole, decays on reacher/pendulum" | RERUN → `p2.*.horizons`. |
| Eq. 4 controls | Now KL + Rec + EMARec + **EMAKL**. Change Eq. 4 and the Table 3 caption. The paper-controls version is also reported for continuity. |
| Table 3 (+0.0280 / +0.0182 / +0.0019) | RERUN → `p2.*.table3.{C, C_past}.delta_r2`. |
| "VIF / partial-correlation analysis confirms the pendulum effect" | **REMOVE.** It was not rerun, and the addendum used the old pipeline. |
| Head-to-head +0.177 / +0.049 / +0.013 | RERUN → `p2.*.head_to_head.C_vs_emarec.ct_beyond_rival`. Add the C_t vs EMAKL head-to-head. |
| Scramble z +19.6 / +21.5 (Fig. 2c) and pendulum −11.7 (§8.5) | RERUN → `p2.*.scrambling_external.{C, C_past}.z`. The paper's older test compared correlations. The rerun uses ΔR² beyond the full controls with the current flag fixed, so reword it as that. |

### §6 Mechanism / §7.3 sparse components / Tables 4 and 6 (atom rows)
P5 only. If P5 doesn't finish, **REMOVE**: Fig. 3, Table 4, the atom rows of Table 6, §6.2, §7.3, and the atom sentences in the abstract, contributions and conclusion. The "Incremental R² carried" row is removed in any case. Its "removal" re-encoded h_t − (h_t·d)d at the same step, which zeroes the atom by construction. If P5 runs, replace it with a non-tautological test or drop it.

### §7 Causal
| Claim | Status |
|---|---|
| Steering protocol text: "σ_v … on a held-out pool" | Now σ_v = std(h·v) on calibration episodes; the same absolute step is used for every null direction. |
| Table 5 values, "every readout z between +5.2 and +19.2", "no E^state z exceeds 1.48" | RERUN → `p3.*` via section 2a Table 5 (min/max printed). The multiseed version needs `run_sprint_p3.py --models all`. |
| §7.2 "Independent replication … partial observability … noise shift (max \|z\| = 1.43)" | **REMOVE.** The POMDP and `run_task4_distribution_shift.py:116-118` both use the W1 pairing. Replacement robustness for the dense null: the 500-direction and subspace-matched nulls (`p3.*.nulls.null500`, `subspace50`). |
| Table 6 dense row | RERUN from Table 5. |

### §8 Interpretation
| Claim | Status |
|---|---|
| Berger-style drift ("C_t is lower in reward-inflated drift states on cartpole and reacher") | **REMOVE** (old Phase 1 §6.4, not rerun). |
| "C_t includes the current step's flag … evidence beyond the current step comes from §5" | Rewrite. C_t^past now answers this directly (Table 2, Table 3 C_t^past columns). |
| Kalman-filter simulation (+0.0004 / +0.0046 / +0.0825) | UNAFFECTED (no world model). |
| §8.4 partial observability (0.871→0.951, 0.761→0.824, 0.062→0.009) | **REMOVE** (W1 path). |
| §8.5 cross-task ordering and pendulum z = −11.7 | RERUN → `p2.*.table3`, `p2.*.scrambling_external`. The P6 autocorrelation (`p6.*`) tests the filtering prediction, which was stated in advance. |

### Figures
- **fig1_emergence**: the sprint figure currently has 2 panels (R² KL / C_t / C_t^past with CIs; the γ grid). The paper caption describes 4 panels (scatter, R², angle, scramble). Either update the caption or extend the figure. Panel (c) must become the projection fraction vs. random, not an angle.
- **fig2_external**: the sprint figure has (a) ΔR² with CIs and (b) horizon correlations. The caption's panel (c) (scramble) is in the tables, not the figure.
- **fig3_mechanism**: P5 only.
- **fig4_dissociation**: dense only unless P5 runs. The caption's (a) summary panel and its atom sentences must change.
- **fig5_robustness**: new. It needs a caption and a reference in the text.
- **LaTeX bug:** `\widefig{H}{fig4_dissociation.pdf}{0.72}{…}{fig:dissociation}` passes 5 arguments to a 4-argument macro, so the file name resolves to "H". Delete the leading `{H}`.

### Answers to the `[VERIFY]` items
1. **E^state norm / actions (§3.3).** Per step it is the L2 (Euclidean) norm in decoded-observation space, averaged over k = 1..K. The imagined rollout uses the real logged actions of the continuation. In the corrected pipeline h_t has not yet consumed a_t, so step k feeds a_{t+k-1} and a_t is used exactly once (tested: `test_imagination_handoff_uses_a_t_once`). In the old pipeline a_t was fed twice (W1).
2. **Geometry angle (§4.4).** It is **not** arccos(‖proj_U v‖/‖v‖). The 88° was the mean over PCs of arccos(|w·pc_k|) (`run_second_env.py:166-175`). "cos 88° ≈ 3.5% of its norm" is therefore incorrect, and the sentence must be replaced with the projection fraction (see §4 above).
3. **Exact top-10 share (Fig. 3).** This is only available from a P5 rerun (the old value is in `outputs/phase3_sae/phase3_results.json`, which is an old-pipeline number and not to be used).
4. **Ensemble disagreement in Table 3.** In the old analysis the ensemble term was included for cartpole only, whenever `ensemble_seed{0,1,2}.pt` loaded (`run_phase1_external_validation.py:367-371,473`). Reacher and pendulum had none. The corrected Table 3 **drops the ensemble term on all tasks** (handoff P2), so the caption should say "no ensemble term".
5. **#156 r(activation, E^state).** P5 only. Otherwise the row is removed.
6. **Removal operation behind "Incremental R² carried".** It re-encoded h_t − (h_t·d)d through the SAE at the same step t and recomputed an in-sample incremental R² (`run_phase3_causal_features.py:258-284`). This zeroes the atom's activation by construction, so the row is tautological. **Remove it.**
7. **k=10 posterior or prior (§7.1).** Posterior. The 10 steps use observe-step updates with the real observations (old: `run_addendum_phase6_hardening.py:73-78`; corrected: `run_sprint_p3.py`, posterior continuation). E^state uses prior-only imagination.
