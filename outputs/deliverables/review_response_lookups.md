# Review response — Part 1 lookups

Answered from the code on branch `claude/dazzling-gates-5wg23z` (commit `b6d604d`).
**What this clone lacks:** `outputs/` is git-ignored. This clone has no result JSONs, no
`outputs/deliverables/*`, no world-model checkpoints and no `training_states.npz`. So every
answer below is either (a) read directly from code, or (b) a number quoted from a committed
summary doc (`RESULTS_CONSOLIDATED.md`, `GAP_CLOSING_FINAL_SUMMARY.md`, `PAPER_SCOPE_FINAL_STATUS.md`,
`README.md`, `DEV_LOG.md`), and says which. Items that need a result JSON are marked **NEEDS LOCAL FILE**
and name the file.

---

## Read first: findings that weaken paper claims

These come out of the lookups themselves. The brief says to stop and report anything like this.

**W1. Online inference feeds the wrong action into h_t (off by one), in every analysis dataset.**
Training uses the previous action with the current observation. `src/model/world_model.py:51,66,105,136`:
`prev_a = actions[:, t-1]` is paired with `obs[t]`, which matches the RSSM docstring
`h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])` (`src/model/rssm.py:12`). But every *online* state-logging loop
samples the action it is **about to** take and feeds it together with the current observation:
- `src/training/trainer.py:64-73`. This produces `training_states.npz`, which is the probe's training data, the C_t/γ fits and the geometry.
- `src/data/collect.py:40-49,108-112`. Produces Sets A/B/C.
- `run_phase1_external_validation.py:190-194`. Produces the held-out trajectories for Phase 1/2/3/6.
- `run_task_g_null.py:86` and `run_task_s_scale.py:62,106` do the same.

So each logged h_t has consumed a_t, the action that will be executed *after* obs_t, and never
a_{t-1}, the action that actually produced obs_t. Every logged KL_t is therefore measured with the prior
lacking the causal action and conditioned on an irrelevant one (actions are i.i.d. uniform). KL_t,
C_t, h_t and the probe are all built from these states. Then imagination and posterior continuation
(`run_phase1_external_validation.py:237`, `run_phase3_causal_features.py:97,117`,
`run_addendum_phase6_hardening.py:76,97`) switch to the training convention (`act[k-1]`). As a result,
a_t is fed into the GRU twice at the handoff step.
*Impact:* unknown in size, but it touches the definition of the paper's central quantity. It can be fixed
by inference only (no retraining): re-log the states with `prev_a` and re-run Phase 1 and the emergence
fit. **Recommend doing this before Part 2**, because Part 2 would otherwise inherit it.

**W2. Phase 1's C_t threshold looks at the future.** `run_phase1_external_validation.py:362` computes the
training-split `kl_median`, but line 384 never passes it: `compute_ct(trj['kl'], zeros, gamma=...)`.
`compute_ct` then defaults to `np.median(kl)` of *that single trajectory*
(`src/probe/intervention.py:165-166`). That median covers the whole episode, including steps t+1…T,
which are the same steps E^state is measured on. The same pattern appears in
`run_phase3_sae_decomposition.py:185` and `run_phase3_causal_features.py:177`. The emergence analyses
(`run_confusion_integral.py:86`, `run_second_env.py:244`) use the global training median, so they are not affected.
*Impact:* the Table 3 incremental R² values (+0.0280 / +0.0182 / +0.0019) use a non-causal regressor.
The leak is probably small (one scalar per episode), but it has to be fixed, and the Part 2 variants should all use the calibration-split median.

**W3. The "88°" angle is barely different from a random direction.** It is the *mean of per-PC angles*
(see item 8). I simulated random unit vectors in 256-d against a random orthonormal basis (numpy, seed 0, 2,000 draws).
They give a mean per-PC angle of **87.1°** for both the top-10 and top-50 PCs (5th percentile 85.9°). The expected
squared-projection fraction is 3.9% for the top-10 and 19.6% for the top-50.
So "88.2° from all top-50 PCs" is about 1° from chance. The meaningful statistic is the projection fraction:
cartpole 9% in the top-50 against 19.6% for a random vector, which is about 2× below chance, not "near-null-space".
Reacher (0.17–0.2% in the top-10) and pendulum (1.1–1.5% in the top-10) are well below the 3.9% chance level.
As the subspace angle arccos(√frac), the cartpole top-50 value is **72.5°**, not 88°.
There is also a conflict in the cartpole top-10 fraction: `RESULTS_CONSOLIDATED.md:87` says 0.090, while
`RESULTS_CONSOLIDATED.md:110` says 0.5%. **NEEDS LOCAL FILE** to resolve.

**W4. Table 5's "incremental R² carried" drop is guaranteed by construction** (item 6). The "after removal"
activation comes from re-encoding h_t − (h_t·d)d *at the same step*, not from the continued dynamics.
Removing the atom's decoder column pushes that atom's own activation toward zero (the script itself prints
"near-zero by construction", `run_phase3_causal_features.py:287`). The collapse therefore tests nothing causal.

**W5. γ is chosen on the same data the headline R² is reported on, and that split is at the step level
inside trajectories** (items 3 and 4). The reported "R²(probe ~ C_t)" is the maximum over 5 γ values on the evaluation set itself.

**W6. RNG audit gaps** (item 11). The z≈−22 ablation null (Task G) and the 512-dim check (Task S) never
seed torch around the RSSM sampling.

---

## 1. `E^state` definition

- **Norm:** a per-step **L2 (Euclidean) norm** in decoded-observation space, **averaged over k = 1..K**.
  `run_phase1_external_validation.py:242`: `np.linalg.norm(dec_imag - obs_real)`.
  `:250-251`: `e_state = mean(state_dist[:K])`. It is not squared and not per-dimension.
  (The docstring at `:19` states the same formula.)
- **Actions:** yes, the real logged action sequence of the continuation. `:237`: `a = traj['act'][kk-1]`
  for step kk = t+k. The imagination uses prior-only `imagine_step` (`:238`), starting from h_t and a
  z sampled from the stored posterior logits at t (`:228-230`).
- **Caveat (W1):** h_t had already consumed act[t], so the first imagined step feeds act[t] a second time.

## 2. Dense-steering readout at k=10

**Posterior-updated.** The k=1..10 steps use `observe_step` with the real observations and real actions
(`run_addendum_phase6_hardening.py:73-78`, `continue_multi_step`), the same as Task G's `continue_probe`
(`run_task_g_null.py:116`) and Phase 3 (`run_phase3_causal_features.py:96-102`). The steered state is only
written in at step t. Prior-only `imagine_step` is used only for the E^state readout
(`run_addendum_phase6_hardening.py:95-101`).
Note: the original `run_phase6_causal_steering.py` reads the probe at k=0 only (`:92`). The k∈{0,1,5,10}
readouts come from the addendum script (`LOOKAHEAD`, `:40`). Variant G (prior-only at k=10) has therefore not been run.

## 3. How γ was selected

There is no ridge regression of C_t onto h_t in the selection path. For each γ in {0.70, 0.80, 0.90, 0.95, 0.99}:
1. fit 1-D OLS `probe_score ~ C_t(γ)`;
2. take its R² on the **test split** (`te_idx`, the 40% of `train_test_split(..., test_size=0.40, stratify=y, random_state=0)`);
3. choose the γ with the largest R², **and report that same R²**.

- cartpole: `run_confusion_integral.py:89-130` (γ=0.95, R²=0.798)
- reacher: `run_second_env.py:239-246` (γ=0.70)
- pendulum and the multi-seed runs: `run_multiseed_env.py:120-126`

**Same data? Yes.** The selection data and the reported-R² data are identical (te_idx). The OLS fit is also
trained and scored on te_idx (in-sample). The ridge `h_t → C_t` sweep in `run_direct_ct_probe.py:105-119`
(fit on train, scored on test) is a separate secondary analysis. It also picks its best γ on the test split.
The γ values are then **hardcoded** into every later script (`run_phase1_external_validation.py:94,101,108`
and the Phase 3/6 files).

## 4. Probe training and evaluation data

- **Data:** `training_states.npz`, which is the 100K states **logged online during world-model training**
  (`src/training/trainer.py:62-86`), so they come from a model whose weights are still changing. Early
  states come from a nearly untrained model.
- **Split:** a stratified 60/40 split at the step level, `random_state=0`
  (`src/probe/linear_probe.py:43-47`, and repeated inline in each script, e.g.
  `run_phase1_external_validation.py:375-377`). It is **not** grouped by trajectory, so train and test steps
  interleave within the same episodes, often as adjacent timesteps.
- **Is the reported AUROC on training data?** No. The ID AUROC uses X_te (`linear_probe.py:69`). Set A/B/C
  AUROCs use freshly collected episodes (`linear_probe.py:70-72`). Phase 1 uses fresh held-out trajectories (seed 2024+).
- **Is R²(probe ~ C_t) on training data?** It uses probe scores on te_idx, so the probe itself is held out.
  But the probe's neighbours at t±1 are in its training set (step-level split), and γ is chosen on te_idx (W5).
- **The probe's label** is `KL_t > median` of the whole training log (`binarise_by_median`, `linear_probe.py:17-18`),
  which is the i=0 term of C_t. This is the circularity that the new `C_t^past` variant addresses.

## 5. Atom ablation: once, or at every step?

**Once, at the intervention step t.** `run_phase3_causal_features.py:131-133`: `h_abl = h_t - (h_t·d)d`.
The continuation then runs freely (posterior `observe_step`, `:96-102`) with no re-ablation. E^state is
imagined from the ablated h_t (`:110-120`). The dense-direction ablation in Task G works the same way
(`run_task_g_null.py:99-121`).
Minor point: the "baseline" continuation re-samples z_t from `post_net` (`:91-92`) rather than using the
stored z_t. Baseline and ablated runs draw different categorical samples, which adds noise, not bias.

## 6. "Incremental R² carried" per atom (Table 5, last row)

`run_phase3_causal_features.py:258-284`, on the ≤800 intervention sites of the causal run (seed 2524):
- the regressors are the atom's SAE activation at t, plus KL_t, Recon_t and EMARecon_t, all z-scored;
- the target is the **unablated** E^state_{K=10};
- the value is R²(full) − R²(KL, Recon, EMA), fit and scored **in-sample** on the same sites (no held-out split, no CI);
- "before" uses `sae.encode(h_t)[atom]`;
- the "removal" operation is: re-encode `h_t − (h_t·d)d`, where d is the atom's unit decoder column, **at the same step t**, and use that activation in place of the original (`:270-272`).

It does not pass through the dynamics, and the drop is close to guaranteed by construction (see W4).
The printed "before" values are +0.0038 for cartpole #156 and +0.0054 for reacher #612 (`PAPER_SCOPE_FINAL_STATUS.md:24-25`).
There is also a separate §9.4 discovery-time number from `run_phase3_sae_decomposition.py:244-256`. It follows the same
in-sample protocol on the atom-selection sites, so it is not the same quantity as the §9.7 value.

## 7. Ensemble disagreement

- **Checkpoints:** cartpole only: `outputs/checkpoints/ensemble_seed{0,1,2}.pt` (`run_phase1_external_validation.py:93`,
  and `train_ensemble_expand.py:10` says "Existing: ensemble_seed{0,1,2}.pt"). Reacher and pendulum have none
  (`:100,107`, `ensemble=None`, "reported N/A").
- **In Table 3?** The code adds `ens_dis` as a regressor whenever the files load (`:367-371,473,481`).
  So cartpole's +0.0280 *should* include ensemble disagreement, and reacher (+0.0182) and pendulum (+0.0019)
  do not. **NEEDS LOCAL FILE** to confirm: `outputs/phase1_external_validation/phase1_results.json` → `cartpole.have_ensemble`.
- **Caveat:** the ensemble signal is a *single-step decode from a zero state* (`:310-328`), not a trajectory-conditioned
  disagreement. It is a weak baseline.
- The CIs come from `run_addendum_phase1_hardening.py:151-190`: a trajectory-stratified bootstrap of the in-sample incremental R².

## 8. Geometry

- **Not** `arccos(‖proj_U v‖/‖v‖)`. It is the **mean over the k PCs of arccos(|w·pc_k|)**, in StandardScaler-transformed
  h space, with w = the logistic coefficient (standardised space) normalised to unit length
  (`run_second_env.py:166-175`, also used by `run_multiseed_env.py`; `run_boundary_geometry.py:70-73,136-152` for the top-50 cartpole number).
  The "fraction" reported next to it is Σ_k (pc_k·w)², i.e. ‖proj_U w‖² (`run_second_env.py:174`).
  How the two definitions compare, and the random baseline, are in **W3**.
- **Top-10 atom share of v:** defined at `run_phase3_sae_decomposition.py:133-155`. v is ridge-reconstructed from all
  2048 unit decoder atoms (λ=1e-2). Each atom's share is |c_i| / Σ|c|, an L1 coefficient share and *not* a variance share.
  The top-10 share is the sum of `top10_shares`. **NEEDS LOCAL FILE** for the exact value:
  `outputs/phase3_sae/phase3_results.json` → `decomposition[<task>][0].top10_shares`. The committed docs say only
  "diffuse" (the top-1 share is under 0.5 and the top-4 share under 0.8).

## 9. Cartpole atom #156: held-out r(activation, E^state)

**NEEDS LOCAL FILE**. The value isn't in any committed doc. The selection and held-out confirmation came from the
split-sample re-run (Task 5/Item 2; `PAPER_SCOPE_FINAL_STATUS.md:53-63`). The file is probably in `outputs/phase3_sae/`
(`phase3b_causal_results_atom156.json` or the split-sample JSON). The causal z-scores are committed:
z=+10.3 on E^state and z=−5.0 on the probe (`GAP_CLOSING_FINAL_SUMMARY.md:31`).

## 10. History vs current KL (Fig. 1b)

These are OLS R² values of `probe ~ C_t` and `probe ~ KL_t` on te_idx, with γ = best (see item 3).

| Task | seed | R²(probe, C_t) | R²(probe, KL_t) | Source |
|---|---|---|---|---|
| cartpole | pilot | 0.798 | 0.519 | `DEV_LOG.md:1046,1064`; `PAPER.md` Table 2 |
| cartpole | 5 seeds | 0.763 ± 0.045 [0.703, 0.828] | **not in committed docs** | `RESULTS_CONSOLIDATED.md:59` |
| reacher | pilot | 0.216 | **NEEDS LOCAL FILE** (`r2_kl_baseline`) | `RESULTS_CONSOLIDATED.md:70` |
| reacher | 4 seeds | 0.261 ± 0.035 | **NEEDS LOCAL FILE** | `RESULTS_CONSOLIDATED.md:85` |
| pendulum | pilot | 0.886 | **NEEDS LOCAL FILE** | `RESULTS_CONSOLIDATED.md:72` |
| pendulum | 4 seeds | 0.855 ± 0.038 | **NEEDS LOCAL FILE** | `RESULTS_CONSOLIDATED.md:85` |

`run_second_env.py:241` writes `r2_kl_baseline` into its results JSON (`outputs/second_env/…`, `outputs/third_env/…`).
`run_multiseed_env.py:120-131` does **not** compute R²(probe, KL_t) at all, so the per-seed KL-only values for reacher and
pendulum do not exist and would need a (cheap, CPU, saved-state-only) rerun. Note also that "C_t" here includes the i=0 term,
which *is* the probe label. This is exactly what `C_t^past` addresses.

## 11. Reproducibility (RNG-fix) coverage of the Section 4 emergence analyses

| Analysis | Script | torch seeded? | Uses RSSM sampling? | Status |
|---|---|---|---|---|
| Closed-form fit (γ sweep, R²) | `run_confusion_integral.py`, `run_second_env.py`, `run_multiseed_env.py` | fit: n/a | no, reads saved `training_states.npz` | deterministic given the saved states (logistic `random_state=0`) |
| Geometry (angle, fraction) | `run_second_env.py:166`, `run_boundary_geometry.py` | n/a | no | deterministic given the saved states (PCA `random_state=0`) |
| KL-matched contrast (Set C) | `run_experiment.py`, `run_multiseed.py`, `src/data/collect.py` | **no** (only `np.random.seed(42)`) | **yes** (`collect.py:49,112`) | **not audited**; Set C AUROC can drift between reruns |
| z≈−22 ablation null (Task G) | `run_task_g_null.py` | **no** | **yes** (`:86,116`) | **not audited** |

The saved `training_states.npz` files are themselves products of unseeded online sampling during training, but they are fixed artifacts now.
**Can it be rerun under an hour?** Probably yes for Task G and Set C on CPU. Task G is 50 directions × 600 sites × 10-step
continuation, the same order as the Phase 3 null runs. The fix is the same one-line `torch.manual_seed(...)` at the top of the
pipeline. I can't time it from this container (no checkpoints, no dm_control installed).

## 12. 512-dim scale check (Task S)

- The numbers are in committed docs: angle 89.4°, top-10 fraction 0.13%, ablation Δprobe@t −0.287 against random −0.050
  (`RESULTS_CONSOLIDATED.md:118-119`).
- On disk: `outputs/scale/cartpole_deter512.pt`, `cartpole_deter512_states.npz` and `task_s_results.json`
  (`run_task_s_scale.py:37-39,220`). None are in this clone. **NEEDS LOCAL FILE** to confirm they still exist.
- **Reproducible?** Only partly. `torch.manual_seed` is called only inside `train()` (`:44`). When the checkpoint is reloaded,
  trajectory collection (`:106`) and continuation (`:128`) are unseeded, so the −0.287 will drift slightly on a rerun.
- The "random −0.050" comes from a **single** random direction, not a null distribution (`:183-194`). This should be
  upgraded to the 50-direction null in Part 2C.
- The 89.4° is the same mean-per-PC-angle statistic, so W3 applies.

## 13. Data availability for Part 2

- **Per-step or per-site arrays were never saved** for Phase 1. `run_addendum_phase1_hardening.py:13-19`: "…did not persist
  per-site arrays or trajectory IDs." The addendum *rebuilds* them deterministically (`rebuild_site_dataset`, `:45-118`,
  seeded) but does not save them either. Phase 3 and Phase 6 rebuild their own sites in the same way.
- What does exist on the local machine: `training_states.npz` per task (h, z, kl, recon; plus traj_id for reacher and
  pendulum, `run_multiseed_env.py:93`). These are the training-time logs, not the Phase 1 held-out trajectories, and they have
  no E^state.
- **Missing everywhere:** saved (h_t, KL_t, Rec_t, E^state_K=10, traj_id) for the held-out Phase 1 sites, for every task.
- **Regeneration:** inference only. Per task this is 100 random-policy episodes plus about 6,000 sites × a 20-step
  imagination, which is what `rebuild_site_dataset` already does. I estimate roughly 10–30 CPU-minutes per task, but that is
  unmeasured. The fix is to add an `np.savez` of the per-step arrays and traj_ids, and fix W1 and W2 in the same pass.
- **In this cloud container specifically, nothing in Part 2 can run yet.** There are no checkpoints and no
  `training_states.npz` (both git-ignored, and too large for plain git), and dm_control, numpy and torch are not installed.
  These files are needed here:
  - `outputs/checkpoints/world_model.pt`, `ensemble_seed{0,1,2}.pt`
  - `outputs/second_env/reacher_easy_world_model.pt`, `outputs/second_env/reacher_easy_training_states.npz`
  - `outputs/third_env/pendulum_swingup_world_model.pt`, `outputs/third_env/pendulum_swingup_training_states.npz`
  - `outputs/data/training_states.npz`
  - `outputs/phase3_sae/sae_seed0.pt`, `outputs/phase3_sae/phase3_results.json`
  - (for 2C) `outputs/canonical_null/random_directions_50_dim256.npz`
