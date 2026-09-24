# Task B — Kaggle package

## Why this runs on Kaggle

Measured locally (this project's CPU-only MacBook Air): ~3.4 min per 6,000
env-step condition, i.e. ~55-57 CPU-min per full 100,000-step run. Task B
needs 27 such runs (3 tasks × 3 conditions × 3 seeds) = ~25 hours serial
locally. Kaggle's free CPU sessions (and parallel notebooks) make this
tractable without tying up the local machine for a day-plus.

## Upload steps

1. Zip is already built: `kaggle/taskB_kaggle_inputs.zip` (contains this
   `taskB/` folder — `src/` and `taskB_program.py`, ~36KB).
2. On kaggle.com: New Dataset → upload `taskB_kaggle_inputs.zip` → note the
   resulting dataset slug.
3. New Notebook → `+ Add Input` → attach that dataset.
4. Upload/paste `taskB_kaggle_notebook.ipynb` (in this same folder) as the
   notebook, or recreate its cells directly.
5. Edit `INPUT_ROOT` in the first code cell to match your dataset's actual
   slug (Kaggle shows it once attached, typically
   `/kaggle/input/<your-dataset-name>`).

## Run order (do not skip ahead)

1. **Smoke test cell** (`RUN_SMOKE_TEST = True`) — mechanical check only,
   ~1-2 min. Confirms the code runs on Kaggle's environment (dm_control
   install, etc.) before spending real compute.
2. **Pilot** — cartpole seed 3101, all 3 conditions, full 100,000 steps
   (~3 × 55 min ≈ 2.75h). This is required by the spec (§3.6) before the
   full run. Check:
   - `constraint_resid_final` for `B_vconstrained` is ~1e-7 or smaller
     (machine precision) — confirms the projection hook is actually zeroing
     `v^T h_tilde` after each refit. Already verified locally at small scale
     (6.4e-8) before this was packaged.
   - Loss curves for all 3 conditions are finite, no divergence.
   - `v_history.json` per condition shows the refit cosines (a finding to
     record either way).
3. **Reproducibility cell** — reruns the first 15,000 steps of condition B
   twice, checks `np.array_equal` on `h`/`kl`/`recon`. Must show
   `True`/`True`/`True` before proceeding.
4. **Full run** — only after 2 and 3 both pass. `FULL_TASKS` defaults to all
   3 tasks (27 runs total, ~25h). If Kaggle session/quota limits make that
   infeasible, set `FULL_TASKS = ['cartpole']` for the declared 9-run
   fallback — **decide this before starting the full run**, per the spec;
   do not switch mid-experiment based on partial results.
   - Kaggle sessions have a runtime limit (historically ~9-12h for CPU
     notebooks, check current limits). The runner is resumable: every cell
     writes `cell_result.json` (+ checkpoint, states, histories) the moment
     it finishes, so restarting the notebook after a timeout skips
     already-done cells and only computes what's left. Plan on 3-4 session
     restarts for the full 27-run pass at current limits.

## Bringing results back

Download `/kaggle/working/taskB_results/` (all cells) and
`/kaggle/working/taskB_final_manifest.json`, then place them at
`outputs/taskB_training_necessity/` in the main project repo (matching the
path `run_taskB_training_necessity.py` and
`run_taskB_reemergence_analysis.py` expect). Each cell directory
(`<task>_<condition>_<seed>/`) contains `model.pt`, `states.npz`,
`loss_history.json`, `v_history.json`, `cell_result.json`.
