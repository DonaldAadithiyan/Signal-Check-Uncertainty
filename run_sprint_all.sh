#!/usr/bin/env bash
# Earlier ("sprint") pipeline, superseded by the camera_ready pipeline (see README).
# Needs: the checkpoints listed in src/rerun/pipeline.py TASKS (primary + multiseed),
#        outputs/canonical_null/random_directions_50_dim256.npz (a copy also lives in kaggle/task12/).
# Deps:  torch, dm_control, mujoco, numpy, scipy, scikit-learn, matplotlib.
# Output: outputs/sprint/{gate.json,results.json}
set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=disable
PY=${PY:-python3}
$PY -m pytest -q tests/test_rerun_pipeline.py
$PY run_sprint_gate.py
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('outputs/sprint/gate.json'))['_gate']['proceed_sept29'] else 1)"; then
  echo "Gate failed: stopping."; exit 2
fi
$PY run_sprint_p1p2.py --primary-only          # P1 + P2 + P6, primary models
$PY run_sprint_p3.py --big-nulls               # P3 primary: null50, null500, subspace50
$PY run_sprint_p4.py                           # P4 probe controls
$PY run_sprint_p1p2.py                         # P1 on every multiseed model
${SPRINT_SEEDS_P3:-true} && $PY run_sprint_p3.py --models all   # P3 multiseed (null50)
