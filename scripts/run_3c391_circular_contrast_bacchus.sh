#!/usr/bin/env bash
# Isolated Bacchus overnight: global circular contrast on native pointings,
# then an independent-hand composite refit on C1.
set -euo pipefail

OLD="${SL1MJAX_OLD_CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
STAGED="${SL1MJAX_POL_CHECKOUT:-/home/stephen/checkouts/SL1MJax-pol-20260827}"
export PYTHONPATH="$STAGED/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

PY="$OLD/.venv/bin/python"
SCRIPT="$STAGED/scripts/diagnose_3c391_circular_contrast.py"
SKY_PROTOCOL="$OLD/outputs/3c391_composite_catalogue_stage3/protocol.json"
SKY_CHECKPOINT="$OLD/outputs/3c391_recovery_policy_fit_zero/sealed_active_only.npz"
NATIVE_ROOT="$OLD/outputs/3c391_native_averaging_ablation"
OUT_ROOT="$STAGED/outputs/3c391_circular_contrast"

# Protocol frozen_directory and similar paths are repo-relative.
cd "$OLD"

mkdir -p "$OUT_ROOT"
echo "=== start $(date -Iseconds) ==="
echo "staged=$STAGED"
echo "cwd=$(pwd)"
echo "python=$PY"
"$PY" -c "import jax; print('jax devices', jax.devices())"

for pointing in C1 C2 C3 C4 C5 C6 C7; do
  fixture="$NATIVE_ROOT/native_${pointing}.zarr"
  if [[ ! -d "$fixture" ]]; then
    echo "skip missing $fixture"
    continue
  fi
  echo "=== global circular contrast $pointing $(date -Iseconds) ==="
  "$PY" "$SCRIPT" \
    --native-fixture "$fixture" \
    --sky-protocol "$SKY_PROTOCOL" \
    --sky-checkpoint "$SKY_CHECKPOINT" \
    --output "$OUT_ROOT/${pointing}_global"
done

echo "=== independent-hand composite refit C1 $(date -Iseconds) ==="
"$PY" "$SCRIPT" \
  --native-fixture "$NATIVE_ROOT/native_C1.zarr" \
  --sky-protocol "$SKY_PROTOCOL" \
  --sky-checkpoint "$SKY_CHECKPOINT" \
  --output "$OUT_ROOT/C1_independent" \
  --fit-independent-hands \
  --steps 250

echo "=== done $(date -Iseconds) ==="
