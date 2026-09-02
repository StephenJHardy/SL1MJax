#!/usr/bin/env bash
# Continue the diagnostic full-Jones image from its 200-step checkpoint.
# Fixed topology. Fold 4 stays sealed. Does not overwrite the ancestor.
set -euo pipefail

SRC="${SL1MJAX_PHASE6_SRC:-$(cd "$(dirname "$0")/.." && pwd)}"
DEST="${SL1MJAX_PHASE6_EXPLICIT_DEST:-/home/stephen/checkouts/SL1MJax-phase6-explicit-20260831}"
LIVE_DEST="${SL1MJAX_PHASE6_DEST:-/home/stephen/checkouts/SL1MJax-phase6-20260830}"
MAIN="${SL1MJAX_OLD_CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
PY="${SL1MJAX_PYTHON:-$MAIN/.venv/bin/python}"
NATIVE="${SL1MJAX_NATIVE_ROOT:-$MAIN/outputs/3c391_native_averaging_ablation}"
GOLDEN="${SL1MJAX_POL_GOLDEN:-$DEST/tests/fixtures/3c391_polarization_golden.npz}"
SOURCE="${SL1MJAX_JONES_CONTINUE_SOURCE:-$DEST/outputs/3c391_fullpol_prep/jones_imaging/baseline/full_jones}"
OUT="${SL1MJAX_JONES_CONTINUE_OUT:-$DEST/outputs/3c391_fullpol_prep/jones_imaging_continue}"
LOG="${SL1MJAX_JONES_CONTINUE_LOG:-$DEST/outputs/3c391_fullpol_prep/jones_imaging_continue.log}"
STEPS="${SL1MJAX_JONES_CONTINUE_STEPS:-400}"

if [[ "$DEST" == "$LIVE_DEST" ]]; then
  echo "continue dest must not be the live checkout" >&2
  exit 2
fi
if ps -eo args= | grep -F "$LIVE_DEST/scripts/run_3c391_phase6_bacchus.py" | grep -vq grep; then
  echo "live Phase 6 job still holds the GPU; refuse" >&2
  exit 2
fi
if pgrep -f diagnose_3c391_voltage_beam_polarization.py >/dev/null; then
  echo "residual diagnostic still holds the GPU; refuse" >&2
  exit 2
fi
if pgrep -f "run_3c391_phase6_bacchus.py" >/dev/null; then
  echo "a Phase 6 imaging job is already running; refuse" >&2
  exit 2
fi

export PYTHONPATH="$DEST/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$LOG")" "$OUT"
echo "=== continue full Jones from checkpoint $(date -Iseconds) ===" | tee -a "$LOG"
"$PY" -c "import jax; print('jax devices', jax.devices())" | tee -a "$LOG"
"$PY" "$DEST/scripts/continue_3c391_diagnostic_full_jones.py" \
  --source "$SOURCE" \
  --output "$OUT" \
  --native-root "$NATIVE" \
  --polarization-golden "$GOLDEN" \
  --steps "$STEPS" \
  --validation-interval 5 \
  --learning-rate 0.02 \
  --patience 200 \
  2>&1 | tee -a "$LOG"
echo "=== continue done $(date -Iseconds) ===" | tee -a "$LOG"
