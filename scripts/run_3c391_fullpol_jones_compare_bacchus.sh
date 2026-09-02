#!/usr/bin/env bash
# Freeze the diagonal ancestor and run diagonal vs full-Jones global q,u.
# Never rsyncs or writes the live 20260830 VJP tree.
set -euo pipefail

SRC="${SL1MJAX_PHASE6_SRC:-$(cd "$(dirname "$0")/.." && pwd)}"
DEST="${SL1MJAX_PHASE6_EXPLICIT_DEST:-/home/stephen/checkouts/SL1MJax-phase6-explicit-20260831}"
LIVE_DEST="${SL1MJAX_PHASE6_DEST:-/home/stephen/checkouts/SL1MJax-phase6-20260830}"
MAIN="${SL1MJAX_OLD_CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
PY="${SL1MJAX_PYTHON:-$MAIN/.venv/bin/python}"
OUT="${SL1MJAX_PHASE6_EXPLICIT_OUTPUT:-$DEST/outputs/3c391_phase6_explicit}"
GOLDEN="${SL1MJAX_POL_GOLDEN:-$DEST/tests/fixtures/3c391_polarization_golden.npz}"
KBG="${SL1MJAX_KBG_GOLDEN:-$DEST/tests/fixtures/3c391_calibration_golden.npz}"
MS="${SL1MJAX_FULLPOL_MS:-$DEST/data/3c391/fullpol_prep/3c391_gkb_only_4corr.ms}"
GATE="${SL1MJAX_CALIBRATOR_GATE:-$DEST/data/3c391/fullpol_prep/calibrator_gate/report.json}"
ANCESTOR="${SL1MJAX_POL_ANCESTOR:-$DEST/outputs/3c391_fullpol_prep/frozen_diagonal_ancestor}"
DIAG_OUT="${SL1MJAX_JONES_COMPARE_OUT:-$DEST/outputs/3c391_fullpol_prep/jones_compare}"
SKY_PROTOCOL="${SL1MJAX_SKY_PROTOCOL:-$MAIN/outputs/3c391_composite_catalogue_stage3/protocol.json}"

if [[ "$DEST" == "$LIVE_DEST" ]]; then
  echo "full-pol Jones compare dest must not be the live checkout" >&2
  exit 2
fi
if ps -eo args= | grep -F "$LIVE_DEST/scripts/run_3c391_phase6_bacchus.py" | grep -vq grep; then
  echo "live Phase 6 job still holds the GPU; refuse" >&2
  exit 2
fi

export PYTHONPATH="$DEST/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

rsync -a \
  --exclude '.venv/' \
  --exclude '.git/' \
  --exclude '/outputs/' \
  --exclude '/data/' \
  --exclude '__pycache__/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.pytest_cache/' \
  "$SRC/" "$DEST/"

mkdir -p "$(dirname "$ANCESTOR")" "$DIAG_OUT"
echo "=== freeze polarisation-test ancestor $(date -Iseconds) ==="
"$PY" "$DEST/scripts/freeze_3c391_polarisation_ancestor.py" \
  --source "$OUT/baseline/diagonal_copolar" \
  --dest "$ANCESTOR"

echo "=== diagonal vs full Jones $(date -Iseconds) ==="
"$PY" -c "import jax; print('jax devices', jax.devices())"
"$PY" "$DEST/scripts/diagnose_3c391_voltage_beam_polarization.py" \
  --polarization-golden "$GOLDEN" \
  --calibration-golden "$KBG" \
  --measurement-set "$MS" \
  --sky-protocol "$SKY_PROTOCOL" \
  --frozen-diagonal-product "$ANCESTOR" \
  --calibrator-gate-report "$GATE" \
  --output "$DIAG_OUT"
echo "=== done $(date -Iseconds) ==="
