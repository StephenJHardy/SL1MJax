#!/usr/bin/env bash
# After the held-out residual diagnostic finishes, inspect it and start a
# seven-point fixed-topology Stokes-I reconstruction under experimental full
# Jones. This is not a production freeze and does not open topology or fold 4.
set -euo pipefail

SRC="${SL1MJAX_PHASE6_SRC:-$(cd "$(dirname "$0")/.." && pwd)}"
DEST="${SL1MJAX_PHASE6_EXPLICIT_DEST:-/home/stephen/checkouts/SL1MJax-phase6-explicit-20260831}"
LIVE_DEST="${SL1MJAX_PHASE6_DEST:-/home/stephen/checkouts/SL1MJax-phase6-20260830}"
MAIN="${SL1MJAX_OLD_CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
PY="${SL1MJAX_PYTHON:-$MAIN/.venv/bin/python}"
NATIVE="${SL1MJAX_NATIVE_ROOT:-$MAIN/outputs/3c391_native_averaging_ablation}"
GOLDEN="${SL1MJAX_POL_GOLDEN:-$DEST/tests/fixtures/3c391_polarization_golden.npz}"
COMPARE_OUT="${SL1MJAX_JONES_COMPARE_OUT:-$DEST/outputs/3c391_fullpol_prep/jones_compare}"
COMPARE_LOG="${SL1MJAX_JONES_COMPARE_LOG:-$DEST/outputs/3c391_fullpol_prep/jones_compare.log}"
IMAGING_OUT="${SL1MJAX_JONES_IMAGING_OUT:-$DEST/outputs/3c391_fullpol_prep/jones_imaging}"
LOG="${SL1MJAX_JONES_IMAGING_LOG:-$DEST/outputs/3c391_fullpol_prep/jones_imaging.log}"
INTERVAL="${SL1MJAX_JONES_WAIT_INTERVAL:-60}"

if [[ "$DEST" == "$LIVE_DEST" ]]; then
  echo "diagnostic Jones imaging dest must not be the live checkout" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG")" "$IMAGING_OUT"
exec >>"$LOG" 2>&1
echo "=== wait for residual diagnostic $(date -Iseconds) pid=$$ ==="

_compare_python() {
  ps -eo args= | grep -F "diagnose_3c391_voltage_beam_polarization.py" | grep -vq grep
}

_compare_wrapper() {
  ps -eo args= | grep -F "run_3c391_fullpol_jones_compare_bacchus.sh" | grep -vq grep
}

while _compare_python || _compare_wrapper; do
  echo "residual diagnostic still running $(date -Iseconds)"
  sleep "$INTERVAL"
done

if [[ ! -f "$COMPARE_OUT/summary.json" ]]; then
  echo "compare summary missing; refuse to start Jones imaging" >&2
  exit 2
fi
if grep -q 'Traceback' "$COMPARE_LOG"; then
  echo "compare log has a traceback; refuse to start Jones imaging" >&2
  exit 2
fi

export PYTHONPATH="$DEST/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

echo "=== inspect residual diagnostic $(date -Iseconds) ==="
"$PY" "$DEST/scripts/inspect_3c391_fullpol_diagnostic.py" --summary "$COMPARE_OUT/summary.json"

echo "=== convention ladder $(date -Iseconds) ==="
set +e
"$PY" -m pytest \
  "$DEST/tests/test_vla_beam_conventions.py" \
  "$DEST/tests/test_cassbeam_beam.py" \
  "$DEST/tests/test_full_jones.py" \
  "$DEST/tests/test_polarization.py" \
  "$DEST/tests/test_3c391_polarization_golden.py" \
  "$DEST/tests/test_fullpol_prep.py" \
  -q
ladder_status=$?
set -e
if [[ "$ladder_status" -ne 0 ]]; then
  echo "convention ladder exited $ladder_status; continue diagnostic imaging"
fi

if ps -eo args= | grep -F "$LIVE_DEST/scripts/run_3c391_phase6_bacchus.py" | grep -vq grep; then
  echo "live Phase 6 job still holds the GPU; refuse" >&2
  exit 2
fi

echo "=== diagnostic full-Jones imaging $(date -Iseconds) ==="
"$PY" -c "import jax; print('jax devices', jax.devices())"
"$PY" "$DEST/scripts/run_3c391_phase6_bacchus.py" \
  --native-root "$NATIVE" \
  --catalogue "$DEST/config/3c391_radio_guard_catalog.json" \
  --polarization-golden "$GOLDEN" \
  --output "$IMAGING_OUT" \
  --stage baseline \
  --beams full_jones \
  --allow-diagnostic-full-jones \
  --operator-mode explicit_jax
echo "=== diagnostic full-Jones imaging done $(date -Iseconds) ==="
