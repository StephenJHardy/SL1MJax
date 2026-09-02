#!/usr/bin/env bash
# Abort the live full-pol diagnostic only for concrete pipeline failures.
# Does not interpret residuals as beam selection or sky evidence.
set -euo pipefail

DEST="${SL1MJAX_PHASE6_EXPLICIT_DEST:-/home/stephen/checkouts/SL1MJax-phase6-explicit-20260831}"
LOG="${SL1MJAX_JONES_COMPARE_LOG:-$DEST/outputs/3c391_fullpol_prep/jones_compare.log}"
SUMMARY="${SL1MJAX_JONES_COMPARE_OUT:-$DEST/outputs/3c391_fullpol_prep/jones_compare}/summary.json"
MS="${SL1MJAX_FULLPOL_MS:-$DEST/data/3c391/fullpol_prep/3c391_gkb_only_4corr.ms}"
PATTERN='diagnose_3c391_voltage_beam_polarization.py'

_fail() {
  echo "stop full-pol diagnostic: $*" >&2
  pkill -f "$PATTERN" || true
  exit 2
}

if [ "$(basename "$MS")" != "3c391_gkb_only_4corr.ms" ]; then
  _fail "wrong MeasurementSet $(basename "$MS")"
fi

STATE="${MS}.calibration_state.json"
python3 - "$STATE" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload.get("calibration_state") == "gkb_only"
assert payload.get("jax_polarisation_input") is True
assert "Kcross" not in payload.get("applied", [])
assert "Df" not in payload.get("applied", [])
PY

while pgrep -f "$PATTERN" >/dev/null; do
  if grep -Eiq 'adaptive regional|opened fold 4|unseal fold' "$LOG"; then
    _fail "run started regional Q/U or opened fold 4"
  fi
  if grep -q 'Traceback' "$LOG"; then
    _fail "known convention or runtime failure while running"
  fi
  sleep 30
done

if [ -f "$SUMMARY" ]; then
  echo "diagnostic finished; summary $SUMMARY"
fi
