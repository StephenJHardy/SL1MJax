#!/usr/bin/env bash
# Calibrate one survey work MS. Never the scientific SPW 4+5 MS.
set -euo pipefail
SPW="${1:?usage: $0 <spw> [lower_c|upper_c]}"
EXEC="${2:-lower_c}"
OUT="${OUT:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_diagonal_survey_v1}"
WORK="$OUT/work_ms/${EXEC}_spw$(printf '%02d' "$SPW").work.ms"
if [[ "$EXEC" == "upper_c" ]]; then
  CAL="$OUT/cal/upper_c/spw$(printf '%02d' "$SPW")"
else
  CAL="$OUT/cal/spw$(printf '%02d' "$SPW")"
fi
CASA="${CASA:-/home/stephen/casa/casa-6.7.6-14-py3.12.el9/bin/casa}"
SCRIPT="${SCRIPT:-/tmp/sl1mjax-evla-c-survey-scripts/create_thol0001_scientific_calibration.py}"
SCIENTIFIC="/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"

if [[ ! -d "$WORK" ]]; then
  echo "missing work MS: $WORK" >&2
  exit 1
fi
if [[ "$EXEC" == "upper_c" || "$WORK" == *upper_c_* ]]; then
  POLICY="${POLICY:-$OUT/upper_c_calibration_policy.json}"
  if [[ ! -f "$POLICY" ]]; then
    echo "refusing lower-C 3C286/scan calibration on an upper-C work MS" >&2
    exit 2
  fi
  eval "$(
    PYTHONPATH="${PYTHONPATH:-/tmp/sl1mjax-evla-c-survey}" \
      "${PY:-python3}" - "$POLICY" <<'PY'
import json, sys
from pathlib import Path
policy = json.loads(Path(sys.argv[1]).read_text())
if policy.get("uses_3c286") or policy.get("lower_c_scan_list_reusable"):
    raise SystemExit("upper-C policy still points at the lower-C 3C286 scan list")
if policy.get("flux_scans") == "2,51":
    raise SystemExit("upper-C policy reused lower-C flux scans 2,51")
mapping = {
    "SL1MJAX_THOL0001_FLUX_SCANS": "flux_scans",
    "SL1MJAX_THOL0001_SOLVE_FLUX_SCAN": "solve_flux_scan",
    "SL1MJAX_THOL0001_HELD_OUT_FLUX_SCAN": "held_out_flux_scan",
    "SL1MJAX_THOL0001_PHASE_SCANS": "phase_scans",
    "SL1MJAX_THOL0001_FLUX_FIELD": "flux_field",
    "SL1MJAX_THOL0001_D_FIELD": "d_field",
    "SL1MJAX_THOL0001_HOLORASTER_FIELD": "holoraster_field",
    "SL1MJAX_THOL0001_ON_AXIS_FIELDS": "on_axis_fields",
    "SL1MJAX_THOL0001_PREDICTION_FIELDS": "prediction_fields",
    "SL1MJAX_THOL0001_APPLY_FIELDS": "apply_fields",
    "SL1MJAX_THOL0001_CHECK_FIELD": "check_field",
}
for env, key in mapping.items():
    value = policy.get(key)
    if value:
        print(f'export {env}={json.dumps(str(value))}')
PY
  )"
fi
if [[ "$(realpath "$WORK")" == "$(realpath "$SCIENTIFIC")" ]]; then
  echo "refusing to calibrate the scientific SPW 4+5 MS" >&2
  exit 1
fi
if [[ "$CAL" == *products/scientific* ]]; then
  echo "refusing to write into products/scientific" >&2
  exit 1
fi
mkdir -p "$CAL"
export SL1MJAX_THOL0001_SCIENTIFIC_MS="$WORK"
export SL1MJAX_THOL0001_SCIENTIFIC_CAL_ROOT="$CAL"
export SL1MJAX_ALLOW_SCIENTIFIC_CAL_ROOT=1
export SL1MJAX_THOL0001_SPW="$SPW"
export SL1MJAX_THOL0001_EDGE_SPW="${SPW}:5~58"
export SL1MJAX_THOL0001_G0_SPW="${SPW}:27~36"
export SL1MJAX_THOL0001_DIAGONAL_ONLY="${SL1MJAX_THOL0001_DIAGONAL_ONLY:-1}"
echo "calibrating $WORK into $CAL"
"$CASA" --nologger --nogui -c "$SCRIPT"
