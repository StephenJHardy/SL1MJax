#!/usr/bin/env bash
# Continue isolated Bacchus survey work. Never writes frozen archives.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-/tmp/sl1mjax-evla-c-survey}"
case "${1:-help}" in
  smoke|imaging-smoke|imaging-mosaic)
    unset JAX_PLATFORMS || true
    ;;
  *)
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
    ;;
esac
PY="${PY:-/home/stephen/checkouts/SL1MJax/.venv/bin/python}"
SCR="${SCR:-/tmp/sl1mjax-evla-c-survey-scripts}"
BEAM="${BEAM:-/media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1}"
PROD="${PROD:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_diagonal_survey_v1}"
NATIVE="${NATIVE:-/home/stephen/checkouts/SL1MJax/outputs/3c391_native_averaging_ablation}"
CHECKOUT="${CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
CATALOGUE="${CATALOGUE:-$SCR/config/3c391_radio_guard_catalog.json}"
POL_GOLDEN="${POL_GOLDEN:-$SCR/fixtures/3c391_polarization_golden.npz}"

need_plane() {
  local mhz="$1"
  local found
  found="$(find "$BEAM" -name "evla-cband-${mhz}-g1024-p32.params" -print -quit)"
  [[ -z "$found" ]]
}

generate_mhz() {
  local mhz="$1"
  if need_plane "$mhz"; then
    echo "GENERATE $mhz $(date -u +%FT%TZ)"
    "$PY" -u "$SCR/generate_cassbeam_highres_model.py" \
      --base-input "$BEAM/reference/base.in" \
      --geometry "$BEAM/reference/vla_geom" \
      --output-dir "$BEAM" \
      --feedtaper-mode evla_c \
      --name-prefix evla-cband \
      --frequencies-mhz "$mhz"
  else
    echo "HAVE $mhz"
  fi
}

rewrite_digest() {
  "$PY" - <<PY
from pathlib import Path
from sl1mjax.cassbeam_highres import write_development_highres_manifest
from sl1mjax.evla_c_survey_compare import SURVEY_MODEL_ID
from sl1mjax.evla_c_survey_beam import survey_catalog_digest
root = Path("$BEAM")
write_development_highres_manifest(root, model_id=SURVEY_MODEL_ID, name_prefix="evla-cband")
print("digest", survey_catalog_digest(root))
PY
}

case "${1:-help}" in
  imaging-nodes)
    generate_mhz 4536
    generate_mhz 4598
    generate_mhz 4662
    rewrite_digest
    ;;
  spw6-passb-planes)
    for mhz in 4772 4804 4836 4868; do
      generate_mhz "$mhz"
    done
    rewrite_digest
    ;;
  score-spw6-passb)
    MS="$PROD/work_ms/lower_c_spw06.work.ms"
    for ch in 8 24 40 56; do
      echo "SCORE SPW6 ch$ch $(date -u +%FT%TZ)"
      "$PY" -u "$SCR/run_thol0001_evla_c_diagonal_survey.py" \
        --stage score --spw 6 --channel "$ch" \
        --scripts-dir "$SCR" --beam-root "$BEAM" --measurement-set "$MS"
    done
    ;;
  smoke)
    DIGEST="$("$PY" - <<PY
from sl1mjax.evla_c_survey_beam import survey_catalog_digest
from pathlib import Path
print(survey_catalog_digest(Path("$BEAM")))
PY
)"
    echo "SMOKE digest=$DIGEST $(date -u +%FT%TZ)"
    "$PY" -u "$SCR/run_3c391_survey_smoke.py" \
      --native-root "$NATIVE" \
      --survey-catalog-root "$BEAM" \
      --survey-catalog-digest "$DIGEST" \
      --catalogue "$CATALOGUE" \
      --polarization-golden "$POL_GOLDEN" \
      --output "$PROD/3c391_survey_smoke"
    ;;
  score-pass-a)
    SPW="${2:?spw}"
    CH="${3:-32}"
    MS="$PROD/work_ms/lower_c_spw$(printf '%02d' "$SPW").work.ms"
    echo "SCORE SPW$SPW ch$CH $(date -u +%FT%TZ)"
    "$PY" -u "$SCR/run_thol0001_evla_c_diagonal_survey.py" \
      --stage score --spw "$SPW" --channel "$CH" \
      --scripts-dir "$SCR" --beam-root "$BEAM" --measurement-set "$MS"
    ;;
  lower-c-spw)
    SPW="${2:?spw}"
    MS="$PROD/work_ms/lower_c_spw$(printf '%02d' "$SPW").work.ms"
    if [[ ! -f "$MS.provenance.json" ]]; then
      echo "WORKCOPY SPW$SPW $(date -u +%FT%TZ)"
      "$PY" -u "$SCR/run_thol0001_evla_c_diagonal_survey.py" --stage workcopy --spw "$SPW"
    fi
    if [[ ! -d "$PROD/cal/spw$(printf '%02d' "$SPW")/diagonal" ]]; then
      echo "CAL SPW$SPW $(date -u +%FT%TZ)"
      "$SCR/run_thol0001_survey_spw_calibration.sh" "$SPW"
    fi
    echo "SCORE_A SPW$SPW $(date -u +%FT%TZ)"
    "$PY" -u "$SCR/run_thol0001_evla_c_diagonal_survey.py" \
      --stage score --spw "$SPW" --channel 32 \
      --scripts-dir "$SCR" --beam-root "$BEAM" --measurement-set "$MS"
    ;;
  upper-c-scan-inventory)
    "$PY" -u "$SCR/run_thol0001_evla_c_diagonal_survey.py" \
      --stage scan-inventory --execution upper_c \
      --scripts-dir "$SCR"
    ;;
  upper-c-workcopy)
    SPW="${2:?spw}"
    echo "WORKCOPY upper_c SPW$SPW $(date -u +%FT%TZ)"
    "$PY" -u "$SCR/run_thol0001_evla_c_diagonal_survey.py" \
      --stage workcopy --execution upper_c --spw "$SPW" \
      --scripts-dir "$SCR"
    ;;
  upper-c-spw)
    SPW="${2:?spw}"
    MS="$PROD/work_ms/upper_c_spw$(printf '%02d' "$SPW").work.ms"
    POLICY="$PROD/upper_c_calibration_policy.json"
    if [[ ! -f "$POLICY" ]]; then
      echo "missing upper-C policy $POLICY" >&2
      exit 2
    fi
    if [[ ! -f "$MS.provenance.json" ]]; then
      echo "WORKCOPY upper_c SPW$SPW $(date -u +%FT%TZ)"
      "$PY" -u "$SCR/run_thol0001_evla_c_diagonal_survey.py" \
        --stage workcopy --execution upper_c --spw "$SPW" \
        --scripts-dir "$SCR"
    fi
    if [[ ! -d "$PROD/cal/upper_c/spw$(printf '%02d' "$SPW")/diagonal" ]]; then
      echo "CAL upper_c SPW$SPW $(date -u +%FT%TZ)"
      "$SCR/run_thol0001_survey_spw_calibration.sh" "$SPW" upper_c
    fi
    echo "SCORE_A upper_c SPW$SPW $(date -u +%FT%TZ)"
    "$PY" -u "$SCR/run_thol0001_evla_c_diagonal_survey.py" \
      --stage score --execution upper_c --spw "$SPW" --channel 32 \
      --scripts-dir "$SCR" --beam-root "$BEAM" --measurement-set "$MS"
    ;;
  imaging-convergence)
    CONV="${CONV:-/media/stephen/astro/vla/beam_models/cassbeam_evla_c_survey_convergence_v1}"
    mkdir -p "$CONV"
    echo "IMAGING_CONVERGENCE $(date -u +%FT%TZ)"
    "$PY" -u "$SCR/run_evla_c_imaging_convergence.py" \
      --production-root "$BEAM" \
      --workspace "$CONV"
    ;;
  native-3c391-planes)
    echo "NATIVE_3C391_PLANES $(date -u +%FT%TZ)"
    if [[ -f "$BEAM/manifest.json" && ! -f "$PROD/manifest_before_native_3c391.json" ]]; then
      cp "$BEAM/manifest.json" "$PROD/manifest_before_native_3c391.json"
    fi
    for mhz in $(seq 4536 2 4662); do
      generate_mhz "$mhz"
    done
    rewrite_digest
    ;;
  verify-native-planes)
    PRIOR=""
    if [[ -f "$PROD/manifest_before_native_3c391.json" ]]; then
      PRIOR="--prior-manifest $PROD/manifest_before_native_3c391.json"
    fi
    "$PY" -u "$SCR/verify_evla_c_native_planes.py" \
      --beam-root "$BEAM" \
      --output "$PROD/native_3c391_plane_verification.json" \
      $PRIOR
    ;;
  imaging-smoke)
    exec "$0" smoke
    ;;
  imaging-mosaic)
    CONV="${CONV:-/media/stephen/astro/vla/beam_models/cassbeam_evla_c_survey_convergence_v1}"
    if [[ ! -f "$CONV/imaging_convergence.json" ]]; then
      echo "missing imaging convergence report $CONV/imaging_convergence.json" >&2
      exit 2
    fi
    if [[ ! -f "$PROD/native_3c391_plane_verification.json" ]]; then
      echo "missing native-plane verification $PROD/native_3c391_plane_verification.json" >&2
      exit 2
    fi
    if [[ ! -f "$PROD/3c391_survey_smoke/smoke/evla_c_diagonal_survey_v1/smoke_summary.json" ]]; then
      echo "missing imaging smoke summary" >&2
      exit 2
    fi
    DIGEST="$("$PY" - <<PY
from sl1mjax.evla_c_survey_beam import survey_catalog_digest
from pathlib import Path
print(survey_catalog_digest(Path("$BEAM")))
PY
)"
    echo "IMAGING_MOSAIC digest=$DIGEST $(date -u +%FT%TZ)"
    "$PY" -u "$SCR/run_3c391_phase6_bacchus.py" \
      --stage baseline \
      --beams evla_c_diagonal_survey_v1 \
      --survey-catalog-root "$BEAM" \
      --survey-catalog-digest "$DIGEST" \
      --native-root "$NATIVE" \
      --catalogue "$CATALOGUE" \
      --polarization-golden "$POL_GOLDEN" \
      --operator-mode explicit_jax \
      --output "$PROD/3c391_survey_mosaic"
    ;;
  reexport-provenance)
    DEST="$PROD/provenance_exports"
    mkdir -p "$DEST"
    echo "REEXPORT $(date -u +%FT%TZ)"
    "$PY" -u "$SCR/reexport_evla_c_survey_slots.py" \
      --report-dir "$PROD" \
      --export-dir "$DEST" \
      --beam-root "$BEAM" \
      --scripts-dir "$SCR"
    ;;
  matched-population)
    "$PY" -u "$SCR/score_evla_c_matched_population.py" \
      --export-dir "$PROD/provenance_exports" \
      --output "$PROD/matched_population.json"
    ;;
  upper-c-crc)
    echo "UPPER_C_CRC $(date -u +%FT%TZ)"
    "$SCR/inspect_upper_c_archive_integrity.sh"
    ;;
  catalog-record)
    "$PY" - <<PY
import json
from pathlib import Path
from sl1mjax.evla_c_survey_beam import survey_catalog_record
from sl1mjax.evla_c_diagonal_survey import write_json_atomic
slots = []
for path in sorted(Path("$PROD").glob("channels/*channel*_report.json")):
    slots.append(json.loads(path.read_text()))
record = survey_catalog_record(root=Path("$BEAM"), scored_slots=slots)
write_json_atomic(Path("$PROD") / "catalog_record.json", record)
print(record["digest"], len(record["planes"]))
PY
    ;;
  *)
    echo "usage: $0 imaging-nodes|imaging-convergence|native-3c391-planes|verify-native-planes|imaging-smoke|imaging-mosaic|reexport-provenance|matched-population|upper-c-crc|spw6-passb-planes|score-spw6-passb|smoke|catalog-record|score-pass-a <spw> [ch]|lower-c-spw <spw>|upper-c-scan-inventory|upper-c-workcopy <spw>|upper-c-spw <spw>" >&2
    exit 2
    ;;
esac
