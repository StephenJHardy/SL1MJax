#!/usr/bin/env bash
# Survey-validation path. Must not take the imaging GPU.
set -euo pipefail
export JAX_PLATFORMS=cpu
SCR="${SCR:-/tmp/sl1mjax-evla-c-survey-scripts}"
PROD="${PROD:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_diagonal_survey_v1}"
LOG="${LOG:-$PROD/survey_path.log}"
mkdir -p "$PROD"
exec >>"$LOG" 2>&1
echo "SURVEY_PATH_START $(date -u +%FT%TZ)"
"$SCR/run_thol0001_survey_continue.sh" upper-c-crc
"$SCR/run_thol0001_survey_continue.sh" reexport-provenance
"$SCR/run_thol0001_survey_continue.sh" matched-population
echo "SURVEY_PATH_DONE $(date -u +%FT%TZ)"
