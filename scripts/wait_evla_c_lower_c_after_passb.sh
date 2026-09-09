#!/usr/bin/env bash
# Serialized remaining lower-C Pass A after SPW 6 Pass B finishes.
set -euo pipefail
PROD="${PROD:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_diagonal_survey_v1}"
SCR="${SCR:-/tmp/sl1mjax-evla-c-survey-scripts}"
MARKER="$PROD/score_spw06_passb.log"
while ! grep -q PASSB_SCORE_DONE "$MARKER" 2>/dev/null; do
  sleep 30
done
"$SCR/run_thol0001_survey_continue.sh" score-pass-a 1 32
for spw in 2 3 7 8 9 10 11 12 13 14 15; do
  "$SCR/run_thol0001_survey_continue.sh" lower-c-spw "$spw"
done
echo LOWER_C_PASS_A_DONE
