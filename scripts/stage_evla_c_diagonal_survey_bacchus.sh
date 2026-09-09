#!/usr/bin/env bash
# Copy an isolated SL1MJax tree to Bacchus. Never rsync onto the checkout src.
set -euo pipefail
HOST="${HOST:-stephen@hardynet.dyndns.org}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/home_pem.pem}"
SSH=(ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o ConnectTimeout=20 "$HOST")
RSYNC=(rsync -az -e "ssh -i $SSH_KEY -o IdentitiesOnly=yes")
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PACKAGE="/tmp/sl1mjax-evla-c-survey"
SCRIPTS="/tmp/sl1mjax-evla-c-survey-scripts"

"${SSH[@]}" "mkdir -p '$PACKAGE/sl1mjax' '$SCRIPTS'"
"${RSYNC[@]}" \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.mypy_cache/' \
  "$ROOT/src/sl1mjax/" \
  "$HOST:$PACKAGE/sl1mjax/"
"${SSH[@]}" "mkdir -p '$SCRIPTS/config' '$SCRIPTS/fixtures'"
"${RSYNC[@]}" \
  "$ROOT/config/3c391_radio_guard_catalog.json" \
  "$HOST:$SCRIPTS/config/"
"${RSYNC[@]}" \
  "$ROOT/tests/fixtures/3c391_polarization_golden.npz" \
  "$HOST:$SCRIPTS/fixtures/"
"${RSYNC[@]}" \
  "$ROOT/scripts/run_thol0001_evla_c_diagonal_survey.py" \
  "$ROOT/scripts/generate_cassbeam_highres_model.py" \
  "$ROOT/scripts/run_thol0001_holoraster_cassbeam_comparison.py" \
  "$ROOT/scripts/run_thol0001_diagonal_recovery.py" \
  "$ROOT/scripts/create_thol0001_scientific_calibration.py" \
  "$ROOT/scripts/run_thol0001_survey_spw_calibration.sh" \
  "$ROOT/scripts/summarize_evla_c_diagonal_survey.py" \
  "$ROOT/scripts/plot_evla_c_diagonal_survey.py" \
  "$ROOT/scripts/plot_evla_c_diagonal_survey_maps.py" \
  "$ROOT/scripts/run_3c391_phase6_bacchus.py" \
  "$ROOT/scripts/run_3c391_survey_smoke.py" \
  "$ROOT/scripts/diagnose_3c391_voltage_beam_transfer.py" \
  "$ROOT/scripts/run_thol0001_survey_continue.sh" \
  "$ROOT/scripts/run_evla_c_imaging_convergence.py" \
  "$ROOT/scripts/verify_evla_c_native_planes.py" \
  "$ROOT/scripts/reexport_evla_c_survey_slots.py" \
  "$ROOT/scripts/score_evla_c_matched_population.py" \
  "$ROOT/scripts/inspect_upper_c_archive_integrity.sh" \
  "$ROOT/scripts/wait_evla_c_imaging_path.sh" \
  "$ROOT/scripts/wait_evla_c_survey_path.sh" \
  "$ROOT/scripts/restart_evla_c_survey_smoke.sh" \
  "$ROOT/scripts/restore_evla_c_lower_c_waiter.sh" \
  "$ROOT/scripts/wait_evla_c_lower_c_after_passb.sh" \
  "$ROOT/scripts/write_vla_c_band_beam_validation_v3_notebook.py" \
  "$ROOT/scripts/render_vla_c_band_beam_validation_v3.py" \
  "$HOST:$SCRIPTS/"
echo "staged $PACKAGE and $SCRIPTS"
