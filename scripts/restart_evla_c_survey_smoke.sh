#!/usr/bin/env bash
# Restart the isolated 3C391 survey smoke. Do not match this filename
# against running job argument lists.
set -euo pipefail
PROD="${PROD:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_diagonal_survey_v1}"
SCR="${SCR:-/tmp/sl1mjax-evla-c-survey-scripts}"
LOG="$PROD/smoke_restart.log"
SUMMARY="$PROD/3c391_survey_smoke/smoke/evla_c_diagonal_survey_v1/smoke_summary.json"

if [[ -f "$SUMMARY" ]]; then
  echo "smoke already complete: $SUMMARY"
  exit 0
fi

python3 - <<'PY'
import os
import signal
import subprocess

out = subprocess.check_output(["ps", "-ax", "-o", "pid=,args="], text=True)
for line in out.splitlines():
    pid_s, args = line.strip().split(" ", 1)
    if "restart_evla_c_survey_smoke.sh" in args:
        continue
    if args.endswith("run_thol0001_survey_continue.sh smoke") or "run_3c391_survey_smoke.py" in args:
        print("stop", pid_s, args[:140])
        os.kill(int(pid_s), signal.SIGTERM)
PY

nohup "$SCR/run_thol0001_survey_continue.sh" smoke > "$LOG" 2>&1 &
echo "SMOKE_RESTARTED pid=$! log=$LOG"
