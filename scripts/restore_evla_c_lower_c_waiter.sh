#!/usr/bin/env bash
# Launch wait_evla_c_lower_c_after_passb.sh if its pid file is stale.
set -euo pipefail
PROD="${PROD:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_diagonal_survey_v1}"
SCR="${SCR:-/tmp/sl1mjax-evla-c-survey-scripts}"
PIDFILE="$PROD/lower_c_after_passb.pid"
LOG="$PROD/lower_c_remaining.log"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "lower-C waiter already running pid=$(cat "$PIDFILE")"
  exit 0
fi
nohup "$SCR/wait_evla_c_lower_c_after_passb.sh" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "LOWER_C_WAITER pid=$! log=$LOG"
