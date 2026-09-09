#!/usr/bin/env bash
# Continue EVLA-C refresh after a running frequency job. Does not overwrite
# frozen products. SPW 5 stays sealed.
set -euo pipefail

OUT="${OUT:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_full_jones_validation_refresh_v1}"
PY="${PY:-/home/stephen/checkouts/SL1MJax/.venv/bin/python}"
SCRIPT="${SCRIPT:-/tmp/sl1mjax-evla-c-refresh-scripts/run_thol0001_evla_c_validation_refresh.py}"
export PYTHONPATH="${PYTHONPATH:-/tmp/sl1mjax-evla-c-refresh}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export PYTHONUNBUFFERED=1

wait_pid="${1:-}"
if [[ -n "$wait_pid" ]]; then
  echo "waiting for pid $wait_pid"
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 15
  done
  if ! grep -q '"ok": true' "$OUT/logs/frequency.log"; then
    echo "frequency stage did not finish ok" >&2
    exit 1
  fi
fi

run_stage() {
  local stage="$1"
  echo "=== stage $stage ==="
  "$PY" -u "$SCRIPT" \
    --stage "$stage" \
    --repo-root /tmp/sl1mjax-evla-c-refresh \
    --output-dir "$OUT" \
    --beam-root /media/stephen/astro/vla/beam_models/cassbeam_evla_c_g1024_p32_ninepub_20260908 \
    > "$OUT/logs/${stage}.log" 2>&1
}

run_stage phase4
run_stage phase5
echo "tail stages complete"
