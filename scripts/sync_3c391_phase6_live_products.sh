#!/usr/bin/env bash
# Read-only copy of live Phase 6 products into the explicit dest.
# Never writes the live checkout and never starts a GPU job.
# Never overwrites a dest beam that already has summary.json and checkpoint.json.
set -euo pipefail

LIVE_DEST="${SL1MJAX_PHASE6_DEST:-/home/stephen/checkouts/SL1MJax-phase6-20260830}"
LIVE_OUT="${SL1MJAX_PHASE6_OUTPUT:-$LIVE_DEST/outputs/3c391_phase6_20260830}"
EXPLICIT_DEST="${SL1MJAX_PHASE6_EXPLICIT_DEST:-/home/stephen/checkouts/SL1MJax-phase6-explicit-20260831}"
EXPLICIT_OUT="${SL1MJAX_PHASE6_EXPLICIT_OUTPUT:-$EXPLICIT_DEST/outputs/3c391_phase6_explicit}"
MAIN="${SL1MJAX_OLD_CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
PY="${SL1MJAX_PYTHON:-$MAIN/.venv/bin/python}"
INTERVAL="${SL1MJAX_PHASE6_SYNC_INTERVAL:-30}"

if [[ "$EXPLICIT_OUT" == "$LIVE_OUT" ]]; then
  echo "explicit output must differ from the live output" >&2
  exit 2
fi

mkdir -p "$EXPLICIT_OUT"
log="$EXPLICIT_OUT/product_sync.log"
export PYTHONPATH="$EXPLICIT_DEST/src"

_copy_products() {
  "$PY" - <<PY
from pathlib import Path
from sl1mjax.phase6_protocol import copy_phase6_products
copied = copy_phase6_products(
    Path("$LIVE_OUT"),
    Path("$EXPLICIT_OUT"),
    require_complete_source=False,
)
print(",".join(copied) or "none")
PY
}

echo "=== product sync start $(date -Iseconds) ===" | tee -a "$log"
while true; do
  copied="$(_copy_products)"
  echo "$(date -Iseconds) copied=$copied commissioning=$(ls -1 "$EXPLICIT_OUT/commissioning" 2>/dev/null | tr '\n' ' ') c4=$(ls -1 "$EXPLICIT_OUT/commissioning-c4" 2>/dev/null | tr '\n' ' ')" >>"$log"
  if [[ -f "$LIVE_OUT/commissioning/diagonal_copolar/summary.json" \
     && -f "$LIVE_OUT/commissioning-c4/diagonal_copolar/summary.json" ]]; then
    echo "=== product sync done $(date -Iseconds) ===" | tee -a "$log"
    exit 0
  fi
  sleep "$INTERVAL"
done
