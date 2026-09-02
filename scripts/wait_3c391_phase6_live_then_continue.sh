#!/usr/bin/env bash
# Wait for the live VJP C1/C4 job, then continue without starting baseline.
# After the live wrapper exits, copy only missing complete products and resume
# any missing C1 or C4 beam on the explicit dest with VJP. Then run the
# real-product operator compare and write a commissioning review.
# Seven-point baseline stays a separate step.
set -euo pipefail

LIVE_DEST="${SL1MJAX_PHASE6_DEST:-/home/stephen/checkouts/SL1MJax-phase6-20260830}"
EXPLICIT_DEST="${SL1MJAX_PHASE6_EXPLICIT_DEST:-/home/stephen/checkouts/SL1MJax-phase6-explicit-20260831}"
MAIN="${SL1MJAX_OLD_CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
PY="${SL1MJAX_PYTHON:-$MAIN/.venv/bin/python}"
LIVE_OUT="${SL1MJAX_PHASE6_OUTPUT:-$LIVE_DEST/outputs/3c391_phase6_20260830}"
EXPLICIT_OUT="${SL1MJAX_PHASE6_EXPLICIT_OUTPUT:-$EXPLICIT_DEST/outputs/3c391_phase6_explicit}"
NATIVE="${SL1MJAX_NATIVE_ROOT:-$MAIN/outputs/3c391_native_averaging_ablation}"
GOLDEN="${SL1MJAX_POL_GOLDEN:-$LIVE_DEST/tests/fixtures/3c391_polarization_golden.npz}"
INTERVAL="${SL1MJAX_PHASE6_WAIT_INTERVAL:-60}"
IDLE_RECHECK="${SL1MJAX_PHASE6_IDLE_RECHECK:-5}"
LOG="${SL1MJAX_PHASE6_WAIT_LOG:-$EXPLICIT_OUT/wait_continue.log}"
SOURCE_JSON="$EXPLICIT_OUT/explicit_source.json"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== wait-continue start $(date -Iseconds) pid=$$ ==="

_live_python() {
  ps -eo args= | grep -F "$LIVE_DEST/scripts/run_3c391_phase6_bacchus.py" | grep -vq grep
}

_live_wrapper() {
  # The live wrapper queues C4 after C1; argv can mention commissioning-c4
  # while no Python process is up yet.
  ps -eo args= | grep -F "$LIVE_DEST" | grep -E 'commissioning-c4|run_3c391_phase6_bacchus\.sh' | grep -vq grep
}

_live_running() {
  _live_python || _live_wrapper
}

_live_idle() {
  if _live_running; then
    return 1
  fi
  sleep "$IDLE_RECHECK"
  if _live_running; then
    return 1
  fi
  return 0
}

_write_staged_source() {
  export PYTHONPATH="$EXPLICIT_DEST/src"
  "$PY" - <<PY
from pathlib import Path
from sl1mjax.phase6_protocol import (
    copy_phase6_products,
    preserve_commissioning_source,
    write_staged_source_manifest,
)
live = Path("$LIVE_OUT")
dest_out = Path("$EXPLICIT_OUT")
copied = copy_phase6_products(live, dest_out, require_complete_source=True)
preserve_commissioning_source(live / "source.json", dest_out)
write_staged_source_manifest(
    Path("$EXPLICIT_DEST"),
    Path("$SOURCE_JSON"),
    git_root=Path("$MAIN"),
)
print("copied", ",".join(copied) or "none")
PY
}

_stage_incomplete() {
  local stage="$1"
  local beam
  for beam in static_scalar streamed_scalar diagonal_copolar; do
    if [[ ! -f "$EXPLICIT_OUT/$stage/$beam/summary.json" ]]; then
      return 0
    fi
  done
  return 1
}

_run_stage() {
  local stage="$1"
  if _live_running; then
    echo "$(date -Iseconds) live job still holds the GPU; refuse $stage"
    exit 2
  fi
  echo "$(date -Iseconds) resume $stage on explicit dest with VJP"
  export PYTHONPATH="$EXPLICIT_DEST/src"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export PYTHONUNBUFFERED=1
  cd "$MAIN"
  "$PY" "$EXPLICIT_DEST/scripts/run_3c391_phase6_bacchus.py" \
    --native-root "$NATIVE" \
    --catalogue "$EXPLICIT_DEST/config/3c391_radio_guard_catalog.json" \
    --polarization-golden "$GOLDEN" \
    --output "$EXPLICIT_OUT" \
    --source-manifest "$SOURCE_JSON" \
    --stage "$stage" \
    --operator-mode vjp
}

while ! _live_idle; do
  echo "$(date -Iseconds) live VJP job still running"
  sleep "$INTERVAL"
done
echo "$(date -Iseconds) live VJP job idle"

_write_staged_source
if _stage_incomplete commissioning; then
  _run_stage commissioning
fi
if _stage_incomplete commissioning-c4; then
  _run_stage commissioning-c4
fi

compared=false
for beam in static_scalar streamed_scalar diagonal_copolar; do
  product="$EXPLICIT_OUT/commissioning/$beam"
  if [[ -f "$product/summary.json" ]]; then
    echo "$(date -Iseconds) compare $beam using $product"
    bash "$EXPLICIT_DEST/scripts/run_3c391_operator_compare_bacchus.sh" "$product" "$beam"
    compared=true
  fi
done
if [[ "$compared" == false ]]; then
  echo "no C1 product with summary.json; compare skipped" >&2
  exit 2
fi

export PYTHONPATH="$EXPLICIT_DEST/src"
"$PY" - <<PY
import json
from pathlib import Path
from sl1mjax.phase6_protocol import (
    phase6_commissioning_complete,
    phase6_output_complete,
    review_phase6_output,
)
root = Path("$EXPLICIT_OUT")
reviews = review_phase6_output(root)
payload = {
    "commissioning_complete": phase6_commissioning_complete(reviews),
    "ladder_complete": phase6_output_complete(reviews),
    "reviews": [
        {
            "stage": item.stage,
            "beam_mode": item.beam_mode,
            "present": item.present,
            "passed": item.passed,
            "failures": list(item.failures),
            "metrics": dict(item.metrics),
        }
        for item in reviews
    ],
}
out = Path("$EXPLICIT_OUT/commissioning_review.json")
out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
print("commissioning_complete", payload["commissioning_complete"])
print("ladder_complete", payload["ladder_complete"])
print("wrote", out)
for item in reviews:
    status = "PASS" if item.passed else ("MISS" if not item.present else "FAIL")
    extra = "; ".join(item.failures) if item.failures else ""
    print(status, f"{item.stage}/{item.beam_mode}", extra)
PY
echo "=== wait-continue done $(date -Iseconds) ==="
