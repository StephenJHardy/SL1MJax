#!/usr/bin/env bash
# CPU-only C1 review as soon as diagonal summary.json exists.
# Does not start a GPU job and does not start baseline.
# Exits if the live producer dies without writing the product.
set -euo pipefail

LIVE_DEST="${SL1MJAX_PHASE6_DEST:-/home/stephen/checkouts/SL1MJax-phase6-20260830}"
LIVE_OUT="${SL1MJAX_PHASE6_OUTPUT:-$LIVE_DEST/outputs/3c391_phase6_20260830}"
EXPLICIT_DEST="${SL1MJAX_PHASE6_EXPLICIT_DEST:-/home/stephen/checkouts/SL1MJax-phase6-explicit-20260831}"
EXPLICIT_OUT="${SL1MJAX_PHASE6_EXPLICIT_OUTPUT:-$EXPLICIT_DEST/outputs/3c391_phase6_explicit}"
PY="${SL1MJAX_PYTHON:-/home/stephen/checkouts/SL1MJax/.venv/bin/python}"
INTERVAL="${SL1MJAX_C1_WATCH_INTERVAL:-30}"
IDLE_RECHECK="${SL1MJAX_C1_WATCH_IDLE_RECHECK:-5}"
LOG="${SL1MJAX_C1_WATCH_LOG:-$EXPLICIT_OUT/c1_review_watch.log}"
PRODUCT="$LIVE_OUT/commissioning/diagonal_copolar/summary.json"

mkdir -p "$EXPLICIT_OUT"
exec >>"$LOG" 2>&1
echo "=== c1-review-watch start $(date -Iseconds) pid=$$ ==="

_live_python() {
  ps -eo args= | grep -F "$LIVE_DEST/scripts/run_3c391_phase6_bacchus.py" | grep -vq grep
}

_live_wrapper() {
  ps -eo args= | grep -F "$LIVE_DEST" | grep -E 'commissioning-c4|run_3c391_phase6_bacchus\.sh' | grep -vq grep
}

_live_running() {
  _live_python || _live_wrapper
}

_producer_gone() {
  if _live_running; then
    return 1
  fi
  sleep "$IDLE_RECHECK"
  if _live_running; then
    return 1
  fi
  return 0
}

while [[ ! -f "$PRODUCT" ]]; do
  if _producer_gone; then
    echo "$(date -Iseconds) C1 producer exited without $PRODUCT"
    exit 2
  fi
  echo "$(date -Iseconds) waiting for $PRODUCT"
  sleep "$INTERVAL"
done
echo "$(date -Iseconds) C1 diagonal summary present"
export PYTHONPATH="$EXPLICIT_DEST/src"
"$PY" - <<PY
import json
from pathlib import Path
from sl1mjax.phase6_protocol import review_phase6_output, select_topology_round_beams

root = Path("$LIVE_OUT")
reviews = review_phase6_output(root, stages=("commissioning",))
c1 = [item for item in reviews if item.stage == "commissioning"]
payload = {
    "c1_complete": bool(c1) and all(item.passed for item in c1),
    "topology_beams": list(select_topology_round_beams(c1)) if all(item.passed for item in c1) else [],
    "reviews": [
        {
            "stage": item.stage,
            "beam_mode": item.beam_mode,
            "present": item.present,
            "passed": item.passed,
            "failures": list(item.failures),
            "metrics": {
                key: item.metrics.get(key)
                for key in (
                    "train_loss",
                    "holdout_loss",
                    "kkt_residual",
                    "audit_under_resolved",
                    "sealed_fold_unused",
                    "stop_reason",
                    "sky_max_depth",
                    "integration_max_depth",
                )
            },
        }
        for item in c1
    ],
}
out = Path("$EXPLICIT_OUT/commissioning_c1_review.json")
out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
print("c1_complete", payload["c1_complete"])
print("topology_beams", payload["topology_beams"])
print("wrote", out)
for item in c1:
    status = "PASS" if item.passed else ("MISS" if not item.present else "FAIL")
    extra = "; ".join(item.failures) if item.failures else ""
    print(status, f"{item.stage}/{item.beam_mode}", extra)
PY
echo "=== c1-review-watch done $(date -Iseconds) ==="
