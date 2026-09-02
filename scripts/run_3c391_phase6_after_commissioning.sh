#!/usr/bin/env bash
# Seven-point baseline and one topology round after C1/C4 pass.
# Refuses if the live VJP job still holds the GPU, if commissioning
# review fails, or if explicit_jax is requested without a passing compare.
# This is a separate step; the live waiter does not call it.
set -euo pipefail

DEST="${SL1MJAX_PHASE6_EXPLICIT_DEST:-/home/stephen/checkouts/SL1MJax-phase6-explicit-20260831}"
LIVE_DEST="${SL1MJAX_PHASE6_DEST:-/home/stephen/checkouts/SL1MJax-phase6-20260830}"
MAIN="${SL1MJAX_OLD_CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
PY="${SL1MJAX_PYTHON:-$MAIN/.venv/bin/python}"
OUT="${SL1MJAX_PHASE6_EXPLICIT_OUTPUT:-$DEST/outputs/3c391_phase6_explicit}"
LIVE_OUT="${SL1MJAX_PHASE6_OUTPUT:-$LIVE_DEST/outputs/3c391_phase6_20260830}"
NATIVE="${SL1MJAX_NATIVE_ROOT:-$MAIN/outputs/3c391_native_averaging_ablation}"
GOLDEN="${SL1MJAX_POL_GOLDEN:-$DEST/tests/fixtures/3c391_polarization_golden.npz}"
COMPARE_DIR="${SL1MJAX_PHASE6_COMPARE_OUTPUT:-$DEST/outputs/3c391_phase6_explicit_compare}"
MODE="${1:-vjp}"
if [[ "$MODE" != "vjp" && "$MODE" != "explicit_jax" ]]; then
  echo "usage: run_3c391_phase6_after_commissioning.sh [vjp|explicit_jax]" >&2
  exit 2
fi

if [[ "$DEST" == "$LIVE_DEST" ]]; then
  echo "after-commissioning dest must not be the live checkout" >&2
  exit 2
fi

_live_python() {
  ps -eo args= | grep -F "$LIVE_DEST/scripts/run_3c391_phase6_bacchus.py" | grep -vq grep
}
_live_wrapper() {
  ps -eo args= | grep -F "$LIVE_DEST" | grep -E 'commissioning-c4|run_3c391_phase6_bacchus\.sh' | grep -vq grep
}
if _live_python || _live_wrapper; then
  echo "live Phase 6 job still holds the GPU; refuse" >&2
  exit 2
fi

export PYTHONPATH="$DEST/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"
SOURCE_JSON="$OUT/explicit_source.json"
copied="$("$PY" - <<PY
from pathlib import Path
from sl1mjax.phase6_protocol import (
    copy_phase6_products,
    preserve_commissioning_source,
    write_staged_source_manifest,
)
live = Path("$LIVE_OUT")
dest_out = Path("$OUT")
copied = copy_phase6_products(live, dest_out, require_complete_source=True)
preserve_commissioning_source(live / "source.json", dest_out)
write_staged_source_manifest(Path("$DEST"), Path("$SOURCE_JSON"), git_root=Path("$MAIN"))
print(",".join(copied))
PY
)"
echo "copied complete live products: ${copied:-none}"
if [[ ! -f "$SOURCE_JSON" ]]; then
  echo "missing $SOURCE_JSON; refuse to start baseline without a staged source manifest" >&2
  exit 2
fi

require_compare=0
if [[ "$MODE" == "explicit_jax" ]]; then
  require_compare=1
fi
gate="$("$PY" - <<PY
import json
from pathlib import Path
from sl1mjax.phase6_protocol import commissioning_ready
report = commissioning_ready(
    Path("$OUT"),
    compare_dir=Path("$COMPARE_DIR"),
    require_compare=bool($require_compare),
)
print(json.dumps({
    "ready": report["ready"],
    "commissioning_complete": report["commissioning_complete"],
    "compare_passed": report["compare_passed"],
    "compare_failures": list(report["compare_failures"]),
    "topology_beams": list(report["topology_beams"]),
}))
if not report["ready"]:
    raise SystemExit(2)
PY
)"
echo "$gate"
gate_json="$OUT/after_commissioning_gate.json"
printf '%s\n' "$gate" >"$gate_json"
beams="$("$PY" -c "import json; print(','.join(json.loads(open('$gate_json').read())['topology_beams']))")"

cd "$MAIN"
echo "=== after-commissioning baseline $MODE start $(date -Iseconds) ==="
"$PY" "$DEST/scripts/run_3c391_phase6_bacchus.py" \
  --native-root "$NATIVE" \
  --catalogue "$DEST/config/3c391_radio_guard_catalog.json" \
  --polarization-golden "$GOLDEN" \
  --output "$OUT" \
  --source-manifest "$SOURCE_JSON" \
  --stage baseline \
  --beams static_scalar,streamed_scalar,diagonal_copolar \
  --operator-mode "$MODE"
echo "=== after-commissioning full_round1 beams=$beams start $(date -Iseconds) ==="
"$PY" "$DEST/scripts/run_3c391_phase6_bacchus.py" \
  --native-root "$NATIVE" \
  --catalogue "$DEST/config/3c391_radio_guard_catalog.json" \
  --polarization-golden "$GOLDEN" \
  --output "$OUT" \
  --source-manifest "$SOURCE_JSON" \
  --stage full \
  --beams "$beams" \
  --operator-mode "$MODE"
echo "=== after-commissioning done $(date -Iseconds) ==="
"$PY" -c "
from pathlib import Path
from sl1mjax.phase6_protocol import review_phase6_output, phase6_ladder_complete
reviews = review_phase6_output(Path('$OUT'))
print('complete', phase6_ladder_complete(reviews))
for item in reviews:
    status = 'PASS' if item.passed else ('MISS' if not item.present else 'FAIL')
    extra = '; '.join(item.failures) if item.failures else ''
    print(status, f'{item.stage}/{item.beam_mode}', extra)
"
