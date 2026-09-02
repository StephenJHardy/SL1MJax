#!/usr/bin/env bash
# Continue Phase 6 on the explicit-JAX dest. Never rsyncs the live 20260830
# checkout. Refuses to start while that job holds the GPU. Copies completed
# live products so finished C1 beams are skipped. Does not start baseline
# unless the caller passes that stage explicitly.
set -euo pipefail

SRC="${SL1MJAX_PHASE6_SRC:-$(cd "$(dirname "$0")/.." && pwd)}"
DEST="${SL1MJAX_PHASE6_EXPLICIT_DEST:-/home/stephen/checkouts/SL1MJax-phase6-explicit-20260831}"
LIVE_DEST="${SL1MJAX_PHASE6_DEST:-/home/stephen/checkouts/SL1MJax-phase6-20260830}"
MAIN="${SL1MJAX_OLD_CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
PY="${SL1MJAX_PYTHON:-$MAIN/.venv/bin/python}"
OUT="${SL1MJAX_PHASE6_EXPLICIT_OUTPUT:-$DEST/outputs/3c391_phase6_explicit}"
LIVE_OUT="${SL1MJAX_PHASE6_OUTPUT:-$LIVE_DEST/outputs/3c391_phase6_20260830}"
NATIVE="${SL1MJAX_NATIVE_ROOT:-$MAIN/outputs/3c391_native_averaging_ablation}"
GOLDEN="${SL1MJAX_POL_GOLDEN:-$DEST/tests/fixtures/3c391_polarization_golden.npz}"
STAGE="${1:-commissioning-c4}"
BEAMS="${2:-static_scalar,streamed_scalar,diagonal_copolar}"

if [[ "$DEST" == "$LIVE_DEST" ]]; then
  echo "explicit ladder dest must not be the live checkout" >&2
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
  ps -eo pid=,args= | grep -F "$LIVE_DEST/scripts/run_3c391_phase6_bacchus.py" | grep -v grep >&2 || true
  exit 2
fi

export PYTHONPATH="$DEST/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

mkdir -p "$DEST" "$OUT"
rsync -a \
  --exclude '.venv/' \
  --exclude '.git/' \
  --exclude '/outputs/' \
  --exclude '/data/' \
  --exclude '__pycache__/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.pytest_cache/' \
  "$SRC/" "$DEST/"

SOURCE_JSON="$OUT/explicit_source.json"
"$PY" - <<PY
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
manifest = write_staged_source_manifest(
    Path("$DEST"),
    Path("$SOURCE_JSON"),
    git_root=Path("$SRC"),
)
print("copied", ",".join(copied) or "none")
print("source", manifest)
PY

cd "$MAIN"
echo "=== explicit ladder $STAGE start $(date -Iseconds) ==="
echo "dest=$DEST"
echo "output=$OUT"
"$PY" -c "import jax; print('jax devices', jax.devices())"
"$PY" "$DEST/scripts/run_3c391_phase6_bacchus.py" \
  --native-root "$NATIVE" \
  --catalogue "$DEST/config/3c391_radio_guard_catalog.json" \
  --polarization-golden "$GOLDEN" \
  --output "$OUT" \
  --source-manifest "$SOURCE_JSON" \
  --stage "$STAGE" \
  --beams "$BEAMS" \
  --operator-mode explicit_jax
echo "=== explicit ladder $STAGE done $(date -Iseconds) ==="
"$PY" -c "
from pathlib import Path
from sl1mjax.phase6_protocol import review_phase6_output, phase6_output_complete
reviews = review_phase6_output(Path('$OUT'))
print('complete', phase6_output_complete(reviews))
for item in reviews:
    status = 'PASS' if item.passed else ('MISS' if not item.present else 'FAIL')
    extra = '; '.join(item.failures) if item.failures else ''
    print(status, f'{item.stage}/{item.beam_mode}', extra)
"
