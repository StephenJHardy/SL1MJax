#!/usr/bin/env bash
# Stage a *new* tree and run the one-batch VJP vs explicit-JAX gate.
# Does not touch the live C1/C4 checkout or start baseline.
set -euo pipefail

SRC="${SL1MJAX_PHASE6_SRC:-$(cd "$(dirname "$0")/.." && pwd)}"
DEST="${SL1MJAX_PHASE6_COMPARE_DEST:-/home/stephen/checkouts/SL1MJax-phase6-explicit-20260831}"
MAIN="${SL1MJAX_OLD_CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
PY="${SL1MJAX_PYTHON:-$MAIN/.venv/bin/python}"
OUT="${SL1MJAX_PHASE6_COMPARE_OUTPUT:-$DEST/outputs/3c391_phase6_explicit_compare}"
NATIVE="${SL1MJAX_NATIVE_ROOT:-$MAIN/outputs/3c391_native_averaging_ablation}"
GOLDEN="${SL1MJAX_POL_GOLDEN:-$SRC/tests/fixtures/3c391_polarization_golden.npz}"
PRODUCT="${1:?usage: run_3c391_operator_compare_bacchus.sh PRODUCT_DIR [BEAMS] [POINTING]}"
BEAMS="${2:-static_scalar,streamed_scalar,diagonal_copolar}"
POINTING="${3:-C1}"

LIVE_DEST="${SL1MJAX_PHASE6_DEST:-/home/stephen/checkouts/SL1MJax-phase6-20260830}"
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
if [[ ! -d "$PRODUCT" ]]; then
  echo "product directory does not exist: $PRODUCT" >&2
  exit 2
fi

export PYTHONPATH="$DEST/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

mkdir -p "$DEST" "$OUT"
src_root="$(cd "$SRC" && pwd)"
dest_root="$(cd "$DEST" && pwd)"
if [[ "$src_root" != "$dest_root" ]]; then
  rsync -a --delete \
    --exclude '.venv/' \
    --exclude '.git/' \
    --exclude '/outputs/' \
    --exclude '/data/' \
    --exclude '__pycache__/' \
    --exclude '.mypy_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '.pytest_cache/' \
    "$SRC/" "$DEST/"
else
  echo "compare dest is already $dest_root; skip self-rsync"
fi

cd "$MAIN"
echo "=== operator compare start $(date -Iseconds) ==="
echo "dest=$DEST"
"$PY" -c "import jax; print('jax devices', jax.devices())"
cmd=(
  "$PY" "$DEST/scripts/compare_3c391_operator_modes.py"
  --native-root "$NATIVE"
  --catalogue "$DEST/config/3c391_radio_guard_catalog.json"
  --polarization-golden "$GOLDEN"
  --output "$OUT/operator_compare_${POINTING}_$(basename "$PRODUCT").json"
  --pointing "$POINTING"
  --beams "$BEAMS"
  --product "$PRODUCT"
)
"${cmd[@]}"
echo "=== operator compare done $(date -Iseconds) ==="
