#!/usr/bin/env bash
# Stage the uncommitted Phase 6 tree and start the overnight protocol.
set -euo pipefail

SRC="${SL1MJAX_PHASE6_SRC:-$(cd "$(dirname "$0")/.." && pwd)}"
DEST="${SL1MJAX_PHASE6_DEST:-/home/stephen/checkouts/SL1MJax-phase6-20260830}"
MAIN="${SL1MJAX_OLD_CHECKOUT:-/home/stephen/checkouts/SL1MJax}"
PY="${SL1MJAX_PYTHON:-$MAIN/.venv/bin/python}"
OUT="${SL1MJAX_PHASE6_OUTPUT:-$DEST/outputs/3c391_phase6_20260830}"
NATIVE="${SL1MJAX_NATIVE_ROOT:-$MAIN/outputs/3c391_native_averaging_ablation}"
GOLDEN="${SL1MJAX_POL_GOLDEN:-$DEST/tests/fixtures/3c391_polarization_golden.npz}"

export PYTHONPATH="$DEST/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

if ps -eo args= | grep -F "$DEST/scripts/run_3c391_phase6_bacchus.py" | grep -vq grep; then
  echo "refusing to rsync over a live dest: $DEST" >&2
  ps -eo pid=,args= | grep -F "$DEST/scripts/run_3c391_phase6_bacchus.py" | grep -v grep >&2 || true
  exit 2
fi

mkdir -p "$(dirname "$DEST")" "$OUT"
SOURCE_JSON="$OUT/source.json"
"$PY" - <<PY
import hashlib, json, subprocess
from pathlib import Path
src = Path("$SRC")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=src, text=True)
lock = src / "uv.lock"
payload = {
    "commit": commit,
    "uncommitted_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
}
Path("$SOURCE_JSON").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("source", payload["commit"], payload["uncommitted_diff_sha256"][:12])
PY
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

cd "$MAIN"
echo "=== start $(date -Iseconds) ==="
echo "src=$SRC"
echo "dest=$DEST"
echo "python=$PY"
"$PY" -c "import jax; print('jax devices', jax.devices())"
"$PY" "$DEST/scripts/run_3c391_phase6_bacchus.py" \
  --native-root "$NATIVE" \
  --catalogue "$DEST/config/3c391_radio_guard_catalog.json" \
  --polarization-golden "$GOLDEN" \
  --output "$OUT" \
  --source-manifest "$SOURCE_JSON" \
  --stage "${1:-commissioning}"
echo "=== done $(date -Iseconds) ==="
