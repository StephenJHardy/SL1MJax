#!/usr/bin/env bash
# Build the v3 survey publication pack. Does not overwrite v1 or v2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEDGER="${LEDGER:-$ROOT/docs/evla_c_diagonal_survey}"
OUT="${OUT:-$ROOT/src/sl1mjax/data/vla_c_band_beam_validation_v3}"
DIGEST="${SURVEY_CATALOG_DIGEST:-}"

if [[ "$OUT" == *vla_c_band_beam_validation_v2* || "$OUT" == *vla_c_band_beam_validation_v1* ]]; then
  echo "refusing to overwrite a frozen publication bundle" >&2
  exit 2
fi

cd "$ROOT"
uv run python - <<PY
from pathlib import Path
from sl1mjax.evla_c_survey_publication import write_survey_v3_bundle
root = Path("$OUT")
digest = "${DIGEST}" or None
path = write_survey_v3_bundle(
    ledger_dir=Path("$LEDGER"),
    output_dir=root,
    catalog_digest=digest,
)
print(path)
print((path / "manifest.json").read_text())
PY
