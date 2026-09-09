#!/usr/bin/env bash
# Build the v2 publication bundle, notebook, and rendered summary from a
# completed EVLA-C refresh product. Does not overwrite v1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_REFRESH="${REMOTE_REFRESH:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_full_jones_validation_refresh_v1}"
REMOTE_COORD="${REMOTE_COORD:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/holoraster_coordinate_feed_comparison_v2}"
STAGE="${STAGE:-$ROOT/docs/evla_c_full_jones_validation_refresh/stage}"
REFRESH="${REFRESH:-$STAGE/refresh}"
COORD="${COORD:-$STAGE/coordinate_feed_v2}"
EXISTING="${EXISTING:-$ROOT/src/sl1mjax/data/vla_c_band_beam_validation_v1}"
OUT="${OUT:-$ROOT/src/sl1mjax/data/vla_c_band_beam_validation_v2}"
RSYNC=(rsync -az -e "ssh -i ${HOME}/.ssh/home_pem.pem -o IdentitiesOnly=yes")

if [[ ! -f "$REFRESH/channels/channel32_export.npz" ]]; then
  echo "fetching refresh products from Bacchus into $STAGE"
  mkdir -p "$REFRESH/channels" "$REFRESH/c147_ring" "$COORD"
  "${RSYNC[@]}" --include='*.json' --exclude='*' \
    stephen@hardynet.dyndns.org:"$REMOTE_REFRESH/channels/" \
    "$REFRESH/channels/"
  "${RSYNC[@]}" \
    stephen@hardynet.dyndns.org:"$REMOTE_REFRESH/channels/channel32_export.npz" \
    "$REFRESH/channels/"
  "${RSYNC[@]}" \
    stephen@hardynet.dyndns.org:"$REMOTE_REFRESH/phase4_classification.json" \
    stephen@hardynet.dyndns.org:"$REMOTE_REFRESH/phase4_injections.json" \
    "$REFRESH/"
  "${RSYNC[@]}" \
    stephen@hardynet.dyndns.org:"$REMOTE_REFRESH/c147_ring/c147_report.json" \
    "$REFRESH/c147_ring/" || true
  "${RSYNC[@]}" \
    stephen@hardynet.dyndns.org:"$REMOTE_COORD/channel32_model_comparison.npz" \
    stephen@hardynet.dyndns.org:"$REMOTE_COORD/report.json" \
    "$COORD/"
fi

need=(
  "$REFRESH/channels/channel00_report.json"
  "$REFRESH/channels/channel08_report.json"
  "$REFRESH/channels/channel16_report.json"
  "$REFRESH/channels/channel24_report.json"
  "$REFRESH/channels/channel32_report.json"
  "$REFRESH/channels/channel40_report.json"
  "$REFRESH/channels/channel48_report.json"
  "$REFRESH/channels/channel56_report.json"
  "$REFRESH/channels/channel63_report.json"
  "$REFRESH/channels/channel32_export.npz"
  "$REFRESH/phase4_classification.json"
)
for path in "${need[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "blocked: missing $path" >&2
    exit 2
  fi
done
if [[ ! -f "$REFRESH/c147_ring/c147_report.json" ]]; then
  echo "warning: C147 ring missing; bundle will mark F17 historical/blocked"
fi

cd "$ROOT"
uv run python scripts/build_vla_c_band_beam_validation_bundle.py \
  --refresh-dir "$REFRESH" \
  --existing-bundle "$EXISTING" \
  --coordinate-feed-dir "$COORD" \
  --output-dir "$OUT"
uv run python scripts/write_vla_c_band_beam_validation_notebook.py
uv run python scripts/execute_vla_c_band_beam_validation_notebook.py --timeout 900
uv run python scripts/render_vla_c_band_beam_validation.py \
  --bundle "$OUT" \
  --asset-dir "$ROOT/docs/assets/vla_c_band_beam_validation" \
  --output "$ROOT/docs/vla_c_band_beam_validation.md"
echo "v2 bundle $OUT"
