#!/usr/bin/env bash
# Copy the immutable commissioning MS and solve the scientific flux-gauge chain.
# Does not overwrite products/diagonal or products/fullpol.
set -euo pipefail

COMMISSIONING_MS="${SL1MJAX_THOL0001_COMMISSIONING_MS:-/media/stephen/astro/vla/extracted/commissioning/THOL0001.sb31628704.eb31629959.lowerC.spw45.ms}"
SCIENTIFIC_MS="${SL1MJAX_THOL0001_SCIENTIFIC_MS:-/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms}"
PRODUCT_ROOT="${SL1MJAX_THOL0001_SCIENTIFIC_CAL_ROOT:-/media/stephen/astro/vla/extracted/commissioning/products/scientific}"
CASA="${SL1MJAX_CASA:-/home/stephen/casa/casa-6.7.6-14-py3.12.el9/bin/casa}"
SCRIPT="${1:-/tmp/sl1mjax-thol0001-scripts/create_thol0001_scientific_calibration.py}"

if [[ ! -d "$COMMISSIONING_MS" ]]; then
  echo "missing commissioning MS: $COMMISSIONING_MS" >&2
  exit 1
fi
if [[ "$PRODUCT_ROOT" != *products/scientific* ]]; then
  echo "refusing to write scientific tables outside products/scientific: $PRODUCT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$SCIENTIFIC_MS" ]]; then
  mkdir -p "$(dirname "$SCIENTIFIC_MS")"
  echo "copying $COMMISSIONING_MS -> $SCIENTIFIC_MS"
  cp -a "$COMMISSIONING_MS" "$SCIENTIFIC_MS"
fi
# The commissioning archive is read-only. cp -a keeps that, and CASA cannot solve.
chmod -R u+w "$SCIENTIFIC_MS"
export SL1MJAX_THOL0001_SCIENTIFIC_MS="$SCIENTIFIC_MS"
export SL1MJAX_THOL0001_SCIENTIFIC_CAL_ROOT="$PRODUCT_ROOT"
mkdir -p "$PRODUCT_ROOT"
"$CASA" --nologger --nogui -c "$SCRIPT"
