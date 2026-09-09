#!/usr/bin/env bash
# Imaging path only. Does not wait for survey closeout or upper-C repair.
set -euo pipefail
SCR="${SCR:-/tmp/sl1mjax-evla-c-survey-scripts}"
PY="${PY:-/home/stephen/checkouts/SL1MJax/.venv/bin/python}"
CONV="${CONV:-/media/stephen/astro/vla/beam_models/cassbeam_evla_c_survey_convergence_v1}"
PROD="${PROD:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_diagonal_survey_v1}"
LOG="${LOG:-$PROD/imaging_path.log}"
mkdir -p "$PROD" "$CONV"
exec >>"$LOG" 2>&1
echo "IMAGING_PATH_START $(date -u +%FT%TZ)"
"$SCR/run_thol0001_survey_continue.sh" imaging-convergence
"$PY" - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$CONV/imaging_convergence.json").read_text())
justified = payload.get("keep_production_g1024_p32")
if justified is None:
    justified = True
    for row in payload.get("slots") or []:
        classification = row.get("classification") or {}
        sampling = row.get("sampling") or {}
        vis = classification.get("main_lobe_visibility_rel_l2")
        if vis is None or not (vis == vis) or float(vis) > 0.01:
            justified = False
        if row.get("comparison") == "aperture" and not sampling.get("same_pixel_scale", True):
            justified = False
        if row.get("comparison") == "angular" and sampling.get("same_pixel_scale", False):
            justified = False
if not justified:
    raise SystemExit(
        "imaging-frequency comparison does not justify keeping g1024/p32; "
        "not generating 64 planes"
    )
print(
    "keep g1024/p32",
    justified,
    "numerically_qualified_0p002",
    payload.get("numerically_qualified_0p002"),
)
PY
"$SCR/run_thol0001_survey_continue.sh" native-3c391-planes
"$SCR/run_thol0001_survey_continue.sh" verify-native-planes
echo "IMAGING_GATES_CPU_DONE $(date -u +%FT%TZ)"
"$SCR/run_thol0001_survey_continue.sh" imaging-smoke
"$SCR/run_thol0001_survey_continue.sh" imaging-mosaic
echo "IMAGING_PATH_DONE $(date -u +%FT%TZ)"
