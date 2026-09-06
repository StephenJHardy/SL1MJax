"""Solve a scan-2-only bandpass and apply it to scan 51.

G1_hold_scan51 is a G holdout only. This ablation is the genuine scan-51
bandpass / source-model test: B is solved from scan 2 and transferred.
Standalone CASA 6.7.6 script; do not import sl1mjax.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from casatasks import applycal, bandpass

WORK_MS = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_WORK_MS",
        "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.work.ms",
    )
)
PRODUCT_ROOT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_CAL_ROOT",
        "/media/stephen/astro/vla/extracted/commissioning/products",
    )
)
REFERENCE_ANTENNA = os.environ.get("SL1MJAX_THOL0001_REFANT", "ea02")
FLUX_FIELD = "0"
SOLVE_FLUX_SCAN = "2"
HOLD_FLUX_SCAN = "51"
SPW = "4,5"


def _table(product: str, name: str) -> str:
    destination = PRODUCT_ROOT / product
    destination.mkdir(parents=True, exist_ok=True)
    return str(destination / name)


if not WORK_MS.is_dir():
    raise FileNotFoundError(WORK_MS)

vis = str(WORK_MS)
antpos = _table("diagonal", "antpos.cal")
g0 = _table("diagonal", "G0.cal")
delay = _table("diagonal", "K0.cal")
gain_hold = _table("diagonal", "G1_hold_scan51.cal")
bandpass_ablation = _table("diagonal", "B2_scan2.cal")

bandpass(
    vis=vis,
    caltable=bandpass_ablation,
    field=FLUX_FIELD,
    scan=SOLVE_FLUX_SCAN,
    spw=SPW,
    refant=REFERENCE_ANTENNA,
    combine="scan",
    solint="inf",
    bandtype="B",
    gaintable=[antpos, g0, delay],
)
applycal(
    vis=vis,
    field=FLUX_FIELD,
    scan=HOLD_FLUX_SCAN,
    gaintable=[antpos, gain_hold, delay, bandpass_ablation],
    gainfield=["", "", "", ""],
    interp=["", "linear", "", "nearest"],
    calwt=False,
    parang=False,
    applymode="calflag",
)
payload = {
    "product": "thol0001_lower_c_b2_scan2_ablation",
    "kind": "bandpass_ablation",
    "vis": vis,
    "solve_scan": SOLVE_FLUX_SCAN,
    "apply_scan": HOLD_FLUX_SCAN,
    "g_holdout": "G1_hold_scan51 is a G holdout only; this table is the scan-51 B test",
    "parang": False,
    "calwt": False,
    "tables": {
        "antpos": antpos,
        "G0": g0,
        "K0": delay,
        "G1_hold_scan51": gain_hold,
        "B2_scan2": bandpass_ablation,
    },
}
destination = PRODUCT_ROOT / "diagonal" / "B2_scan2.product.json"
destination.write_text(json.dumps(payload, indent=2) + "\n")
print(destination)
print(bandpass_ablation)
