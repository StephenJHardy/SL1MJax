"""Solve Df+QU on on-axis 3C147 and leave Df(Q=U=0) unchanged.

Field 9 spans much more of the observation than 3C286. This script does
not use C147-* . Standalone CASA 6.7.6 script; do not import sl1mjax.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from casatasks import polcal, setjy

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
ON_AXIS_FIELDS = "0,9"
D_FIELD = "9"
EDGE_SPW = "4:5~58,5:5~58"
SPW = "4,5"


def _table(product: str, name: str) -> str:
    destination = PRODUCT_ROOT / product
    destination.mkdir(parents=True, exist_ok=True)
    return str(destination / name)


if not WORK_MS.is_dir():
    raise FileNotFoundError(WORK_MS)

vis = str(WORK_MS)
antpos = _table("diagonal", "antpos.cal")
gain = _table("diagonal", "G1.cal")
delay = _table("diagonal", "K0.cal")
bandpass_table = _table("diagonal", "B0.cal")
kcross = _table("fullpol", "Kcross.cal")
dterms_qu = _table("fullpol", "Df_QU.cal")

setjy(
    vis=vis,
    field=D_FIELD,
    standard="manual",
    fluxdensity=[1.0, 0.0, 0.0, 0.0],
    spix=[0.0],
    reffreq="4.6GHz",
    usescratch=True,
    scalebychan=True,
    spw=SPW,
)
polcal(
    vis=vis,
    caltable=dterms_qu,
    field=ON_AXIS_FIELDS,
    spw=EDGE_SPW,
    solint="inf",
    combine="scan",
    poltype="Df+QU",
    refant=REFERENCE_ANTENNA,
    minsnr=3,
    gaintable=[antpos, gain, delay, bandpass_table, kcross],
)
payload = {
    "product": "thol0001_lower_c_df_qu",
    "vis": vis,
    "poltype": "Df+QU",
    "fields": ON_AXIS_FIELDS,
    "c147_offset_used_for_d": False,
    "compare_with": _table("fullpol", "Df.cal"),
    "holdout": {
        "field_id": 9,
        "axes": ["time", "antenna", "channel"],
        "held_out_reference": "ea26",
    },
    "tables": {"Df_QU": dterms_qu},
    "notes": [
        "Df assumes Q=U=0; Df+QU solves a global 3C147 Q/U",
        "Difference on held-out field 9 is the leakage systematic floor",
    ],
}
destination = PRODUCT_ROOT / "fullpol" / "Df_QU.product.json"
destination.write_text(json.dumps(payload, indent=2) + "\n")
print(destination)
print(dterms_qu)
