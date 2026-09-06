"""Apply existing THOL0001 tables to a small HOLORASTER golden MS.

The Measurement Set must already be a native-resolution copy. This script
does not average, solve, or import sl1mjax. Diagonal uses parang=False;
full-pol uses parang=True. calwt is always False.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from casatasks import applycal

PRODUCT = os.environ.get("SL1MJAX_THOL0001_GOLDEN_PRODUCT", "diagonal")
VIS = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_GOLDEN_MS",
        "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.golden.diagonal.ms",
    )
)
PRODUCT_ROOT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_CAL_ROOT",
        "/media/stephen/astro/vla/extracted/commissioning/products",
    )
)

DIAGONAL = [
    PRODUCT_ROOT / "diagonal" / "antpos.cal",
    PRODUCT_ROOT / "diagonal" / "G1.cal",
    PRODUCT_ROOT / "diagonal" / "K0.cal",
    PRODUCT_ROOT / "diagonal" / "B0.cal",
]
FULLPOL = DIAGONAL + [
    PRODUCT_ROOT / "fullpol" / "Kcross.cal",
    PRODUCT_ROOT / "fullpol" / "Df.cal",
    PRODUCT_ROOT / "fullpol" / "Xf.cal",
]

if PRODUCT not in {"diagonal", "fullpol"}:
    raise ValueError(f"unknown golden product {PRODUCT!r}")
if not VIS.is_dir():
    raise FileNotFoundError(VIS)

gaintable = [str(path) for path in (FULLPOL if PRODUCT == "fullpol" else DIAGONAL)]
missing = [path for path in gaintable if not Path(path).exists()]
if missing:
    raise FileNotFoundError(missing)

applycal(
    vis=str(VIS),
    field="",
    gaintable=gaintable,
    interp=["", "linear", "", "nearest"] + ([""] * (len(gaintable) - 4)),
    calwt=False,
    parang=PRODUCT == "fullpol",
    applymode="calflag",
)
payload = {
    "product": PRODUCT,
    "vis": str(VIS),
    "parang": PRODUCT == "fullpol",
    "calwt": False,
    "gaintable": gaintable,
    "apply_from": "DATA",
    "conventions": [
        "table_order",
        "conjugation_and_baseline_orientation",
        "correlation_packing",
        "rl_handedness",
        "parallactic_angle_treatment",
        "unsupported_antenna_channel_masking",
        "no_double_application",
    ],
}
destination = VIS.with_name(VIS.name + ".apply.json")
destination.write_text(json.dumps(payload, indent=2) + "\n")
print(destination)
