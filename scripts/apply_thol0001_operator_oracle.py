"""Apply one cumulative chain to a tiny injected-basis MS.

Standalone CASA 6.7.6 script. DATA must already contain one basis
visibility. Writes CORRECTED_DATA. sl1mjax is not imported.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from casatasks import applycal

PRODUCT_ROOT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_CAL_ROOT",
        "/media/stephen/astro/vla/extracted/commissioning/products",
    )
)
VIS = Path(os.environ["SL1MJAX_THOL0001_ORACLE_MS"])
STAGE = os.environ.get("SL1MJAX_THOL0001_ORACLE_STAGE", "K+B+G+Kcross+Df+Xf+P")

TABLES = {
    "antpos": str(PRODUCT_ROOT / "diagonal" / "antpos.cal"),
    "G1": str(PRODUCT_ROOT / "diagonal" / "G1.cal"),
    "K0": str(PRODUCT_ROOT / "diagonal" / "K0.cal"),
    "B0": str(PRODUCT_ROOT / "diagonal" / "B0.cal"),
    "Kcross": str(PRODUCT_ROOT / "fullpol" / "Kcross.cal"),
    "Df": str(PRODUCT_ROOT / "fullpol" / "Df.cal"),
    "Xf": str(PRODUCT_ROOT / "fullpol" / "Xf.cal"),
}
STAGE_TABLES = {
    "antpos": ["antpos"],
    "K": ["K0"],
    "K+B": ["K0", "B0"],
    "K+B+G": ["antpos", "G1", "K0", "B0"],
    "K+B+G+Kcross": ["antpos", "G1", "K0", "B0", "Kcross"],
    "K+B+G+Kcross+Df": ["antpos", "G1", "K0", "B0", "Kcross", "Df"],
    "K+B+G+Kcross+Df+Xf": ["antpos", "G1", "K0", "B0", "Kcross", "Df", "Xf"],
    "K+B+G+Kcross+Df+Xf+P": ["antpos", "G1", "K0", "B0", "Kcross", "Df", "Xf"],
}

if STAGE not in STAGE_TABLES:
    raise ValueError(STAGE)
if not VIS.is_dir():
    raise FileNotFoundError(VIS)

keys = STAGE_TABLES[STAGE]
gaintable = [TABLES[key] for key in keys]
parang = STAGE.endswith("+P")
interp = [""] * len(gaintable)
if keys[:4] == ["antpos", "G1", "K0", "B0"]:
    interp = ["", "linear", "", "nearest"] + ([""] * (len(keys) - 4))
applycal(
    vis=str(VIS),
    field="",
    gaintable=gaintable,
    interp=interp,
    calwt=False,
    parang=parang,
    applymode="calflag",
)
payload = {
    "stage": STAGE,
    "vis": str(VIS),
    "gaintable": gaintable,
    "parang": parang,
    "apply_from": "DATA",
}
VIS.with_name(VIS.name + ".apply.json").write_text(json.dumps(payload, indent=2) + "\n")
print(VIS)
