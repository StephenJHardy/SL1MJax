"""Apply scientific Df and Df+QU from DATA onto dedicated field-9/11 copies.

Does not modify the scientific work MS, archives, or compatibility fixtures.
Each chain is applied once from DATA onto a versioned destination.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from casatasks import applycal, split

SOURCE = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_SCIENTIFIC_MS",
        "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms",
    )
)
PRODUCT_ROOT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_SCIENTIFIC_CAL_ROOT",
        "/media/stephen/astro/vla/extracted/commissioning/products/scientific",
    )
)
DEST_ROOT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_UNBLOCK",
        "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock",
    )
)
VARIANT = os.environ.get("SL1MJAX_THOL0001_POL_VARIANT", "field9_df")

TABLES = {
    "field9_df": {
        "field": "9",
        "d_table": PRODUCT_ROOT / "fullpol" / "Df.cal",
        "dest": DEST_ROOT / "ms" / "field9_df.apply.ms",
    },
    "field9_df_qu": {
        "field": "9",
        "d_table": PRODUCT_ROOT / "fullpol" / "Df_QU.cal",
        "dest": DEST_ROOT / "ms" / "field9_df_qu.apply.ms",
    },
    "field11_df": {
        "field": "11",
        "d_table": PRODUCT_ROOT / "fullpol" / "Df.cal",
        "dest": DEST_ROOT / "ms" / "field11_df.apply.ms",
    },
}

if VARIANT not in TABLES:
    raise SystemExit(f"unknown variant {VARIANT}")
spec = TABLES[VARIANT]
DEST = spec["dest"]
DEST.parent.mkdir(parents=True, exist_ok=True)
if not DEST.exists():
    split(
        vis=str(SOURCE),
        outputvis=str(DEST),
        field=spec["field"],
        spw="4",
        datacolumn="data",
    )
# After split the sole field is usually id 0 and SPW 4 is reindexed to 0.
gaintable = [
    str(PRODUCT_ROOT / "diagonal" / "antpos.cal"),
    str(PRODUCT_ROOT / "diagonal" / "G1.cal"),
    str(PRODUCT_ROOT / "diagonal" / "K0.cal"),
    str(PRODUCT_ROOT / "diagonal" / "B0.cal"),
    str(PRODUCT_ROOT / "fullpol" / "Kcross.cal"),
    str(spec["d_table"]),
    str(PRODUCT_ROOT / "fullpol" / "Xf.cal"),
]
applycal(
    vis=str(DEST),
    field="",
    gaintable=gaintable,
    interp=["", "linear", "", "nearest", "", "", ""],
    spwmap=[[4]] * len(gaintable),
    calwt=False,
    parang=True,
    applymode="calflag",
)
payload = {
    "vis": str(DEST),
    "source": str(SOURCE),
    "field": spec["field"],
    "spw": "4",
    "d_table": str(spec["d_table"]),
    "applied_from": "DATA",
    "parang": True,
    "calwt": False,
    "scientific_ms_unmodified": True,
}
DEST.with_name(DEST.name + ".apply.json").write_text(json.dumps(payload, indent=2) + "\n")
print(DEST)
