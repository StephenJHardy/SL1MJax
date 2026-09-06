"""Apply the full-pol chain to 3C286 samples excluded from the Xf solve set.

Standalone CASA 6.7.6 script. Xf used field 11 with solint=inf, so a
channel-edge holdout is the exclusion that does not require another solve.
This writes CORRECTED_DATA on a dedicated copy; the commissioning MS is
not modified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from casatasks import applycal, split

SOURCE = Path(
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
DEST = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_C286_MS",
        "/media/stephen/astro/vla/extracted/commissioning/validation/"
        "THOL0001.lowerC.three_c286.applyback.ms",
    )
)

# Xf was solved on 4:5~58,5:5~58. Channels 0-4 and 59-63 were excluded.
HOLDOUT_SPW = "4:0~4,4:59~63,5:0~4,5:59~63"

if DEST.exists():
    import shutil

    shutil.rmtree(DEST)
split(
    vis=str(SOURCE),
    outputvis=str(DEST),
    field="11",
    spw="4,5",
    datacolumn="data",
)
applycal(
    vis=str(DEST),
    field="11",
    gaintable=[
        str(PRODUCT_ROOT / "diagonal" / "antpos.cal"),
        str(PRODUCT_ROOT / "diagonal" / "G1.cal"),
        str(PRODUCT_ROOT / "diagonal" / "K0.cal"),
        str(PRODUCT_ROOT / "diagonal" / "B0.cal"),
        str(PRODUCT_ROOT / "fullpol" / "Kcross.cal"),
        str(PRODUCT_ROOT / "fullpol" / "Df.cal"),
        str(PRODUCT_ROOT / "fullpol" / "Xf.cal"),
    ],
    interp=["", "linear", "", "nearest", "", "", ""],
    calwt=False,
    parang=True,
    applymode="calflag",
)
payload = {
    "vis": str(DEST),
    "field": 11,
    "xf_solve_channels": "4:5~58,5:5~58",
    "holdout_spw": HOLDOUT_SPW,
    "independence": "apply_back_on_xf_excluded_edge_channels",
    "parang": True,
    "calwt": False,
}
DEST.with_name(DEST.name + ".apply.json").write_text(json.dumps(payload, indent=2) + "\n")
print(DEST)
