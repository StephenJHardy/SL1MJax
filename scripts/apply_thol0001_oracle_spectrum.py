"""Apply K+B+G and K+B+G+Kcross to the all-channel oracle row.

Standalone CASA 6.7.6 script. Does not import sl1mjax.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from casatasks import applycal

PRODUCT_ROOT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_CAL_ROOT",
        "/media/stephen/astro/vla/extracted/commissioning/products",
    )
)
SPECTRUM_ROOT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_ORACLE_SPECTRUM",
        "/media/stephen/astro/vla/extracted/commissioning/validation/oracle/spectrum",
    )
)
OUTPUT = SPECTRUM_ROOT / "applied"
TABLES = {
    "antpos": str(PRODUCT_ROOT / "diagonal" / "antpos.cal"),
    "G1": str(PRODUCT_ROOT / "diagonal" / "G1.cal"),
    "K0": str(PRODUCT_ROOT / "diagonal" / "K0.cal"),
    "B0": str(PRODUCT_ROOT / "diagonal" / "B0.cal"),
    "Kcross": str(PRODUCT_ROOT / "fullpol" / "Kcross.cal"),
}
STAGES = {
    "K+B+G": ["antpos", "G1", "K0", "B0"],
    "K+B+G+Kcross": ["antpos", "G1", "K0", "B0", "Kcross"],
}

manifest = json.loads((SPECTRUM_ROOT / "oracle_cases.json").read_text())
OUTPUT.mkdir(parents=True, exist_ok=True)
written = []
for stage, keys in STAGES.items():
    gaintable = [TABLES[key] for key in keys]
    interp = ["", "linear", "", "nearest"] + ([""] * (len(keys) - 4))
    for case in manifest["cases"]:
        dest = OUTPUT / stage.replace("+", "_") / Path(case["ms"]).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(case["ms"], dest)
        applycal(
            vis=str(dest),
            field="",
            gaintable=gaintable,
            interp=interp,
            calwt=False,
            parang=False,
            applymode="calflag",
        )
        written.append({"stage": stage, "basis": case["basis"], "ms": str(dest)})
        print(dest)
(OUTPUT / "applied.json").write_text(json.dumps(written, indent=2) + "\n")
print(OUTPUT / "applied.json")
