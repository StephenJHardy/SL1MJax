"""Apply cumulative CASA chains to every injected-basis oracle MS.

Standalone CASA 6.7.6 script. Does not import sl1mjax. Each stage is a
fresh copy so DATA stays the injected basis.
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
ORACLE_ROOT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_ORACLE_ROOT",
        "/media/stephen/astro/vla/extracted/commissioning/validation/oracle",
    )
)
OUTPUT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_ORACLE_APPLIED",
        str(ORACLE_ROOT / "applied"),
    )
)
STAGES = (
    "K+B+G",
    "K+B+G+Kcross",
    "K+B+G+Kcross+Df",
    "K+B+G+Kcross+Df+Xf",
    "K+B+G+Kcross+Df+Xf+P",
)
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
    "K+B+G": ["antpos", "G1", "K0", "B0"],
    "K+B+G+Kcross": ["antpos", "G1", "K0", "B0", "Kcross"],
    "K+B+G+Kcross+Df": ["antpos", "G1", "K0", "B0", "Kcross", "Df"],
    "K+B+G+Kcross+Df+Xf": ["antpos", "G1", "K0", "B0", "Kcross", "Df", "Xf"],
    "K+B+G+Kcross+Df+Xf+P": ["antpos", "G1", "K0", "B0", "Kcross", "Df", "Xf"],
}

manifest = json.loads((ORACLE_ROOT / "oracle_cases.json").read_text())
OUTPUT.mkdir(parents=True, exist_ok=True)
written = []
for stage in STAGES:
    keys = STAGE_TABLES[stage]
    gaintable = [TABLES[key] for key in keys]
    interp = ["", "linear", "", "nearest"] + ([""] * (len(keys) - 4))
    parang = stage.endswith("+P")
    for case in manifest["cases"]:
        dest = OUTPUT / stage.replace("+", "_") / Path(case["ms"]).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and (dest / "table.dat").exists():
            written.append(
                {
                    "stage": stage,
                    "label": case["label"],
                    "basis": case["basis"],
                    "ms": str(dest),
                    "reused": True,
                }
            )
            print("reuse", dest)
            continue
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(case["ms"], dest)
        applycal(
            vis=str(dest),
            field="",
            gaintable=gaintable,
            interp=interp,
            calwt=False,
            parang=parang,
            applymode="calflag",
        )
        written.append(
            {"stage": stage, "label": case["label"], "basis": case["basis"], "ms": str(dest)}
        )
        print(dest)
(OUTPUT / "applied.json").write_text(json.dumps(written, indent=2) + "\n")
print(OUTPUT / "applied.json")
