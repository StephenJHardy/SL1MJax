"""Apply cumulative THOL0001 tables to copies of the golden HOLORASTER MS.

Standalone CASA 6.7.6 script. Does not import sl1mjax. Each stage starts
from DATA. antpos is first so the 0.2 deg diagonal floor can be isolated.
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
SOURCE_MS = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_BISECTION_SOURCE",
        "/media/stephen/astro/vla/extracted/commissioning/validation/"
        "THOL0001.lowerC.golden.fullpol.ms",
    )
)
OUTPUT = Path(
    os.environ.get(
        "SL1MJAX_THOL0001_BISECTION_OUT",
        "/media/stephen/astro/vla/extracted/commissioning/validation/bisection",
    )
)

TABLES = {
    "antpos": PRODUCT_ROOT / "diagonal" / "antpos.cal",
    "K0": PRODUCT_ROOT / "diagonal" / "K0.cal",
    "B0": PRODUCT_ROOT / "diagonal" / "B0.cal",
    "G1": PRODUCT_ROOT / "diagonal" / "G1.cal",
    "Kcross": PRODUCT_ROOT / "fullpol" / "Kcross.cal",
    "Df": PRODUCT_ROOT / "fullpol" / "Df.cal",
    "Xf": PRODUCT_ROOT / "fullpol" / "Xf.cal",
}
STAGES = (
    ("antpos", ["antpos"], False),
    ("K", ["K0"], False),
    ("K+B", ["K0", "B0"], False),
    ("K+B+G", ["antpos", "G1", "K0", "B0"], False),
    ("K+B+G+Kcross", ["antpos", "G1", "K0", "B0", "Kcross"], False),
    ("K+B+G+Kcross+Df", ["antpos", "G1", "K0", "B0", "Kcross", "Df"], False),
    ("K+B+G+Kcross+Df+Xf", ["antpos", "G1", "K0", "B0", "Kcross", "Df", "Xf"], False),
    ("K+B+G+Kcross+Df+Xf+P", ["antpos", "G1", "K0", "B0", "Kcross", "Df", "Xf"], True),
)


def _interp(n_table: int) -> list[str]:
    mapping = {
        "antpos": "",
        "G1": "linear",
        "K0": "",
        "B0": "nearest",
        "Kcross": "",
        "Df": "",
        "Xf": "",
    }
    return [mapping.get(name, "") for name in ("antpos", "G1", "K0", "B0", "Kcross", "Df", "Xf")][
        :n_table
    ]


if not SOURCE_MS.is_dir():
    raise FileNotFoundError(SOURCE_MS)
OUTPUT.mkdir(parents=True, exist_ok=True)
written = []
for name, keys, parang in STAGES:
    dest = OUTPUT / f"stage_{name.replace('+', '_')}.ms"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SOURCE_MS, dest)
    gaintable = [str(TABLES[key]) for key in keys]
    applycal(
        vis=str(dest),
        field="",
        gaintable=gaintable,
        interp=_interp(len(gaintable))
        if keys[:4] == ["antpos", "G1", "K0", "B0"]
        else [""] * len(gaintable),
        calwt=False,
        parang=parang,
        applymode="calflag",
    )
    payload = {
        "stage": name,
        "vis": str(dest),
        "gaintable": gaintable,
        "parang": parang,
        "calwt": False,
        "apply_from": "DATA",
    }
    sidecar = dest.with_name(dest.name + ".apply.json")
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")
    written.append(payload)
    print(sidecar)
(OUTPUT / "stages.json").write_text(json.dumps(written, indent=2) + "\n")
print(OUTPUT / "stages.json")
