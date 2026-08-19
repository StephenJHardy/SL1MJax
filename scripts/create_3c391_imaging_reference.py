"""Create CASA dirty and multiscale CLEAN references for 3C391 C1."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from casatasks import exportfits, imstat, tclean

MEASUREMENT_SET = Path(os.environ["SL1MJAX_3C391_MS"]).resolve()
OUTPUT = Path(
    os.environ.get(
        "SL1MJAX_3C391_IMAGE_REFERENCE",
        "outputs/3c391_casa_imaging",
    )
).resolve()
FIELD = os.environ.get("SL1MJAX_3C391_IMAGE_FIELD", "3C391 C1")
IMAGE_SIZE = 96
CELL = "4arcsec"


def _remove_products(prefix: Path) -> None:
    for path in prefix.parent.glob(prefix.name + ".*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _export(prefix: Path, suffix: str) -> Path:
    output = prefix.with_name(prefix.name + f".{suffix}.fits")
    exportfits(
        imagename=str(prefix) + f".{suffix}",
        fitsimage=str(output),
        overwrite=True,
        dropdeg=True,
    )
    return output


def _number(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


OUTPUT.mkdir(parents=True, exist_ok=True)
dirty = OUTPUT / "3c391_c1_dirty"
clean = OUTPUT / "3c391_c1_multiscale"
hogbom = OUTPUT / "3c391_c1_hogbom"
_remove_products(dirty)
_remove_products(clean)
_remove_products(hogbom)

common: dict[str, Any] = {
    "vis": str(MEASUREMENT_SET),
    "field": FIELD,
    "spw": "0",
    "datacolumn": "corrected",
    "specmode": "mfs",
    "stokes": "I",
    "imsize": [IMAGE_SIZE, IMAGE_SIZE],
    "cell": CELL,
    "gridder": "wproject",
    "wprojplanes": -1,
    "weighting": "natural",
    "pblimit": -1.0,
    "interactive": False,
    "savemodel": "none",
}

tclean(imagename=str(dirty), niter=0, **common)
statistics = imstat(imagename=str(dirty) + ".residual")
dirty_rms_jy_beam = _number(statistics["rms"])
threshold_jy = float(
    os.environ.get("SL1MJAX_3C391_CLEAN_THRESHOLD_JY", "0.001")
)
_export(dirty, "residual")
_export(dirty, "psf")

tclean(
    imagename=str(clean),
    niter=20000,
    threshold=f"{threshold_jy}Jy",
    deconvolver="multiscale",
    scales=[0, 3, 9, 27],
    usemask="auto-multithresh",
    sidelobethreshold=2.0,
    noisethreshold=4.25,
    lownoisethreshold=1.5,
    minbeamfrac=0.3,
    growiterations=75,
    negativethreshold=0.0,
    **common,
)
for suffix in ("image", "model", "residual", "psf", "mask"):
    _export(clean, suffix)

clean_statistics = imstat(imagename=str(clean) + ".residual")
tclean(
    imagename=str(hogbom),
    niter=20000,
    threshold=f"{threshold_jy}Jy",
    deconvolver="hogbom",
    usemask="user",
    **common,
)
for suffix in ("image", "model", "residual", "psf"):
    _export(hogbom, suffix)
hogbom_statistics = imstat(imagename=str(hogbom) + ".residual")
result = {
    "schema_version": 1,
    "measurement_set": MEASUREMENT_SET.name,
    "field": FIELD,
    "data_column": "CORRECTED_DATA",
    "image_size": IMAGE_SIZE,
    "cell": CELL,
    "gridder": "wproject",
    "weighting": "natural",
    "deconvolver": "multiscale",
    "scales_pixels": [0, 3, 9, 27],
    "niter": 20000,
    "threshold_jy": threshold_jy,
    "dirty_rms_jy_beam": dirty_rms_jy_beam,
    "clean_residual_rms_jy_beam": _number(clean_statistics["rms"]),
    "hogbom_residual_rms_jy_beam": _number(hogbom_statistics["rms"]),
    "hogbom_residual_max_jy_beam": _number(hogbom_statistics["max"]),
}
(OUTPUT / "result.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2, sort_keys=True))
