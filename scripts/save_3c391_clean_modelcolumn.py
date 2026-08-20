"""Write an existing CASA CLEAN model into MODEL_DATA without re-cleaning."""

from __future__ import annotations

import os
from pathlib import Path

from casatasks import tclean

MEASUREMENT_SET = Path(os.environ["SL1MJAX_3C391_MS"]).resolve()
IMAGE_NAME = Path(
    os.environ.get(
        "SL1MJAX_3C391_CLEAN_IMAGE",
        "outputs/3c391_casa_imaging_128/3c391_c1_multiscale",
    )
).resolve()
FIELD = os.environ.get("SL1MJAX_3C391_IMAGE_FIELD", "3C391 C1")

tclean(
    vis=str(MEASUREMENT_SET),
    imagename=str(IMAGE_NAME),
    field=FIELD,
    spw="0",
    datacolumn="corrected",
    specmode="mfs",
    stokes="I",
    imsize=[128, 128],
    cell="4arcsec",
    gridder="wproject",
    wprojplanes=-1,
    weighting="natural",
    pblimit=-1.0,
    interactive=False,
    niter=0,
    restart=True,
    calcres=False,
    calcpsf=False,
    savemodel="modelcolumn",
)
print(f"wrote MODEL_DATA from {IMAGE_NAME} into {MEASUREMENT_SET}")
