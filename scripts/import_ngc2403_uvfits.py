"""Run with CASA to import the local NGC2403 UVFITS validation dataset."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from casatasks import importuvfits

INPUT = Path(
    os.environ.get("SL1MJAX_NGC2403_UVFITS", "outputs/n2403.uvfits")
).resolve()
OUTPUT = Path(
    os.environ.get("SL1MJAX_NGC2403_MS", "outputs/n2403.uvfits.ms")
).resolve()

if not INPUT.is_file():
    raise FileNotFoundError(INPUT)
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

importuvfits(fitsfile=str(INPUT), vis=str(OUTPUT))
print(OUTPUT)
