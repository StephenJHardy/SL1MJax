"""Export compact CASA VLA configuration sampling statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.rime import SPEED_OF_LIGHT_M_S

ROOT = Path("outputs/casa_sampling")
OUTPUT = Path("tests/fixtures/casa_vla_sampling.json")
statistics: dict[str, Any] = {"casa_version": "6.7.6.14", "configurations": {}}
for configuration in ("A", "C", "D"):
    lower = configuration.lower()
    block = extract_measurement_set(
        ROOT / lower / f"{lower}.vla.{lower}.ms", data_column="DATA"
    ).blocks[0]
    uv_distance_m = np.linalg.norm(block.uvw_m[:, :2], axis=1)
    p99 = float(np.percentile(uv_distance_m, 99))
    wavelength_m = SPEED_OF_LIGHT_M_S / float(np.mean(block.frequency_hz))
    statistics["configurations"][configuration] = {
        "rows": block.shape[0],
        "uv_distance_m_p50": float(np.percentile(uv_distance_m, 50)),
        "uv_distance_m_p99": p99,
        "uv_distance_m_max": float(np.max(uv_distance_m)),
        "nominal_resolution_arcsec_p99": float(
            wavelength_m / p99 * np.rad2deg(1) * 3600
        ),
    }

OUTPUT.write_text(
    json.dumps(statistics, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(OUTPUT)
