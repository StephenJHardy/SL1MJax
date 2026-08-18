"""Export compact deterministic visibility subsets from CASA fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.data.ms import extract_measurement_set

CASES = ("center", "east", "north", "diagonal", "gaussian")
ROOT = Path(os.environ.get("SL1MJAX_CASA_CASE_ROOT", "outputs/casa_cases"))
OUTPUT = Path(
    os.environ.get("SL1MJAX_CASA_GOLDEN", "tests/fixtures/casa_vla_golden.npz")
)

payload: dict[str, Any] = {}
metadata: dict[str, Any] = {
    "schema_version": 1,
    "casa_version": "6.7.6.14",
    "cases": {},
}
for case in CASES:
    measurement_set = ROOT / case / f"{case}.vla.d.ms"
    block = extract_measurement_set(measurement_set, data_column="DATA").blocks[0]
    rows = np.linspace(0, block.shape[0] - 1, 64, dtype=np.int64)
    prefix = f"{case}_"
    payload[prefix + "uvw_m"] = block.uvw_m[rows]
    payload[prefix + "frequency_hz"] = block.frequency_hz
    payload[prefix + "visibility"] = block.visibility[rows]
    payload[prefix + "weight"] = block.weight[rows]
    payload[prefix + "flag"] = block.flag[rows]
    payload[prefix + "antenna1"] = block.antenna1[rows]
    payload[prefix + "antenna2"] = block.antenna2[rows]
    truth_path = ROOT / f"{case}.truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    l, m, n = radec_to_lmn(
        *block.phase_centre_rad,
        [source["ra_rad"] for source in truth["sources"]],
        [source["dec_rad"] for source in truth["sources"]],
    )
    for source, source_l, source_m, source_n in zip(
        truth["sources"], l, m, n, strict=True
    ):
        source.update(l=float(source_l), m=float(source_m), n=float(source_n))
    metadata["cases"][case] = {
        "truth": truth,
        "correlations": [value.value for value in block.correlations],
        "receptor_basis": block.receptor_basis.value,
        "phase_centre_rad": list(block.phase_centre_rad),
        "selected_rows": rows.tolist(),
        "effects": {"noise": "none", "calibration": "identity"},
    }

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("wb") as stream:
    np.savez_compressed(stream, **payload)
OUTPUT.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(OUTPUT)
