"""Prove THOL0001 DIRECTION-TARGET=POINTING_OFFSET. SPW 5 stays sealed."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_diagonal_correction import refuse_spw5
from sl1mjax.holography_holoraster_coordinates import holoraster_ms_geometry_oracle
from sl1mjax.holography_ms import read_pointing_table_from_ms

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
PRODUCT_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_geometry_oracle_v1"
)
FROZEN_NAMES = (
    "holoraster_cassbeam_comparison",
    "holoraster_cassbeam_correction",
    "holoraster_physical_squint_width_v1",
    "holoraster_spatial_convention_v1",
    "vla_c_band_beam_validation_v1",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=PRODUCT_DIR)
    arguments = parser.parse_args()
    refuse_spw5(spectral_window_id=4, opened=False)
    output_dir = arguments.output_dir.resolve()
    if output_dir.name in FROZEN_NAMES:
        raise RuntimeError(f"refusing to write into a frozen product path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    table = read_pointing_table_from_ms(arguments.measurement_set)
    # AZELGEO columns are used unchanged; J2000 would need the field centre.
    phase_centre = (0.0, 0.0)
    report = holoraster_ms_geometry_oracle(table, phase_centre)
    payload = {
        "artifact": "holoraster_geometry_oracle_v1",
        "development_only": True,
        "spw5_closed": True,
        "measurement_set": str(arguments.measurement_set),
        "environment": {"hostname": platform.node(), "python": sys.version},
        "oracle": report,
    }
    write_json(payload, output_dir / "report.json")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["passes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
