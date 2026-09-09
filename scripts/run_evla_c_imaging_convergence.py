#!/usr/bin/env python3
"""Generate finer EVLA-C planes at 3C391 nodes and score imaging convergence.

Writes a separate workspace. Does not overwrite the production survey catalog
or frozen CASSBEAM archives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sl1mjax.evla_c_diagonal_survey import refuse_frozen_write, write_json_atomic
from sl1mjax.evla_c_imaging_convergence import (
    assert_sampling_change,
    classify_imaging_convergence,
    compare_on_production_nodes,
    imaging_node_mhz,
    parse_params,
    pixel_scale_rad,
    raster_description,
)

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_cassbeam_highres_model import generate


def _load_plane(prefix: Path, geometry: Path) -> dict[str, object]:
    params = parse_params(prefix.with_suffix(".params"))
    raw = np.loadtxt(prefix.with_suffix(".jones.dat"), dtype=np.float64)
    size = int(round(np.sqrt(raw.shape[0])))
    jones = np.empty((size, size, 2, 2), dtype=np.complex128)
    jones[..., 0, 0] = (raw[:, 0] + 1j * raw[:, 1]).reshape(size, size)
    jones[..., 1, 0] = (raw[:, 2] + 1j * raw[:, 3]).reshape(size, size)
    jones[..., 0, 1] = (raw[:, 4] + 1j * raw[:, 5]).reshape(size, size)
    jones[..., 1, 1] = (raw[:, 6] + 1j * raw[:, 7]).reshape(size, size)
    gridsize = int(float(params["gridsize"]))
    half = gridsize // 2
    if half % 2:
        half += 1
    aperture_n = 2 * half
    crop_start = aperture_n // 4
    l_origin = aperture_n // 2 - 1 - crop_start
    m_origin = aperture_n // 2 - crop_start
    scale = pixel_scale_rad(params, geometry)
    l_rad = (np.arange(size, dtype=np.float64) - l_origin) * scale
    m_rad = (np.arange(size, dtype=np.float64) - m_origin) * scale
    center = jones[m_origin, l_origin]
    normalized = np.einsum("ij,...jk->...ik", np.linalg.inv(center), jones)
    description = raster_description(params, geometry, size=size)
    return {
        "jones": normalized,
        "l_rad": l_rad,
        "m_rad": m_rad,
        **description,
    }


def _find_production(beam_root: Path, mhz: int) -> Path:
    matches = list(Path(beam_root).glob(f"**/evla-cband-{mhz}-g1024-p32.params"))
    if not matches:
        raise FileNotFoundError(f"missing production plane {mhz} MHz")
    return matches[0].with_suffix("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("/usr/bin/cassbeam"))
    parser.add_argument("--frequencies-mhz", type=str, default="")
    arguments = parser.parse_args()
    production_root = arguments.production_root
    workspace = arguments.workspace
    refuse_frozen_write(workspace)
    refuse_frozen_write(production_root)
    geometry = production_root / "reference" / "vla_geom"
    base_input = production_root / "reference" / "base.in"
    frequencies = (
        [int(item) for item in arguments.frequencies_mhz.split(",") if item.strip()]
        if arguments.frequencies_mhz.strip()
        else list(imaging_node_mhz())
    )
    reports: list[dict[str, object]] = []
    for mhz in frequencies:
        production_prefix = _find_production(production_root, mhz)
        for gridsize, pixels, kind in ((2048, 32, "aperture"), (1024, 64, "angular")):
            generate(
                binary=arguments.binary,
                base_input=base_input,
                geometry=geometry,
                output_dir=workspace,
                frequencies_mhz=[mhz],
                reference_frequencies_mhz=set(),
                gridsize=gridsize,
                pixelsperbeam=pixels,
                name_prefix="evla-cband",
                feedtaper_mode="evla_c",
            )
            candidate_prefix = workspace / "spw4" / f"evla-cband-{mhz}-g{gridsize}-p{pixels}"
            production = _load_plane(production_prefix, geometry)
            candidate = _load_plane(candidate_prefix, geometry)
            sampling = assert_sampling_change(production, candidate, kind=kind)
            compared = compare_on_production_nodes(production, candidate)
            classified = classify_imaging_convergence(compared)
            reports.append(
                {
                    "frequency_mhz": mhz,
                    "comparison": kind,
                    "production": {
                        key: production[key]
                        for key in (
                            "frequency_hz",
                            "gridsize",
                            "pixelsperbeam",
                            "raster_size",
                            "pixel_scale_arcmin",
                            "half_extent_arcmin",
                        )
                    },
                    "candidate": {
                        key: candidate[key]
                        for key in (
                            "frequency_hz",
                            "gridsize",
                            "pixelsperbeam",
                            "raster_size",
                            "pixel_scale_arcmin",
                            "half_extent_arcmin",
                        )
                    },
                    "sampling": sampling,
                    **compared,
                    "classification": classified,
                }
            )
            print(
                f"{mhz} {kind} main={classified['main_lobe_jones_rel_l2']:.4g} "
                f"vis={classified['main_lobe_visibility_rel_l2']:.4g} "
                f"pass={classified['imaging_prerequisite_passed']}",
                flush=True,
            )
    payload = {
        "schema_version": 1,
        "purpose": "3C391 imaging-frequency numerical convergence",
        "production_root": str(production_root),
        "workspace": str(workspace),
        "imaging_prerequisite_passed": all(
            bool(row["classification"]["keep_production_g1024_p32"]) for row in reports
        ),
        "numerically_qualified_0p002": all(
            bool(row["classification"]["numerically_qualified_0p002"]) for row in reports
        ),
        "keep_production_g1024_p32": all(
            bool(row["classification"]["keep_production_g1024_p32"]) for row in reports
        ),
        "slots": reports,
    }
    path = write_json_atomic(workspace / "imaging_convergence.json", payload)
    print(path, payload["imaging_prerequisite_passed"])


if __name__ == "__main__":
    main()
