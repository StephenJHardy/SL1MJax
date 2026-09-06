"""Validate and checksum a generated high-resolution CASSBEAM model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("%", 1)[0].strip()
        if stripped and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: stream.read(1024 * 1024), b""))


def _convergence_passed(report: dict[str, object]) -> tuple[bool, dict[str, float]]:
    limits = {
        "main_lobe_20percent": 0.002,
        "through_1percent_power": 0.005,
        "full_overlap": 0.01,
    }
    passed = True
    maxima: dict[str, float] = {}
    regions = report["regions"]
    for region, limit in limits.items():
        elements = regions[region]["elements"]
        maximum = max(float(metrics["relative_l2"]) for metrics in elements.values())
        maxima[region] = maximum
        passed &= maximum <= limit
    return passed, maxima


def finalize(root: Path) -> dict[str, object]:
    frequencies = [*range(4500, 4628, 2), 4692]
    expected_rows = 513 * 513
    planes: list[dict[str, object]] = []
    files: dict[str, str] = {}
    for frequency_mhz in frequencies:
        group = "reference" if frequency_mhz == 4692 else "spw4"
        prefix = root / group / f"vla-cband-{frequency_mhz}-g1024-p32"
        data_path = prefix.with_suffix(".jones.dat")
        params_path = prefix.with_suffix(".params")
        if not data_path.is_file() or not params_path.is_file():
            raise FileNotFoundError(prefix)
        params = _params(params_path)
        if int(float(params["gridsize"])) != 1024:
            raise ValueError(f"wrong gridsize in {params_path}")
        if int(float(params["pixelsperbeam"])) != 32:
            raise ValueError(f"wrong pixelsperbeam in {params_path}")
        if abs(float(params["freq"]) * 1000.0 - frequency_mhz) > 1.0e-6:
            raise ValueError(f"wrong frequency in {params_path}")
        rows = _line_count(data_path)
        if rows != expected_rows:
            raise ValueError(f"{data_path} has {rows} rows, expected {expected_rows}")
        for path in (data_path, params_path):
            files[str(path.relative_to(root))] = _sha256(path)
        planes.append(
            {
                "frequency_mhz": frequency_mhz,
                "group": group,
                "shape": [513, 513, 2, 2],
                "native_columns": [
                    "Re_RR",
                    "Im_RR",
                    "Re_LR",
                    "Im_LR",
                    "Re_RL",
                    "Im_RL",
                    "Re_LL",
                    "Im_LL",
                ],
                "data": str(data_path.relative_to(root)),
                "params": str(params_path.relative_to(root)),
            }
        )

    for relative in ("reference/base.in", "reference/vla_geom"):
        files[relative] = _sha256(root / relative)
    convergence_path = root / "convergence/g1024-p32_vs_g2048-p64.json"
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    convergence_passed, convergence_maxima = _convergence_passed(convergence)
    if not convergence_passed:
        raise ValueError("1024/32 candidate did not pass convergence limits")
    files[str(convergence_path.relative_to(root))] = _sha256(convergence_path)
    lower_path = root / "convergence/g512-p16_vs_g1024-p32.json"
    files[str(lower_path.relative_to(root))] = _sha256(lower_path)
    return {
        "schema_version": 1,
        "model_id": "cassbeam_vla_cband_full_jones_g1024_p32_spw4_v1",
        "artifact_kind": "electromagnetic_voltage_jones",
        "generator": {
            "host": "bacchus",
            "package": "cassbeam 1.1-4build2",
            "reported_version": "1.0",
            "binary": "/usr/bin/cassbeam",
            "binary_sha256": "61a66c263f478cd1ec8bf0da9773ded3b6c493ef28bff4241d86c7e2c5d25b64",
            "arguments": {
                "gridsize": 1024,
                "pixelsperbeam": 32,
                "compute": "jp",
            },
        },
        "frequency_policy": {
            "spw4": "one exact plane per native 2 MHz channel",
            "spw4_frequency_mhz": list(range(4500, 4628, 2)),
            "additional_reference_frequency_mhz": [4692],
            "no_frequency_interpolation_used_in_generation": True,
        },
        "raster": {
            "shape": [513, 513, 2, 2],
            "approximately_pixels_across_fwhm": 32,
            "approximately_half_extent_arcmin_at_4564_mhz": 72.26,
            "stored_quantity": "native unnormalized CASSBEAM transmit Jones",
            "science_normalization": "inv(E(0)) @ E(s)",
            "origin": "dephased FFT DC after even-N l reflection",
        },
        "convergence": {
            "numerically_converged": True,
            "candidate": "gridsize=1024,pixelsperbeam=32",
            "reference": "gridsize=2048,pixelsperbeam=64",
            "same_approximately_72_arcmin_half_extent": True,
            "maximum_element_relative_l2_by_region": convergence_maxima,
            "limits": {
                "main_lobe_20percent": 0.002,
                "through_1percent_power": 0.005,
                "full_overlap": 0.01,
            },
            "report": str(convergence_path.relative_to(root)),
            "lower_resolution_report": str(lower_path.relative_to(root)),
        },
        "scientific_status": {
            "numerically_converged": True,
            "full_jones_frozen": False,
            "production_accepted": False,
            "reason": (
                "Numerical generation convergence does not establish receive/transmit, "
                "axis, hand, or empirical holography validity."
            ),
        },
        "planes": planes,
        "files_sha256": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    manifest = finalize(arguments.root)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output = arguments.output or arguments.root / "manifest.json"
    output.write_text(encoded, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
