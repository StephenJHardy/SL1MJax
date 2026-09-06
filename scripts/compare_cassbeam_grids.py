"""Compare two nested CASSBEAM Jones rasters on their physical sky grid.

This is a generation-time diagnostic.  It deliberately does not use the
runtime C-band artifact, whose current frequency and support policy are
independent of a candidate high-resolution raster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0


def _params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("%", 1)[0].strip()
        if stripped and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _geometry_radius_m(path: Path) -> float:
    radii = [
        float(line.split()[0])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not radii or radii[-1] <= 0.0:
        raise ValueError("geometry must end with a positive primary radius")
    return radii[-1]


def _load(prefix: Path, geometry: Path) -> dict[str, object]:
    params = _params(prefix.with_suffix(".params"))
    raw = np.loadtxt(prefix.with_suffix(".jones.dat"), dtype=np.float64)
    size = int(round(np.sqrt(raw.shape[0])))
    if raw.shape != (size * size, 8) or size % 2 != 1:
        raise ValueError(f"{prefix} is not an odd square eight-column raster")
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
    if size != aperture_n - 2 * crop_start + 1:
        raise ValueError("raster size disagrees with CASSBEAM crop convention")
    spacing_m = _geometry_radius_m(geometry) / half
    wavelength_m = SPEED_OF_LIGHT_M_S / (float(params["freq"]) * 1.0e9)
    scale_rad = wavelength_m / (float(params["pixelsperbeam"]) * aperture_n * spacing_m)
    l_rad = (np.arange(size, dtype=np.float64) - l_origin) * scale_rad
    m_rad = (np.arange(size, dtype=np.float64) - m_origin) * scale_rad
    center = jones[m_origin, l_origin]
    normalized = np.einsum("ij,...jk->...ik", np.linalg.inv(center), jones)
    return {
        "jones": normalized,
        "l_rad": l_rad,
        "m_rad": m_rad,
        "scale_rad": scale_rad,
        "origin": (m_origin, l_origin),
        "params": params,
    }


def _bilinear(
    plane: np.ndarray,
    l_axis: np.ndarray,
    m_axis: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ii = np.interp(l, l_axis, np.arange(l_axis.size), left=np.nan, right=np.nan)
    jj = np.interp(m, m_axis, np.arange(m_axis.size), left=np.nan, right=np.nan)
    valid = np.isfinite(ii) & np.isfinite(jj)
    i0 = np.clip(np.floor(np.where(valid, ii, 0.0)).astype(int), 0, l_axis.size - 2)
    j0 = np.clip(np.floor(np.where(valid, jj, 0.0)).astype(int), 0, m_axis.size - 2)
    di = np.where(valid, ii - i0, 0.0)[:, None, None]
    dj = np.where(valid, jj - j0, 0.0)[:, None, None]
    out = (1.0 - dj) * ((1.0 - di) * plane[j0, i0] + di * plane[j0, i0 + 1]) + dj * (
        (1.0 - di) * plane[j0 + 1, i0] + di * plane[j0 + 1, i0 + 1]
    )
    return out, valid


def _element_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = candidate - reference
    norm = float(np.linalg.norm(reference))
    return {
        "relative_l2": float(np.linalg.norm(delta) / norm) if norm else float("nan"),
        "rms_abs": float(np.sqrt(np.mean(np.abs(delta) ** 2))),
        "p95_abs": float(np.quantile(np.abs(delta), 0.95)),
        "max_abs": float(np.max(np.abs(delta))),
        "reference_p90_abs": float(np.quantile(np.abs(reference), 0.90)),
    }


def compare(coarse_prefix: Path, fine_prefix: Path, geometry: Path) -> dict[str, object]:
    coarse = _load(coarse_prefix, geometry)
    fine = _load(fine_prefix, geometry)
    coarse_l, coarse_m = np.meshgrid(coarse["l_rad"], coarse["m_rad"])
    sampled, valid = _bilinear(
        fine["jones"],
        fine["l_rad"],
        fine["m_rad"],
        coarse_l.ravel(),
        coarse_m.ravel(),
    )
    sampled = sampled.reshape(*coarse_l.shape, 2, 2)
    valid = valid.reshape(coarse_l.shape)
    candidate = coarse["jones"]
    power = 0.5 * (np.abs(sampled[..., 0, 0]) ** 2 + np.abs(sampled[..., 1, 1]) ** 2)
    peak = float(np.max(power[valid]))
    masks = {
        "main_lobe_20percent": valid & (power >= 0.20 * peak),
        "through_1percent_power": valid & (power >= 0.01 * peak),
        "full_overlap": valid,
    }
    elements = ((0, 0, "RR"), (0, 1, "RL"), (1, 0, "LR"), (1, 1, "LL"))
    regions: dict[str, object] = {}
    for name, mask in masks.items():
        regions[name] = {
            "n": int(np.sum(mask)),
            "elements": {
                label: _element_metrics(
                    sampled[..., row, col][mask], candidate[..., row, col][mask]
                )
                for row, col, label in elements
            },
        }

    def description(item: dict[str, object]) -> dict[str, object]:
        scale = float(item["scale_rad"])
        return {
            "shape": list(np.asarray(item["jones"]).shape),
            "frequency_hz": float(item["params"]["freq"]) * 1.0e9,
            "gridsize": int(float(item["params"]["gridsize"])),
            "pixelsperbeam": int(float(item["params"]["pixelsperbeam"])),
            "pixel_scale_arcmin": float(np.rad2deg(scale) * 60.0),
            "l_extent_arcmin": [float(np.rad2deg(x) * 60.0) for x in item["l_rad"][[0, -1]]],
            "m_extent_arcmin": [float(np.rad2deg(x) * 60.0) for x in item["m_rad"][[0, -1]]],
            "origin_index_m_l": list(item["origin"]),
        }

    return {
        "schema_version": 1,
        "normalization": "E_norm(s) = inv(E(0)) @ E(s)",
        "coarse": description(coarse),
        "fine": description(fine),
        "regions": regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coarse_prefix", type=Path)
    parser.add_argument("fine_prefix", type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = compare(arguments.coarse_prefix, arguments.fine_prefix, arguments.geometry)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
