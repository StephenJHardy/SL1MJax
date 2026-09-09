"""Numerical convergence for the 3C391 imaging nodes.

Compares a production g1024/p32 plane with a finer aperture or angular
sample after interpolating onto the production sky nodes. Changing
gridsize or pixelsperbeam must not be treated as the same field or the
same image sampling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.evla_c_survey_beam import IMAGING_NODE_MHZ, NATIVE_3C391_MHZ
from sl1mjax.evla_c_survey_compare import FIXED_MAIN_LOBE_ARCMIN, FIXED_MID_ARCMIN

SPEED_OF_LIGHT_M_S = 299_792_458.0
MAIN_LOBE_REL_L2_MAX = 0.002
MID_REL_L2_MAX = 0.01
VISIBILITY_MAIN_REL_L2_MAX = 0.002
# Keep the locked g1024/p32 catalog if visibilities on the same sky nodes stay
# inside the survey 1% main-lobe cut. The 0.002 Jones target is numerical
# qualification, not a reason to change image sampling.
VISIBILITY_KEEP_PRODUCTION_MAX = 0.01


def native_3c391_mhz() -> tuple[int, ...]:
    return NATIVE_3C391_MHZ


def imaging_node_mhz() -> tuple[int, ...]:
    return IMAGING_NODE_MHZ


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.split("%", 1)[0].strip()
        if stripped and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def geometry_radius_m(path: Path) -> float:
    radii = [
        float(line.split()[0])
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not radii or radii[-1] <= 0.0:
        raise ValueError("geometry must end with a positive primary radius")
    return radii[-1]


def pixel_scale_rad(params: Mapping[str, str], geometry: Path) -> float:
    gridsize = int(float(params["gridsize"]))
    half = gridsize // 2
    if half % 2:
        half += 1
    aperture_n = 2 * half
    spacing_m = geometry_radius_m(geometry) / half
    wavelength_m = SPEED_OF_LIGHT_M_S / (float(params["freq"]) * 1.0e9)
    return wavelength_m / (float(params["pixelsperbeam"]) * aperture_n * spacing_m)


def raster_description(params: Mapping[str, str], geometry: Path, *, size: int) -> dict[str, object]:
    scale = pixel_scale_rad(params, geometry)
    half = (size - 1) / 2.0
    extent = float(np.rad2deg(scale * half) * 60.0)
    return {
        "frequency_hz": float(params["freq"]) * 1.0e9,
        "gridsize": int(float(params["gridsize"])),
        "pixelsperbeam": int(float(params["pixelsperbeam"])),
        "raster_size": int(size),
        "pixel_scale_arcmin": float(np.rad2deg(scale) * 60.0),
        "half_extent_arcmin": extent,
    }


def bilinear_jones(
    plane: NDArray[np.complex128],
    l_axis: NDArray[np.float64],
    m_axis: NDArray[np.float64],
    l: NDArray[np.float64],
    m: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
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


def radius_region_masks(l_rad: ArrayLike, m_rad: ArrayLike) -> dict[str, NDArray[np.bool_]]:
    ll, mm = np.meshgrid(np.asarray(l_rad, dtype=np.float64), np.asarray(m_rad, dtype=np.float64))
    radius = np.hypot(ll, mm) * (180.0 / np.pi) * 60.0
    return {
        "main_lobe": radius < FIXED_MAIN_LOBE_ARCMIN,
        "mid": (radius >= FIXED_MAIN_LOBE_ARCMIN) & (radius < FIXED_MID_ARCMIN),
        "outer_diagnostic": radius >= FIXED_MID_ARCMIN,
    }


def _element_metrics(reference: NDArray, candidate: NDArray) -> dict[str, float]:
    delta = candidate - reference
    norm = float(np.linalg.norm(reference))
    return {
        "relative_l2": float(np.linalg.norm(delta) / norm) if norm else float("nan"),
        "rms_abs": float(np.sqrt(np.mean(np.abs(delta) ** 2))),
        "p95_abs": float(np.quantile(np.abs(delta), 0.95)) if delta.size else float("nan"),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else float("nan"),
        "n": int(delta.size),
    }


def stokes_i_visibility(jones_p: NDArray, jones_q: NDArray) -> NDArray:
    sky = np.eye(2, dtype=np.complex128)
    return np.einsum("...ij,jk,...lk->...il", jones_p, sky, np.conjugate(jones_q))


def compare_on_production_nodes(
    production: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Interpolate candidate Jones onto the production (l, m) nodes."""

    prod_l = np.asarray(production["l_rad"], dtype=np.float64)
    prod_m = np.asarray(production["m_rad"], dtype=np.float64)
    coarse_l, coarse_m = np.meshgrid(prod_l, prod_m)
    sampled, valid = bilinear_jones(
        np.asarray(candidate["jones"]),
        np.asarray(candidate["l_rad"], dtype=np.float64),
        np.asarray(candidate["m_rad"], dtype=np.float64),
        coarse_l.ravel(),
        coarse_m.ravel(),
    )
    sampled = sampled.reshape(*coarse_l.shape, 2, 2)
    valid = valid.reshape(coarse_l.shape)
    production_j = np.asarray(production["jones"])
    regions = radius_region_masks(prod_l, prod_m)
    elements = ((0, 0, "RR"), (0, 1, "RL"), (1, 0, "LR"), (1, 1, "LL"))
    jones_regions: dict[str, object] = {}
    vis_regions: dict[str, object] = {}
    vis_prod = stokes_i_visibility(production_j, production_j)
    vis_cand = stokes_i_visibility(sampled, sampled)
    for name, mask in regions.items():
        keep = mask & valid
        jones_regions[name] = {
            "n": int(np.sum(keep)),
            "elements": {
                label: _element_metrics(production_j[..., row, col][keep], sampled[..., row, col][keep])
                for row, col, label in elements
            },
        }
        vis_regions[name] = {
            "n": int(np.sum(keep)),
            "elements": {
                label: _element_metrics(vis_prod[..., row, col][keep], vis_cand[..., row, col][keep])
                for row, col, label in elements
            },
        }
    return {
        "common_nodes": int(np.sum(valid)),
        "production_nodes": int(valid.size),
        "coverage_fraction": float(np.mean(valid)),
        "jones": jones_regions,
        "visibilities": vis_regions,
    }


def classify_imaging_convergence(comparison: Mapping[str, object]) -> dict[str, object]:
    jones = dict(comparison.get("jones") or {})
    vis = dict(comparison.get("visibilities") or {})

    def _worst(payload: Mapping[str, object], region: str) -> float:
        elements = ((payload.get(region) or {}).get("elements") or {})
        values = [float(item.get("relative_l2", float("nan"))) for item in elements.values()]
        finite = [value for value in values if np.isfinite(value)]
        return max(finite) if finite else float("nan")

    main_j = _worst(jones, "main_lobe")
    mid_j = _worst(jones, "mid")
    main_v = _worst(vis, "main_lobe")
    mid_v = _worst(vis, "mid")
    main_n = int(((jones.get("main_lobe") or {}).get("n")) or 0)
    jones_ok = bool(np.isfinite(main_j) and main_j <= MAIN_LOBE_REL_L2_MAX and main_n > 0)
    vis_ok = bool(np.isfinite(main_v) and main_v <= VISIBILITY_MAIN_REL_L2_MAX and main_n > 0)
    keep_production = bool(
        np.isfinite(main_v) and main_v <= VISIBILITY_KEEP_PRODUCTION_MAX and main_n > 0
    )
    mid_ok = bool(np.isfinite(mid_j) and mid_j <= MID_REL_L2_MAX)
    return {
        "main_lobe_passed": jones_ok and vis_ok,
        "mid_passed": mid_ok,
        "outer_diagnostic": True,
        "numerically_qualified_0p002": jones_ok and vis_ok,
        "keep_production_g1024_p32": keep_production,
        "imaging_prerequisite_passed": keep_production,
        "main_lobe_nodes": main_n,
        "main_lobe_jones_rel_l2": main_j,
        "mid_jones_rel_l2": mid_j,
        "main_lobe_visibility_rel_l2": main_v,
        "mid_visibility_rel_l2": mid_v,
        "limits": {
            "main_lobe_jones": MAIN_LOBE_REL_L2_MAX,
            "mid_jones": MID_REL_L2_MAX,
            "main_lobe_visibility": VISIBILITY_MAIN_REL_L2_MAX,
            "keep_production_visibility": VISIBILITY_KEEP_PRODUCTION_MAX,
        },
    }


def assert_sampling_change(
    production: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    kind: str,
) -> dict[str, object]:
    """Refuse to treat a different field or pixel scale as the same grid."""

    prod_scale = float(production["pixel_scale_arcmin"])
    cand_scale = float(candidate["pixel_scale_arcmin"])
    prod_ext = float(production["half_extent_arcmin"])
    cand_ext = float(candidate["half_extent_arcmin"])
    same_scale = abs(prod_scale - cand_scale) <= 1.0e-6 * max(prod_scale, 1.0e-12)
    same_extent = abs(prod_ext - cand_ext) <= 0.05 * prod_ext
    if kind == "aperture" and not same_scale:
        raise ValueError("aperture refinement silently changed image sampling")
    if kind == "angular" and same_scale:
        raise ValueError("angular refinement did not change image sampling")
    return {
        "kind": kind,
        "same_pixel_scale": same_scale,
        "same_half_extent": same_extent,
        "production_pixel_scale_arcmin": prod_scale,
        "candidate_pixel_scale_arcmin": cand_scale,
        "production_half_extent_arcmin": prod_ext,
        "candidate_half_extent_arcmin": cand_ext,
    }


def unsupported_direction_account(valid: ArrayLike) -> dict[str, object]:
    """Count supported vs unsupported beam samples. Missing is not zero."""

    mask = np.asarray(valid, dtype=bool)
    n = int(mask.size)
    n_ok = int(np.sum(mask))
    return {
        "n_samples": n,
        "n_supported": n_ok,
        "n_unsupported": n - n_ok,
        "unsupported_fraction": float((n - n_ok) / n) if n else float("nan"),
        "missing_treated_as_zero": False,
    }
