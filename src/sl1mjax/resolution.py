"""Synthesized-beam estimates and resolution-aware quadtree depth limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import stokes_i_to_correlations

SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class SynthesizedBeamEstimate:
    """Gaussian beam inferred from the weighted PSF curvature at its peak."""

    major_fwhm_rad: float
    minor_fwhm_rad: float
    position_angle_rad: float
    method: str = "weighted_uv_second_moment"


def estimate_synthesized_beam(
    blocks: tuple[VisibilityBlock, ...],
    masks: tuple[np.ndarray, ...] | None = None,
) -> SynthesizedBeamEstimate:
    """Estimate a natural-weight Gaussian beam without forming a PSF image.

    The normalized dirty beam has local curvature ``4 pi^2 E[(u,v)(u,v)^T]``.
    Matching that curvature to a Gaussian gives its major and minor FWHM. The
    calculation uses sampling coordinates and weights only, never visibility
    values, so it cannot leak held-out sky measurements into model selection.
    """

    if not blocks:
        raise ValueError("blocks must contain at least one visibility block")
    selected_masks = (
        tuple(np.ones(block.shape, dtype=bool) for block in blocks) if masks is None else masks
    )
    if len(selected_masks) != len(blocks):
        raise ValueError("masks must contain one mask per visibility block")

    moment = np.zeros((2, 2), dtype=np.float64)
    weight_sum = 0.0
    for index, (block, mask) in enumerate(zip(blocks, selected_masks, strict=True)):
        if mask.shape != block.shape:
            raise ValueError(f"masks[{index}] must match its visibility block")
        factors = np.asarray(
            stokes_i_to_correlations(1.0, block.correlations),
            dtype=np.complex128,
        )
        sample_weight = np.sum(
            np.where(block.active & mask, block.weight, 0.0) * np.abs(factors)[None, None, :] ** 2,
            axis=2,
        )
        uv_wavelengths = (
            block.uvw_m[:, None, :2] * block.frequency_hz[None, :, None] / SPEED_OF_LIGHT_M_S
        )
        moment += np.einsum(
            "rc,rci,rcj->ij",
            sample_weight,
            uv_wavelengths,
            uv_wavelengths,
        )
        weight_sum += float(np.sum(sample_weight))
    if weight_sum <= 0:
        raise ValueError("beam estimate requires active positive-weight samples")

    eigenvalues, eigenvectors = np.linalg.eigh(moment / weight_sum)
    maximum_eigenvalue = float(eigenvalues[-1])
    if not np.isfinite(maximum_eigenvalue) or maximum_eigenvalue <= 0:
        raise ValueError("beam estimate requires a non-zero projected baseline")
    fwhm_factor = np.sqrt(2.0 * np.log(2.0)) / np.pi
    minor_fwhm = fwhm_factor / np.sqrt(maximum_eigenvalue)
    minimum_eigenvalue = float(eigenvalues[0])
    major_fwhm = (
        np.inf
        if minimum_eigenvalue <= np.finfo(np.float64).eps * maximum_eigenvalue
        else fwhm_factor / np.sqrt(minimum_eigenvalue)
    )
    major_axis = eigenvectors[:, 0]
    position_angle = float(np.arctan2(major_axis[0], major_axis[1]))
    return SynthesizedBeamEstimate(
        major_fwhm_rad=float(major_fwhm),
        minor_fwhm_rad=float(minor_fwhm),
        position_angle_rad=position_angle,
    )


def resolution_limited_max_depth(
    root_pixel_size_rad: float,
    minor_fwhm_rad: float,
    *,
    maximum_pixels_per_beam: float = 5.0,
) -> int:
    """Return the deepest dyadic level that does not oversample the beam.

    A level-``d`` leaf has width ``root_pixel_size_rad / 2**d``. The returned
    depth is the largest integer for which the minor-axis beam spans no more
    than ``maximum_pixels_per_beam`` leaves. A result of zero means the root
    pixels already meet or exceed the requested sampling density.
    """

    values = (root_pixel_size_rad, minor_fwhm_rad, maximum_pixels_per_beam)
    if not np.all(np.isfinite(values)) or any(value <= 0 for value in values):
        raise ValueError("pixel size, beam FWHM, and sampling limit must be finite and positive")
    ratio = maximum_pixels_per_beam * root_pixel_size_rad / minor_fwhm_rad
    if ratio <= 1.0:
        return 0
    return max(0, int(np.floor(np.log2(ratio))))
