"""Independent direct-DFT diagnostic imaging products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import (
    DirectDFTConfig,
    direct_scalar_adjoint,
)
from sl1mjax.polarization import stokes_i_to_correlations
from sl1mjax.rime import SPEED_OF_LIGHT_M_S
from sl1mjax.sky import RegularGrid


@dataclass(frozen=True)
class ResidualEvaluation:
    full_dirty: NDArray[np.float64]
    train_dirty: NDArray[np.float64]
    holdout_dirty: NDArray[np.float64]
    psf: NDArray[np.float64]
    diagnostics: dict[str, Any]


def _image_statistics(image: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(image, dtype=np.float64)
    median = float(np.median(values))
    robust_rms = float(1.4826 * np.median(np.abs(values - median)))
    rms = float(np.sqrt(np.mean(np.square(values))))
    variance = float(np.mean(np.square(values - np.mean(values))))
    horizontal = (
        float(
            np.mean(
                (values[:, :-1] - np.mean(values))
                * (values[:, 1:] - np.mean(values))
            )
            / variance
        )
        if variance > 0
        else 0.0
    )
    vertical = (
        float(
            np.mean(
                (values[:-1, :] - np.mean(values))
                * (values[1:, :] - np.mean(values))
            )
            / variance
        )
        if variance > 0
        else 0.0
    )
    peak = float(np.max(np.abs(values)))
    return {
        "mean": float(np.mean(values)),
        "median": median,
        "rms": rms,
        "robust_rms": robust_rms,
        "peak_absolute": peak,
        "peak_to_robust_rms": peak / robust_rms if robust_rms > 0 else None,
        "horizontal_lag1_correlation": horizontal,
        "vertical_lag1_correlation": vertical,
    }


def _visibility_statistics(
    observation: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    active = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(weight)
        & (weight > 0)
        & np.isfinite(observation.real)
        & np.isfinite(observation.imag)
        & np.isfinite(residual.real)
        & np.isfinite(residual.imag)
    )
    count = int(np.count_nonzero(active))
    if count == 0:
        return {
            "active_count": 0,
            "weighted_complex_mse": None,
            "normalized_residual_power": None,
            "weighted_mean_real": None,
            "weighted_mean_imag": None,
        }
    selected_weight = weight[active]
    selected_residual = residual[active]
    selected_observation = observation[active]
    weight_sum = float(np.sum(selected_weight))
    residual_power = float(
        np.sum(selected_weight * np.abs(selected_residual) ** 2)
    )
    signal_power = float(
        np.sum(selected_weight * np.abs(selected_observation) ** 2)
    )
    weighted_mean = np.sum(selected_weight * selected_residual) / weight_sum
    return {
        "active_count": count,
        "weighted_complex_mse": residual_power / weight_sum,
        "normalized_residual_power": (
            residual_power / signal_power if signal_power > 0 else None
        ),
        "weighted_mean_real": float(weighted_mean.real),
        "weighted_mean_imag": float(weighted_mean.imag),
    }


def _binned_statistics(
    coordinate: np.ndarray,
    observation: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    *,
    bin_count: int,
) -> list[dict[str, Any]]:
    active_coordinate = coordinate[mask]
    if active_coordinate.size == 0:
        return []
    edges = np.unique(
        np.quantile(
            active_coordinate,
            np.linspace(0.0, 1.0, bin_count + 1),
        )
    )
    results: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        selected = mask & (coordinate >= lower)
        selected &= (
            coordinate <= upper
            if index == edges.size - 2
            else coordinate < upper
        )
        results.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                **_visibility_statistics(
                    observation, residual, weight, selected
                ),
            }
        )
    return results


def _grouped_visibility_diagnostics(
    block: VisibilityBlock,
    residual: np.ndarray,
) -> dict[str, Any]:
    active = block.active
    uv_distance = np.linalg.norm(block.uvw_m[:, :2], axis=1)
    uv_wavelengths = (
        uv_distance[:, None] * block.frequency_hz[None, :] / SPEED_OF_LIGHT_M_S
    )
    uv_coordinate = np.broadcast_to(
        uv_wavelengths[:, :, None], block.shape
    )
    time_coordinate = np.broadcast_to(
        block.time_s[:, None, None], block.shape
    )
    channels = [
        {
            "channel": index,
            "frequency_hz": float(frequency),
            **_visibility_statistics(
                block.visibility,
                residual,
                block.weight,
                active
                & (
                    np.arange(block.shape[1])[None, :, None]
                    == index
                ),
            ),
        }
        for index, frequency in enumerate(block.frequency_hz)
    ]
    correlations = [
        {
            "correlation": correlation.value,
            **_visibility_statistics(
                block.visibility,
                residual,
                block.weight,
                active
                & (
                    np.arange(block.shape[2])[None, None, :]
                    == index
                ),
            ),
        }
        for index, correlation in enumerate(block.correlations)
    ]
    baseline_ids = np.minimum(block.antenna1, block.antenna2) * block.antenna_count
    baseline_ids += np.maximum(block.antenna1, block.antenna2)
    baselines: list[dict[str, Any]] = []
    for baseline_id in np.unique(baseline_ids):
        rows = baseline_ids == baseline_id
        first = int(baseline_id // block.antenna_count)
        second = int(baseline_id % block.antenna_count)
        baselines.append(
            {
                "antenna1": first,
                "antenna2": second,
                **_visibility_statistics(
                    block.visibility,
                    residual,
                    block.weight,
                    active & rows[:, None, None],
                ),
            }
        )
    baselines.sort(
        key=lambda value: float(value["normalized_residual_power"] or -1),
        reverse=True,
    )
    return {
        "uv_distance_wavelengths": _binned_statistics(
            uv_coordinate,
            block.visibility,
            residual,
            block.weight,
            active,
            bin_count=8,
        ),
        "time_s": _binned_statistics(
            time_coordinate,
            block.visibility,
            residual,
            block.weight,
            active,
            bin_count=8,
        ),
        "channels": channels,
        "correlations": correlations,
        "worst_baselines": baselines[:10],
    }


def _adjoint_image(
    block: VisibilityBlock,
    visibility: np.ndarray,
    grid: RegularGrid,
    mask: np.ndarray,
    config: DirectDFTConfig,
) -> NDArray[np.float64]:
    factors = np.asarray(
        stokes_i_to_correlations(1.0, block.correlations),
        dtype=np.complex128,
    )
    active_weight = np.where(mask & block.active, block.weight, 0.0)
    weighted_visibility = np.sum(
        active_weight * np.conj(factors)[None, None, :] * visibility,
        axis=2,
    )
    weighted_response = np.sum(
        active_weight * np.abs(factors)[None, None, :] ** 2,
        axis=2,
    )
    normalization = float(np.sum(weighted_response))
    if normalization <= 0:
        raise ValueError("residual image mask has no active positive-weight samples")
    uvw_wavelengths = (
        block.uvw_m[:, None, :]
        * block.frequency_hz[None, :, None]
        / SPEED_OF_LIGHT_M_S
    ).reshape(-1, 3)
    l, m = grid.coordinates
    image = direct_scalar_adjoint(
        weighted_visibility.ravel(),
        l,
        m,
        uvw_wavelengths,
        config=config,
    )
    return np.asarray(image, dtype=np.float64).reshape(grid.size, grid.size) / normalization


def evaluate_residuals(
    block: VisibilityBlock,
    prediction: np.ndarray,
    grid: RegularGrid,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    *,
    config: DirectDFTConfig | None = None,
) -> ResidualEvaluation:
    """Evaluate visibility and model-independent dirty-image residual structure."""

    if prediction.shape != block.shape:
        raise ValueError("prediction must match the visibility block")
    if train_mask.shape != block.shape or holdout_mask.shape != block.shape:
        raise ValueError("residual masks must match the visibility block")
    selected_config = config or DirectDFTConfig()
    residual = block.visibility - prediction
    full_mask = train_mask | holdout_mask
    full_dirty = _adjoint_image(
        block, residual, grid, full_mask, selected_config
    )
    train_dirty = _adjoint_image(
        block, residual, grid, train_mask, selected_config
    )
    holdout_dirty = _adjoint_image(
        block, residual, grid, holdout_mask, selected_config
    )
    psf = _adjoint_image(
        block,
        np.ones(block.shape, dtype=np.complex128),
        grid,
        full_mask,
        selected_config,
    )
    return ResidualEvaluation(
        full_dirty=full_dirty,
        train_dirty=train_dirty,
        holdout_dirty=holdout_dirty,
        psf=psf,
        diagnostics={
            "sign_convention": "observed_minus_model",
            "images": {
                "full": _image_statistics(full_dirty),
                "train": _image_statistics(train_dirty),
                "holdout": _image_statistics(holdout_dirty),
            },
            "visibility": {
                "full": _visibility_statistics(
                    block.visibility,
                    residual,
                    block.weight,
                    full_mask,
                ),
                "train": _visibility_statistics(
                    block.visibility,
                    residual,
                    block.weight,
                    train_mask,
                ),
                "holdout": _visibility_statistics(
                    block.visibility,
                    residual,
                    block.weight,
                    holdout_mask,
                ),
                "grouped_full": _grouped_visibility_diagnostics(
                    block, residual
                ),
            },
        },
    )


def dirty_image_and_psf(
    block: VisibilityBlock,
    grid: RegularGrid,
    *,
    chunk_size: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Form naturally weighted dirty image and PSF by a direct adjoint DFT."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    factors = np.asarray(
        stokes_i_to_correlations(1.0, block.correlations), dtype=np.complex128
    )
    active_weight = np.where(block.active, block.weight, 0.0)
    weighted_visibility = np.sum(
        active_weight * np.conj(factors)[None, None, :] * block.visibility,
        axis=2,
    )
    weighted_response = np.sum(
        active_weight * np.abs(factors)[None, None, :] ** 2,
        axis=2,
    )
    normalization = float(np.sum(weighted_response))
    if normalization <= 0:
        raise ValueError("visibility block has no active positive-weight samples")

    uvw_wavelengths = (
        block.uvw_m[:, None, :]
        * block.frequency_hz[None, :, None]
        / SPEED_OF_LIGHT_M_S
    )
    l, m = grid.coordinates
    dirty = np.empty(l.size, dtype=np.float64)
    psf = np.empty(l.size, dtype=np.float64)
    for start in range(0, l.size, chunk_size):
        stop = min(start + chunk_size, l.size)
        source_l = l[start:stop]
        source_m = m[start:stop]
        source_n = np.sqrt(1 - source_l * source_l - source_m * source_m)
        phase = 2j * np.pi * (
            uvw_wavelengths[..., 0, None] * source_l
            + uvw_wavelengths[..., 1, None] * source_m
            + uvw_wavelengths[..., 2, None] * (source_n - 1)
        )
        adjoint = np.exp(-phase)
        dirty[start:stop] = (
            np.real(np.sum(adjoint * weighted_visibility[..., None], axis=(0, 1)))
            / normalization
        )
        psf[start:stop] = (
            np.real(np.sum(adjoint * weighted_response[..., None], axis=(0, 1)))
            / normalization
        )
    shape = (grid.size, grid.size)
    return dirty.reshape(shape), psf.reshape(shape)
