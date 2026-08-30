"""Controlled Stokes-I flux refit under a streamed voltage beam.

Geometry, source positions, and the L1 recipe stay frozen. Only leaf
fluxes move. This is a diagnostic path; default imaging still uses
static Airy / ``predict_stokes_i_explicit``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from sl1mjax.beam_operator import BeamOperatorConfig, SkyStokesPlanes
from sl1mjax.composite import (
    MosaicSkyComponent,
    _component_coordinates,
)
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.inference import InferenceConfig, _time_groups_for_row_uniform_sgd
from sl1mjax.polarization import Correlation
from sl1mjax.voltage_beam import VoltageBeamModel
from sl1mjax.voltage_operator_jax import (
    predict_voltage_beam_jax,
    predict_voltage_beam_jax_value_and_grad,
)


@dataclass(frozen=True)
class FlattenedSky:
    """Frozen mosaic atoms as one flux vector."""

    l_rad: np.ndarray
    m_rad: np.ndarray
    flux: np.ndarray
    names: tuple[str, ...]
    sizes: tuple[int, ...]

    @property
    def slices(self) -> tuple[slice, ...]:
        starts = np.cumsum((0, *self.sizes[:-1]))
        return tuple(
            slice(int(start), int(start + size))
            for start, size in zip(starts, self.sizes, strict=True)
        )


@dataclass(frozen=True)
class VoltageFluxRefitResult:
    """Positive flux after a time-grouped proximal SGD refit."""

    flux: np.ndarray
    objective_history: tuple[float, ...]
    holdout_history: tuple[float, ...]
    holdout_steps: tuple[int, ...]
    best_step: int
    steps: int
    sparsity_weight: float


def flatten_sky_atoms(components: tuple[MosaicSkyComponent, ...]) -> FlattenedSky:
    """Return every frozen atom, including zeros. Geometry is not filtered."""

    if not components:
        raise ValueError("components must contain at least one sky component")
    l_parts: list[np.ndarray] = []
    m_parts: list[np.ndarray] = []
    flux_parts: list[np.ndarray] = []
    names: list[str] = []
    sizes: list[int] = []
    for component in components:
        l_rad, m_rad = _component_coordinates(component)
        flux = np.asarray(component.flux, dtype=np.float64).reshape(-1)
        if flux.size != l_rad.size:
            raise ValueError(f"component {component.name!r} flux must match its atoms")
        l_parts.append(np.asarray(l_rad, dtype=np.float64))
        m_parts.append(np.asarray(m_rad, dtype=np.float64))
        flux_parts.append(flux)
        names.append(component.name)
        sizes.append(int(flux.size))
    return FlattenedSky(
        l_rad=np.concatenate(l_parts),
        m_rad=np.concatenate(m_parts),
        flux=np.concatenate(flux_parts),
        names=tuple(names),
        sizes=tuple(sizes),
    )


def replace_component_fluxes(
    components: tuple[MosaicSkyComponent, ...], flux: np.ndarray
) -> tuple[MosaicSkyComponent, ...]:
    """Write a flat flux vector back onto the frozen component geometry."""

    sky = flatten_sky_atoms(components)
    values = np.asarray(flux, dtype=np.float64).reshape(-1)
    if values.size != sky.flux.size:
        raise ValueError("flux length must match the frozen atom count")
    updated: list[MosaicSkyComponent] = []
    for component, selected in zip(components, sky.slices, strict=True):
        updated.append(replace(component, flux=values[selected].copy()))
    return tuple(updated)


def mosaic_local_directions(
    l_mosaic: np.ndarray,
    m_mosaic: np.ndarray,
    mosaic_phase_centre_rad: tuple[float, float],
    block_phase_centre_rad: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate mosaic-frame directions into a pointing's tangent plane."""

    sky_ra, sky_dec = lmn_to_radec(
        mosaic_phase_centre_rad[0], mosaic_phase_centre_rad[1], l_mosaic, m_mosaic
    )
    local_l, local_m, _n = radec_to_lmn(
        block_phase_centre_rad[0], block_phase_centre_rad[1], sky_ra, sky_dec
    )
    return np.asarray(local_l, dtype=np.float64), np.asarray(local_m, dtype=np.float64)


def _row_mask(block: VisibilityBlock, rows: np.ndarray, base: np.ndarray) -> np.ndarray:
    mask = np.zeros(block.shape, dtype=bool)
    if rows.size:
        mask[np.asarray(rows, dtype=np.int32)] = True
    return mask & np.asarray(base, dtype=bool) & block.active


def _time_batches(
    blocks: tuple[VisibilityBlock, ...], masks: tuple[np.ndarray, ...]
) -> tuple[tuple[int, np.ndarray], ...]:
    batches: list[tuple[int, np.ndarray]] = []
    for block_index, (block, mask) in enumerate(zip(blocks, masks, strict=True)):
        selected = np.asarray(mask, dtype=bool) & block.active
        rows = np.flatnonzero(np.any(selected, axis=(1, 2)))
        if rows.size == 0:
            continue
        groups, _probabilities = _time_groups_for_row_uniform_sgd(rows, block.time_s)
        batches.extend((block_index, group) for group in groups)
    if not batches:
        raise ValueError("masks contain no eligible rows")
    return tuple(batches)


def mosaic_weighted_mse(
    predictions: tuple[np.ndarray, ...],
    blocks: tuple[VisibilityBlock, ...],
    masks: tuple[np.ndarray, ...],
) -> float:
    """Normalised mosaic MSE: total residual power over total weight."""

    numerator = 0.0
    denominator = 0.0
    for prediction, block, mask in zip(predictions, blocks, masks, strict=True):
        selected = np.asarray(mask, dtype=bool) & block.active
        finite_weight = np.isfinite(block.weight) & (block.weight > 0)
        weight = np.where(selected & finite_weight, block.weight, 0.0)
        residual = prediction - block.visibility
        finite = np.isfinite(residual.real) & np.isfinite(residual.imag)
        usable = (weight > 0) & finite
        if not np.any(usable):
            continue
        numerator += float(np.sum(weight[usable] * np.abs(residual[usable]) ** 2))
        denominator += float(np.sum(weight[usable]))
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def predict_voltage_mosaic(
    flux: np.ndarray,
    blocks: tuple[VisibilityBlock, ...],
    local_directions: tuple[tuple[np.ndarray, np.ndarray], ...],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: str,
    config: BeamOperatorConfig,
    stokes_q: np.ndarray | None = None,
    stokes_u: np.ndarray | None = None,
    stokes_v: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Predict each pointing with the JAX voltage operator."""

    intensity = np.asarray(flux, dtype=np.float64).reshape(-1)
    sky = SkyStokesPlanes(
        stokes_i=intensity,
        stokes_q=stokes_q,
        stokes_u=stokes_u,
        stokes_v=stokes_v,
    )
    predictions: list[np.ndarray] = []
    for block, (l_rad, m_rad) in zip(blocks, local_directions, strict=True):
        result = predict_voltage_beam_jax(
            block,
            l_rad,
            m_rad,
            sky,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
        )
        predictions.append(np.asarray(result.visibility))
    return tuple(predictions)


def refit_stokes_i_fluxes(
    blocks: tuple[VisibilityBlock, ...],
    train_masks: tuple[np.ndarray, ...],
    local_directions: tuple[tuple[np.ndarray, np.ndarray], ...],
    initial_flux: np.ndarray,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: str = "casa_parang_true",
    config: InferenceConfig | None = None,
    operator_config: BeamOperatorConfig | None = None,
    holdout_masks: tuple[np.ndarray, ...] | None = None,
    sparsity_weights: np.ndarray | None = None,
) -> VoltageFluxRefitResult:
    """Refit only Stokes-I fluxes with time-grouped proximal SGD.

    ``batch_grouping`` is forced to ``times`` so one Jones plane is reused
    for the sampled integration. FISTA is refused: it is full-batch and
    would rescan every unique time on every backtracking trial.
    """

    selected = config or InferenceConfig(
        solver="proximal_sgd",
        batch_grouping="times",
        steps=50,
        operator_mode="autodiff",
    )
    if selected.solver == "fista":
        raise ValueError(
            "voltage flux refit uses proximal SGD with time grouping; "
            "FISTA is full-batch over every unique time"
        )
    if selected.smoothness_weight != 0:
        raise ValueError("smoothness_weight is not defined for voltage flux refit")
    if len(blocks) != len(train_masks) or len(blocks) != len(local_directions):
        raise ValueError("blocks, train_masks, and local_directions must align")
    flux = np.asarray(initial_flux, dtype=np.float64).reshape(-1)
    if flux.size == 0 or np.any(~np.isfinite(flux)) or np.any(flux < 0):
        raise ValueError("initial flux must be finite, non-negative, and nonempty")
    if sparsity_weights is None:
        penalty = np.full(flux.size, selected.sparsity_weight, dtype=np.float64)
    else:
        weights = np.asarray(sparsity_weights, dtype=np.float64).reshape(-1)
        if weights.shape != flux.shape:
            raise ValueError("sparsity_weights must match the flux vector")
        penalty = selected.sparsity_weight * weights
    operator = operator_config or BeamOperatorConfig(
        visibility_chunk_size=256, pixel_chunk_size=512
    )
    batches = _time_batches(blocks, train_masks)
    batch_weights = np.asarray([group.size for _index, group in batches], dtype=np.float64)
    batch_probabilities = batch_weights / batch_weights.sum()
    current = jnp.asarray(flux)
    history: list[float] = []
    holdout_history: list[float] = []
    holdout_steps: list[int] = []
    best_flux = np.asarray(current)
    best_holdout = np.inf
    best_step = 0
    rng = np.random.default_rng(selected.random_seed)

    def evaluate(values: np.ndarray) -> tuple[float, float]:
        predictions = predict_voltage_mosaic(
            values,
            blocks,
            local_directions,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=operator,
        )
        train = mosaic_weighted_mse(predictions, blocks, train_masks)
        holdout = (
            mosaic_weighted_mse(predictions, blocks, holdout_masks)
            if holdout_masks is not None
            else float("nan")
        )
        return train, holdout

    start_train, start_holdout = evaluate(np.asarray(current))
    history.append(start_train + float(np.sum(penalty * np.asarray(current))))
    if holdout_masks is not None:
        holdout_history.append(start_holdout)
        holdout_steps.append(0)
        best_holdout = start_holdout
    for step in range(1, selected.steps + 1):
        block_index, rows = batches[int(rng.choice(len(batches), p=batch_probabilities))]
        block = blocks[block_index]
        l_rad, m_rad = local_directions[block_index]
        row_mask = _row_mask(block, rows, train_masks[block_index])
        _loss, gradient = predict_voltage_beam_jax_value_and_grad(
            current,
            block,
            l_rad,
            m_rad,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=operator,
            train_mask=row_mask,
        )
        progress = (step - 1) / max(selected.steps - 1, 1)
        decay = 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))
        step_size = selected.learning_rate * decay
        current = jnp.maximum(current - step_size * (gradient + penalty), 0.0)
        if step % selected.validation_interval == 0 or step == selected.steps:
            train, holdout = evaluate(np.asarray(current))
            history.append(train + float(np.sum(penalty * np.asarray(current))))
            if holdout_masks is not None:
                holdout_history.append(holdout)
                holdout_steps.append(step)
                if np.isfinite(holdout) and holdout < best_holdout - selected.min_delta:
                    best_holdout = holdout
                    best_flux = np.asarray(current)
                    best_step = step
            else:
                best_flux = np.asarray(current)
                best_step = step
    if holdout_masks is None:
        best_flux = np.asarray(current)
        best_step = selected.steps
    return VoltageFluxRefitResult(
        flux=np.asarray(best_flux, dtype=np.float64),
        objective_history=tuple(history),
        holdout_history=tuple(holdout_history),
        holdout_steps=tuple(holdout_steps),
        best_step=best_step,
        steps=selected.steps,
        sparsity_weight=float(selected.sparsity_weight),
    )


def weighted_residual_power(
    residual: np.ndarray, weight: np.ndarray, mask: np.ndarray
) -> float:
    selected = np.asarray(mask, dtype=bool) & np.isfinite(weight) & (weight > 0)
    if not np.any(selected):
        return float("nan")
    values = residual[selected]
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if not np.any(finite):
        return float("nan")
    return float(np.sum(weight[selected][finite] * np.abs(values[finite]) ** 2))


def correlation_mask(
    block: VisibilityBlock, base: np.ndarray, correlation: Correlation
) -> np.ndarray:
    if correlation not in block.correlations:
        return np.zeros(block.shape, dtype=bool)
    mask = np.asarray(base, dtype=bool).copy()
    index = block.correlations.index(correlation)
    for other, _name in enumerate(block.correlations):
        if other != index:
            mask[..., other] = False
    return mask & block.active


def score_visibility_prediction(
    block: VisibilityBlock,
    prediction: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Held-out residual power by correlation, channel, and time."""

    residual = block.visibility - prediction
    selected = block.active if mask is None else np.asarray(mask, dtype=bool) & block.active
    unique_times, time_index = np.unique(block.time_s, return_inverse=True)
    payload: dict[str, Any] = {
        "total": weighted_residual_power(residual, block.weight, selected),
        "data_power": weighted_residual_power(block.visibility, block.weight, selected),
        "model_power": weighted_residual_power(prediction, block.weight, selected),
        "correlations": {},
        "by_channel": [],
        "by_time": [],
    }
    for correlation in (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL):
        if correlation in block.correlations:
            hand = correlation_mask(block, selected, correlation)
            payload["correlations"][correlation.value] = {
                "held_out_loss": weighted_residual_power(residual, block.weight, hand),
                "data_power": weighted_residual_power(block.visibility, block.weight, hand),
                "model_power": weighted_residual_power(prediction, block.weight, hand),
                "in_data": True,
            }
        else:
            payload["correlations"][correlation.value] = {
                "held_out_loss": None,
                "data_power": None,
                "model_power": None,
                "in_data": False,
            }
    for channel in range(block.frequency_hz.size):
        channel_mask = np.zeros(block.shape, dtype=bool)
        channel_mask[:, channel, :] = selected[:, channel, :]
        payload["by_channel"].append(
            {
                "channel": channel,
                "frequency_hz": float(block.frequency_hz[channel]),
                "held_out_loss": weighted_residual_power(residual, block.weight, channel_mask),
            }
        )
    for index, time_s in enumerate(unique_times):
        time_mask = np.zeros(block.shape, dtype=bool)
        time_mask[time_index == index] = selected[time_index == index]
        payload["by_time"].append(
            {
                "time_s": float(time_s),
                "held_out_loss": weighted_residual_power(residual, block.weight, time_mask),
            }
        )
    return payload


def paired_score_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """``candidate - reference``. Negative total or hand loss is an improvement."""

    def _delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return float(right) - float(left)

    channel_delta = []
    for left, right in zip(reference["by_channel"], candidate["by_channel"], strict=True):
        channel_delta.append(
            {
                "channel": left["channel"],
                "frequency_hz": left["frequency_hz"],
                "held_out_loss": _delta(left["held_out_loss"], right["held_out_loss"]),
            }
        )
    time_delta = []
    for left, right in zip(reference["by_time"], candidate["by_time"], strict=True):
        time_delta.append(
            {
                "time_s": left["time_s"],
                "held_out_loss": _delta(left["held_out_loss"], right["held_out_loss"]),
            }
        )
    correlations = {}
    for name in ("RR", "RL", "LR", "LL"):
        correlations[name] = _delta(
            reference["correlations"][name]["held_out_loss"],
            candidate["correlations"][name]["held_out_loss"],
        )
    return {
        "total": _delta(reference["total"], candidate["total"]),
        "correlations": correlations,
        "by_channel": channel_delta,
        "by_time": time_delta,
    }


def off_axis_atom_report(
    sky: FlattenedSky,
    initial_flux: np.ndarray,
    fitted_flux: np.ndarray,
    *,
    radius_arcmin_cut: float = 8.0,
) -> dict[str, Any]:
    """Flux movement of atoms outside the mosaic-frame radius cut."""

    radius_arcmin = np.rad2deg(np.hypot(sky.l_rad, sky.m_rad)) * 60.0
    initial = np.asarray(initial_flux, dtype=np.float64).reshape(-1)
    fitted = np.asarray(fitted_flux, dtype=np.float64).reshape(-1)
    if initial.shape != fitted.shape or initial.shape != sky.flux.shape:
        raise ValueError("flux vectors must match the flattened sky")
    outside = radius_arcmin >= radius_arcmin_cut
    by_component = []
    offset = 0
    for name, size in zip(sky.names, sky.sizes, strict=True):
        selected = outside[offset : offset + size]
        before = initial[offset : offset + size]
        after = fitted[offset : offset + size]
        by_component.append(
            {
                "name": name,
                "n_outside": int(np.count_nonzero(selected)),
                "flux_jy_initial": float(before[selected].sum()) if np.any(selected) else 0.0,
                "flux_jy_fitted": float(after[selected].sum()) if np.any(selected) else 0.0,
                "flux_jy_delta": (
                    float(after[selected].sum() - before[selected].sum())
                    if np.any(selected)
                    else 0.0
                ),
            }
        )
        offset += size
    brightest = []
    if np.any(outside):
        order = np.argsort(-(initial + fitted)[outside])[:8]
        indices = np.flatnonzero(outside)[order]
        for index in indices:
            brightest.append(
                {
                    "atom": int(index),
                    "component": sky.names[
                        next(
                            component
                            for component, selected in enumerate(sky.slices)
                            if selected.start <= index < selected.stop
                        )
                    ],
                    "radius_arcmin": float(radius_arcmin[index]),
                    "flux_jy_initial": float(initial[index]),
                    "flux_jy_fitted": float(fitted[index]),
                    "flux_jy_delta": float(fitted[index] - initial[index]),
                }
            )
    return {
        "radius_arcmin_cut": radius_arcmin_cut,
        "n_outside": int(np.count_nonzero(outside)),
        "flux_jy_initial": float(initial[outside].sum()),
        "flux_jy_fitted": float(fitted[outside].sum()),
        "flux_jy_delta": float(fitted[outside].sum() - initial[outside].sum()),
        "by_component": by_component,
        "brightest": brightest,
    }


def transfer_diagonal_is_consistent(summary: dict[str, Any]) -> dict[str, Any]:
    """True when diagonal beats Airy on every scored pointing total."""

    pointings = summary.get("pointings", {})
    comparisons = []
    for name, pointing in pointings.items():
        beams = pointing.get("beams", {})
        airy = beams.get("static_scalar", {}).get("total")
        diagonal = beams.get("diagonal_copolar", {}).get("total")
        if airy is None or diagonal is None:
            continue
        comparisons.append(
            {
                "pointing": name,
                "airy": float(airy),
                "diagonal": float(diagonal),
                "diagonal_beats_airy": float(diagonal) < float(airy),
            }
        )
    wins = [item["diagonal_beats_airy"] for item in comparisons]
    return {
        "n_pointings": len(comparisons),
        "n_diagonal_beats_airy": int(sum(wins)),
        "consistent": bool(wins) and all(wins),
        "pointings": comparisons,
    }


__all__ = [
    "FlattenedSky",
    "VoltageFluxRefitResult",
    "flatten_sky_atoms",
    "replace_component_fluxes",
    "mosaic_local_directions",
    "mosaic_weighted_mse",
    "predict_voltage_mosaic",
    "refit_stokes_i_fluxes",
    "score_visibility_prediction",
    "paired_score_delta",
    "off_axis_atom_report",
    "transfer_diagonal_is_consistent",
]
