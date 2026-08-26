"""Composite fixed-support sky inference for wide-field mosaics.

The adaptive quadtree remains responsible for discovering compact central
structure.  This module refits that fixed topology together with lower
resolution wide-field quadtrees and exact point-source atoms.  All component
groups share one visibility objective, so overlapping dictionaries compete
under the same positive L1 prior rather than being subtracted in sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sl1mjax.beam import VLAPrimaryBeam, predict_beam_weights
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.inference import (
    InferenceConfig,
    _fit_physical_flux,
    _inference_dtypes,
    _validate_inference_inputs,
)
from sl1mjax.objective import effective_weight
from sl1mjax.polarization import Correlation
from sl1mjax.quadtree import QuadtreeTopology, predict_quadtree_stokes_i_explicit
from sl1mjax.sky import GaussianApproximation


@dataclass(frozen=True)
class MosaicQuadtreeComponent:
    """One fixed quadtree dictionary in the common mosaic tangent plane."""

    name: str
    topology: QuadtreeTopology
    flux: np.ndarray
    sparsity_weights: np.ndarray | None = None


@dataclass(frozen=True)
class MosaicPointComponent:
    """Exact delta-function atoms in the common mosaic tangent plane."""

    name: str
    l_rad: np.ndarray
    m_rad: np.ndarray
    flux: np.ndarray
    sparsity_weights: np.ndarray | None = None


type MosaicSkyComponent = MosaicQuadtreeComponent | MosaicPointComponent


@dataclass(frozen=True)
class MosaicCompositeInferenceResult:
    """Joint fit of several fixed sky dictionaries to mosaic visibilities."""

    components: tuple[MosaicSkyComponent, ...]
    predictions: tuple[np.ndarray, ...]
    component_predictions: tuple[tuple[np.ndarray, ...], ...]
    residuals: tuple[np.ndarray, ...]
    mosaic_phase_centre_rad: tuple[float, float]
    objective_history: tuple[float, ...]
    data_history: tuple[float, ...]
    prior_history: tuple[float, ...]
    holdout_history: tuple[float, ...]
    holdout_steps: tuple[int, ...]
    best_step: int
    steps: int
    converged: bool
    solver: str
    objective_steps: tuple[int, ...]
    stationarity_history: tuple[float, ...]
    stationarity_steps: tuple[int, ...]
    kkt_residual: float


@dataclass(frozen=True)
class _PreparedComponent:
    component: MosaicSkyComponent
    parameter_slice: slice
    local_centres: tuple[tuple[np.ndarray, np.ndarray], ...]
    beam_arrays: tuple[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None], ...]


def _component_coordinates(component: MosaicSkyComponent) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(component, MosaicQuadtreeComponent):
        return component.topology.centers()
    return (
        np.asarray(component.l_rad, dtype=np.float64).reshape(-1),
        np.asarray(component.m_rad, dtype=np.float64).reshape(-1),
    )


def _component_size(component: MosaicSkyComponent) -> int:
    if isinstance(component, MosaicQuadtreeComponent):
        return len(component.topology.leaves)
    return np.asarray(component.l_rad).size


def _validated_components(
    components: tuple[MosaicSkyComponent, ...],
) -> tuple[MosaicSkyComponent, ...]:
    if not components:
        raise ValueError("components must contain at least one sky component")
    names = [component.name for component in components]
    if any(not name.strip() for name in names):
        raise ValueError("component names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("component names must be unique")

    validated: list[MosaicSkyComponent] = []
    for component in components:
        size = _component_size(component)
        if size == 0:
            raise ValueError(f"component {component.name!r} must contain at least one atom")
        l_rad, m_rad = _component_coordinates(component)
        if l_rad.shape != (size,) or m_rad.shape != (size,):
            raise ValueError(
                f"component {component.name!r} coordinates must contain one value per atom"
            )
        if np.any(~np.isfinite(l_rad)) or np.any(~np.isfinite(m_rad)):
            raise ValueError(f"component {component.name!r} coordinates must be finite")
        if np.any(l_rad * l_rad + m_rad * m_rad >= 1):
            raise ValueError(
                f"component {component.name!r} coordinates must lie in the visible hemisphere"
            )
        flux = np.asarray(component.flux, dtype=np.float64).reshape(-1)
        if flux.shape != (size,):
            raise ValueError(f"component {component.name!r} flux must contain one value per atom")
        if np.any(~np.isfinite(flux)) or np.any(flux < 0):
            raise ValueError(f"component {component.name!r} flux must be finite and non-negative")
        weights = component.sparsity_weights
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if weights.shape != (size,):
                raise ValueError(
                    f"component {component.name!r} sparsity_weights must contain one value per atom"
                )
            if np.any(~np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError(
                    f"component {component.name!r} sparsity_weights must be finite and non-negative"
                )
        validated.append(replace(component, flux=flux, sparsity_weights=weights))
    return tuple(validated)


def _prepare_components(
    blocks: tuple[VisibilityBlock, ...],
    components: tuple[MosaicSkyComponent, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    primary_beam: VLAPrimaryBeam | None,
) -> tuple[_PreparedComponent, ...]:
    prepared: list[_PreparedComponent] = []
    start = 0
    for component in components:
        reference_l, reference_m = _component_coordinates(component)
        sky_ra, sky_dec = lmn_to_radec(
            mosaic_phase_centre_rad[0],
            mosaic_phase_centre_rad[1],
            reference_l,
            reference_m,
        )
        local_centres = tuple(
            radec_to_lmn(
                block.phase_centre_rad[0],
                block.phase_centre_rad[1],
                sky_ra,
                sky_dec,
            )[:2]
            for block in blocks
        )
        beam_arrays = tuple(
            predict_beam_weights(primary_beam, l_rad, m_rad, block.frequency_hz)
            for block, (l_rad, m_rad) in zip(blocks, local_centres, strict=True)
        )
        stop = start + _component_size(component)
        prepared.append(
            _PreparedComponent(
                component=component,
                parameter_slice=slice(start, stop),
                local_centres=local_centres,
                beam_arrays=beam_arrays,
            )
        )
        start = stop
    return tuple(prepared)


def _predict_prepared_component(
    prepared: _PreparedComponent,
    flux: jax.Array,
    block: VisibilityBlock,
    block_index: int,
    *,
    approximation: GaussianApproximation,
    config: DirectDFTConfig,
    real_dtype: Any,
) -> jax.Array:
    local_l, local_m = prepared.local_centres[block_index]
    beam_i, beam_rr, beam_ll = prepared.beam_arrays[block_index]
    uvw_m = jnp.asarray(block.uvw_m, dtype=real_dtype)
    frequency_hz = jnp.asarray(block.frequency_hz, dtype=real_dtype)
    antenna1 = jnp.asarray(block.antenna1)
    antenna2 = jnp.asarray(block.antenna2)
    beam_i_array = None if beam_i is None else jnp.asarray(beam_i, dtype=real_dtype)
    beam_rr_array = None if beam_rr is None else jnp.asarray(beam_rr, dtype=real_dtype)
    beam_ll_array = None if beam_ll is None else jnp.asarray(beam_ll, dtype=real_dtype)
    if isinstance(prepared.component, MosaicQuadtreeComponent):
        return predict_quadtree_stokes_i_explicit(
            flux,
            prepared.component.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            block.correlations,
            approximation=approximation,
            config=config,
            beam_weights=beam_i_array,
            beam_weights_rr=beam_rr_array,
            beam_weights_ll=beam_ll_array,
            centers_lm=(local_l, local_m),
        )
    return predict_stokes_i_explicit(
        flux,
        local_l,
        local_m,
        uvw_m,
        frequency_hz,
        antenna1,
        antenna2,
        block.correlations,
        config=config,
        beam_weights=beam_i_array,
        beam_weights_rr=beam_rr_array,
        beam_weights_ll=beam_ll_array,
    )


def predict_mosaic_composite(
    blocks: tuple[VisibilityBlock, ...],
    components: tuple[MosaicSkyComponent, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    config: DirectDFTConfig | None = None,
    fixed_predictions: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, ...]:
    """Predict mosaic visibilities from a sum of fixed sky dictionaries."""

    if not blocks:
        raise ValueError("blocks must contain at least one visibility block")
    if len(mosaic_phase_centre_rad) != 2 or np.any(~np.isfinite(mosaic_phase_centre_rad)):
        raise ValueError("mosaic_phase_centre_rad must contain finite RA and Dec")
    selected_components = _validated_components(components)
    selected_config = config or DirectDFTConfig()
    selected_approximation = GaussianApproximation(approximation)
    prepared = _prepare_components(
        blocks,
        selected_components,
        mosaic_phase_centre_rad,
        primary_beam,
    )
    selected_fixed = _validate_fixed_predictions(blocks, fixed_predictions)
    real_dtype = selected_config.real_dtype
    predictions: list[np.ndarray] = []
    for block_index, block in enumerate(blocks):
        total = jnp.asarray(selected_fixed[block_index], dtype=selected_config.complex_dtype)
        for item in prepared:
            total = total + _predict_prepared_component(
                item,
                jnp.asarray(item.component.flux, dtype=real_dtype),
                block,
                block_index,
                approximation=selected_approximation,
                config=selected_config,
                real_dtype=real_dtype,
            )
        predictions.append(np.asarray(total))
    return tuple(predictions)


def _validate_fixed_predictions(
    blocks: tuple[VisibilityBlock, ...],
    fixed_predictions: tuple[np.ndarray, ...] | None,
) -> tuple[np.ndarray, ...]:
    if fixed_predictions is None:
        return tuple(np.zeros(block.shape, dtype=np.complex128) for block in blocks)
    if len(fixed_predictions) != len(blocks):
        raise ValueError("fixed_predictions must contain one array per block")
    selected: list[np.ndarray] = []
    for block, prediction in zip(blocks, fixed_predictions, strict=True):
        array = np.asarray(prediction, dtype=np.complex128)
        if array.shape != block.shape:
            raise ValueError("each fixed prediction must match its visibility block")
        if np.any(~np.isfinite(array)):
            raise ValueError("fixed_predictions must be finite")
        selected.append(array)
    return tuple(selected)


def _beam_response_for_correlations(
    block: VisibilityBlock,
    beam_arrays: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    component_count: int,
) -> np.ndarray:
    beam_i, beam_rr, beam_ll = beam_arrays
    if beam_i is None and beam_rr is None and beam_ll is None:
        return np.ones(
            (component_count, block.frequency_hz.size, len(block.correlations)),
            dtype=np.float64,
        )
    response = np.zeros(
        (component_count, block.frequency_hz.size, len(block.correlations)),
        dtype=np.float64,
    )
    for correlation_index, correlation in enumerate(block.correlations):
        if correlation in {Correlation.I, Correlation.XX, Correlation.YY}:
            if beam_i is not None:
                values = beam_i
            else:
                assert beam_rr is not None and beam_ll is not None
                values = 0.5 * (beam_rr + beam_ll)
        elif correlation is Correlation.RR:
            if beam_i is not None:
                values = beam_i
            else:
                assert beam_rr is not None
                values = beam_rr
        elif correlation is Correlation.LL:
            if beam_i is not None:
                values = beam_i
            else:
                assert beam_ll is not None
                values = beam_ll
        else:
            values = np.zeros((component_count, block.frequency_hz.size))
        response[:, :, correlation_index] = values
    return response


def mosaic_beam_sensitivity_weights(
    blocks: tuple[VisibilityBlock, ...],
    components: tuple[MosaicSkyComponent, ...],
    train_masks: tuple[np.ndarray, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    primary_beam: VLAPrimaryBeam | None,
) -> tuple[np.ndarray, ...]:
    """Return globally normalized beam-weighted column sensitivities.

    The weight for atom ``j`` is proportional to
    ``sqrt(sum(w * B_j**2))`` over selected mosaic samples.  Dividing by the
    largest value gives unit penalty at peak sensitivity and a smaller L1
    penalty where the primary beam attenuates the sky.  This removes the most
    direct centre-versus-edge selection bias.  The estimate intentionally
    omits the square-pixel Fourier envelope, so validation must still compare
    dictionaries with very different pixel widths.
    """

    if not blocks or len(train_masks) != len(blocks):
        raise ValueError("blocks and train_masks must contain the same non-zero count")
    selected_components = _validated_components(components)
    prepared = _prepare_components(
        blocks,
        selected_components,
        mosaic_phase_centre_rad,
        primary_beam,
    )
    sensitivities = [np.zeros(_component_size(item.component)) for item in prepared]
    for block_index, (block, mask) in enumerate(zip(blocks, train_masks, strict=True)):
        if mask.shape != block.shape:
            raise ValueError("each train mask must match its visibility block")
        active_weight = np.where(mask & block.active, block.weight, 0.0)
        channel_correlation_weight = np.sum(active_weight, axis=0)
        for component_index, item in enumerate(prepared):
            response = _beam_response_for_correlations(
                block,
                item.beam_arrays[block_index],
                _component_size(item.component),
            )
            sensitivities[component_index] += np.sum(
                channel_correlation_weight[None, :, :] * response**2,
                axis=(1, 2),
            )
    roots = [np.sqrt(values) for values in sensitivities]
    maximum = max(float(np.max(values)) for values in roots)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("training samples have zero component sensitivity")
    return tuple(values / maximum for values in roots)


def infer_mosaic_composite(
    blocks: tuple[VisibilityBlock, ...],
    components: tuple[MosaicSkyComponent, ...],
    train_masks: tuple[np.ndarray, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    config: InferenceConfig | None = None,
    *,
    holdout_masks: tuple[np.ndarray, ...] | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    fixed_predictions: tuple[np.ndarray, ...] | None = None,
) -> MosaicCompositeInferenceResult:
    """Jointly fit positive flux in several fixed mosaic sky dictionaries.

    Component coordinates use the common mosaic tangent plane.  Each group can
    supply dimensionless per-atom sparsity weights.  They multiply the single
    ``InferenceConfig.sparsity_weight`` and should be normalized together when
    weights need to be comparable across groups.
    """

    configuration = config or InferenceConfig(solver="fista")
    if not blocks:
        raise ValueError("blocks must contain at least one visibility block")
    if len(train_masks) != len(blocks):
        raise ValueError("train_masks must contain one mask per block")
    if holdout_masks is not None and len(holdout_masks) != len(blocks):
        raise ValueError("holdout_masks must contain one mask per block")
    if configuration.solver != "fista":
        raise ValueError("composite mosaic inference currently requires solver='fista'")
    if configuration.operator_mode != "explicit":
        raise ValueError("composite mosaic inference requires operator_mode='explicit'")
    if configuration.smoothness_weight != 0:
        raise ValueError("smoothness_weight is not defined for composite inference")
    if len(mosaic_phase_centre_rad) != 2 or np.any(~np.isfinite(mosaic_phase_centre_rad)):
        raise ValueError("mosaic_phase_centre_rad must contain finite RA and Dec")

    selected_components = _validated_components(components)
    selected_holdout_masks = (
        tuple(np.zeros(block.shape, dtype=bool) for block in blocks)
        if holdout_masks is None
        else holdout_masks
    )
    for block, train_mask, holdout_mask in zip(
        blocks, train_masks, selected_holdout_masks, strict=True
    ):
        _validate_inference_inputs(
            block,
            train_mask,
            None if holdout_masks is None else holdout_mask,
            configuration,
        )

    prepared = _prepare_components(
        blocks,
        selected_components,
        mosaic_phase_centre_rad,
        primary_beam,
    )
    selected_fixed = _validate_fixed_predictions(blocks, fixed_predictions)
    selected_approximation = GaussianApproximation(approximation)
    real_dtype, complex_dtype = _inference_dtypes(configuration)
    observations = tuple(jnp.asarray(block.visibility, dtype=complex_dtype) for block in blocks)
    fixed_arrays = tuple(jnp.asarray(value, dtype=complex_dtype) for value in selected_fixed)
    training_flags = tuple(jnp.asarray(~mask) for mask in train_masks)
    training_weights = tuple(
        effective_weight(observation, jnp.asarray(block.weight, dtype=real_dtype), flag)
        for block, observation, flag in zip(blocks, observations, training_flags, strict=True)
    )
    training_weight_sum = jnp.asarray(0.0, dtype=real_dtype)
    for value in training_weights:
        training_weight_sum = training_weight_sum + jnp.sum(value)
    if float(training_weight_sum) <= 0:
        raise ValueError("train_masks must contain positive-weight finite samples")

    has_holdout = holdout_masks is not None and any(
        np.any(mask & block.active)
        for block, mask in zip(blocks, selected_holdout_masks, strict=True)
    )
    holdout_flags = tuple(jnp.asarray(~mask) for mask in selected_holdout_masks)
    holdout_weights = tuple(
        effective_weight(observation, jnp.asarray(block.weight, dtype=real_dtype), flag)
        for block, observation, flag in zip(blocks, observations, holdout_flags, strict=True)
    )
    holdout_weight_sum = jnp.asarray(0.0, dtype=real_dtype)
    for value in holdout_weights:
        holdout_weight_sum = holdout_weight_sum + jnp.sum(value)

    initial_flux = np.concatenate([item.component.flux for item in prepared])
    sparsity_weights = np.concatenate(
        [
            np.ones(_component_size(item.component), dtype=np.float64)
            if item.component.sparsity_weights is None
            else item.component.sparsity_weights
            for item in prepared
        ]
    )

    def predict_component(flux: jax.Array, component_index: int, block_index: int) -> jax.Array:
        item = prepared[component_index]
        return _predict_prepared_component(
            item,
            flux[item.parameter_slice],
            blocks[block_index],
            block_index,
            approximation=selected_approximation,
            config=configuration.direct_dft,
            real_dtype=real_dtype,
        )

    def predict_block(flux: jax.Array, block_index: int) -> jax.Array:
        total = fixed_arrays[block_index]
        for component_index in range(len(prepared)):
            total = total + predict_component(flux, component_index, block_index)
        return total

    def weighted_data_term(
        flux: jax.Array,
        selected_weights: tuple[jax.Array, ...],
        denominator: jax.Array,
    ) -> jax.Array:
        numerator = jnp.asarray(0.0, dtype=real_dtype)
        for block_index, active_weight in enumerate(selected_weights):
            residual = jnp.where(
                active_weight > 0,
                predict_block(flux, block_index) - observations[block_index],
                0.0,
            )
            numerator = numerator + jnp.sum(active_weight * jnp.abs(residual) ** 2)
        return numerator / denominator

    def smooth_objective(flux: jax.Array) -> jax.Array:
        return weighted_data_term(flux, training_weights, training_weight_sum)

    sparsity_array = jnp.asarray(sparsity_weights, dtype=real_dtype)

    def full_terms(
        flux: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        data_term = smooth_objective(flux)
        prior_term = configuration.sparsity_weight * jnp.sum(sparsity_array * flux)
        return data_term + prior_term, (data_term, prior_term)

    def holdout_data(flux: jax.Array) -> jax.Array:
        if not has_holdout:
            return jnp.asarray(jnp.inf, dtype=real_dtype)
        return weighted_data_term(flux, holdout_weights, holdout_weight_sum)

    fit = _fit_physical_flux(
        initial_flux.size,
        configuration,
        full_terms,
        smooth_objective,
        None,
        holdout_data,
        has_holdout=has_holdout,
        eligible_rows=np.ones(1, dtype=bool),
        initial_flux=initial_flux,
        sparsity_weights=sparsity_weights,
    )

    fitted_components: list[MosaicSkyComponent] = []
    component_predictions: list[tuple[np.ndarray, ...]] = []
    fitted_flux = jnp.asarray(fit.flux, dtype=real_dtype)
    for component_index, item in enumerate(prepared):
        component_flux = fit.flux[item.parameter_slice]
        fitted_components.append(replace(item.component, flux=component_flux))
        component_predictions.append(
            tuple(
                np.asarray(predict_component(fitted_flux, component_index, block_index))
                for block_index in range(len(blocks))
            )
        )
    predictions = tuple(
        np.asarray(predict_block(fitted_flux, block_index)) for block_index in range(len(blocks))
    )
    residuals = tuple(
        prediction - block.visibility for prediction, block in zip(predictions, blocks, strict=True)
    )
    return MosaicCompositeInferenceResult(
        components=tuple(fitted_components),
        predictions=predictions,
        component_predictions=tuple(component_predictions),
        residuals=residuals,
        mosaic_phase_centre_rad=(
            float(mosaic_phase_centre_rad[0]),
            float(mosaic_phase_centre_rad[1]),
        ),
        objective_history=fit.objective_history,
        data_history=fit.data_history,
        prior_history=fit.prior_history,
        holdout_history=fit.holdout_history,
        holdout_steps=fit.holdout_steps,
        best_step=fit.best_step,
        steps=fit.steps,
        converged=fit.converged,
        solver=fit.solver,
        objective_steps=fit.objective_steps,
        stationarity_history=fit.stationarity_history,
        stationarity_steps=fit.stationarity_steps,
        kkt_residual=fit.kkt_residual,
    )
