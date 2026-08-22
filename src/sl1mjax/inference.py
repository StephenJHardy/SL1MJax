"""Optax gradient inference for positive fixed-support Stokes-I models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sl1mjax.beam import VLAPrimaryBeam, predict_beam_weights
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.objective import sky_prior, weighted_complex_mse
from sl1mjax.quadtree import (
    QuadtreeTopology,
    predict_quadtree_stokes_i,
    predict_quadtree_stokes_i_explicit,
)
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import (
    DeltaPixelBasis,
    GaussianApproximation,
    PixelBasis,
    RegularGrid,
    physical_intensity,
    raw_from_intensity,
)


@dataclass(frozen=True)
class InferenceConfig:
    steps: int = 500
    learning_rate: float = 0.05
    sparsity_weight: float = 1e-4
    smoothness_weight: float = 0.0
    chunk_size: int = 4096
    initial_intensity: float = 1e-2
    patience: int = 100
    min_delta: float = 1e-9
    validation_interval: int = 10
    operator_mode: Literal["autodiff", "explicit"] = "autodiff"
    direct_dft: DirectDFTConfig = DirectDFTConfig()


@dataclass(frozen=True)
class InferenceResult:
    image: np.ndarray
    raw_parameters: np.ndarray
    optimizer_state: Any
    objective_history: tuple[float, ...]
    data_history: tuple[float, ...]
    prior_history: tuple[float, ...]
    holdout_history: tuple[float, ...]
    holdout_steps: tuple[int, ...]
    best_step: int
    steps: int
    converged: bool


@dataclass(frozen=True)
class QuadtreeInferenceResult:
    """Fixed-topology positive-flux fit and its visibility residual."""

    topology: QuadtreeTopology
    flux: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    raw_parameters: np.ndarray
    optimizer_state: Any
    objective_history: tuple[float, ...]
    data_history: tuple[float, ...]
    prior_history: tuple[float, ...]
    holdout_history: tuple[float, ...]
    holdout_steps: tuple[int, ...]
    leaf_penalty: float
    topology_penalty: float
    best_step: int
    steps: int
    converged: bool


@dataclass(frozen=True)
class _PositiveFluxFit:
    flux: np.ndarray
    raw_parameters: np.ndarray
    optimizer_state: Any
    objective_history: tuple[float, ...]
    data_history: tuple[float, ...]
    prior_history: tuple[float, ...]
    holdout_history: tuple[float, ...]
    holdout_steps: tuple[int, ...]
    best_step: int
    steps: int
    converged: bool


def create_optimizer(config: InferenceConfig) -> optax.GradientTransformation:
    schedule = optax.cosine_decay_schedule(config.learning_rate, max(config.steps, 1), 0.1)
    return optax.adam(schedule)


def _inference_dtypes(config: InferenceConfig) -> tuple[Any, Any]:
    if (
        config.operator_mode == "explicit"
        and config.direct_dft.precision == "float32"
    ):
        return jnp.float32, jnp.complex64
    return jnp.float64, jnp.complex128


def _validate_inference_inputs(
    block: VisibilityBlock,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray | None,
    config: InferenceConfig,
) -> None:
    if train_mask.shape != block.shape:
        raise ValueError("train_mask must match the visibility block")
    if holdout_mask is not None:
        if holdout_mask.shape != block.shape:
            raise ValueError("holdout_mask must match the visibility block")
        if np.any(train_mask & holdout_mask):
            raise ValueError("train_mask and holdout_mask must be disjoint")
        if not np.any(holdout_mask):
            raise ValueError("holdout_mask must contain active samples")
    if config.steps < 1 or config.learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive")
    if config.validation_interval < 1:
        raise ValueError("validation_interval must be positive")
    if config.operator_mode not in {"autodiff", "explicit"}:
        raise ValueError("operator_mode must be autodiff or explicit")


def _fit_positive_flux(
    parameter_count: int,
    config: InferenceConfig,
    terms: Callable[[jax.Array], tuple[jax.Array, tuple[jax.Array, jax.Array]]],
    holdout_data: Callable[[jax.Array], jax.Array],
    *,
    has_holdout: bool,
    initial_raw: np.ndarray | None,
    initial_optimizer_state: Any | None,
) -> _PositiveFluxFit:
    """Run the shared fixed-support optimizer for a positive flux vector."""

    real_dtype, _ = _inference_dtypes(config)
    value_and_gradient = jax.jit(jax.value_and_grad(terms, has_aux=True))
    optimizer = create_optimizer(config)
    if initial_raw is None:
        raw = jnp.full(
            parameter_count,
            raw_from_intensity(config.initial_intensity),
            dtype=real_dtype,
        )
    else:
        raw = jnp.asarray(initial_raw, dtype=real_dtype).reshape(-1)
        if raw.shape != (parameter_count,):
            raise ValueError(
                f"initial_raw has {raw.size} parameters; expected {parameter_count}"
            )
    optimizer_state = (
        optimizer.init(raw) if initial_optimizer_state is None else initial_optimizer_state
    )
    best_raw = raw
    best_optimizer_state = optimizer_state
    best_objective = np.inf
    best_holdout = np.inf
    best_step = 0
    stale_steps = 0
    objectives: list[float] = []
    data_values: list[float] = []
    prior_values: list[float] = []
    holdout_values: list[float] = []
    holdout_steps: list[int] = []
    converged = False

    @jax.jit
    def update(
        parameters: jax.Array, state: Any
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, jax.Array]:
        (_, _), gradient = value_and_gradient(parameters)
        updates, next_state = optimizer.update(gradient, state, parameters)
        next_parameters = optax.apply_updates(parameters, updates)
        objective, (data_term, prior_term) = terms(next_parameters)
        return (
            next_parameters,
            next_state,
            objective,
            data_term,
            prior_term,
        )

    last_validation_step = 0
    for step_index in range(config.steps):
        raw, optimizer_state, objective, data_term, prior_term = update(raw, optimizer_state)
        step = step_index + 1
        objective_value = float(objective)
        objectives.append(objective_value)
        data_values.append(float(data_term))
        prior_values.append(float(prior_term))
        if not has_holdout:
            if objective_value < best_objective - config.min_delta:
                best_objective = objective_value
                best_raw = raw
                best_optimizer_state = optimizer_state
                best_step = step
                stale_steps = 0
            else:
                stale_steps += 1
        elif step % config.validation_interval == 0 or step == config.steps:
            holdout_value = float(holdout_data(raw))
            holdout_values.append(holdout_value)
            holdout_steps.append(step)
            elapsed_steps = step - last_validation_step
            last_validation_step = step
            if holdout_value < best_holdout - config.min_delta:
                best_holdout = holdout_value
                best_raw = raw
                best_optimizer_state = optimizer_state
                best_step = step
                stale_steps = 0
            else:
                stale_steps += elapsed_steps
        if stale_steps >= config.patience:
            converged = True
            break

    return _PositiveFluxFit(
        flux=np.asarray(physical_intensity(best_raw)),
        raw_parameters=np.asarray(best_raw),
        optimizer_state=best_optimizer_state,
        objective_history=tuple(objectives),
        data_history=tuple(data_values),
        prior_history=tuple(prior_values),
        holdout_history=tuple(holdout_values),
        holdout_steps=tuple(holdout_steps),
        best_step=best_step,
        steps=len(objectives),
        converged=converged,
    )


def infer_regular_grid(
    block: VisibilityBlock,
    grid: RegularGrid,
    train_mask: np.ndarray,
    config: InferenceConfig | None = None,
    *,
    holdout_mask: np.ndarray | None = None,
    fixed_gains: np.ndarray | None = None,
    pixel_basis: PixelBasis | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    initial_raw: np.ndarray | None = None,
    initial_optimizer_state: Any | None = None,
) -> InferenceResult:
    configuration = config or InferenceConfig()
    selected_basis = pixel_basis or DeltaPixelBasis()
    _validate_inference_inputs(block, train_mask, holdout_mask, configuration)
    real_dtype, complex_dtype = _inference_dtypes(configuration)
    l, m = grid.coordinates
    observation = jnp.asarray(block.visibility, dtype=complex_dtype)
    weight = jnp.asarray(block.weight, dtype=real_dtype)
    training_flag = jnp.asarray(~train_mask)
    beam_i, beam_rr, beam_ll = predict_beam_weights(
        primary_beam, l, m, block.frequency_hz
    )
    beam_i_array = None if beam_i is None else jnp.asarray(beam_i, dtype=real_dtype)
    beam_rr_array = (
        None if beam_rr is None else jnp.asarray(beam_rr, dtype=real_dtype)
    )
    beam_ll_array = (
        None if beam_ll is None else jnp.asarray(beam_ll, dtype=real_dtype)
    )

    def predict(raw_parameters: jax.Array) -> jax.Array:
        intensity = physical_intensity(raw_parameters)
        if configuration.operator_mode == "explicit":
            return predict_stokes_i_explicit(
                intensity,
                l,
                m,
                block.uvw_m,
                block.frequency_hz,
                block.antenna1,
                block.antenna2,
                block.correlations,
                fixed_gains=fixed_gains,
                pixel_basis=selected_basis,
                pixel_size_rad=grid.pixel_size_rad,
                config=configuration.direct_dft,
                beam_weights=beam_i_array,
                beam_weights_rr=beam_rr_array,
                beam_weights_ll=beam_ll_array,
            )
        return predict_stokes_i(
            intensity,
            l,
            m,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            fixed_gains=fixed_gains,
            chunk_size=configuration.chunk_size,
            pixel_basis=selected_basis,
            pixel_size_rad=grid.pixel_size_rad,
            beam_weights=beam_i_array,
            beam_weights_rr=beam_rr_array,
            beam_weights_ll=beam_ll_array,
        )

    def terms(raw_parameters: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        intensity = physical_intensity(raw_parameters)
        prediction = predict(raw_parameters)
        data_term = weighted_complex_mse(prediction, observation, weight, training_flag)
        prior_term = sky_prior(
            intensity,
            size=grid.size,
            sparsity_weight=configuration.sparsity_weight,
            smoothness_weight=configuration.smoothness_weight,
        )
        return data_term + prior_term, (data_term, prior_term)

    holdout_flag = None if holdout_mask is None else jnp.asarray(~holdout_mask)

    @jax.jit
    def holdout_data(raw_parameters: jax.Array) -> jax.Array:
        if holdout_flag is None:
            return jnp.asarray(jnp.inf, dtype=real_dtype)
        return weighted_complex_mse(
            predict(raw_parameters),
            observation,
            weight,
            holdout_flag,
        )

    fit = _fit_positive_flux(
        grid.size * grid.size,
        configuration,
        terms,
        holdout_data,
        has_holdout=holdout_mask is not None,
        initial_raw=initial_raw,
        initial_optimizer_state=initial_optimizer_state,
    )
    return InferenceResult(
        image=fit.flux.reshape(grid.size, grid.size),
        raw_parameters=fit.raw_parameters,
        optimizer_state=fit.optimizer_state,
        objective_history=fit.objective_history,
        data_history=fit.data_history,
        prior_history=fit.prior_history,
        holdout_history=fit.holdout_history,
        holdout_steps=fit.holdout_steps,
        best_step=fit.best_step,
        steps=fit.steps,
        converged=fit.converged,
    )


def infer_quadtree(
    block: VisibilityBlock,
    topology: QuadtreeTopology,
    train_mask: np.ndarray,
    config: InferenceConfig | None = None,
    *,
    holdout_mask: np.ndarray | None = None,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    leaf_penalty: float = 0.0,
    initial_flux: np.ndarray | None = None,
    initial_raw: np.ndarray | None = None,
    initial_optimizer_state: Any | None = None,
) -> QuadtreeInferenceResult:
    """Fit positive leaf flux while holding a quadtree topology fixed.

    Topology changes belong outside this function. Each call is one optimization
    epoch with fixed array shapes, so the objective remains jittable. The prior
    is positive L1 flux plus ``leaf_penalty * len(topology.leaves)``. The latter
    is constant within an epoch but makes objectives comparable across proposed
    topologies; a split therefore costs ``3 * leaf_penalty``.

    ``smoothness_weight`` is rejected because the regular-grid neighbour prior
    has no unambiguous meaning on unequal-area leaves. Pass either physical
    ``initial_flux`` or ``initial_raw`` to warm-start an epoch, but not both.
    """

    configuration = config or InferenceConfig()
    _validate_inference_inputs(block, train_mask, holdout_mask, configuration)
    if not topology.leaves:
        raise ValueError("quadtree topology must contain at least one leaf")
    if configuration.smoothness_weight != 0:
        raise ValueError("smoothness_weight is not defined for quadtree inference")
    if not np.isfinite(leaf_penalty) or leaf_penalty < 0:
        raise ValueError("leaf_penalty must be finite and non-negative")
    if initial_flux is not None and initial_raw is not None:
        raise ValueError("pass either initial_flux or initial_raw, not both")

    approximation = GaussianApproximation(approximation)
    real_dtype, complex_dtype = _inference_dtypes(configuration)
    l, m = topology.centers()
    observation = jnp.asarray(block.visibility, dtype=complex_dtype)
    weight = jnp.asarray(block.weight, dtype=real_dtype)
    training_flag = jnp.asarray(~train_mask)
    beam_i, beam_rr, beam_ll = predict_beam_weights(
        primary_beam, l, m, block.frequency_hz
    )
    beam_i_array = None if beam_i is None else jnp.asarray(beam_i, dtype=real_dtype)
    beam_rr_array = (
        None if beam_rr is None else jnp.asarray(beam_rr, dtype=real_dtype)
    )
    beam_ll_array = (
        None if beam_ll is None else jnp.asarray(beam_ll, dtype=real_dtype)
    )
    topology_penalty = float(leaf_penalty * len(topology.leaves))

    def predict_flux(flux: jax.Array) -> jax.Array:
        if configuration.operator_mode == "explicit":
            return predict_quadtree_stokes_i_explicit(
                flux,
                topology,
                block.uvw_m,
                block.frequency_hz,
                block.antenna1,
                block.antenna2,
                block.correlations,
                approximation=approximation,
                fixed_gains=fixed_gains,
                config=configuration.direct_dft,
                beam_weights=beam_i_array,
                beam_weights_rr=beam_rr_array,
                beam_weights_ll=beam_ll_array,
            )
        return predict_quadtree_stokes_i(
            flux,
            topology,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            approximation=approximation,
            fixed_gains=fixed_gains,
            chunk_size=configuration.chunk_size,
            beam_weights=beam_i_array,
            beam_weights_rr=beam_rr_array,
            beam_weights_ll=beam_ll_array,
        )

    def predict(raw_parameters: jax.Array) -> jax.Array:
        return predict_flux(physical_intensity(raw_parameters))

    def terms(raw_parameters: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        flux = physical_intensity(raw_parameters)
        prediction = predict_flux(flux)
        data_term = weighted_complex_mse(prediction, observation, weight, training_flag)
        prior_term = configuration.sparsity_weight * jnp.sum(flux) + jnp.asarray(
            topology_penalty, dtype=real_dtype
        )
        return data_term + prior_term, (data_term, prior_term)

    holdout_flag = None if holdout_mask is None else jnp.asarray(~holdout_mask)

    @jax.jit
    def holdout_data(raw_parameters: jax.Array) -> jax.Array:
        if holdout_flag is None:
            return jnp.asarray(jnp.inf, dtype=real_dtype)
        return weighted_complex_mse(
            predict(raw_parameters),
            observation,
            weight,
            holdout_flag,
        )

    selected_initial_raw = initial_raw
    if initial_flux is not None:
        initial_flux_array = np.asarray(initial_flux, dtype=np.float64).reshape(-1)
        if initial_flux_array.shape != (len(topology.leaves),):
            raise ValueError(
                "initial_flux must contain exactly one value per topology leaf"
            )
        if not np.all(np.isfinite(initial_flux_array)):
            raise ValueError("initial_flux must be finite")
        if np.any(initial_flux_array < 0):
            raise ValueError("initial_flux must be non-negative")
        selected_initial_raw = np.asarray(
            raw_from_intensity(jnp.asarray(initial_flux_array, dtype=real_dtype))
        )

    fit = _fit_positive_flux(
        len(topology.leaves),
        configuration,
        terms,
        holdout_data,
        has_holdout=holdout_mask is not None,
        initial_raw=selected_initial_raw,
        initial_optimizer_state=initial_optimizer_state,
    )
    prediction = np.asarray(predict_flux(jnp.asarray(fit.flux, dtype=real_dtype)))
    return QuadtreeInferenceResult(
        topology=topology,
        flux=fit.flux,
        prediction=prediction,
        residual=prediction - block.visibility,
        raw_parameters=fit.raw_parameters,
        optimizer_state=fit.optimizer_state,
        objective_history=fit.objective_history,
        data_history=fit.data_history,
        prior_history=fit.prior_history,
        holdout_history=fit.holdout_history,
        holdout_steps=fit.holdout_steps,
        leaf_penalty=float(leaf_penalty),
        topology_penalty=topology_penalty,
        best_step=fit.best_step,
        steps=fit.steps,
        converged=fit.converged,
    )


def save_checkpoint(
    path: str | Path, result: InferenceResult | QuadtreeInferenceResult
) -> None:
    leaves, _ = jax.tree_util.tree_flatten(
        (jnp.asarray(result.raw_parameters), result.optimizer_state)
    )
    with Path(path).open("wb") as stream:
        np.savez(
            stream,
            *(np.asarray(leaf) for leaf in leaves),
            step=result.best_step,
        )


def load_checkpoint(
    path: str | Path,
    config: InferenceConfig,
    parameter_count: int,
) -> tuple[np.ndarray, Any, int]:
    optimizer = create_optimizer(config)
    real_dtype, _ = _inference_dtypes(config)
    template_raw = jnp.zeros(parameter_count, dtype=real_dtype)
    template = (template_raw, optimizer.init(template_raw))
    template_leaves, tree = jax.tree_util.tree_flatten(template)
    with np.load(Path(path)) as stored:
        leaves = [jnp.asarray(stored[f"arr_{index}"]) for index in range(len(template_leaves))]
        step = int(stored["step"])
    raw, optimizer_state = jax.tree_util.tree_unflatten(tree, leaves)
    if raw.shape != (parameter_count,):
        raise ValueError(
            f"checkpoint has {raw.size} parameters; expected {parameter_count}"
        )
    return np.asarray(raw), optimizer_state, step
