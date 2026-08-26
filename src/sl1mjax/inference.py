"""Positive fixed-support Stokes-I inference with proximal and Adam solvers."""

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
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.objective import effective_weight, sky_prior, weighted_complex_mse
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
    solver: Literal["softplus_adam", "fista", "proximal_sgd", "hybrid"] = "softplus_adam"
    batch_size_rows: int = 1024
    random_seed: int = 0
    hybrid_sgd_fraction: float = 0.5
    backtracking_factor: float = 2.0
    kkt_tolerance: float = 1e-5
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
    solver: str = "softplus_adam"
    objective_steps: tuple[int, ...] = ()
    stationarity_history: tuple[float, ...] = ()
    stationarity_steps: tuple[int, ...] = ()
    kkt_residual: float = float("nan")


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
    solver: str = "softplus_adam"
    objective_steps: tuple[int, ...] = ()
    stationarity_history: tuple[float, ...] = ()
    stationarity_steps: tuple[int, ...] = ()
    kkt_residual: float = float("nan")


@dataclass(frozen=True)
class MosaicQuadtreeInferenceResult:
    """One shared quadtree sky fitted to several pointing-specific blocks."""

    topology: QuadtreeTopology
    flux: np.ndarray
    predictions: tuple[np.ndarray, ...]
    residuals: tuple[np.ndarray, ...]
    mosaic_phase_centre_rad: tuple[float, float]
    raw_parameters: np.ndarray
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
    solver: str
    objective_steps: tuple[int, ...]
    stationarity_history: tuple[float, ...]
    stationarity_steps: tuple[int, ...]
    kkt_residual: float


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
    solver: str = "softplus_adam"
    objective_steps: tuple[int, ...] = ()
    stationarity_history: tuple[float, ...] = ()
    stationarity_steps: tuple[int, ...] = ()
    kkt_residual: float = float("nan")


def create_optimizer(config: InferenceConfig) -> optax.GradientTransformation:
    schedule = optax.cosine_decay_schedule(config.learning_rate, max(config.steps, 1), 0.1)
    return optax.adam(schedule)


def _inference_dtypes(config: InferenceConfig) -> tuple[Any, Any]:
    if config.operator_mode == "explicit" and config.direct_dft.precision == "float32":
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
    if config.solver not in {
        "softplus_adam",
        "fista",
        "proximal_sgd",
        "hybrid",
    }:
        raise ValueError("solver must be softplus_adam, fista, proximal_sgd, or hybrid")
    if config.batch_size_rows < 1:
        raise ValueError("batch_size_rows must be positive")
    if not 0 < config.hybrid_sgd_fraction < 1:
        raise ValueError("hybrid_sgd_fraction must be between zero and one")
    if config.backtracking_factor <= 1:
        raise ValueError("backtracking_factor must be greater than one")
    if not np.isfinite(config.kkt_tolerance) or config.kkt_tolerance <= 0:
        raise ValueError("kkt_tolerance must be finite and positive")
    if config.operator_mode not in {"autodiff", "explicit"}:
        raise ValueError("operator_mode must be autodiff or explicit")


def positive_l1_kkt_residual(
    flux: jax.Array,
    smooth_gradient: jax.Array,
    sparsity_weight: float | jax.Array,
) -> jax.Array:
    """Return the maximum physical KKT violation for positive L1 flux.

    At a positive component the smooth gradient must equal ``-lambda``. At
    zero it may be larger than ``-lambda``, but it must not point into the
    feasible positive half-line strongly enough to overcome the L1 penalty.
    """

    flux_array = jnp.asarray(flux)
    shifted_gradient = jnp.asarray(smooth_gradient) + jnp.asarray(
        sparsity_weight, dtype=flux_array.dtype
    )
    component_residual = jnp.where(
        flux_array > 0,
        jnp.abs(shifted_gradient),
        jnp.maximum(-shifted_gradient, 0.0),
    )
    return jnp.max(component_residual)


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
            raise ValueError(f"initial_raw has {raw.size} parameters; expected {parameter_count}")
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
        solver="softplus_adam",
        objective_steps=tuple(range(1, len(objectives) + 1)),
    )


def _fit_physical_flux(
    parameter_count: int,
    config: InferenceConfig,
    full_terms: Callable[[jax.Array], tuple[jax.Array, tuple[jax.Array, jax.Array]]],
    smooth_objective: Callable[[jax.Array], jax.Array],
    batch_smooth_objective: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None,
    holdout_data: Callable[[jax.Array], jax.Array],
    *,
    has_holdout: bool,
    eligible_rows: np.ndarray,
    initial_flux: np.ndarray,
    sparsity_weights: np.ndarray | None = None,
) -> _PositiveFluxFit:
    """Fit non-negative flux directly with proximal stochastic or full steps."""

    real_dtype, _ = _inference_dtypes(config)
    flux = jnp.asarray(initial_flux, dtype=real_dtype).reshape(-1)
    if flux.shape != (parameter_count,):
        raise ValueError(f"initial flux has {flux.size} parameters; expected {parameter_count}")
    if not np.all(np.isfinite(np.asarray(flux))) or np.any(np.asarray(flux) < 0):
        raise ValueError("initial flux must be finite and non-negative")
    if sparsity_weights is None:
        proximal_penalty = jnp.asarray(config.sparsity_weight, dtype=real_dtype)
    else:
        selected_sparsity_weights = np.asarray(sparsity_weights, dtype=np.float64).reshape(-1)
        if selected_sparsity_weights.shape != (parameter_count,):
            raise ValueError(
                "sparsity_weights must contain exactly one value per flux parameter"
            )
        if not np.all(np.isfinite(selected_sparsity_weights)) or np.any(
            selected_sparsity_weights < 0
        ):
            raise ValueError("sparsity_weights must be finite and non-negative")
        proximal_penalty = jnp.asarray(
            config.sparsity_weight * selected_sparsity_weights,
            dtype=real_dtype,
        )

    row_indices = np.flatnonzero(np.asarray(eligible_rows, dtype=bool))
    if row_indices.size == 0:
        raise ValueError("train_mask must contain positive-weight finite samples")

    evaluated_terms = jax.jit(full_terms)
    evaluated_smooth = jax.jit(smooth_objective)
    smooth_value_and_gradient = jax.jit(jax.value_and_grad(smooth_objective))
    evaluated_holdout = jax.jit(holdout_data)
    evaluated_batch_gradient = (
        None if batch_smooth_objective is None else jax.jit(jax.grad(batch_smooth_objective))
    )

    objectives: list[float] = []
    objective_steps: list[int] = []
    data_values: list[float] = []
    prior_values: list[float] = []
    holdout_values: list[float] = []
    holdout_steps: list[int] = []
    stationarity_values: list[float] = []
    stationarity_steps: list[int] = []

    initial_objective, _ = evaluated_terms(flux)
    best_objective = float(initial_objective)
    best_holdout = float(evaluated_holdout(flux)) if has_holdout else np.inf
    best_flux = flux
    best_step = 0
    converged = False
    completed_steps = 0

    def record_evaluation(candidate: jax.Array, step: int) -> float:
        nonlocal best_flux, best_holdout, best_objective, best_step, converged
        objective, (data_term, prior_term) = evaluated_terms(candidate)
        objective_value = float(objective)
        objectives.append(objective_value)
        objective_steps.append(step)
        data_values.append(float(data_term))
        prior_values.append(float(prior_term))
        if has_holdout:
            holdout_value = float(evaluated_holdout(candidate))
            holdout_values.append(holdout_value)
            holdout_steps.append(step)
        _, smooth_gradient = smooth_value_and_gradient(candidate)
        stationarity = float(
            positive_l1_kkt_residual(candidate, smooth_gradient, proximal_penalty)
        )
        stationarity_values.append(stationarity)
        stationarity_steps.append(step)
        if has_holdout and holdout_value < best_holdout - config.min_delta:
            best_holdout = holdout_value
            best_flux = candidate
            best_step = step
        elif not has_holdout and objective_value < best_objective:
            best_objective = objective_value
            best_flux = candidate
            best_step = step
        if stationarity <= config.kkt_tolerance:
            converged = True
        return objective_value

    def run_sgd(current: jax.Array, step_count: int, start_step: int) -> jax.Array:
        nonlocal completed_steps
        if step_count == 0:
            return current
        if evaluated_batch_gradient is None:
            raise ValueError("proximal SGD requires a batch objective")
        rng = np.random.default_rng(config.random_seed)
        order = rng.permutation(row_indices)
        cursor = 0
        batch_size = config.batch_size_rows
        for local_step in range(step_count):
            if cursor >= order.size:
                order = rng.permutation(row_indices)
                cursor = 0
            count = min(batch_size, order.size - cursor)
            selected = np.zeros(batch_size, dtype=np.int32)
            selected[:count] = order[cursor : cursor + count]
            valid = np.zeros(batch_size, dtype=bool)
            valid[:count] = True
            cursor += count
            progress = local_step / max(step_count - 1, 1)
            decay = 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))
            step_size = config.learning_rate * decay
            gradient = evaluated_batch_gradient(
                current,
                jnp.asarray(selected),
                jnp.asarray(valid),
            )
            current = jnp.maximum(
                current - step_size * gradient - step_size * proximal_penalty,
                0.0,
            )
            step = start_step + local_step + 1
            completed_steps = step
            if step % config.validation_interval == 0 or local_step + 1 == step_count:
                record_evaluation(current, step)
                if converged:
                    break
        return current

    def run_fista(current: jax.Array, step_count: int, start_step: int) -> jax.Array:
        nonlocal best_flux, best_holdout, best_objective, best_step, completed_steps, converged
        if step_count == 0:
            return current
        momentum_point = current
        momentum_scale = 1.0
        inverse_step = 1.0 / config.learning_rate
        current_objective = float(evaluated_terms(current)[0])

        def proximal_step(
            base: jax.Array, trial_inverse_step: float
        ) -> tuple[jax.Array, float, jax.Array]:
            smooth_value, gradient = smooth_value_and_gradient(base)
            smooth_scalar = float(smooth_value)
            if not np.isfinite(smooth_scalar) or not np.all(np.isfinite(np.asarray(gradient))):
                raise RuntimeError(
                    "FISTA encountered a non-finite smooth value or gradient at its base point"
                )
            local_inverse_step = max(
                trial_inverse_step / config.backtracking_factor,
                np.finfo(np.float64).tiny,
            )
            candidate_scalar = float("nan")
            majorant_scalar = float("nan")
            for _ in range(60):
                candidate = jnp.maximum(
                    base
                    - gradient / local_inverse_step
                    - proximal_penalty / local_inverse_step,
                    0.0,
                )
                difference = candidate - base
                candidate_smooth = evaluated_smooth(candidate)
                majorant = (
                    smooth_value
                    + jnp.vdot(gradient, difference).real
                    + 0.5 * local_inverse_step * jnp.vdot(difference, difference).real
                )
                candidate_scalar = float(candidate_smooth)
                majorant_scalar = float(majorant)
                roundoff_scale = max(
                    abs(smooth_scalar),
                    abs(candidate_scalar),
                    abs(majorant_scalar),
                    np.finfo(np.float64).tiny,
                )
                roundoff_tolerance = 64.0 * np.finfo(np.dtype(real_dtype)).eps * roundoff_scale
                if (
                    np.isfinite(candidate_scalar)
                    and np.isfinite(majorant_scalar)
                    and candidate_scalar <= majorant_scalar + roundoff_tolerance
                ):
                    return candidate, local_inverse_step, gradient
                local_inverse_step *= config.backtracking_factor
            gradient_max = float(np.max(np.abs(np.asarray(gradient))))
            raise RuntimeError(
                "FISTA backtracking failed to find a finite step: "
                f"smooth={smooth_scalar:.9g}, candidate={candidate_scalar:.9g}, "
                f"majorant={majorant_scalar:.9g}, gradient_max={gradient_max:.9g}, "
                f"inverse_step={local_inverse_step:.9g}, dtype={np.dtype(real_dtype)}"
            )

        for local_step in range(step_count):
            candidate, inverse_step, _ = proximal_step(momentum_point, inverse_step)
            candidate_objective = float(evaluated_terms(candidate)[0])
            if candidate_objective > current_objective + config.min_delta:
                momentum_point = current
                momentum_scale = 1.0
                candidate, inverse_step, _ = proximal_step(momentum_point, inverse_step)
                candidate_objective = float(evaluated_terms(candidate)[0])
                while candidate_objective > current_objective + config.min_delta:
                    inverse_step *= config.backtracking_factor
                    candidate, inverse_step, _ = proximal_step(
                        momentum_point,
                        inverse_step * config.backtracking_factor,
                    )
                    candidate_objective = float(evaluated_terms(candidate)[0])

            next_momentum_scale = (1.0 + np.sqrt(1.0 + 4.0 * momentum_scale**2)) / 2.0
            extrapolation = (momentum_scale - 1.0) / next_momentum_scale
            momentum_point = candidate + extrapolation * (candidate - current)
            current = candidate
            momentum_scale = next_momentum_scale
            current_objective = candidate_objective
            step = start_step + local_step + 1
            completed_steps = step

            objective, (data_term, prior_term) = evaluated_terms(current)
            objectives.append(float(objective))
            objective_steps.append(step)
            data_values.append(float(data_term))
            prior_values.append(float(prior_term))
            evaluated_holdout_here = has_holdout and (
                step % config.validation_interval == 0 or local_step + 1 == step_count
            )
            if evaluated_holdout_here:
                holdout_value = float(evaluated_holdout(current))
                holdout_values.append(holdout_value)
                holdout_steps.append(step)
            if evaluated_holdout_here and holdout_value < best_holdout - config.min_delta:
                best_holdout = holdout_value
                best_flux = current
                best_step = step
            elif not has_holdout and float(objective) < best_objective:
                best_objective = float(objective)
                best_flux = current
                best_step = step
            if step % config.validation_interval == 0 or local_step + 1 == step_count:
                _, smooth_gradient = smooth_value_and_gradient(current)
                stationarity = float(
                    positive_l1_kkt_residual(
                        current,
                        smooth_gradient,
                        proximal_penalty,
                    )
                )
                stationarity_values.append(stationarity)
                stationarity_steps.append(step)
                if stationarity <= config.kkt_tolerance:
                    converged = True
                    break
        return current

    if config.solver == "proximal_sgd":
        flux = run_sgd(flux, config.steps, 0)
    elif config.solver == "fista":
        flux = run_fista(flux, config.steps, 0)
    elif config.solver == "hybrid":
        stochastic_steps = int(round(config.steps * config.hybrid_sgd_fraction))
        stochastic_steps = min(max(stochastic_steps, 1), config.steps - 1)
        flux = run_sgd(flux, stochastic_steps, 0)
        if not converged:
            flux = best_flux
            run_fista(flux, config.steps - stochastic_steps, stochastic_steps)
    else:
        raise ValueError("physical flux solver received softplus_adam")

    _, best_smooth_gradient = smooth_value_and_gradient(best_flux)
    best_kkt_residual = float(
        positive_l1_kkt_residual(
            best_flux,
            best_smooth_gradient,
            proximal_penalty,
        )
    )
    selected_converged = best_kkt_residual <= config.kkt_tolerance
    best_flux_array = np.asarray(best_flux)
    return _PositiveFluxFit(
        flux=best_flux_array,
        raw_parameters=np.asarray(
            raw_from_intensity(jnp.asarray(best_flux_array, dtype=real_dtype))
        ),
        optimizer_state=None,
        objective_history=tuple(objectives),
        data_history=tuple(data_values),
        prior_history=tuple(prior_values),
        holdout_history=tuple(holdout_values),
        holdout_steps=tuple(holdout_steps),
        best_step=best_step,
        steps=completed_steps,
        converged=selected_converged,
        solver=config.solver,
        objective_steps=tuple(objective_steps),
        stationarity_history=tuple(stationarity_values),
        stationarity_steps=tuple(stationarity_steps),
        kkt_residual=best_kkt_residual,
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
    uvw = jnp.asarray(block.uvw_m, dtype=real_dtype)
    frequency = jnp.asarray(block.frequency_hz, dtype=real_dtype)
    antenna1 = jnp.asarray(block.antenna1)
    antenna2 = jnp.asarray(block.antenna2)
    beam_i, beam_rr, beam_ll = predict_beam_weights(primary_beam, l, m, block.frequency_hz)
    beam_i_array = None if beam_i is None else jnp.asarray(beam_i, dtype=real_dtype)
    beam_rr_array = None if beam_rr is None else jnp.asarray(beam_rr, dtype=real_dtype)
    beam_ll_array = None if beam_ll is None else jnp.asarray(beam_ll, dtype=real_dtype)

    def predict_flux(intensity: jax.Array) -> jax.Array:
        if configuration.operator_mode == "explicit":
            return predict_stokes_i_explicit(
                intensity,
                l,
                m,
                uvw,
                frequency,
                antenna1,
                antenna2,
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
            uvw,
            frequency,
            antenna1,
            antenna2,
            block.correlations,
            fixed_gains=fixed_gains,
            chunk_size=configuration.chunk_size,
            pixel_basis=selected_basis,
            pixel_size_rad=grid.pixel_size_rad,
            beam_weights=beam_i_array,
            beam_weights_rr=beam_rr_array,
            beam_weights_ll=beam_ll_array,
        )

    def predict_flux_rows(intensity: jax.Array, rows: jax.Array) -> jax.Array:
        if configuration.operator_mode == "explicit":
            return predict_stokes_i_explicit(
                intensity,
                l,
                m,
                uvw[rows],
                frequency,
                antenna1[rows],
                antenna2[rows],
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
            uvw[rows],
            frequency,
            antenna1[rows],
            antenna2[rows],
            block.correlations,
            fixed_gains=fixed_gains,
            chunk_size=configuration.chunk_size,
            pixel_basis=selected_basis,
            pixel_size_rad=grid.pixel_size_rad,
            beam_weights=beam_i_array,
            beam_weights_rr=beam_rr_array,
            beam_weights_ll=beam_ll_array,
        )

    def predict(raw_parameters: jax.Array) -> jax.Array:
        return predict_flux(physical_intensity(raw_parameters))

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

    def smooth_prior(intensity: jax.Array) -> jax.Array:
        return sky_prior(
            intensity,
            size=grid.size,
            sparsity_weight=0.0,
            smoothness_weight=configuration.smoothness_weight,
        )

    def physical_terms(
        intensity: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        prediction = predict_flux_rows(intensity, training_rows)
        data_term = weighted_complex_mse(
            prediction,
            observation[training_rows],
            weight[training_rows],
            training_flag[training_rows],
        )
        prior_term = configuration.sparsity_weight * jnp.sum(intensity) + smooth_prior(intensity)
        return data_term + prior_term, (data_term, prior_term)

    def physical_smooth_objective(intensity: jax.Array) -> jax.Array:
        prediction = predict_flux_rows(intensity, training_rows)
        return weighted_complex_mse(
            prediction,
            observation[training_rows],
            weight[training_rows],
            training_flag[training_rows],
        ) + smooth_prior(intensity)

    training_weight = effective_weight(observation, weight, training_flag)
    training_weight_sum = jnp.sum(training_weight)
    eligible_rows = np.any(
        np.asarray(training_weight) > 0,
        axis=(1, 2),
    )
    eligible_row_count = int(np.count_nonzero(eligible_rows))
    training_rows = jnp.asarray(np.flatnonzero(eligible_rows), dtype=jnp.int32)

    def batch_smooth_objective(
        intensity: jax.Array, rows: jax.Array, valid_rows: jax.Array
    ) -> jax.Array:
        prediction = predict_flux_rows(intensity, rows)
        selected_weight = training_weight[rows] * valid_rows[:, None, None]
        residual = jnp.where(
            selected_weight > 0,
            prediction - observation[rows],
            0.0,
        )
        selected_row_count = jnp.sum(valid_rows)
        data_term = (
            eligible_row_count
            * jnp.sum(selected_weight * jnp.abs(residual) ** 2)
            / (selected_row_count * training_weight_sum)
        )
        return data_term + smooth_prior(intensity)

    holdout_flag = None if holdout_mask is None else jnp.asarray(~holdout_mask)
    holdout_rows = (
        None
        if holdout_mask is None
        else jnp.asarray(
            np.flatnonzero(np.any(holdout_mask & block.active, axis=(1, 2))),
            dtype=jnp.int32,
        )
    )

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

    def physical_holdout_data(intensity: jax.Array) -> jax.Array:
        if holdout_flag is None or holdout_rows is None:
            return jnp.asarray(jnp.inf, dtype=real_dtype)
        return weighted_complex_mse(
            predict_flux_rows(intensity, holdout_rows),
            observation[holdout_rows],
            weight[holdout_rows],
            holdout_flag[holdout_rows],
        )

    if configuration.solver == "softplus_adam":
        fit = _fit_positive_flux(
            grid.size * grid.size,
            configuration,
            terms,
            holdout_data,
            has_holdout=holdout_mask is not None,
            initial_raw=initial_raw,
            initial_optimizer_state=initial_optimizer_state,
        )
    else:
        if initial_optimizer_state is not None:
            raise ValueError("initial_optimizer_state is only supported by softplus_adam")
        physical_initial_flux = (
            np.full(
                grid.size * grid.size,
                configuration.initial_intensity,
                dtype=np.float64,
            )
            if initial_raw is None
            else np.asarray(physical_intensity(jnp.asarray(initial_raw)))
        )
        fit = _fit_physical_flux(
            grid.size * grid.size,
            configuration,
            physical_terms,
            physical_smooth_objective,
            batch_smooth_objective,
            physical_holdout_data,
            has_holdout=holdout_mask is not None,
            eligible_rows=eligible_rows,
            initial_flux=physical_initial_flux,
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
        solver=fit.solver,
        objective_steps=fit.objective_steps,
        stationarity_history=fit.stationarity_history,
        stationarity_steps=fit.stationarity_steps,
        kkt_residual=fit.kkt_residual,
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
    uvw = jnp.asarray(block.uvw_m, dtype=real_dtype)
    frequency = jnp.asarray(block.frequency_hz, dtype=real_dtype)
    antenna1 = jnp.asarray(block.antenna1)
    antenna2 = jnp.asarray(block.antenna2)
    beam_i, beam_rr, beam_ll = predict_beam_weights(primary_beam, l, m, block.frequency_hz)
    beam_i_array = None if beam_i is None else jnp.asarray(beam_i, dtype=real_dtype)
    beam_rr_array = None if beam_rr is None else jnp.asarray(beam_rr, dtype=real_dtype)
    beam_ll_array = None if beam_ll is None else jnp.asarray(beam_ll, dtype=real_dtype)
    topology_penalty = float(leaf_penalty * len(topology.leaves))

    def predict_flux(flux: jax.Array) -> jax.Array:
        if configuration.operator_mode == "explicit":
            return predict_quadtree_stokes_i_explicit(
                flux,
                topology,
                uvw,
                frequency,
                antenna1,
                antenna2,
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
            uvw,
            frequency,
            antenna1,
            antenna2,
            block.correlations,
            approximation=approximation,
            fixed_gains=fixed_gains,
            chunk_size=configuration.chunk_size,
            beam_weights=beam_i_array,
            beam_weights_rr=beam_rr_array,
            beam_weights_ll=beam_ll_array,
        )

    def predict_flux_rows(flux: jax.Array, rows: jax.Array) -> jax.Array:
        if configuration.operator_mode == "explicit":
            return predict_quadtree_stokes_i_explicit(
                flux,
                topology,
                uvw[rows],
                frequency,
                antenna1[rows],
                antenna2[rows],
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
            uvw[rows],
            frequency,
            antenna1[rows],
            antenna2[rows],
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

    def physical_terms(
        flux: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        prediction = predict_flux_rows(flux, training_rows)
        data_term = weighted_complex_mse(
            prediction,
            observation[training_rows],
            weight[training_rows],
            training_flag[training_rows],
        )
        prior_term = configuration.sparsity_weight * jnp.sum(flux) + jnp.asarray(
            topology_penalty, dtype=real_dtype
        )
        return data_term + prior_term, (data_term, prior_term)

    def physical_smooth_objective(flux: jax.Array) -> jax.Array:
        return weighted_complex_mse(
            predict_flux_rows(flux, training_rows),
            observation[training_rows],
            weight[training_rows],
            training_flag[training_rows],
        )

    training_weight = effective_weight(observation, weight, training_flag)
    training_weight_sum = jnp.sum(training_weight)
    eligible_rows = np.any(
        np.asarray(training_weight) > 0,
        axis=(1, 2),
    )
    eligible_row_count = int(np.count_nonzero(eligible_rows))
    training_rows = jnp.asarray(np.flatnonzero(eligible_rows), dtype=jnp.int32)

    def batch_smooth_objective(
        flux: jax.Array, rows: jax.Array, valid_rows: jax.Array
    ) -> jax.Array:
        prediction = predict_flux_rows(flux, rows)
        selected_weight = training_weight[rows] * valid_rows[:, None, None]
        residual = jnp.where(
            selected_weight > 0,
            prediction - observation[rows],
            0.0,
        )
        selected_row_count = jnp.sum(valid_rows)
        return (
            eligible_row_count
            * jnp.sum(selected_weight * jnp.abs(residual) ** 2)
            / (selected_row_count * training_weight_sum)
        )

    holdout_flag = None if holdout_mask is None else jnp.asarray(~holdout_mask)
    holdout_rows = (
        None
        if holdout_mask is None
        else jnp.asarray(
            np.flatnonzero(np.any(holdout_mask & block.active, axis=(1, 2))),
            dtype=jnp.int32,
        )
    )

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

    def physical_holdout_data(flux: jax.Array) -> jax.Array:
        if holdout_flag is None or holdout_rows is None:
            return jnp.asarray(jnp.inf, dtype=real_dtype)
        return weighted_complex_mse(
            predict_flux_rows(flux, holdout_rows),
            observation[holdout_rows],
            weight[holdout_rows],
            holdout_flag[holdout_rows],
        )

    selected_initial_raw = initial_raw
    if initial_flux is not None:
        initial_flux_array = np.asarray(initial_flux, dtype=np.float64).reshape(-1)
        if initial_flux_array.shape != (len(topology.leaves),):
            raise ValueError("initial_flux must contain exactly one value per topology leaf")
        if not np.all(np.isfinite(initial_flux_array)):
            raise ValueError("initial_flux must be finite")
        if np.any(initial_flux_array < 0):
            raise ValueError("initial_flux must be non-negative")
        selected_initial_raw = np.asarray(
            raw_from_intensity(jnp.asarray(initial_flux_array, dtype=real_dtype))
        )

    if configuration.solver == "softplus_adam":
        fit = _fit_positive_flux(
            len(topology.leaves),
            configuration,
            terms,
            holdout_data,
            has_holdout=holdout_mask is not None,
            initial_raw=selected_initial_raw,
            initial_optimizer_state=initial_optimizer_state,
        )
    else:
        if initial_optimizer_state is not None:
            raise ValueError("initial_optimizer_state is only supported by softplus_adam")
        if initial_flux is not None:
            physical_initial_flux = initial_flux_array
        elif initial_raw is not None:
            physical_initial_flux = np.asarray(
                physical_intensity(jnp.asarray(initial_raw, dtype=real_dtype))
            )
        else:
            physical_initial_flux = np.full(
                len(topology.leaves),
                configuration.initial_intensity,
                dtype=np.float64,
            )
        fit = _fit_physical_flux(
            len(topology.leaves),
            configuration,
            physical_terms,
            physical_smooth_objective,
            batch_smooth_objective,
            physical_holdout_data,
            has_holdout=holdout_mask is not None,
            eligible_rows=eligible_rows,
            initial_flux=physical_initial_flux,
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
        solver=fit.solver,
        objective_steps=fit.objective_steps,
        stationarity_history=fit.stationarity_history,
        stationarity_steps=fit.stationarity_steps,
        kkt_residual=fit.kkt_residual,
    )


def infer_mosaic_quadtree(
    blocks: tuple[VisibilityBlock, ...],
    topology: QuadtreeTopology,
    train_masks: tuple[np.ndarray, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    config: InferenceConfig | None = None,
    *,
    holdout_masks: tuple[np.ndarray, ...] | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    leaf_penalty: float = 0.0,
    initial_flux: np.ndarray | None = None,
    sparsity_weights: np.ndarray | None = None,
) -> MosaicQuadtreeInferenceResult:
    """Fit one positive quadtree sky to several mosaic pointings.

    The topology is expressed in direction cosines about
    ``mosaic_phase_centre_rad``. Each block evaluates those same celestial
    leaf centres in its own phase-centred ``(l, m)`` frame. The Fourier phase
    and primary beam are therefore specific to each pointing, while leaf flux
    and its L1 penalty are shared once across the mosaic.

    The square width is reused in each local tangent frame. This differs from
    an exactly reprojected square only at second order in the separation of
    the pointing centres. The approximation is negligible for the few-arcmin
    3C391 mosaic, but should be revisited for degree-scale mosaics.

    This first joint implementation deliberately supports deterministic FISTA
    only. A stochastic mosaic solver needs a block-aware row sampler so its
    gradient remains unbiased across pointings.
    """

    configuration = config or InferenceConfig(solver="fista")
    if not blocks:
        raise ValueError("blocks must contain at least one visibility block")
    if len(train_masks) != len(blocks):
        raise ValueError("train_masks must contain one mask per block")
    if holdout_masks is not None and len(holdout_masks) != len(blocks):
        raise ValueError("holdout_masks must contain one mask per block")
    if configuration.solver != "fista":
        raise ValueError("mosaic quadtree inference currently requires solver='fista'")
    if configuration.operator_mode != "explicit":
        raise ValueError("mosaic quadtree inference requires operator_mode='explicit'")
    if configuration.smoothness_weight != 0:
        raise ValueError("smoothness_weight is not defined for quadtree inference")
    if not topology.leaves:
        raise ValueError("quadtree topology must contain at least one leaf")
    if not np.isfinite(leaf_penalty) or leaf_penalty < 0:
        raise ValueError("leaf_penalty must be finite and non-negative")
    selected_sparsity_weights = (
        np.ones(len(topology.leaves), dtype=np.float64)
        if sparsity_weights is None
        else np.asarray(sparsity_weights, dtype=np.float64).reshape(-1)
    )
    if selected_sparsity_weights.shape != (len(topology.leaves),):
        raise ValueError(
            "sparsity_weights must contain exactly one value per topology leaf"
        )
    if not np.all(np.isfinite(selected_sparsity_weights)) or np.any(
        selected_sparsity_weights < 0
    ):
        raise ValueError("sparsity_weights must be finite and non-negative")
    if len(mosaic_phase_centre_rad) != 2 or not np.all(np.isfinite(mosaic_phase_centre_rad)):
        raise ValueError("mosaic_phase_centre_rad must contain finite RA and Dec")

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

    approximation = GaussianApproximation(approximation)
    real_dtype, complex_dtype = _inference_dtypes(configuration)
    reference_l, reference_m = topology.centers()
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

    observations = tuple(jnp.asarray(block.visibility, dtype=complex_dtype) for block in blocks)
    weights = tuple(jnp.asarray(block.weight, dtype=real_dtype) for block in blocks)
    uvw = tuple(jnp.asarray(block.uvw_m, dtype=real_dtype) for block in blocks)
    frequencies = tuple(jnp.asarray(block.frequency_hz, dtype=real_dtype) for block in blocks)
    antenna1 = tuple(jnp.asarray(block.antenna1) for block in blocks)
    antenna2 = tuple(jnp.asarray(block.antenna2) for block in blocks)
    beam_arrays = tuple(
        predict_beam_weights(primary_beam, l, m, block.frequency_hz)
        for block, (l, m) in zip(blocks, local_centres, strict=True)
    )
    beam_i = tuple(
        None if values[0] is None else jnp.asarray(values[0], dtype=real_dtype)
        for values in beam_arrays
    )
    beam_rr = tuple(
        None if values[1] is None else jnp.asarray(values[1], dtype=real_dtype)
        for values in beam_arrays
    )
    beam_ll = tuple(
        None if values[2] is None else jnp.asarray(values[2], dtype=real_dtype)
        for values in beam_arrays
    )

    training_flags = tuple(jnp.asarray(~mask) for mask in train_masks)
    training_weights = tuple(
        effective_weight(observation, weight, flag)
        for observation, weight, flag in zip(observations, weights, training_flags, strict=True)
    )
    training_weight_sum = sum(jnp.sum(value) for value in training_weights)
    if float(training_weight_sum) <= 0:
        raise ValueError("train_masks must contain positive-weight finite samples")

    has_holdout = holdout_masks is not None and any(
        np.any(mask & block.active)
        for block, mask in zip(blocks, selected_holdout_masks, strict=True)
    )
    holdout_flags = tuple(jnp.asarray(~mask) for mask in selected_holdout_masks)
    holdout_weights = tuple(
        effective_weight(observation, weight, flag)
        for observation, weight, flag in zip(observations, weights, holdout_flags, strict=True)
    )
    holdout_weight_sum = sum(jnp.sum(value) for value in holdout_weights)
    topology_penalty = float(leaf_penalty * len(topology.leaves))
    sparsity_weight_array = jnp.asarray(selected_sparsity_weights, dtype=real_dtype)

    def predict_block(flux: jax.Array, index: int) -> jax.Array:
        return predict_quadtree_stokes_i_explicit(
            flux,
            topology,
            uvw[index],
            frequencies[index],
            antenna1[index],
            antenna2[index],
            blocks[index].correlations,
            approximation=approximation,
            config=configuration.direct_dft,
            beam_weights=beam_i[index],
            beam_weights_rr=beam_rr[index],
            beam_weights_ll=beam_ll[index],
            centers_lm=local_centres[index],
        )

    def weighted_data_term(
        flux: jax.Array,
        selected_weights: tuple[jax.Array, ...],
        denominator: jax.Array,
    ) -> jax.Array:
        numerator = jnp.asarray(0.0, dtype=real_dtype)
        for index, active_weight in enumerate(selected_weights):
            prediction = predict_block(flux, index)
            residual = jnp.where(
                active_weight > 0,
                prediction - observations[index],
                0.0,
            )
            numerator = numerator + jnp.sum(active_weight * jnp.abs(residual) ** 2)
        return numerator / denominator

    def smooth_objective(flux: jax.Array) -> jax.Array:
        return weighted_data_term(flux, training_weights, training_weight_sum)

    def full_terms(
        flux: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        data_term = smooth_objective(flux)
        prior_term = configuration.sparsity_weight * jnp.sum(
            sparsity_weight_array * flux
        ) + jnp.asarray(topology_penalty, dtype=real_dtype)
        return data_term + prior_term, (data_term, prior_term)

    def holdout_data(flux: jax.Array) -> jax.Array:
        if not has_holdout:
            return jnp.asarray(jnp.inf, dtype=real_dtype)
        return weighted_data_term(flux, holdout_weights, holdout_weight_sum)

    if initial_flux is None:
        physical_initial_flux = np.full(
            len(topology.leaves),
            configuration.initial_intensity,
            dtype=np.float64,
        )
    else:
        physical_initial_flux = np.asarray(initial_flux, dtype=np.float64).reshape(-1)
        if physical_initial_flux.shape != (len(topology.leaves),):
            raise ValueError("initial_flux must contain exactly one value per topology leaf")
        if not np.all(np.isfinite(physical_initial_flux)):
            raise ValueError("initial_flux must be finite")
        if np.any(physical_initial_flux < 0):
            raise ValueError("initial_flux must be non-negative")

    fit = _fit_physical_flux(
        len(topology.leaves),
        configuration,
        full_terms,
        smooth_objective,
        None,
        holdout_data,
        has_holdout=has_holdout,
        eligible_rows=np.ones(1, dtype=bool),
        initial_flux=physical_initial_flux,
        sparsity_weights=selected_sparsity_weights,
    )
    fitted_flux = jnp.asarray(fit.flux, dtype=real_dtype)
    predictions = tuple(
        np.asarray(predict_block(fitted_flux, index)) for index in range(len(blocks))
    )
    residuals = tuple(
        prediction - block.visibility for prediction, block in zip(predictions, blocks, strict=True)
    )
    return MosaicQuadtreeInferenceResult(
        topology=topology,
        flux=fit.flux,
        predictions=predictions,
        residuals=residuals,
        mosaic_phase_centre_rad=(
            float(mosaic_phase_centre_rad[0]),
            float(mosaic_phase_centre_rad[1]),
        ),
        raw_parameters=fit.raw_parameters,
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
        solver=fit.solver,
        objective_steps=fit.objective_steps,
        stationarity_history=fit.stationarity_history,
        stationarity_steps=fit.stationarity_steps,
        kkt_residual=fit.kkt_residual,
    )


def predict_mosaic_quadtree(
    blocks: tuple[VisibilityBlock, ...],
    topology: QuadtreeTopology,
    flux: np.ndarray,
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    config: DirectDFTConfig | None = None,
) -> tuple[np.ndarray, ...]:
    """Predict several pointing blocks from one fixed mosaic quadtree sky.

    This is the non-fitting counterpart of :func:`infer_mosaic_quadtree`. It is
    useful for evaluating held-out or already flagged samples without allowing
    those samples to alter either topology or leaf flux.
    """

    if not blocks:
        raise ValueError("blocks must contain at least one visibility block")
    if not topology.leaves:
        raise ValueError("quadtree topology must contain at least one leaf")
    if len(mosaic_phase_centre_rad) != 2 or not np.all(
        np.isfinite(mosaic_phase_centre_rad)
    ):
        raise ValueError("mosaic_phase_centre_rad must contain finite RA and Dec")
    flux_array = np.asarray(flux, dtype=np.float64).reshape(-1)
    if flux_array.shape != (len(topology.leaves),):
        raise ValueError("flux must contain exactly one value per topology leaf")
    if np.any(~np.isfinite(flux_array)) or np.any(flux_array < 0):
        raise ValueError("flux must be finite and non-negative")
    selected_config = config or DirectDFTConfig()
    approximation = GaussianApproximation(approximation)
    real_dtype = selected_config.real_dtype
    reference_l, reference_m = topology.centers()
    sky_ra, sky_dec = lmn_to_radec(
        mosaic_phase_centre_rad[0],
        mosaic_phase_centre_rad[1],
        reference_l,
        reference_m,
    )
    predictions: list[np.ndarray] = []
    for block in blocks:
        local_l, local_m, _ = radec_to_lmn(
            block.phase_centre_rad[0],
            block.phase_centre_rad[1],
            sky_ra,
            sky_dec,
        )
        beam_i, beam_rr, beam_ll = predict_beam_weights(
            primary_beam, local_l, local_m, block.frequency_hz
        )
        predictions.append(
            np.asarray(
                predict_quadtree_stokes_i_explicit(
                    jnp.asarray(flux_array, dtype=real_dtype),
                    topology,
                    jnp.asarray(block.uvw_m, dtype=real_dtype),
                    jnp.asarray(block.frequency_hz, dtype=real_dtype),
                    jnp.asarray(block.antenna1),
                    jnp.asarray(block.antenna2),
                    block.correlations,
                    approximation=approximation,
                    config=selected_config,
                    beam_weights=(
                        None
                        if beam_i is None
                        else jnp.asarray(beam_i, dtype=real_dtype)
                    ),
                    beam_weights_rr=(
                        None
                        if beam_rr is None
                        else jnp.asarray(beam_rr, dtype=real_dtype)
                    ),
                    beam_weights_ll=(
                        None
                        if beam_ll is None
                        else jnp.asarray(beam_ll, dtype=real_dtype)
                    ),
                    centers_lm=(local_l, local_m),
                )
            )
        )
    return tuple(predictions)


def save_checkpoint(path: str | Path, result: InferenceResult | QuadtreeInferenceResult) -> None:
    leaves, _ = jax.tree_util.tree_flatten(
        (jnp.asarray(result.raw_parameters), result.optimizer_state)
    )
    with Path(path).open("wb") as stream:
        np.savez(
            stream,
            *(np.asarray(leaf) for leaf in leaves),
            step=result.best_step,
            solver=result.solver,
        )


def load_checkpoint(
    path: str | Path,
    config: InferenceConfig,
    parameter_count: int,
) -> tuple[np.ndarray, Any, int]:
    real_dtype, _ = _inference_dtypes(config)
    template_raw = jnp.zeros(parameter_count, dtype=real_dtype)
    template_state = (
        create_optimizer(config).init(template_raw) if config.solver == "softplus_adam" else None
    )
    template = (template_raw, template_state)
    template_leaves, tree = jax.tree_util.tree_flatten(template)
    with np.load(Path(path)) as stored:
        if "solver" in stored and str(stored["solver"]) != config.solver:
            raise ValueError(
                f"checkpoint solver {str(stored['solver'])!r} does not match "
                f"configured solver {config.solver!r}"
            )
        leaves = [jnp.asarray(stored[f"arr_{index}"]) for index in range(len(template_leaves))]
        step = int(stored["step"])
    raw, optimizer_state = jax.tree_util.tree_unflatten(tree, leaves)
    if raw.shape != (parameter_count,):
        raise ValueError(f"checkpoint has {raw.size} parameters; expected {parameter_count}")
    return np.asarray(raw), optimizer_state, step
