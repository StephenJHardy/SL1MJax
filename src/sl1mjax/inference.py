"""Optax gradient inference for a positive regular Stokes-I grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.objective import sky_prior, weighted_complex_mse
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import (
    DeltaPixelBasis,
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


def infer_regular_grid(
    block: VisibilityBlock,
    grid: RegularGrid,
    train_mask: np.ndarray,
    config: InferenceConfig | None = None,
    *,
    fixed_gains: np.ndarray | None = None,
    pixel_basis: PixelBasis | None = None,
    initial_raw: np.ndarray | None = None,
    initial_optimizer_state: Any | None = None,
) -> InferenceResult:
    configuration = config or InferenceConfig()
    selected_basis = pixel_basis or DeltaPixelBasis()
    if train_mask.shape != block.shape:
        raise ValueError("train_mask must match the visibility block")
    if configuration.steps < 1 or configuration.learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive")
    if configuration.operator_mode not in {"autodiff", "explicit"}:
        raise ValueError("operator_mode must be autodiff or explicit")
    real_dtype, complex_dtype = _inference_dtypes(configuration)
    l, m = grid.coordinates
    observation = jnp.asarray(block.visibility, dtype=complex_dtype)
    weight = jnp.asarray(block.weight, dtype=real_dtype)
    training_flag = jnp.asarray(~train_mask)

    def terms(raw_parameters: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        intensity = physical_intensity(raw_parameters)
        if configuration.operator_mode == "explicit":
            prediction = predict_stokes_i_explicit(
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
            )
        else:
            prediction = predict_stokes_i(
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
            )
        data_term = weighted_complex_mse(prediction, observation, weight, training_flag)
        prior_term = sky_prior(
            intensity,
            size=grid.size,
            sparsity_weight=configuration.sparsity_weight,
            smoothness_weight=configuration.smoothness_weight,
        )
        return data_term + prior_term, (data_term, prior_term)

    value_and_gradient = jax.jit(jax.value_and_grad(terms, has_aux=True))
    optimizer = create_optimizer(configuration)
    if initial_raw is None:
        raw = jnp.full(
            grid.size * grid.size,
            raw_from_intensity(configuration.initial_intensity),
            dtype=real_dtype,
        )
    else:
        raw = jnp.asarray(initial_raw, dtype=real_dtype).reshape(-1)
    optimizer_state = (
        optimizer.init(raw) if initial_optimizer_state is None else initial_optimizer_state
    )
    best_raw = raw
    best_optimizer_state = optimizer_state
    best_objective = np.inf
    stale_steps = 0
    objectives: list[float] = []
    data_values: list[float] = []
    prior_values: list[float] = []
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

    for _ in range(configuration.steps):
        raw, optimizer_state, objective, data_term, prior_term = update(raw, optimizer_state)
        objective_value = float(objective)
        objectives.append(objective_value)
        data_values.append(float(data_term))
        prior_values.append(float(prior_term))
        if objective_value < best_objective - configuration.min_delta:
            best_objective = objective_value
            best_raw = raw
            best_optimizer_state = optimizer_state
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= configuration.patience:
            converged = True
            break

    image = np.asarray(physical_intensity(best_raw)).reshape(grid.size, grid.size)
    return InferenceResult(
        image=image,
        raw_parameters=np.asarray(best_raw),
        optimizer_state=best_optimizer_state,
        objective_history=tuple(objectives),
        data_history=tuple(data_values),
        prior_history=tuple(prior_values),
        steps=len(objectives),
        converged=converged,
    )


def save_checkpoint(path: str | Path, result: InferenceResult) -> None:
    leaves, _ = jax.tree_util.tree_flatten(
        (jnp.asarray(result.raw_parameters), result.optimizer_state)
    )
    with Path(path).open("wb") as stream:
        np.savez(stream, *(np.asarray(leaf) for leaf in leaves), step=result.steps)


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
