#!/usr/bin/env python3
"""Diagnose initialization and KKT behaviour of the 3C391 positive grid fit."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import InferenceConfig, infer_quadtree
from sl1mjax.objective import weighted_complex_mse
from sl1mjax.quadtree import (
    predict_quadtree_stokes_i_explicit,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.refinement import quadtree_objective_metrics
from sl1mjax.sky import GaussianApproximation
from sl1mjax.split import uv_cell_split


def _select_even_rows(
    block: VisibilityBlock,
    maximum_rows: int | None,
) -> VisibilityBlock:
    if maximum_rows is None or maximum_rows >= block.shape[0]:
        return block
    if maximum_rows < 2:
        raise ValueError("maximum_rows must be at least two")
    rows = np.linspace(0, block.shape[0] - 1, maximum_rows, dtype=np.int64)
    row_fields = (
        "field_id",
        "scan_id",
        "state_id",
        "observation_id",
        "feed1",
        "feed2",
        "interval_s",
    )
    replacements: dict[str, Any] = {
        "uvw_m": block.uvw_m[rows],
        "visibility": block.visibility[rows],
        "weight": block.weight[rows],
        "flag": block.flag[rows],
        "time_s": block.time_s[rows],
        "antenna1": block.antenna1[rows],
        "antenna2": block.antenna2[rows],
        "model_visibility": (
            None
            if block.model_visibility is None
            else block.model_visibility[rows]
        ),
    }
    for field in row_fields:
        value = getattr(block, field)
        if value is None:
            raise ValueError(f"visibility block has no {field}")
        replacements[field] = value[rows]
    return replace(block, **replacements)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"q50": 0.0, "q90": 0.0, "q99": 0.0, "max": 0.0}
    absolute = np.abs(values)
    return {
        "q50": float(np.quantile(absolute, 0.50)),
        "q90": float(np.quantile(absolute, 0.90)),
        "q99": float(np.quantile(absolute, 0.99)),
        "max": float(np.max(absolute)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--maximum-rows", type=int)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--pixel-arcsec", type=float, default=4.0)
    parser.add_argument("--initial-intensity", type=float, default=1e-3)
    parser.add_argument("--sparsity-weight", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--fit-all", action="store_true")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument(
        "--approximation",
        choices=("paraxial", "wide-field"),
        default="paraxial",
    )
    parser.add_argument("--flux-floor", type=float, default=1e-7)
    arguments = parser.parse_args()

    dataset = read_dataset(arguments.fixture)
    if not 0 <= arguments.block < len(dataset.blocks):
        raise ValueError(f"block must be between 0 and {len(dataset.blocks) - 1}")
    block = _select_even_rows(dataset.blocks[arguments.block], arguments.maximum_rows)
    if arguments.fit_all:
        train_mask = block.active
        holdout_mask = None
    else:
        split = uv_cell_split(
            block,
            holdout_fraction=arguments.holdout_fraction,
            seed=arguments.split_seed,
        )
        train_mask = split.train
        holdout_mask = split.holdout

    approximation = (
        GaussianApproximation.PARAXIAL
        if arguments.approximation == "paraxial"
        else GaussianApproximation.WIDE_FIELD
    )
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    config = InferenceConfig(
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        sparsity_weight=arguments.sparsity_weight,
        initial_intensity=arguments.initial_intensity,
        patience=arguments.patience,
        validation_interval=10,
        operator_mode="explicit",
        direct_dft=direct,
    )
    pixel_size_rad = np.deg2rad(arguments.pixel_arcsec / 3600.0)
    initial_sky = quadtree_sky_from_regular_grid(
        arguments.size,
        pixel_size_rad,
        np.zeros(arguments.size**2),
    )
    fit = infer_quadtree(
        block,
        initial_sky.topology,
        train_mask,
        config,
        holdout_mask=holdout_mask,
        approximation=approximation,
    )
    metrics = quadtree_objective_metrics(
        block,
        fit,
        train_mask,
        config,
        holdout_mask=holdout_mask,
    )

    real_dtype = direct.real_dtype
    observation = jnp.asarray(block.visibility, dtype=direct.complex_dtype)
    weight = jnp.asarray(block.weight, dtype=real_dtype)
    training_flag = jnp.asarray(~train_mask)

    def physical_objective(flux: jax.Array) -> jax.Array:
        prediction = predict_quadtree_stokes_i_explicit(
            flux,
            fit.topology,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            approximation=approximation,
            config=direct,
        )
        objective: jax.Array = (
            jnp.asarray(
                weighted_complex_mse(
                    prediction,
                    observation,
                    weight,
                    training_flag,
                )
            )
            + arguments.sparsity_weight * jnp.sum(flux)
        )
        return objective

    physical_gradient = np.asarray(
        jax.jit(jax.grad(physical_objective))(
            jnp.asarray(fit.flux, dtype=real_dtype)
        )
    )
    raw_gradient = physical_gradient * np.asarray(
        jax.nn.sigmoid(jnp.asarray(fit.raw_parameters, dtype=real_dtype))
    )
    at_floor = fit.flux <= arguments.flux_floor
    projected_gradient = np.where(
        ~at_floor | (physical_gradient < 0),
        physical_gradient,
        0.0,
    )

    image = np.asarray(fit.flux).reshape(arguments.size, arguments.size)
    edge_flux = float(
        image[0, :].sum()
        + image[-1, :].sum()
        + image[1:-1, 0].sum()
        + image[1:-1, -1].sum()
    )
    peak = np.unravel_index(np.argmax(image), image.shape)
    result = {
        "configuration": {
            "shape": list(block.shape),
            "size": arguments.size,
            "pixel_arcsec": arguments.pixel_arcsec,
            "initial_intensity": arguments.initial_intensity,
            "initial_total_flux": arguments.initial_intensity * arguments.size**2,
            "sparsity_weight": arguments.sparsity_weight,
            "fit_all": arguments.fit_all,
            "steps": arguments.steps,
            "patience": arguments.patience,
            "precision": arguments.precision,
            "approximation": approximation.value,
        },
        "fit": {
            "steps": fit.steps,
            "best_step": fit.best_step,
            "converged": fit.converged,
            "training_data": metrics.training_data,
            "training_objective": metrics.objective,
            "holdout_data": metrics.holdout_data,
            "total_flux": float(np.sum(image)),
            "edge_flux": edge_flux,
            "edge_flux_fraction": edge_flux / float(np.sum(image)),
            "peak_iy_ix": [int(peak[0]), int(peak[1])],
            "peak_flux": float(image[peak]),
            "corners": {
                "lower_left": float(image[0, 0]),
                "lower_right": float(image[0, -1]),
                "upper_left": float(image[-1, 0]),
                "upper_right": float(image[-1, -1]),
            },
            "floor_count": int(np.sum(at_floor)),
        },
        "kkt": {
            "flux_floor": arguments.flux_floor,
            "physical_gradient": _quantiles(physical_gradient),
            "raw_gradient": _quantiles(raw_gradient),
            "projected_physical_gradient": _quantiles(projected_gradient),
            "negative_gradient_at_floor_count": int(
                np.sum(at_floor & (physical_gradient < 0))
            ),
        },
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        arguments.output / "fit.npz",
        image=image,
        prediction=fit.prediction,
        residual=fit.residual,
        train_mask=train_mask,
        holdout_mask=(
            np.zeros(block.shape, dtype=bool)
            if holdout_mask is None
            else holdout_mask
        ),
        physical_gradient=physical_gradient,
        raw_gradient=raw_gradient,
        objective_history=np.asarray(fit.objective_history),
        data_history=np.asarray(fit.data_history),
        holdout_history=np.asarray(fit.holdout_history),
        holdout_steps=np.asarray(fit.holdout_steps),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
