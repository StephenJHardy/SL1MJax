from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest

import sl1mjax
import sl1mjax.inference as inference_api
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.data.synthetic import simulate_dataset
from sl1mjax.imaging import ImagingConfig, reconstruct
from sl1mjax.inference import (
    InferenceConfig,
    InferenceResult,
    infer_regular_grid,
    load_checkpoint,
    save_checkpoint,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.sky import RegularGrid
from sl1mjax.split import uv_cell_split


@pytest.fixture(scope="module")
def reconstruction() -> Iterator[
    tuple[RegularGrid, VisibilityBlock, InferenceConfig, InferenceResult]
]:
    grid = RegularGrid(6, np.deg2rad(12 / 3600))
    block = simulate_dataset(
        grid,
        basis=ReceptorBasis.LINEAR,
        rows=72,
        channels=2,
        noise_std=0.0,
        seed=11,
    ).blocks[0]
    config = InferenceConfig(
        steps=120,
        learning_rate=0.12,
        sparsity_weight=1e-5,
        smoothness_weight=0.0,
        initial_intensity=0.02,
        patience=200,
        chunk_size=128,
    )
    result = infer_regular_grid(block, grid, block.active, config)
    yield grid, block, config, result
    jax.clear_caches()


def test_optax_reconstruction_is_deterministic_and_recovers_sources(
    reconstruction: tuple[RegularGrid, VisibilityBlock, InferenceConfig, InferenceResult],
) -> None:
    grid, block, config, first = reconstruction
    second = infer_regular_grid(block, grid, block.active, config)

    assert block.shape == (72, 2, 4)
    assert block.correlations == (
        Correlation.XX,
        Correlation.XY,
        Correlation.YX,
        Correlation.YY,
    )
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.raw_parameters, second.raw_parameters)
    np.testing.assert_array_equal(first.objective_history, second.objective_history)

    assert first.steps == config.steps
    assert first.objective_history[-1] < first.objective_history[0] * 0.01
    assert first.data_history[-1] < first.data_history[0] * 0.01
    assert np.all(first.image > 0)

    expected_sources = {(3, 3): 1.0, (1, 5): 0.5, (5, 1): 0.25}
    for pixel, flux in expected_sources.items():
        assert first.image[pixel] == pytest.approx(flux, abs=0.04)
    background = first.image.copy()
    for pixel in expected_sources:
        background[pixel] = 0
    assert float(background.max()) < 0.04
    assert float(first.image.sum()) == pytest.approx(1.75, abs=0.25)


def test_checkpoint_round_trip(
    tmp_path: Path,
    reconstruction: tuple[RegularGrid, VisibilityBlock, InferenceConfig, InferenceResult],
) -> None:
    grid, _block, config, result = reconstruction
    checkpoint = tmp_path / "fit.checkpoint.npz"

    save_checkpoint(checkpoint, result)
    raw, optimizer_state, step = load_checkpoint(
        checkpoint,
        config,
        grid.size * grid.size,
    )

    assert step == result.steps
    np.testing.assert_array_equal(raw, result.raw_parameters)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(result.optimizer_state)
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(optimizer_state)
    assert cast(Any, actual_tree) == expected_tree
    assert len(actual_leaves) == len(expected_leaves)
    for actual, expected in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(actual, expected)

    with pytest.raises(ValueError, match="checkpoint has 36 parameters; expected 35"):
        load_checkpoint(checkpoint, config, 35)


def test_structured_holdout_and_diagnostics_are_correlation_aware(
    reconstruction: tuple[RegularGrid, VisibilityBlock, InferenceConfig, InferenceResult],
) -> None:
    grid, block, _config, _result = reconstruction
    split = uv_cell_split(block, holdout_fraction=0.25, cells_per_axis=6, seed=9)

    assert split.strategy == "uv_cell"
    assert split.train.shape == block.shape
    assert split.holdout.shape == block.shape
    assert np.any(split.train)
    assert np.any(split.holdout)
    assert not np.any(split.train & split.holdout)
    # A uv-cell split holds out complete rows, including every channel/correlation.
    assert np.all(split.holdout == split.holdout[:, :1, :1])

    result = reconstruct(
        block,
        ImagingConfig(
            size=grid.size,
            pixel_size_rad=grid.pixel_size_rad,
            inference=InferenceConfig(
                steps=50,
                learning_rate=0.12,
                sparsity_weight=1e-5,
                patience=60,
                chunk_size=128,
            ),
            holdout_fraction=0.25,
            split_seed=9,
        ),
    )
    diagnostics = result.diagnostics()
    metrics = diagnostics["metrics"]
    assert diagnostics["split"] == {"strategy": "uv_cell", "seed": 9}
    assert diagnostics["correlations"] == ["XX", "XY", "YX", "YY"]
    assert np.isfinite(metrics["train_weighted_complex_mse"])
    assert np.isfinite(metrics["holdout_weighted_complex_mse"])
    assert metrics["train_weighted_complex_mse"] == pytest.approx(result.train_loss)
    assert metrics["holdout_weighted_complex_mse"] == pytest.approx(result.holdout_loss)
    assert len(diagnostics["history"]["objective"]) == metrics["steps"]


def test_legacy_fista_and_proximal_apis_are_absent() -> None:
    forbidden = ("fista", "proximal", "soft_threshold")
    exposed_names = {
        name.lower()
        for module in (sl1mjax, inference_api)
        for name in dir(module)
        if not name.startswith("_")
    }
    assert not {
        name for name in exposed_names if any(term in name for term in forbidden)
    }
