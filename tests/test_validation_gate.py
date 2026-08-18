from time import perf_counter

import numpy as np
import pytest

from sl1mjax.data.synthetic import default_sources, simulate_dataset
from sl1mjax.inference import InferenceConfig, infer_regular_grid
from sl1mjax.polarization import ReceptorBasis
from sl1mjax.sky import RegularGrid


def _truth_image(grid: RegularGrid) -> np.ndarray:
    l, m = grid.coordinates
    image = np.zeros((grid.size, grid.size), dtype=np.float64)
    for source in default_sources(grid):
        index = int(np.argmin((l - source.l) ** 2 + (m - source.m) ** 2))
        image.ravel()[index] = source.flux
    return image


def test_complete_internal_synthetic_release_gate() -> None:
    grid = RegularGrid(8, np.deg2rad(10 / 3600))
    block = simulate_dataset(
        grid,
        basis=ReceptorBasis.LINEAR,
        rows=128,
        channels=2,
        seed=2026,
    ).blocks[0]
    config = InferenceConfig(
        steps=160,
        learning_rate=0.12,
        sparsity_weight=1e-5,
        initial_intensity=0.02,
        patience=200,
        chunk_size=31,
    )
    started = perf_counter()
    first = infer_regular_grid(block, grid, block.active, config)
    elapsed = perf_counter() - started
    second = infer_regular_grid(block, grid, block.active, config)
    truth = _truth_image(grid)

    assert elapsed < 30
    assert first.objective_history[-1] <= 0.05 * first.objective_history[0]
    assert np.linalg.norm(first.image - truth) / np.linalg.norm(truth) < 0.15
    for source in default_sources(grid):
        nearest = np.argmin(
            (grid.coordinates[0] - source.l) ** 2
            + (grid.coordinates[1] - source.m) ** 2
        )
        assert first.image.ravel()[nearest] == pytest.approx(source.flux, rel=0.1)
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.objective_history, second.objective_history)
