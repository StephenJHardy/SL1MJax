from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.data.synthetic import simulate_dataset
from sl1mjax.inference import (
    InferenceConfig,
    infer_quadtree,
    infer_regular_grid,
    load_checkpoint,
    positive_l1_kkt_residual,
    save_checkpoint,
)
from sl1mjax.polarization import ReceptorBasis
from sl1mjax.quadtree import quadtree_sky_from_regular_grid
from sl1mjax.sky import RegularGrid, raw_from_intensity


@pytest.fixture(scope="module")
def sparse_problem() -> tuple[RegularGrid, VisibilityBlock]:
    grid = RegularGrid(6, np.deg2rad(12 / 3600))
    block = simulate_dataset(
        grid,
        basis=ReceptorBasis.LINEAR,
        rows=72,
        channels=2,
        noise_std=0.0,
        seed=11,
    ).blocks[0]
    return grid, block


def test_positive_l1_kkt_residual_distinguishes_active_and_zero_flux() -> None:
    flux = jnp.asarray([2.0, 0.0, 0.0])
    gradient = jnp.asarray([-0.1, -0.05, -0.3])

    residual = positive_l1_kkt_residual(flux, gradient, 0.1)

    assert float(residual) == pytest.approx(0.2)


def test_positive_l1_kkt_residual_accepts_per_parameter_penalties() -> None:
    flux = jnp.asarray([2.0, 0.0, 0.0])
    gradient = jnp.asarray([-0.1, -0.05, -0.3])

    residual = positive_l1_kkt_residual(
        flux,
        gradient,
        jnp.asarray([0.1, 0.02, 0.25]),
    )

    assert float(residual) == pytest.approx(0.05)


@pytest.mark.parametrize("solver", ["fista", "hybrid"])
def test_deterministic_physical_solvers_recover_sparse_sky(
    sparse_problem: tuple[RegularGrid, VisibilityBlock], solver: str
) -> None:
    grid, block = sparse_problem
    config = InferenceConfig(
        solver=solver,  # type: ignore[arg-type]
        steps=220,
        learning_rate=0.03,
        sparsity_weight=1e-5,
        initial_intensity=0.02,
        batch_size_rows=17,
        validation_interval=10,
        kkt_tolerance=1e-5,
    )

    result = infer_regular_grid(block, grid, block.active, config)

    assert result.solver == solver
    assert result.converged
    assert result.kkt_residual < config.kkt_tolerance
    assert np.all(result.image >= 0)
    assert np.count_nonzero(result.image == 0) >= 20
    assert result.image[3, 3] == pytest.approx(1.0, abs=2e-3)
    assert result.image[1, 5] == pytest.approx(0.5, abs=2e-3)
    assert result.image[5, 1] == pytest.approx(0.25, abs=2e-3)
    assert result.objective_steps == tuple(sorted(result.objective_steps))
    if solver == "fista":
        assert np.all(np.diff(result.objective_history) <= 1e-10)


def test_proximal_sgd_is_deterministic_and_handles_partial_last_batch(
    sparse_problem: tuple[RegularGrid, VisibilityBlock],
) -> None:
    grid, block = sparse_problem
    config = InferenceConfig(
        solver="proximal_sgd",
        steps=2000,
        learning_rate=0.03,
        sparsity_weight=1e-5,
        initial_intensity=0.02,
        batch_size_rows=17,
        random_seed=4,
        validation_interval=100,
        kkt_tolerance=1e-5,
    )

    first = infer_regular_grid(block, grid, block.active, config)
    second = infer_regular_grid(block, grid, block.active, config)

    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.objective_history, second.objective_history)
    assert first.objective_history[-1] < 3e-5
    assert first.kkt_residual < 1e-4
    assert np.count_nonzero(first.image == 0) >= 20
    assert first.image[3, 3] == pytest.approx(1.0, abs=3e-3)


def test_physical_solver_checkpoint_has_no_synthetic_adam_state(
    tmp_path: Path,
    sparse_problem: tuple[RegularGrid, VisibilityBlock],
) -> None:
    grid, block = sparse_problem
    config = InferenceConfig(
        solver="fista",
        steps=40,
        learning_rate=0.03,
        sparsity_weight=1e-5,
        initial_intensity=0.02,
        validation_interval=10,
    )
    result = infer_regular_grid(block, grid, block.active, config)
    checkpoint = tmp_path / "physical.checkpoint.npz"

    save_checkpoint(checkpoint, result)
    raw, optimizer_state, step = load_checkpoint(checkpoint, config, grid.size * grid.size)

    assert optimizer_state is None
    assert step == result.best_step
    np.testing.assert_allclose(raw, raw_from_intensity(result.image.ravel()))

    with pytest.raises(ValueError, match="checkpoint solver"):
        load_checkpoint(
            checkpoint,
            InferenceConfig(solver="softplus_adam"),
            grid.size * grid.size,
        )


def test_fista_returns_lowest_holdout_checkpoint() -> None:
    rows = 20
    shape = (rows, 1, 1)
    visibility = np.ones(shape, dtype=np.complex128)
    visibility[rows // 2 :] = 0.25
    block = VisibilityBlock(
        uvw_m=np.zeros((rows, 3)),
        frequency_hz=np.asarray([1e9]),
        visibility=visibility,
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=np.zeros(rows, dtype=np.int32),
        antenna2=np.ones(rows, dtype=np.int32),
        correlations=("I",),
        receptor_basis="stokes",
    )
    train_mask = np.zeros(shape, dtype=bool)
    train_mask[: rows // 2] = True
    holdout_mask = np.zeros(shape, dtype=bool)
    holdout_mask[rows // 2 :] = True
    topology = quadtree_sky_from_regular_grid(1, 1e-4, [0.0]).topology
    config = InferenceConfig(
        solver="fista",
        operator_mode="explicit",
        steps=40,
        learning_rate=0.03,
        sparsity_weight=0.0,
        validation_interval=1,
        kkt_tolerance=1e-12,
    )

    result = infer_quadtree(
        block,
        topology,
        train_mask,
        config,
        holdout_mask=holdout_mask,
        initial_flux=np.asarray([0.01]),
    )

    assert result.best_step == int(np.argmin(result.holdout_history)) + 1
    assert result.flux[0] < 0.5
    assert not result.converged
    assert result.kkt_residual > config.kkt_tolerance
