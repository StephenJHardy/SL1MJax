from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from sl1mjax.calibration import CalibrationSolution, identity_solution
from sl1mjax.calibration_inference import (
    CalibrationSolveConfig,
    load_calibration_checkpoint,
    save_calibration_checkpoint,
    solve_bandpass,
    solve_delays,
    solve_staged_calibration,
    solve_time_gains,
)
from sl1mjax.data.synthetic import CalibrationSyntheticCase, simulate_calibration_case
from sl1mjax.polarization import Correlation
from sl1mjax.split import calibration_split


def _identity(case: CalibrationSyntheticCase) -> CalibrationSolution:
    solution = identity_solution(
        antenna_count=case.truth.antenna_count,
        correlations=case.block.correlations,
        frequency_hz=case.block.frequency_hz,
        time_s=np.unique(case.block.time_s),
        reference_antenna=case.truth.reference_antenna,
    )
    return replace(solution, reference_frequency_hz=case.truth.reference_frequency_hz)


def test_calibration_split_holds_out_data_and_preserves_connected_intervals() -> None:
    case = simulate_calibration_case()
    split = calibration_split(case.block, holdout_fraction=0.25, seed=3)

    assert np.any(split.train)
    assert np.any(split.holdout)
    assert not np.any(split.train & split.holdout)
    for time in np.unique(case.block.time_s):
        rows = np.flatnonzero(np.any(split.train, axis=(1, 2)) & (case.block.time_s == time))
        antennas = set(case.block.antenna1[rows]) | set(case.block.antenna2[rows])
        assert antennas == set(range(case.truth.antenna_count))


def test_calibration_split_rejects_disconnected_solution_interval() -> None:
    case = simulate_calibration_case()
    flag = case.block.flag.copy()
    first_group = np.isin(case.block.antenna1, [0, 1, 2])
    second_group = np.isin(case.block.antenna2, [3, 4, 5])
    disconnected = (case.block.time_s == 0) & (first_group & second_group)
    flag[disconnected] = True
    block = replace(case.block, flag=flag)

    with pytest.raises(ValueError, match="disconnected"):
        calibration_split(block)


@pytest.mark.parametrize("term", ["G", "K", "B"])
def test_single_term_solver_recovers_noiseless_holdout(term: str) -> None:
    case = simulate_calibration_case(terms=(term,), seed=2)
    split = calibration_split(case.block, seed=2)
    config = CalibrationSolveConfig(iterations=220, learning_rate=0.04)
    if term == "G":
        result = solve_time_gains(case.block, _identity(case), split=split, config=config)
    elif term == "K":
        result = solve_delays(case.block, _identity(case), split=split, config=config)
    else:
        result = solve_bandpass(case.block, _identity(case), split=split, config=config)

    assert result.train_rms < 2e-3
    assert result.holdout_rms < 3e-3
    assert np.isfinite(result.losses).all()


def test_time_gain_solver_accepts_explicit_shared_solution_times() -> None:
    case = simulate_calibration_case(terms=(), time_count=4, seed=4)
    native_times = np.unique(case.block.time_s)
    row_gain_time = np.where(
        case.block.time_s <= native_times[1],
        native_times[0],
        native_times[2],
    )

    result = solve_time_gains(
        case.block,
        _identity(case),
        split=calibration_split(case.block, seed=4),
        config=CalibrationSolveConfig(iterations=100, learning_rate=0.04),
        gain_time_s=row_gain_time,
    )

    np.testing.assert_array_equal(
        result.solution.gain_time_s,
        np.array([native_times[0], native_times[2]]),
    )
    assert result.solution.gains.shape[0] == 2
    assert result.holdout_rms < 3e-3
    assert result.solution.provenance["gain_time_coordinate"] == "explicit_per_row"


def test_time_gain_solver_rejects_misaligned_solution_times() -> None:
    case = simulate_calibration_case(terms=(), seed=5)

    with pytest.raises(ValueError, match="one coordinate per visibility row"):
        solve_time_gains(
            case.block,
            _identity(case),
            gain_time_s=np.zeros(case.block.shape[0] - 1),
        )


def test_staged_solver_recovers_combined_independent_parallel_hands() -> None:
    case = simulate_calibration_case(seed=7)
    results = solve_staged_calibration(
        case.block,
        config=CalibrationSolveConfig(iterations=300, learning_rate=0.04, seed=7),
    )

    assert [result.stage for result in results] == [
        "phase_gain",
        "delay",
        "bandpass",
        "time_gain",
    ]
    assert results[-1].train_rms < 0.025
    assert results[-1].holdout_rms < 0.025


def test_complex_gain_parameterization_gradient_matches_finite_difference() -> None:
    observed = jnp.asarray([1.2 + 0.4j, 0.8 - 0.2j])

    def loss(parameters: jax.Array) -> jax.Array:
        gain = jnp.exp(parameters[:2] + 1j * parameters[2:])
        predicted = gain[0] * jnp.conj(gain[1])
        return jnp.sum(jnp.abs(observed - predicted) ** 2)

    check_grads(
        loss,
        (jnp.asarray([0.1, -0.2, 0.3, 0.0]),),
        order=1,
        modes=["rev"],
        atol=2e-5,
        rtol=2e-5,
    )


def test_calibration_checkpoint_round_trip(tmp_path: Path) -> None:
    case = simulate_calibration_case(terms=("G",))
    result = solve_time_gains(
        case.block,
        _identity(case),
        split=calibration_split(case.block),
        config=CalibrationSolveConfig(iterations=30),
    )
    path = tmp_path / "gain-checkpoint.npz"
    save_calibration_checkpoint(result, path)

    restored = load_calibration_checkpoint(path)

    assert restored.stage == result.stage
    assert restored.losses == result.losses
    np.testing.assert_array_equal(restored.solution.gains, result.solution.gains)


def test_diagonal_solver_rejects_four_product_blocks() -> None:
    case = simulate_calibration_case(terms=("G",), seed=1)
    block = replace(
        case.block,
        visibility=np.concatenate(
            [case.block.visibility, np.zeros_like(case.block.visibility)],
            axis=-1,
        ),
        weight=np.concatenate(
            [case.block.weight, case.block.weight],
            axis=-1,
        ),
        flag=np.concatenate(
            [case.block.flag, np.ones_like(case.block.flag)],
            axis=-1,
        ),
        correlations=(
            Correlation.RR,
            Correlation.LL,
            Correlation.RL,
            Correlation.LR,
        ),
        model_visibility=np.concatenate(
            [case.block.model_visibility, np.zeros_like(case.block.model_visibility)],
            axis=-1,
        ),
    )
    with pytest.raises(ValueError, match="one parallel-hand product per receptor"):
        solve_time_gains(block, _identity(case), config=CalibrationSolveConfig(iterations=1))
