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


def _circular_identity(antenna_count: int, frequency_hz: np.ndarray, reference: int = 0):
    return identity_solution(
        antenna_count=antenna_count,
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        frequency_hz=frequency_hz,
        time_s=np.array([0.0]),
        reference_antenna=reference,
    )


def _fully_connected_pairs(antenna_count: int) -> tuple[np.ndarray, np.ndarray]:
    first, second = [], []
    for antenna in range(antenna_count):
        for other in range(antenna + 1, antenna_count):
            first.append(antenna)
            second.append(other)
    return np.asarray(first, dtype=np.int32), np.asarray(second, dtype=np.int32)


def _four_product_block(
    visibility: np.ndarray,
    model: np.ndarray,
    frequency_hz: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
):
    from sl1mjax.data.canonical import VisibilityBlock
    from sl1mjax.polarization import ReceptorBasis

    return VisibilityBlock(
        uvw_m=np.zeros((antenna1.size, 3)),
        frequency_hz=frequency_hz,
        visibility=visibility,
        model_visibility=model,
        weight=np.ones_like(visibility, dtype=np.float64),
        flag=np.zeros(visibility.shape, dtype=bool),
        time_s=np.zeros(antenna1.size),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )


def test_solve_cross_hand_delay_recovers_global_right_receptor_delay() -> None:
    from sl1mjax.calibration import corrupt_model
    from sl1mjax.polarization_inference import solve_cross_hand_delay

    frequency_hz = np.linspace(4.5e9, 4.7e9, 16)
    antenna1, antenna2 = _fully_connected_pairs(4)
    solution = _circular_identity(4, frequency_hz)
    delay = np.zeros((4, 2))
    delay[:, 0] = 5.0e-9
    solution = replace(
        solution,
        cross_hand_delay_s=delay,
        cross_hand_delay_valid=np.ones((4, 2), dtype=bool),
        reference_frequency_hz=float(np.mean(frequency_hz)),
    )
    sky = np.zeros((antenna1.size, frequency_hz.size, 4), dtype=np.complex128)
    sky[..., 0] = 4.0
    sky[..., 3] = 4.0
    sky[..., 1] = 0.5 + 0.2j
    sky[..., 2] = 0.5 - 0.2j
    visibility = np.asarray(
        corrupt_model(
            sky,
            solution,
            time_s=np.zeros(antenna1.size),
            frequency_hz=frequency_hz,
            antenna1=antenna1,
            antenna2=antenna2,
            extrapolate=True,
        )
    )
    block = _four_product_block(visibility, sky, frequency_hz, antenna1, antenna2)
    start = _circular_identity(4, frequency_hz)
    start = replace(start, reference_frequency_hz=solution.reference_frequency_hz)
    result = solve_cross_hand_delay(block, start, apply_parallactic_angle=False)
    assert result.solution.cross_hand_delay_s is not None
    assert result.solution.cross_hand_delay_s[0, 0] == pytest.approx(5.0e-9, abs=1e-12)
    assert result.solution.leakage is None
    assert result.solution.rl_phase is None
    assert result.solution.provenance["frequency_parameterization"] == "global_delay"


def test_solve_leakage_recovers_first_order_d_terms() -> None:
    from sl1mjax.polarization_inference import solve_leakage

    frequency_hz = np.array([4.6e9, 4.61e9])
    antenna1, antenna2 = _fully_connected_pairs(4)
    leakage = np.zeros((4, 2, 2), dtype=np.complex128)
    leakage[1, :, 0] = 0.05 + 0.02j
    leakage[2, :, 1] = -0.04 + 0.01j
    leakage[3, :, 0] = 0.03 - 0.02j
    leakage[3, :, 1] = 0.02 + 0.03j
    stokes_i = 4.0
    visibility = np.zeros((antenna1.size, frequency_hz.size, 4), dtype=np.complex128)
    for row, (first, second) in enumerate(zip(antenna1, antenna2, strict=True)):
        visibility[row, :, 0] = stokes_i
        visibility[row, :, 3] = stokes_i
        visibility[row, :, 1] = stokes_i * (
            leakage[first, :, 0] + np.conjugate(leakage[second, :, 1])
        )
        visibility[row, :, 2] = stokes_i * (
            leakage[first, :, 1] + np.conjugate(leakage[second, :, 0])
        )
    sky = np.zeros_like(visibility)
    sky[..., 0] = stokes_i
    sky[..., 3] = stokes_i
    block = _four_product_block(visibility, sky, frequency_hz, antenna1, antenna2)
    start = _circular_identity(4, frequency_hz)
    result = solve_leakage(block, start, apply_parallactic_angle=False)
    assert result.solution.leakage is not None
    assert result.solution.leakage_application == "casa_parallel_preserving"
    assert result.solution.rl_phase is None
    np.testing.assert_allclose(result.solution.leakage, leakage, atol=1e-10)
    from sl1mjax.calibration import apply_calibration

    corrected = apply_calibration(block, result.solution, extrapolate=True)
    selected = corrected.active
    casa_rms = float(
        np.sqrt(
            np.mean(np.abs(corrected.visibility[selected] - sky[selected]) ** 2)
            / np.mean(np.abs(sky[selected]) ** 2)
        )
    )
    exact = replace(result.solution, leakage_application="exact")
    exact_corrected = apply_calibration(block, exact, extrapolate=True)
    exact_rms = float(
        np.sqrt(
            np.mean(
                np.abs(exact_corrected.visibility[selected] - sky[selected]) ** 2
            )
            / np.mean(np.abs(sky[selected]) ** 2)
        )
    )
    assert casa_rms < 1e-3
    assert casa_rms < exact_rms


def test_solve_leakage_rejects_polarised_or_resolved_models() -> None:
    from sl1mjax.polarization_inference import solve_leakage

    frequency_hz = np.array([4.6e9, 4.61e9])
    antenna1, antenna2 = _fully_connected_pairs(3)
    polarised = np.zeros((antenna1.size, frequency_hz.size, 4), dtype=np.complex128)
    polarised[..., 0] = 4.0
    polarised[..., 3] = 4.0
    polarised[..., 1] = 0.4
    polarised[..., 2] = 0.4
    block = _four_product_block(polarised, polarised, frequency_hz, antenna1, antenna2)
    start = _circular_identity(3, frequency_hz)
    with pytest.raises(ValueError, match="unpolarised point-calibrator"):
        solve_leakage(block, start, apply_parallactic_angle=False)

    resolved = np.zeros_like(polarised)
    resolved[..., 0] = 4.0
    resolved[..., 3] = 4.0
    resolved[0, :, 0] = 5.0
    resolved[0, :, 3] = 5.0
    block = _four_product_block(resolved, resolved, frequency_hz, antenna1, antenna2)
    with pytest.raises(ValueError, match="phase-centred point-calibrator"):
        solve_leakage(block, start, apply_parallactic_angle=False)


def test_solve_leakage_marks_rank_deficient_and_unobserved_parameters_invalid() -> None:
    from sl1mjax.polarization_inference import solve_leakage
    from sl1mjax.split import VisibilitySplit

    frequency_hz = np.array([4.6e9, 4.61e9])
    antenna1 = np.array([0, 2], dtype=np.int32)
    antenna2 = np.array([1, 3], dtype=np.int32)
    stokes_i = 4.0
    visibility = np.zeros((2, frequency_hz.size, 4), dtype=np.complex128)
    visibility[..., 0] = stokes_i
    visibility[..., 3] = stokes_i
    sky = visibility.copy()
    block = _four_product_block(visibility, sky, frequency_hz, antenna1, antenna2)
    holdout = np.zeros_like(block.active)
    holdout[0] = True
    split = VisibilitySplit(block.active & ~holdout, holdout, "all_train")
    result = solve_leakage(
        block,
        _circular_identity(4, frequency_hz),
        split=split,
        apply_parallactic_angle=False,
    )
    assert result.solution.leakage_valid is not None
    assert not np.any(result.solution.leakage_valid)

    antenna1, antenna2 = _fully_connected_pairs(4)
    leakage = np.zeros((4, 2, 2), dtype=np.complex128)
    leakage[1, :, 0] = 0.05 + 0.02j
    leakage[2, :, 1] = -0.04 + 0.01j
    visibility = np.zeros((antenna1.size, frequency_hz.size, 4), dtype=np.complex128)
    for row, (first, second) in enumerate(zip(antenna1, antenna2, strict=True)):
        visibility[row, :, 0] = stokes_i
        visibility[row, :, 3] = stokes_i
        visibility[row, :, 1] = stokes_i * (
            leakage[first, :, 0] + np.conjugate(leakage[second, :, 1])
        )
        visibility[row, :, 2] = stokes_i * (
            leakage[first, :, 1] + np.conjugate(leakage[second, :, 0])
        )
    sky = np.zeros_like(visibility)
    sky[..., 0] = stokes_i
    sky[..., 3] = stokes_i
    flag = np.zeros(visibility.shape, dtype=bool)
    involving_last = (antenna1 == 3) | (antenna2 == 3)
    flag[involving_last, 1, :] = True
    block = _four_product_block(visibility, sky, frequency_hz, antenna1, antenna2)
    block = replace(block, flag=flag)
    holdout = np.zeros_like(block.active)
    holdout[2] = True
    split = VisibilitySplit(block.active & ~holdout, holdout, "all_train")
    result = solve_leakage(
        block,
        _circular_identity(4, frequency_hz),
        split=split,
        apply_parallactic_angle=False,
    )
    assert result.solution.leakage_valid is not None
    assert np.all(result.solution.leakage_valid[:3, 0, :])
    assert np.any(result.solution.leakage_valid[3, 0, :])
    assert np.all(result.solution.leakage_valid[:3, 1, :])
    assert not np.any(result.solution.leakage_valid[3, 1, :])


def test_solve_rl_phase_recovers_shared_right_receptor_phase() -> None:
    from sl1mjax.calibration import corrupt_model
    from sl1mjax.polarization_inference import solve_rl_phase

    frequency_hz = np.array([4.6e9, 4.61e9, 4.62e9])
    antenna1, antenna2 = _fully_connected_pairs(3)
    solution = _circular_identity(3, frequency_hz)
    phase = np.exp(1j * np.array([0.3, 0.5, -0.2]))
    solution = replace(
        solution,
        rl_phase=np.broadcast_to(phase, (3, 3)).copy(),
        rl_phase_frequency_hz=frequency_hz,
        rl_phase_valid=np.ones((3, 3), dtype=bool),
    )
    sky = np.zeros((antenna1.size, frequency_hz.size, 4), dtype=np.complex128)
    sky[..., 0] = 4.0
    sky[..., 3] = 4.0
    sky[..., 1] = 0.4 + 0.2j
    sky[..., 2] = 0.4 - 0.2j
    visibility = np.asarray(
        corrupt_model(
            sky,
            solution,
            time_s=np.zeros(antenna1.size),
            frequency_hz=frequency_hz,
            antenna1=antenna1,
            antenna2=antenna2,
            extrapolate=True,
        )
    )
    block = _four_product_block(visibility, sky, frequency_hz, antenna1, antenna2)
    result = solve_rl_phase(
        block, _circular_identity(3, frequency_hz), apply_parallactic_angle=False
    )
    assert result.solution.rl_phase is not None
    recovered = result.solution.rl_phase[0]
    np.testing.assert_allclose(recovered, phase, atol=1e-10)
    assert result.solution.provenance["frequency_parameterization"] == "per_channel"
    assert result.solution.provenance["frequency_holdout"] is False
