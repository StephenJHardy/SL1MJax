from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.calibration import (
    apply_calibration,
    import_casa_golden_solution,
    load_casa_calibration_golden,
)
from sl1mjax.calibration_inference import (
    CalibrationFitResult,
    CalibrationSolveConfig,
    flux_scale_solution,
    solve_staged_calibration,
    solve_time_gains,
    transfer_flux_scale,
)
from sl1mjax.split import calibration_split

FIXTURE = Path(__file__).parent / "fixtures" / "3c391_calibration_golden.npz"


def _normalized_rms(actual: np.ndarray, expected: np.ndarray, selected: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.sum(np.abs(actual[selected] - expected[selected]) ** 2)
            / np.sum(np.abs(expected[selected]) ** 2)
        )
    )


@pytest.fixture(scope="module")
def solved_cases() -> tuple[CalibrationFitResult, CalibrationFitResult, float]:
    primary = load_casa_calibration_golden(FIXTURE, label="flux_bandpass")
    secondary = load_casa_calibration_golden(FIXTURE, label="time_gain")
    casa = import_casa_golden_solution(FIXTURE, field_id=primary.field_id)
    config = CalibrationSolveConfig(
        iterations=300,
        learning_rate=0.03,
        seed=11,
    )
    primary_result = solve_staged_calibration(
        primary.block,
        reference_antenna=casa.reference_antenna,
        config=config,
    )[-1]
    secondary_result = solve_time_gains(
        secondary.block,
        primary_result.solution,
        split=calibration_split(secondary.block, seed=config.seed),
        config=config,
    )
    return (
        primary_result,
        secondary_result,
        transfer_flux_scale(primary_result.solution, secondary_result.solution),
    )


def test_portable_fixture_has_sanitized_provenance_and_expected_oracles() -> None:
    case = load_casa_calibration_golden(FIXTURE, label="flux_bandpass")

    assert not Path(case.metadata["measurement_set"]).is_absolute()
    assert not Path(case.metadata["reference_directory"]).is_absolute()
    assert case.block.model_visibility is not None
    selected = ~case.post_apply_flag & case.block.active
    assert (
        _normalized_rms(
            case.corrected_visibility,
            case.block.model_visibility,
            selected,
        )
        < 0.12
    )


def test_imported_casa_kbg_application_reproduces_corrected_visibilities() -> None:
    case = load_casa_calibration_golden(FIXTURE, label="flux_bandpass")
    solution = import_casa_golden_solution(FIXTURE, field_id=case.field_id)
    block = replace(
        case.block,
        flag=case.block.flag | case.post_apply_flag,
    )

    corrected = apply_calibration(block, solution, extrapolate=True)
    selected = corrected.active

    assert (
        _normalized_rms(
            corrected.visibility,
            case.corrected_visibility,
            selected,
        )
        < 2e-3
    )
    assert solution.provenance["antenna_position_application"] == "ecef_phase_applied"


def test_imported_casa_gain_table_exposes_flux_transfer_ablation() -> None:
    unscaled = import_casa_golden_solution(
        FIXTURE,
        field_id=1,
        gain_table="gain",
    )
    scaled = import_casa_golden_solution(
        FIXTURE,
        field_id=1,
        gain_table="flux_gain",
    )
    selected = unscaled.gain_valid & scaled.gain_valid

    np.testing.assert_array_equal(unscaled.gain_time_s, scaled.gain_time_s)
    assert np.median(np.abs(unscaled.gains[selected] / scaled.gains[selected])) == (
        pytest.approx(np.sqrt(2.296007665849216), rel=2e-3)
    )
    assert unscaled.provenance["gain_table"] == "gain"
    assert scaled.provenance["gain_table"] == "flux_gain"


def test_jax_3c286_solve_and_j1822_flux_transfer_meet_acceptance(
    solved_cases: tuple[CalibrationFitResult, CalibrationFitResult, float],
) -> None:
    primary, secondary, flux_jy = solved_cases

    assert primary.holdout_rms <= 0.12
    assert abs(primary.train_rms - primary.holdout_rms) < 0.03
    assert secondary.holdout_rms <= 0.12
    assert abs(secondary.train_rms - secondary.holdout_rms) < 0.03
    assert flux_jy == pytest.approx(2.296007665849216, rel=0.02)
    secondary_case = load_casa_calibration_golden(FIXTURE, label="time_gain")
    assert secondary_case.block.model_visibility is not None
    scaled = flux_scale_solution(secondary.solution, flux_jy)
    corrected = apply_calibration(secondary_case.block, scaled, extrapolate=True)
    assert (
        _normalized_rms(
            corrected.visibility,
            flux_jy * secondary_case.block.model_visibility,
            corrected.active,
        )
        < 0.12
    )
    for result in (primary, secondary):
        solution = result.solution
        assert np.all(np.isfinite(solution.gains[solution.gain_valid]))
        assert np.all(np.isfinite(solution.delays_s[solution.delay_valid]))
        assert np.all(np.isfinite(solution.bandpass[solution.bandpass_valid]))
