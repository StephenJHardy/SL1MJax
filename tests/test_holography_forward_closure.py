from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_holography_full_jones import _FULL, _FULL_GRAD, _diagonal_observation, _grid_observation

from sl1mjax.finite_pixel import ManufacturedVoltageBeam
from sl1mjax.holography_diagonal import _median_spacing
from sl1mjax.holography_forward_closure import (
    algebraic_closure_cases,
    cell_aggregation_closure,
    classify_forward_closure,
    dummy_support_artifact,
    gauge_refactorization_closure,
    interpolator_support_category,
    per_sample_map_closure,
    recover_row_jones,
    single_sample_algebraic_closure,
    spatial_support_contract,
)
from sl1mjax.holography_full_jones import freeze_interpolation_support as freeze_support
from sl1mjax.holography_full_jones import (
    interpolate_holography_full_jones,
    interpolate_holography_full_jones_batch,
    recover_holography_full_jones,
)


def test_single_sample_algebraic_closure_recovers_and_reconstructs() -> None:
    report = single_sample_algebraic_closure()
    assert report["status"] == "pass"
    assert report["max_complex_relative_error"] < 1.0e-8
    case = algebraic_closure_cases()[0]
    beam = recover_row_jones(
        case["visibility"],
        case["source"],
        case["r_moving"],
        case["r_ref"],
        moving_is_p=bool(case["moving_is_p"]),
    )
    np.testing.assert_allclose(beam, case["beam"], atol=1.0e-9)


def test_gauge_refactorization_preserves_nondiagonal_complex_A() -> None:
    report = gauge_refactorization_closure()
    assert report["status"] == "pass"
    assert report["tested_nondiagonal_complex_A"] is True


def test_manufactured_per_sample_map_closes() -> None:
    observation = _diagonal_observation(n_time=3, n_ref=2)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    observation = replace(
        observation,
        block=replace(observation.block, visibility=predicted.visibility),
    )
    residual = {int(ant): np.eye(2, dtype=np.complex128) for ant in range(3)}
    chi = np.zeros((3, 3), dtype=np.float64)
    report = per_sample_map_closure(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        row_mask=np.ones(observation.block.time_s.size, dtype=bool),
    )
    assert report["status"] == "pass"
    assert report["max_complex_relative_error"] < 1.0e-8
    aggregation = cell_aggregation_closure(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        row_mask=np.ones(observation.block.time_s.size, dtype=bool),
    )
    assert aggregation["first_failing_operation"] is None
    assert aggregation["status"] == "pass"


def test_duplicate_offsets_do_not_collapse_interpolation_spacing() -> None:
    spacing = 0.002
    unique = np.array([[0.0, 0.0], [spacing, 0.0], [2.0 * spacing, 0.0]], dtype=np.float64)
    duplicated = np.vstack([unique, unique, unique])
    assert _median_spacing(duplicated) == pytest.approx(_median_spacing(unique))
    artifact = dummy_support_artifact({0: duplicated})
    plane, ok, _leak = interpolate_holography_full_jones(
        artifact,
        np.array([0.5 * spacing, 0.0]),
        moving_antenna_id=0,
        frequency_hz=4.564e9,
    )
    assert ok
    assert np.isfinite(plane[0, 0])
    support = freeze_support({0: duplicated})
    interior = interpolator_support_category(support, 0, [0.5 * spacing, 0.0])
    assert interior["category"] == "interpolation"
    assert interior["finite"] is True


def test_support_predicate_matches_finite_interpolator() -> None:
    spacing = 0.002
    support = freeze_support(
        {0: np.array([[0.0, 0.0], [spacing, 0.0], [2.0 * spacing, 0.0]], dtype=np.float64)}
    )
    report = spatial_support_contract(
        support,
        {
            "exact": (0, [spacing, 0.0]),
            "interior": (0, [0.5 * spacing, 0.0]),
            "boundary": (0, [2.0 * spacing, 0.0]),
            "extrapolation": (0, [4.0 * spacing, 0.0]),
            "missing_antenna": (1, [0.0, 0.0]),
            "nan_query": (0, [np.nan, 0.0]),
        },
    )
    assert report["status"] == "pass"
    assert report["cases"]["exact"]["category"] == "exact"
    assert report["cases"]["interior"]["category"] == "interpolation"
    assert report["cases"]["interior"]["finite"] is True
    assert report["cases"]["extrapolation"]["category"] == "extrapolation"
    assert report["cases"]["extrapolation"]["finite"] is False
    assert report["cases"]["missing_antenna"]["category"] == "unsupported"
    assert report["cases"]["nan_query"]["category"] == "unsupported"
    missing = dummy_support_artifact(
        {0: [[0.0, 0.0], [spacing, 0.0]]},
        copolar_valid=False,
        off_diagonal_valid=False,
    )
    _plane, ok, _leak = interpolate_holography_full_jones(
        missing,
        np.array([0.0, 0.0]),
        moving_antenna_id=0,
        frequency_hz=4.564e9,
    )
    assert not ok


def test_batch_and_scalar_agree_on_grid() -> None:
    observation = _grid_observation(n_visit=1, n_ref=2, n_move=1, n_l=3, n_m=1)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_full_jones(observation, predicted.visibility)
    queries = np.stack([sample.offset_lm_rad for sample in artifact.samples])
    movers = np.array([sample.moving_antenna_id for sample in artifact.samples], dtype=np.int32)
    freqs = np.array([sample.frequency_hz for sample in artifact.samples])
    batch, ok, _ = interpolate_holography_full_jones_batch(
        artifact, queries, moving_antenna_id=movers, frequency_hz=freqs
    )
    assert bool(np.all(ok))
    for index, sample in enumerate(artifact.samples):
        plane, valid, _ = interpolate_holography_full_jones(
            artifact,
            sample.offset_lm_rad,
            moving_antenna_id=sample.moving_antenna_id,
            frequency_hz=sample.frequency_hz,
        )
        assert valid
        np.testing.assert_allclose(batch[index], plane, atol=1e-10)


def test_forward_closure_classifier_records_first_failure() -> None:
    gate = classify_forward_closure(
        {
            "single_sample_algebraic_closure": {"status": "pass"},
            "gauge_refactorization_closure": {"status": "pass"},
            "per_sample_map_closure": {"status": "fail"},
            "cell_aggregation_closure": {"status": "fail"},
        }
    )
    assert gate["first_failure"] == "per_sample_map_closure"
    assert gate["comparison_interpretable"] is False
    assert gate["deterministic_closure_passed"] is False
