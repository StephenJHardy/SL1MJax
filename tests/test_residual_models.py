from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.residual_models import (
    add_real_linear_statistics,
    fit_real_linear_statistics,
    real_linear_statistics,
    scan_residual_response_matrix,
    score_real_linear_fit,
)


def test_tiled_statistics_recover_real_coefficients_and_score_holdout() -> None:
    rng = np.random.default_rng(12)
    shape = (18, 4, 2)
    responses = rng.normal(size=shape + (3,)) + 1j * rng.normal(size=shape + (3,))
    truth = np.asarray([0.3, -1.2, 0.7])
    residual = responses @ truth
    weight = rng.uniform(0.5, 2.0, size=shape)
    discovery = np.zeros(shape, dtype=bool)
    discovery[:12] = True
    evaluation = ~discovery

    first = real_linear_statistics(
        residual[:6], weight[:6], discovery[:6], responses[:6]
    )
    second = real_linear_statistics(
        residual[6:12], weight[6:12], discovery[6:12], responses[6:12]
    )
    fit = fit_real_linear_statistics(add_real_linear_statistics(first, second))
    evaluation_statistics = real_linear_statistics(
        residual, weight, evaluation, responses
    )
    _, evaluation_mse = score_real_linear_fit(evaluation_statistics, fit.coefficients)

    np.testing.assert_allclose(fit.coefficients, truth, rtol=1e-12, atol=1e-12)
    assert fit.weighted_complex_mse == pytest.approx(0.0, abs=1e-12)
    assert evaluation_mse == pytest.approx(0.0, abs=1e-12)


def test_ridge_penalty_can_leave_nuisance_parameter_unpenalised() -> None:
    response = np.ones((4, 1, 1, 2), dtype=np.complex128)
    response[..., 1] = 2.0
    residual = np.full((4, 1, 1), 3.0 + 0j)
    statistics = real_linear_statistics(
        residual,
        np.ones(residual.shape),
        np.ones(residual.shape, dtype=bool),
        response,
    )

    fit = fit_real_linear_statistics(
        statistics,
        ridge_fraction=1.0,
        penalty=np.asarray([0.0, 1.0]),
    )

    assert fit.coefficients[0] == pytest.approx(3.0)
    assert fit.coefficients[1] == pytest.approx(0.0, abs=1e-12)


def _small_scan_inputs() -> tuple[np.ndarray, ...]:
    rows = 6
    shape = (rows, 2, 1)
    phase = np.arange(rows)[:, None, None] * 0.2
    model = np.broadcast_to(np.exp(1j * phase), shape).copy()
    local = np.broadcast_to(np.exp(1j * (phase + 0.7)), shape).copy()
    pointing_l = 0.01 * local
    pointing_m = -0.02j * local
    antenna1 = np.asarray([0, 0, 1, 0, 1, 2])
    antenna2 = np.asarray([1, 2, 2, 1, 2, 0])
    event = np.asarray([False, False, False, True, True, True])
    return model, local, pointing_l, pointing_m, antenna1, antenna2, event


def test_scan_response_families_have_shared_nuisance_and_physical_columns() -> None:
    inputs = _small_scan_inputs()
    common_names, common, common_penalty = scan_residual_response_matrix(
        "common_pointing_event",
        *inputs,
        antenna_ids=(0, 1, 2),
        reference_antenna=0,
    )
    gain_names, gain, gain_penalty = scan_residual_response_matrix(
        "antenna_gain_event",
        *inputs,
        antenna_ids=(0, 1, 2),
        reference_antenna=0,
    )

    assert common_names == (
        "static_fractional_scale",
        "static_local_sky_jy",
        "event_fractional_scale",
        "event_pointing_l_arcsec",
        "event_pointing_m_arcsec",
    )
    assert common.shape == inputs[0].shape + (5,)
    np.testing.assert_array_equal(common_penalty, np.zeros(5))
    assert gain.shape == inputs[0].shape + (7,)
    assert gain_names[-2:] == ("antenna_2_log_amplitude", "antenna_2_phase_rad")
    np.testing.assert_array_equal(gain_penalty, [0, 0, 0, 1, 1, 1, 1])
    assert np.all(gain[:3, ..., 2:] == 0)


def test_antenna_gain_family_identifies_injected_gain_on_unseen_baselines() -> None:
    inputs = _small_scan_inputs()
    names, responses, penalty = scan_residual_response_matrix(
        "antenna_gain_event",
        *inputs,
        antenna_ids=(0, 1, 2),
        reference_antenna=0,
    )
    truth = np.zeros(len(names))
    truth[names.index("event_fractional_scale")] = -0.05
    truth[names.index("antenna_1_log_amplitude")] = 0.08
    truth[names.index("antenna_2_phase_rad")] = -0.12
    residual = responses @ truth
    selected = np.ones(residual.shape, dtype=bool)
    statistics = real_linear_statistics(
        residual,
        np.ones(residual.shape),
        selected,
        responses,
    )

    fit = fit_real_linear_statistics(statistics, penalty=penalty)

    np.testing.assert_allclose(fit.coefficients, truth, rtol=1e-11, atol=1e-11)
    assert fit.weighted_complex_mse == pytest.approx(0.0, abs=1e-12)
