from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.sky_recovery import (
    fit_real_sky_component,
    inject_sky_component,
    spectral_support_mask,
    split_native_baselines,
    temporal_support_mask,
)


def _block() -> tuple[VisibilityBlock, np.ndarray]:
    baselines = [(0, 1), (0, 2), (1, 2), (1, 3)]
    times = np.arange(6, dtype=np.float64) * 10.0
    antenna1 = np.tile([pair[0] for pair in baselines], times.size)
    antenna2 = np.tile([pair[1] for pair in baselines], times.size)
    time_s = np.repeat(times, len(baselines))
    rows = time_s.size
    channels = 4
    phase = (
        np.arange(rows, dtype=np.float64)[:, None, None] * 0.13
        + np.arange(channels, dtype=np.float64)[None, :, None] * 0.31
    )
    response = np.exp(1j * phase)
    shape = (rows, channels, 1)
    block = VisibilityBlock(
        uvw_m=np.zeros((rows, 3)),
        frequency_hz=4.5e9 + np.arange(channels) * 2e6,
        visibility=np.zeros(shape, dtype=np.complex128),
        model_visibility=np.zeros(shape, dtype=np.complex128),
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=time_s,
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=(Correlation.RR,),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    return block, response


def test_baseline_holdout_is_disjoint_complete_and_deterministic() -> None:
    block, _ = _block()
    first = split_native_baselines(block, evaluation_fraction=0.25, seed=17)
    second = split_native_baselines(block, evaluation_fraction=0.25, seed=17)

    assert set(first.discovery_baselines).isdisjoint(first.evaluation_baselines)
    assert len(first.discovery_baselines) == 3
    assert len(first.evaluation_baselines) == 1
    np.testing.assert_array_equal(first.discovery_mask | first.evaluation_mask, block.active)
    assert not np.any(first.discovery_mask & first.evaluation_mask)
    np.testing.assert_array_equal(first.evaluation_mask, second.evaluation_mask)


def test_temporal_injection_recovers_amplitude_on_unseen_baseline() -> None:
    block, response = _block()
    holdout = split_native_baselines(block, evaluation_fraction=0.25, seed=3)
    support = temporal_support_mask(block, start_s=20.0, duration_s=20.0)
    injected = inject_sky_component(block, response, support, 2.5)

    fit = fit_real_sky_component(
        injected,
        response,
        support,
        holdout.discovery_mask,
        holdout.evaluation_mask,
    )

    assert fit.coefficient == pytest.approx(2.5)
    assert fit.discovery_sample_count == 2 * 3 * 4
    assert fit.evaluation_sample_count == 2 * 1 * 4
    assert fit.component_supported_evaluation.weighted_complex_mse == pytest.approx(0.0)
    assert fit.supported_evaluation_relative_improvement == pytest.approx(1.0)


def test_spectral_injection_beats_static_component_on_held_out_data() -> None:
    block, response = _block()
    holdout = split_native_baselines(block, evaluation_fraction=0.5, seed=5)
    spectral = spectral_support_mask(block, first_channel=1, channel_count=1)
    injected = inject_sky_component(block, response, spectral, 1.75)

    recovered = fit_real_sky_component(
        injected,
        response,
        spectral,
        holdout.discovery_mask,
        holdout.evaluation_mask,
    )
    static = fit_real_sky_component(
        injected,
        response,
        np.ones(block.shape, dtype=bool),
        holdout.discovery_mask,
        holdout.evaluation_mask,
    )

    assert recovered.coefficient == pytest.approx(1.75)
    assert recovered.component_evaluation.weighted_complex_mse == pytest.approx(0.0)
    assert static.coefficient == pytest.approx(1.75 / 4.0)
    assert (
        static.component_evaluation.weighted_complex_mse
        > recovered.component_evaluation.weighted_complex_mse
    )


def test_signed_component_can_recover_a_decrement() -> None:
    block, response = _block()
    holdout = split_native_baselines(block, evaluation_fraction=0.25, seed=0)
    support = temporal_support_mask(block, start_s=0.0, duration_s=10.0)
    injected = inject_sky_component(block, response, support, -0.4)

    signed = fit_real_sky_component(
        injected,
        response,
        support,
        holdout.discovery_mask,
        holdout.evaluation_mask,
    )
    positive = fit_real_sky_component(
        injected,
        response,
        support,
        holdout.discovery_mask,
        holdout.evaluation_mask,
        nonnegative=True,
    )

    assert signed.coefficient == pytest.approx(-0.4)
    assert positive.coefficient == 0.0


def test_support_validation_rejects_bad_ranges() -> None:
    block, _ = _block()
    with pytest.raises(ValueError, match="exceeds"):
        spectral_support_mask(block, first_channel=3, channel_count=2)
    with pytest.raises(ValueError, match="duration"):
        temporal_support_mask(block, start_s=0.0, duration_s=0.0)
