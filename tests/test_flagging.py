import numpy as np
import pytest

from sl1mjax.data.synthetic import simulate_calibration_case
from sl1mjax.flagging import (
    ResidualHandlingMode,
    apply_residual_handling,
    audit_existing_flags,
    baseline_group_masks,
    fit_grouped_real_sky_component,
)


def test_report_only_never_changes_flags_or_weights() -> None:
    result = apply_residual_handling(
        np.asarray([0.0, 7.0, 100.0]), mode=ResidualHandlingMode.REPORT_ONLY
    )

    assert not np.any(result.proposed_flag)
    np.testing.assert_array_equal(result.weight_multiplier, 1.0)


def test_robust_mode_downweights_without_hard_flagging() -> None:
    result = apply_residual_handling(
        np.asarray([0.0, 6.0, 12.0]), mode=ResidualHandlingMode.ROBUST_WEIGHTS
    )

    assert not np.any(result.proposed_flag)
    np.testing.assert_allclose(result.weight_multiplier, [1.0, 1.0, 0.5])


def test_static_sky_mode_proposes_hard_residual_flags() -> None:
    result = apply_residual_handling(
        np.asarray([0.0, 7.0, 12.0]), mode=ResidualHandlingMode.STATIC_SKY
    )

    np.testing.assert_array_equal(result.proposed_flag, [False, True, True])
    np.testing.assert_array_equal(result.weight_multiplier, [1.0, 0.0, 0.0])


def test_transient_safe_mode_protects_sky_coherent_outlier() -> None:
    result = apply_residual_handling(
        np.asarray([0.0, 12.0, 20.0]),
        mode=ResidualHandlingMode.TRANSIENT_SAFE,
        sky_coherent=np.asarray([False, True, False]),
    )

    np.testing.assert_array_equal(result.proposed_flag, [False, False, True])
    np.testing.assert_array_equal(result.weight_multiplier, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(result.sky_protected, [False, True, False])


def test_instrumental_flag_overrides_transient_protection() -> None:
    result = apply_residual_handling(
        np.asarray([20.0]),
        mode=ResidualHandlingMode.TRANSIENT_SAFE,
        sky_coherent=np.asarray([True]),
        instrumental_flag=np.asarray([True]),
    )

    assert result.proposed_flag[0]
    assert result.weight_multiplier[0] == 0.0
    assert not result.sky_protected[0]


def test_transient_safe_mode_requires_coherence_information() -> None:
    with pytest.raises(ValueError, match="sky_coherent"):
        apply_residual_handling(
            np.asarray([20.0]), mode=ResidualHandlingMode.TRANSIENT_SAFE
        )


def test_existing_flag_audit_reports_both_error_directions() -> None:
    audit = audit_existing_flags(
        np.asarray([1.0, 10.0, 2.0, 20.0]),
        np.asarray([True, True, False, False]),
        threshold=6.0,
    )

    assert audit.flagged_residual_bulk_count == 1
    assert audit.flagged_residual_tail_count == 1
    assert audit.unflagged_residual_bulk_count == 1
    assert audit.unflagged_residual_tail_count == 1
    assert audit.flagged_residual_bulk_fraction == 0.5
    assert audit.unflagged_residual_tail_fraction == 0.5


def test_baseline_group_masks_normalize_antenna_order() -> None:
    block = simulate_calibration_case(
        antenna_count=4, time_count=2, channel_count=2, seed=21
    ).block

    mask = baseline_group_masks((block,), [(2, 0)])[0]

    expected_rows = (
        (np.minimum(block.antenna1, block.antenna2) == 0)
        & (np.maximum(block.antenna1, block.antenna2) == 2)
    )
    np.testing.assert_array_equal(mask, np.broadcast_to(expected_rows[:, None, None], block.shape))


def test_cross_baseline_sky_component_protects_coherent_variability() -> None:
    rng = np.random.default_rng(8)
    rows = 80
    shape = (rows, 2, 1)
    phase = rng.uniform(-np.pi, np.pi, size=shape)
    response = np.exp(1j * phase)
    residual = 2.5 * response + 0.05 * (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    )
    discovery = np.zeros(shape, dtype=bool)
    discovery[::2] = True
    evaluation = ~discovery

    result = fit_grouped_real_sky_component(
        residual,
        response,
        np.ones(shape),
        np.zeros(rows, dtype=np.int32),
        discovery,
        evaluation,
        minimum_evaluation_relative_improvement=0.9,
    )

    assert len(result.groups) == 1
    assert result.groups[0].coefficient == pytest.approx(2.5, abs=0.02)
    assert result.groups[0].evaluation_relative_improvement > 0.99
    assert result.groups[0].protected
    assert np.all(result.protected_mask)


def test_cross_baseline_sky_component_rejects_incoherent_interference() -> None:
    rng = np.random.default_rng(9)
    rows = 80
    shape = (rows, 2, 1)
    response = np.exp(1j * rng.uniform(-np.pi, np.pi, size=shape))
    residual = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    discovery = np.zeros(shape, dtype=bool)
    discovery[::2] = True
    evaluation = ~discovery

    result = fit_grouped_real_sky_component(
        residual,
        response,
        np.ones(shape),
        np.zeros(rows, dtype=np.int32),
        discovery,
        evaluation,
        minimum_evaluation_relative_improvement=0.25,
    )

    assert not result.groups[0].protected
    assert not np.any(result.protected_mask)
