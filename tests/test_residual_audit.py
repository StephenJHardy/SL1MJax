from dataclasses import replace

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.residual_audit import (
    apply_robust_residual_scales,
    audit_visibility_residuals,
    masks_excluding_groups,
    robust_residual_scores,
)


def _case(
    *, discovery_only_corruption: bool = False
) -> tuple[VisibilityBlock, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    pairs = np.asarray([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int32)
    repetitions = 100
    antenna1 = np.repeat(pairs[:, 0], repetitions)
    antenna2 = np.repeat(pairs[:, 1], repetitions)
    rows = antenna1.size
    shape = (rows, 2, 2)
    prediction = np.ones(shape, dtype=np.complex128)
    noise = 0.05 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    visibility = prediction + noise
    row_within_baseline = np.tile(np.arange(repetitions), pairs.shape[0])
    discovery_rows = row_within_baseline % 2 == 0
    evaluation_rows = ~discovery_rows
    corrupt_rows = (antenna1 == 0) & (antenna2 == 1)
    if discovery_only_corruption:
        corrupt_rows &= discovery_rows
    visibility[corrupt_rows] += 2.0 + 1.0j
    block = VisibilityBlock(
        uvw_m=rng.normal(size=(rows, 3)),
        frequency_hz=np.asarray([1.0e9, 1.1e9]),
        visibility=visibility,
        weight=np.full(shape, 400.0),
        flag=np.zeros(shape, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=antenna1,
        antenna2=antenna2,
        scan_id=np.where(discovery_rows, 1, 2),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    discovery = np.broadcast_to(discovery_rows[:, None, None], shape) & block.active
    evaluation = np.broadcast_to(evaluation_rows[:, None, None], shape) & block.active
    return block, prediction, discovery, evaluation


def test_stable_baseline_corruption_is_selected_and_validated() -> None:
    block, prediction, discovery, evaluation = _case()

    audit = audit_visibility_residuals(
        (block,),
        (prediction,),
        (discovery,),
        (evaluation,),
        group_kinds=("baseline", "antenna"),
        score_threshold=6.0,
        minimum_group_samples=20,
        minimum_group_outlier_fraction=0.2,
    )

    bad = next(group for group in audit.groups if group.kind == "baseline" and group.key == (0, 1))
    good = next(group for group in audit.groups if group.kind == "baseline" and group.key == (2, 3))
    assert bad.candidate
    assert bad.validated
    assert bad.discovery.outlier_fraction > 0.9
    assert bad.evaluation.outlier_fraction > 0.9
    assert not good.candidate
    assert audit.discovery.top_one_percent_residual_power_fraction > 0.05


def test_discovery_only_corruption_does_not_validate() -> None:
    block, prediction, discovery, evaluation = _case(discovery_only_corruption=True)

    audit = audit_visibility_residuals(
        (block,),
        (prediction,),
        (discovery,),
        (evaluation,),
        group_kinds=("baseline",),
        score_threshold=6.0,
        minimum_group_samples=20,
        minimum_group_outlier_fraction=0.2,
    )

    bad = next(group for group in audit.groups if group.key == (0, 1))
    assert bad.candidate
    assert not bad.validated
    assert bad.evaluation.outlier_fraction < 0.05


def test_robust_scale_is_fitted_only_from_discovery_samples() -> None:
    block, prediction, discovery, evaluation = _case()
    scores, scales = robust_residual_scores((block,), (prediction,), (discovery,))
    changed = replace(block, visibility=block.visibility.copy())
    changed.visibility[evaluation] += 1000.0

    changed_scores, changed_scales = robust_residual_scores((changed,), (prediction,), (discovery,))

    assert scales == changed_scales
    np.testing.assert_allclose(scores[0][discovery], changed_scores[0][discovery])
    assert np.nanmax(changed_scores[0][evaluation]) > np.nanmax(scores[0][evaluation])


def test_fitted_robust_scales_can_score_a_different_prediction() -> None:
    block, prediction, discovery, _ = _case()
    original_scores, scales = robust_residual_scores((block,), (prediction,), (discovery,))
    improved_prediction = prediction + 0.01

    rescored = apply_robust_residual_scales((block,), (improved_prediction,), scales)
    unchanged = apply_robust_residual_scales((block,), (prediction,), scales)

    np.testing.assert_allclose(unchanged[0], original_scores[0], equal_nan=True)
    assert not np.allclose(rescored[0][discovery], original_scores[0][discovery])


def test_masks_excluding_groups_removes_only_selected_baseline() -> None:
    block, prediction, discovery, evaluation = _case()
    audit = audit_visibility_residuals(
        (block,),
        (prediction,),
        (discovery,),
        (evaluation,),
        group_kinds=("baseline",),
        minimum_group_samples=20,
        minimum_group_outlier_fraction=0.2,
    )
    bad = tuple(group for group in audit.groups if group.validated)

    filtered = masks_excluding_groups((block,), (block.active,), bad)[0]

    bad_rows = (block.antenna1 == 0) & (block.antenna2 == 1)
    assert not np.any(filtered[bad_rows])
    np.testing.assert_array_equal(filtered[~bad_rows], block.active[~bad_rows])


def test_audit_rejects_overlapping_discovery_and_evaluation() -> None:
    block, prediction, discovery, _ = _case()

    with pytest.raises(ValueError, match="overlap"):
        audit_visibility_residuals(
            (block,),
            (prediction,),
            (discovery,),
            (discovery,),
        )
