from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.hierarchical_imaging import (
    AdaptiveRefinementConfig,
    reconstruct_hierarchical,
)
from sl1mjax.inference import InferenceConfig
from sl1mjax.polarization import Correlation, ReceptorBasis

_SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_3c391_frozen_protocol.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_3c391_frozen_protocol", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
frozen_protocol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(frozen_protocol)
accepted_split_leaves = frozen_protocol.accepted_split_leaves
outer_scan_masks = frozen_protocol.outer_scan_masks
select_outer_test_scans = frozen_protocol.select_outer_test_scans
subset_rows = frozen_protocol.subset_rows


def _block() -> VisibilityBlock:
    rows = 12
    visibility = np.ones((rows, 1, 1), dtype=np.complex128)
    return VisibilityBlock(
        uvw_m=np.column_stack(
            (
                np.linspace(-100.0, 100.0, rows),
                np.linspace(50.0, -50.0, rows),
                np.zeros(rows),
            )
        ),
        frequency_hz=np.asarray([1.0e9]),
        visibility=visibility,
        weight=np.ones(visibility.shape),
        flag=np.zeros(visibility.shape, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=np.zeros(rows, dtype=np.int32),
        antenna2=np.ones(rows, dtype=np.int32),
        correlations=(Correlation.I,),
        receptor_basis=ReceptorBasis.STOKES,
        scan_id=np.repeat(np.asarray([5, 13, 21, 29]), 3),
    )


def test_outer_scan_selection_and_masks_are_reproducible_and_disjoint() -> None:
    block = _block()

    first = select_outer_test_scans(block, fraction=0.25, seed=101)
    second = select_outer_test_scans(block, fraction=0.25, seed=101)
    train, test = outer_scan_masks(block, first)

    assert first == second
    assert len(first) == 1
    assert not np.any(train & test)
    assert np.array_equal(train | test, block.active)
    assert np.all(test[np.isin(block.scan_id, first)])
    assert not np.any(test[~np.isin(block.scan_id, first)])


def test_outer_scan_helpers_reject_invalid_splits() -> None:
    block = _block()

    with pytest.raises(ValueError, match="between zero and one"):
        select_outer_test_scans(block, fraction=1.0, seed=1)
    with pytest.raises(ValueError, match="unknown test scans"):
        outer_scan_masks(block, (999,))


def test_subset_rows_preserves_row_metadata_and_active_values() -> None:
    block = _block()
    rows = np.asarray([1, 4, 8, 11])

    subset = subset_rows(block, rows)

    assert subset.shape == (4, 1, 1)
    assert np.array_equal(subset.scan_id, block.scan_id[rows])
    assert np.array_equal(subset.uvw_m, block.uvw_m[rows])
    assert np.array_equal(subset.active, block.active[rows])


def test_accepted_split_leaves_reflects_only_validated_changes() -> None:
    block = _block()
    config = AdaptiveRefinementConfig(
        root_size=1,
        root_pixel_size_rad=1e-4,
        inference=InferenceConfig(
            solver="fista",
            steps=4,
            sparsity_weight=0.0,
            validation_interval=1,
            operator_mode="explicit",
        ),
        split_strategy="random_row",
        max_rounds=0,
        max_depth=1,
    )

    result = reconstruct_hierarchical(block, config)

    assert accepted_split_leaves(result) == ()
