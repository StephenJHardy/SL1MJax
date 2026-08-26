from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
from test_3c391_composite_script import _block

from sl1mjax.residual_audit import audit_visibility_residuals
from sl1mjax.split import interleaved_time_fold_masks

SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_3c391_composite_residual_flags.py"


def _module():
    spec = importlib.util.spec_from_file_location("compare_composite_flags", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_predictions_validates_checkpoint_shapes(tmp_path) -> None:
    module = _module()
    block = _block()
    path = tmp_path / "candidate.npz"
    np.savez(path, prediction_C1=np.ones(block.shape, dtype=np.complex128))

    loaded = module._load_predictions(path, (block,))

    np.testing.assert_array_equal(loaded[0], 1.0)


def test_paired_group_changes_tracks_sealed_residual_reduction() -> None:
    module = _module()
    block = replace(_block(), visibility=np.ones(_block().shape, dtype=np.complex128))
    discovery, _, evaluation = interleaved_time_fold_masks((block,), bin_seconds=2.0)
    reference = audit_visibility_residuals(
        (block,),
        (np.zeros(block.shape, dtype=np.complex128),),
        discovery,
        evaluation,
        group_kinds=("pointing",),
        minimum_group_samples=1,
        minimum_group_outlier_fraction=0.0,
    )
    candidate = audit_visibility_residuals(
        (block,),
        (0.5 * np.ones(block.shape, dtype=np.complex128),),
        discovery,
        evaluation,
        group_kinds=("pointing",),
        minimum_group_samples=1,
        minimum_group_outlier_fraction=0.0,
    )

    changes = module._paired_group_changes(reference, candidate)

    assert len(changes) == 1
    assert changes[0]["evaluation_normalized_residual_power_change"] < 0
