from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from sl1mjax.calibration_inference import CalibrationSolveConfig
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.split import calibration_split

SCRIPT = Path(__file__).parents[1] / "scripts" / "study_3c391_time_half_selfcal.py"


def _module():
    spec = importlib.util.spec_from_file_location("time_half_selfcal_study", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block() -> VisibilityBlock:
    pairs = np.asarray(
        [(first, second) for _ in range(4) for first in range(4) for second in range(first + 1, 4)]
    )
    rows = pairs.shape[0]
    gains = np.asarray([1.0, 1.08 * np.exp(0.08j), 0.94 * np.exp(-0.12j), 1.03 * np.exp(0.05j)])
    visibility = (gains[pairs[:, 0]] * np.conj(gains[pairs[:, 1]]))[:, None, None]
    return VisibilityBlock(
        uvw_m=np.zeros((rows, 3)),
        frequency_hz=np.asarray([1.0e9]),
        visibility=visibility,
        weight=np.ones((rows, 1, 1)),
        flag=np.zeros((rows, 1, 1), dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=pairs[:, 0],
        antenna2=pairs[:, 1],
        correlations=(Correlation.I,),
        receptor_basis=ReceptorBasis.STOKES,
    )


def test_static_calibration_block_preserves_samples_but_uses_one_interval() -> None:
    module = _module()
    block = _block()
    first, _, _ = module.time_half_masks(block)

    static = module.static_calibration_block(block, np.ones(block.shape), first)

    np.testing.assert_array_equal(static.active, first)
    assert np.unique(static.time_s).tolist() == [0.0]
    np.testing.assert_array_equal(static.model_visibility, np.ones(block.shape))


def test_static_gain_fit_transfers_to_other_half_on_heldout_baselines() -> None:
    module = _module()
    block = _block()
    first, second, _ = module.time_half_masks(block)

    result = module.fit_direction(
        block,
        np.ones(block.shape, dtype=np.complex128),
        first,
        second,
        phase_only=False,
        config=CalibrationSolveConfig(
            iterations=180,
            learning_rate=0.04,
            holdout_fraction=0.2,
            seed=3,
        ),
    )

    assert result["training_heldout_baselines"][
        "relative_weighted_complex_mse_change"
    ] < -0.98
    assert result["cross_time_heldout_baselines"][
        "relative_weighted_complex_mse_change"
    ] < -0.98


def test_heldout_baseline_mask_reuses_pairs_in_other_half() -> None:
    module = _module()
    block = _block()
    first, second, _ = module.time_half_masks(block)
    static = module.static_calibration_block(block, np.ones(block.shape), first)
    split = calibration_split(static, holdout_fraction=0.2, seed=7)

    evaluation = module.heldout_baseline_mask(block, split, second)

    assert np.any(evaluation)
    assert not np.any(evaluation & first)
