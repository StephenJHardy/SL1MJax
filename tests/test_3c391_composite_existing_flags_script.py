from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from test_3c391_composite_script import _block

SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_3c391_composite_existing_flags.py"


def _module():
    spec = importlib.util.spec_from_file_location("compare_composite_existing_flags", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combine_cohorts_keeps_training_discovery_and_flagged_evaluation_separate() -> None:
    module = _module()
    active = _block()
    flagged = _block()
    active_prediction = np.zeros(active.shape, dtype=np.complex128)
    flagged_prediction = np.ones(flagged.shape, dtype=np.complex128)
    discovery = np.zeros(active.shape, dtype=bool)
    discovery[:5] = active.active[:5]

    blocks, predictions, discovery_masks, evaluation_masks = module._combine_cohorts(
        (active,),
        (active_prediction,),
        (flagged,),
        (flagged_prediction,),
        (discovery,),
    )

    assert blocks[0].shape[0] == active.shape[0] + flagged.shape[0]
    np.testing.assert_array_equal(discovery_masks[0][: active.shape[0]], discovery)
    assert not np.any(discovery_masks[0][active.shape[0] :])
    assert not np.any(evaluation_masks[0][: active.shape[0]])
    np.testing.assert_array_equal(
        evaluation_masks[0][active.shape[0] :],
        flagged.active,
    )
    np.testing.assert_array_equal(predictions[0][: active.shape[0]], 0.0)
    np.testing.assert_array_equal(predictions[0][active.shape[0] :], 1.0)
