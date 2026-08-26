from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.data.synthetic import simulate_calibration_case

SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_3c391_calibration_flags.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))


def _module():
    spec = importlib.util.spec_from_file_location("calibration_flags", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matched_active_uses_intersection_without_changing_values() -> None:
    module = _module()
    case = simulate_calibration_case(antenna_count=4, time_count=3, channel_count=2, seed=4)
    first = replace(case.block, flag=np.zeros(case.block.shape, dtype=bool))
    second_flag = np.zeros(case.block.shape, dtype=bool)
    second_flag[0, 0, 0] = True
    second = replace(case.block, flag=second_flag, visibility=case.block.visibility * 2.0)

    matched_first, matched_second = module._matched_active((first,), (second,))

    np.testing.assert_array_equal(matched_first[0].active, matched_second[0].active)
    assert not matched_first[0].active[0, 0, 0]
    np.testing.assert_array_equal(matched_first[0].visibility, first.visibility)
    np.testing.assert_array_equal(matched_second[0].visibility, second.visibility)


def test_alignment_rejects_different_times() -> None:
    module = _module()
    case = simulate_calibration_case(antenna_count=4, time_count=3, channel_count=2, seed=5)
    shifted = replace(case.block, time_s=case.block.time_s + 1.0)

    try:
        module._matched_active((case.block,), (shifted,))
    except ValueError as error:
        assert "time_s" in str(error)
    else:
        raise AssertionError("misaligned fixtures should be rejected")
