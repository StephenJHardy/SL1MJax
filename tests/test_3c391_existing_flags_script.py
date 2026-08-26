from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.calibration import apply_calibration
from sl1mjax.data.synthetic import simulate_calibration_case

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_3c391_existing_flags.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))


def _module():
    spec = importlib.util.spec_from_file_location("existing_flags", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_flagged_cohort_calibrates_only_originally_flagged_samples() -> None:
    module = _module()
    case = simulate_calibration_case(antenna_count=5, time_count=4, channel_count=3, seed=17)
    flags = np.zeros(case.block.shape, dtype=bool)
    flags[::2] = True
    source = replace(case.block, flag=flags)

    actual = module._prepare_flagged_cohort(source, case.truth)
    expected = apply_calibration(replace(source, flag=~flags), case.truth, extrapolate=True)

    np.testing.assert_allclose(actual.visibility, expected.visibility)
    np.testing.assert_array_equal(actual.flag, expected.flag)
    assert np.all(~actual.active[~flags])


def test_match_active_samples_intersects_masks_before_averaging() -> None:
    module = _module()
    case = simulate_calibration_case(antenna_count=5, time_count=4, channel_count=3, seed=18)
    reference = replace(case.block, flag=np.zeros(case.block.shape, dtype=bool))
    candidate_flag = np.zeros(case.block.shape, dtype=bool)
    candidate_flag[0, 1, 0] = True
    candidate = replace(case.block, flag=candidate_flag)

    matched_reference, matched_candidate = module._match_active_samples(
        reference, candidate
    )

    np.testing.assert_array_equal(matched_reference.active, matched_candidate.active)
    assert not matched_reference.active[0, 1, 0]
