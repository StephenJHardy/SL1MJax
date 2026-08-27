from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis

SCRIPT = Path(__file__).parents[1] / "scripts" / "ablate_3c391_native_averaging.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))


def _module():
    spec = importlib.util.spec_from_file_location("ablate_3c391_native_averaging", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _block(*, model: bool = True) -> VisibilityBlock:
    rows = 6
    channels = 4
    visibility = np.arange(rows * channels, dtype=np.float64).reshape(rows, channels, 1)
    visibility = visibility + 1j * visibility[::-1]
    prediction = 0.8 * visibility if model else None
    return VisibilityBlock(
        uvw_m=np.column_stack((np.arange(rows) * 10.0, np.zeros(rows), np.zeros(rows))),
        frequency_hz=np.arange(channels) * 2e6 + 4.5e9,
        visibility=visibility,
        model_visibility=prediction,
        weight=np.ones((rows, channels, 1)),
        flag=np.zeros((rows, channels, 1), dtype=bool),
        time_s=np.arange(rows) * 10.0 + 5.0,
        antenna1=np.zeros(rows, dtype=np.int32),
        antenna2=np.ones(rows, dtype=np.int32),
        field_id=np.full(rows, 2, dtype=np.int32),
        scan_id=np.ones(rows, dtype=np.int32),
        interval_s=np.full(rows, 10.0),
        correlations=(Correlation.RR,),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )


def test_parse_cases_and_reject_invalid_values() -> None:
    module = _module()
    cases = module._parse_cases("10:2,20:4,60:32")
    assert [case.label for case in cases] == ["10s_2MHz", "20s_4MHz", "60s_32MHz"]
    with pytest.raises(argparse.ArgumentTypeError):
        module._parse_cases("10x2")
    with pytest.raises(argparse.ArgumentTypeError):
        module._parse_cases("10:2,10:2")


def test_average_case_averages_attached_native_prediction() -> None:
    module = _module()
    source = _block()
    averaged = module._average_case(source, module.AveragingCase(20.0, 4.0))
    assert averaged.shape == (3, 2, 1)
    assert averaged.model_visibility is not None
    np.testing.assert_allclose(averaged.model_visibility, 0.8 * averaged.visibility)
    np.testing.assert_allclose(averaged.weight, 4.0)


def test_native_case_is_identified_as_identity_averaging() -> None:
    module = _module()
    source = _block()
    averaged = module._average_case(source, module.AveragingCase(10.0, 2.0))
    assert module._is_identity_averaging(source, averaged)
    coarse = module._average_case(source, module.AveragingCase(20.0, 4.0))
    assert not module._is_identity_averaging(source, coarse)


def test_metric_moments_are_additive() -> None:
    module = _module()
    block = _block(model=False)
    prediction = 0.75 * block.visibility
    all_moments = module._metric_moments(block, prediction)
    first = module._metric_moments(
        block, prediction, np.arange(block.shape[0])[:, None, None] < 3
    )
    second = module._metric_moments(
        block, prediction, np.arange(block.shape[0])[:, None, None] >= 3
    )
    first.add(second)
    assert first.sample_count == all_moments.sample_count
    assert first.weight_sum == all_moments.weight_sum
    assert first.residual_power == pytest.approx(all_moments.residual_power)
    assert first.signal_power == pytest.approx(all_moments.signal_power)
    payload = module._metric_payload(all_moments)
    assert payload["mean_weighted_squared_residual"] == pytest.approx(
        all_moments.residual_power / all_moments.sample_count
    )


def test_holdout_time_bins_follow_interleaved_fold_assignment() -> None:
    module = _module()
    block = _block(model=False)
    bins = module._holdout_time_bins(
        (block,), fold_bin_seconds=10.0, fold_count=3, holdout_fold=2
    )
    assert len(bins) == 1
    np.testing.assert_array_equal(bins[0], np.asarray([2, 5]))


def test_frequency_case_must_divide_native_channels() -> None:
    module = _module()
    block = _block()
    with pytest.raises(ValueError, match="does not divide"):
        module._frequency_bin_count(block, module.AveragingCase(10.0, 6.0))
