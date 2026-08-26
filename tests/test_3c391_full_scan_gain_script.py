from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.data.synthetic import simulate_calibration_case

SCRIPT = Path(__file__).parents[1] / "scripts" / "fit_3c391_full_scan_gains.py"


def _module():
    spec = importlib.util.spec_from_file_location("full_scan_gain", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scan_gain_coordinates_use_active_weight_centroid() -> None:
    module = _module()
    case = simulate_calibration_case(
        antenna_count=4,
        time_count=4,
        channel_count=3,
        noise_std=0.0,
        seed=3,
    )
    scan_id = np.where(case.block.time_s < 120.0, 4, 12).astype(np.int32)
    weight = case.block.weight.copy()
    weight[case.block.time_s == 60.0] *= 3.0
    block = replace(case.block, scan_id=scan_id, weight=weight)

    coordinates = module._scan_gain_coordinates(block)

    row_weight = np.sum(block.weight, axis=(1, 2))
    for scan in (4, 12):
        rows = scan_id == scan
        expected = np.sum(block.time_s[rows] * row_weight[rows]) / np.sum(row_weight[rows])
        np.testing.assert_allclose(coordinates[rows], expected)
    assert np.unique(coordinates).size == 2


def test_scan_gain_coordinates_reject_fully_flagged_scan() -> None:
    module = _module()
    case = simulate_calibration_case(
        antenna_count=4,
        time_count=2,
        channel_count=3,
        noise_std=0.0,
        seed=4,
    )
    scan_id = np.where(case.block.time_s == 0.0, 4, 12).astype(np.int32)
    flag = case.block.flag.copy()
    flag[scan_id == 12] = True
    block = replace(case.block, scan_id=scan_id, flag=flag)

    try:
        module._scan_gain_coordinates(block)
    except ValueError as error:
        assert "scan 12 has no active weight" in str(error)
    else:
        raise AssertionError("fully flagged scan should have been rejected")
