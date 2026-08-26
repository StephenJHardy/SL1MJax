from __future__ import annotations

import csv
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.calibration import apply_calibration, identity_solution
from sl1mjax.data.synthetic import simulate_calibration_case
from sl1mjax.polarization import Correlation

SCRIPT = Path(__file__).parents[1] / "scripts" / "sweep_3c391_gain_time_models.py"


def _module():
    spec = importlib.util.spec_from_file_location("gain_time_model_sweep", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _solution():
    return identity_solution(
        antenna_count=3,
        correlations=(Correlation.RR, Correlation.LL),
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0, 60.0, 180.0, 300.0]),
        reference_antenna=0,
    )


def test_dense_time_grid_contains_native_knots_and_requested_spacing() -> None:
    module = _module()

    grid = module._dense_time_grid(_solution(), 30.0)

    assert grid[0] == 0.0
    assert grid[-1] == 300.0
    assert set(_solution().gain_time_s).issubset(set(grid))
    assert np.max(np.diff(grid)) <= 30.0


def test_candidate_specs_cover_native_smoothing_and_gp_grid() -> None:
    module = _module()

    candidates = module._candidate_specs(
        _solution(),
        smoothing_strengths=(0.1, 1.0),
        gp_length_scales_s=(60.0, 180.0),
        gp_noise_variances=(0.01, 0.1),
        dense_step_s=30.0,
    )
    labels = [label for label, _, _ in candidates]

    assert len(candidates) == 7
    assert labels[0] == "native_linear"
    assert "smooth_0.1" in labels
    assert "circular_gp_l180_n0.1" in labels
    for _, solution, _ in candidates:
        assert solution.interpolation == "linear"
        np.testing.assert_allclose(np.angle(solution.gains[:, 0, :]), 0.0, atol=1e-14)


def test_candidate_specs_reject_duplicate_hyperparameters() -> None:
    module = _module()

    with pytest.raises(ValueError, match="not unique"):
        module._candidate_specs(
            _solution(),
            smoothing_strengths=(0.1, 0.1),
            gp_length_scales_s=(),
            gp_noise_variances=(),
            dense_step_s=30.0,
        )


def test_ranking_table_accepts_family_specific_metadata(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "ranking.csv"

    module._write_table(
        path,
        [
            {"label": "native", "family": "native", "score": 1.0},
            {
                "label": "smooth",
                "family": "second_derivative",
                "smoothing_strength": 0.1,
                "score": 0.9,
            },
        ],
    )

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["smoothing_strength"] == ""
    assert rows[1]["smoothing_strength"] == "0.1"


def test_shared_fixed_term_gain_ratio_matches_full_candidate_application() -> None:
    module = _module()
    case = simulate_calibration_case(
        antenna_count=5,
        time_count=4,
        channel_count=7,
        terms=("G", "K", "B"),
        seed=8,
    )
    gain_factor = np.exp(
        0.03 * np.arange(case.truth.antenna_count)[None, :, None]
        + 0.1j * np.arange(case.truth.gain_time_s.size)[:, None, None]
    )
    candidate = replace(case.truth, gains=case.truth.gains * gain_factor)

    expected = apply_calibration(case.block, candidate, extrapolate=True)
    actual = module._apply_shared_fixed_calibration(
        case.block,
        case.truth,
        candidate,
    )

    np.testing.assert_allclose(actual.visibility, expected.visibility, atol=2e-14)
    np.testing.assert_array_equal(actual.flag, expected.flag)
    np.testing.assert_array_equal(actual.weight, expected.weight)
