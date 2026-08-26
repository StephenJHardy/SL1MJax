from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.calibration import identity_solution
from sl1mjax.data.synthetic import simulate_calibration_case
from sl1mjax.polarization import Correlation

SCRIPT = Path(__file__).parents[1] / "scripts" / "sweep_3c391_calibration_interpolation.py"
IMAGE_SCRIPT = Path(__file__).parents[1] / "scripts" / "image_3c391_target.py"


def _module():
    spec = importlib.util.spec_from_file_location("calibration_interpolation_sweep", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_module():
    spec = importlib.util.spec_from_file_location("image_3c391_target_for_test", IMAGE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alignment_gate_accepts_identical_blocks_and_rejects_reordering() -> None:
    module = _module()
    block = simulate_calibration_case(noise_std=0.0).block

    module._assert_aligned(block, block, "same")
    reordered = np.arange(block.shape[0])[::-1]
    changed = replace(
        block,
        uvw_m=block.uvw_m[reordered],
        visibility=block.visibility[reordered],
        weight=block.weight[reordered],
        flag=block.flag[reordered],
        model_visibility=block.model_visibility[reordered],
        time_s=block.time_s[reordered],
        antenna1=block.antenna1[reordered],
        antenna2=block.antenna2[reordered],
    )
    with pytest.raises(ValueError, match="antenna1|time|UVW"):
        module._assert_aligned(block, changed, "reordered")


def test_polynomial_time_models_reduce_gain_complexity_and_preserve_gauge() -> None:
    module = _module()
    times = np.linspace(0.0, 50.0, 6)
    solution = identity_solution(
        antenna_count=2,
        correlations=(Correlation.RR, Correlation.LL),
        frequency_hz=np.array([1.0e9]),
        time_s=times,
        reference_antenna=0,
    )
    coordinate = (times - np.mean(times)) / np.max(np.abs(times - np.mean(times)))
    gains = solution.gains.copy()
    gains[:, 1, 0] = np.exp(0.1 + 0.2 * coordinate + 1j * (0.3 - 0.4 * coordinate))
    gains[:, 1, 1] = np.exp(-0.2 + 0.1 * coordinate**2 + 1j * (-0.1 + 0.2 * coordinate**2))
    gain_valid = solution.gain_valid.copy()
    gain_valid[2, 1, :] = False
    solution = replace(solution, gains=gains, gain_valid=gain_valid)

    constant = module._polynomial_time_model(solution, "constant")
    linear = module._polynomial_time_model(solution, "linear_trend")
    quadratic = module._polynomial_time_model(solution, "quadratic_trend")

    assert np.all(constant.gain_valid)
    np.testing.assert_allclose(constant.gains[:, 1, 0], constant.gains[0, 1, 0])
    np.testing.assert_allclose(linear.gains[:, 1, 0], gains[:, 1, 0], atol=1e-13)
    np.testing.assert_allclose(quadratic.gains[:, 1, 1], gains[:, 1, 1], atol=1e-13)
    np.testing.assert_allclose(np.angle(quadratic.gains[:, 0, :]), 0.0, atol=1e-14)


def test_candidate_specs_avoid_duplicate_constant_interpolation() -> None:
    module = _module()
    solution = identity_solution(
        antenna_count=2,
        correlations=(Correlation.RR, Correlation.LL),
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0, 10.0, 20.0]),
    )

    candidates = module._candidate_specs(
        solution,
        ["constant", "native"],
        ["nearest", "linear"],
    )

    assert [candidate[0] for candidate in candidates] == [
        "constant_nearest",
        "native_nearest",
        "native_linear",
    ]
    assert candidates[-1][1].provenance["gain_time_model"] == "native"
    assert candidates[-1][1].provenance["gain_time_native_knot_count"] == 3


def test_composite_prediction_loader_checks_shape_and_finiteness(tmp_path) -> None:
    module = _module()
    block = simulate_calibration_case(noise_std=0.0).block
    checkpoint = tmp_path / "composite.npz"
    np.savez(checkpoint, prediction_C1=np.ones(block.shape, dtype=np.complex128))

    predictions = module._load_composite_predictions(checkpoint, (block,))

    np.testing.assert_array_equal(predictions[0], 1.0)
    np.savez(
        checkpoint,
        prediction_C1=np.full(block.shape, np.nan + 0.0j, dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="non-finite"):
        module._load_composite_predictions(checkpoint, (block,))


def test_post_application_flags_do_not_require_saved_flag_table(monkeypatch, tmp_path) -> None:
    module = _image_module()

    def unexpected_open(*args, **kwargs):
        raise AssertionError("saved flag table should not be opened")

    monkeypatch.setattr(module.tables, "table", unexpected_open)
    with module._input_flag_table_context(tmp_path / "target.ms", "post_application") as table:
        assert table is None
