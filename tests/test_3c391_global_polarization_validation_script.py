from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_3c391_global_polarization.py"
POL_FIXTURE = Path(__file__).parent / "fixtures" / "3c391_polarization_golden.npz"
KBG_FIXTURE = Path(__file__).parent / "fixtures" / "3c391_calibration_golden.npz"


def _module():
    spec = importlib.util.spec_from_file_location(
        "validate_3c391_global_polarization", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_script_golden_only_labels_apply_back_and_holdout(
    tmp_path: Path,
) -> None:
    if not POL_FIXTURE.is_file() or not KBG_FIXTURE.is_file():
        pytest.skip("polarisation or K/B/G golden is missing")
    module = _module()
    assert (
        module.main(
            [
                "--polarization-golden",
                str(POL_FIXTURE),
                "--calibration-golden",
                str(KBG_FIXTURE),
                "--output",
                str(tmp_path),
                "--skip-images",
                "--no-measurement-set",
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["floors"]["jax_3c286_apply_back"]["independence"] == "apply_back"
    assert report["floors"]["casa_3c84_in_sample"]["independence"] == "in_sample"
    holdout = report["three_c286_holdout"]["holdout_floor"]
    assert holdout["independence"] == "held_out_calibrator"
    assert holdout["fractional_linear"] == pytest.approx(0.112, abs=0.03)
    assert "independent" not in json.dumps(report)
    assert (
        report["global_quv"]["jax_3c286_apply_back"]["provenance"]["regressor"]
        == "complex_stokes_i_model"
    )
    assert "pointings" not in report
