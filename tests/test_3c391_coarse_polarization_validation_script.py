from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_3c391_coarse_polarization.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "validate_3c391_coarse_polarization", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coarse_script_records_v_zero_without_measurement_set(tmp_path: Path) -> None:
    module = _module()
    assert (
        module.main(
            [
                "--output",
                str(tmp_path),
                "--no-measurement-set",
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["v_in_sky_model"] == 0.0
    assert report["milestone"] == "coarse_joint_qu_activation"
    assert "spatial_v_activation" in report["next_step_not_started"]
    assert "self_cal" in report["next_step_not_started"]
    assert "rm" in report["next_step_not_started"]
    assert "leave_one_pointing_out" in report["c7_status"]
