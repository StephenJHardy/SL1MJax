from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_3c391_fullpol_calibrator_gate.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "validate_3c391_fullpol_calibrator_gate", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibrator_gate_defaults_use_the_two_immutable_products() -> None:
    module = _module()
    arguments = module.parse_args([])
    assert arguments.gkb_measurement_set.name == "3c391_gkb_only_4corr.ms"
    assert arguments.casa_measurement_set.name == "3c391_casa_fullpol_4corr.ms"
    assert arguments.field == 0
    assert "BagOfWinds" not in str(arguments.gkb_measurement_set)
    assert "fullpol_prep" in str(arguments.gkb_measurement_set)
