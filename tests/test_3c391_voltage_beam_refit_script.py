from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from sl1mjax.cassbeam_beam import voltage_beam_for_mode
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam
from sl1mjax.voltage_flux_refit import transfer_diagonal_is_consistent

SCRIPT = Path(__file__).parents[1] / "scripts" / "refit_3c391_voltage_beam_fluxes.py"


def _module():
    spec = importlib.util.spec_from_file_location("refit_3c391_voltage_beam_fluxes", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_construct_beams_leaves_factory_frozen() -> None:
    module = _module()
    beams = module.construct_beams(0.07)
    assert isinstance(beams["static_scalar"], AnalyticAiryVoltageBeam)
    assert beams["full_jones_unfrozen"].allow_unfrozen is True
    assert beams["full_jones_unfrozen"].off_diagonal is True
    factory = voltage_beam_for_mode("diagonal_copolar")
    assert factory.off_diagonal is False


def test_script_records_inconsistent_transfer_without_running() -> None:
    gate = transfer_diagonal_is_consistent(
        {
            "pointings": {
                "C1": {
                    "beams": {
                        "static_scalar": {"total": 1.0},
                        "diagonal_copolar": {"total": 0.9},
                    }
                },
                "C4": {
                    "beams": {
                        "static_scalar": {"total": 1.0},
                        "diagonal_copolar": {"total": 1.2},
                    }
                },
            }
        }
    )
    assert gate["consistent"] is False
    assert gate["n_pointings"] == 2
