from __future__ import annotations

import importlib.util
from pathlib import Path

DIAGNOSE = Path(__file__).parents[1] / "scripts" / "diagnose_3c391_voltage_beam_polarization.py"
FREEZE = Path(__file__).parents[1] / "scripts" / "freeze_3c391_polarisation_ancestor.py"
COMPARE = Path(__file__).parents[1] / "scripts" / "run_3c391_fullpol_jones_compare_bacchus.sh"
IMAGING = Path(__file__).parents[1] / "scripts" / "run_3c391_diagnostic_full_jones_imaging.sh"
CONTINUE = Path(__file__).parents[1] / "scripts" / "run_3c391_continue_full_jones_imaging.sh"
CONTINUE_PY = Path(__file__).parents[1] / "scripts" / "continue_3c391_diagnostic_full_jones.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnose_defaults_require_ancestor_and_gate() -> None:
    module = _module(DIAGNOSE, "diagnose_3c391_voltage_beam_polarization")
    arguments = module.parse_args([])
    assert arguments.frozen_diagonal_product.name == "frozen_diagonal_ancestor"
    assert arguments.calibrator_gate_report.name == "report.json"
    assert arguments.measurement_set.name == "3c391_gkb_only_4corr.ms"
    source = DIAGNOSE.read_text(encoding="utf-8")
    assert "fold3_mosaic_score" in source
    assert "leakage_modelling_evidence" in source
    assert "require_polarisation_ancestor" in source
    assert "diagnostic_run_metadata" in source
    assert "stamp_diagnostic_interpretation" in source
    assert "require_fullpol_diagnostic_ok" in source


def test_freeze_script_defaults_to_explicit_diagonal_product() -> None:
    module = _module(FREEZE, "freeze_3c391_polarisation_ancestor")
    arguments = module.parse_args([])
    assert arguments.source.name == "diagonal_copolar"
    assert arguments.dest.name == "frozen_diagonal_ancestor"


def test_jones_compare_wrapper_refuses_live_dest() -> None:
    text = COMPARE.read_text(encoding="utf-8")
    assert "SL1MJax-phase6-explicit-20260831" in text
    assert "must not be the live checkout" in text
    assert "live Phase 6 job still holds the GPU" in text
    assert 'rsync -a "$LIVE_OUT/' not in text
    assert "/home/stephen/checkouts/SL1MJax-phase6-20260830" in text
    assert "freeze_3c391_polarisation_ancestor.py" in text


def test_jones_imaging_wrapper_waits_and_does_not_freeze() -> None:
    text = IMAGING.read_text(encoding="utf-8")
    assert "must not be the live checkout" in text
    assert "diagnose_3c391_voltage_beam_polarization.py" in text
    assert "inspect_3c391_fullpol_diagnostic.py" in text
    assert "--allow-diagnostic-full-jones" in text
    assert "--stage baseline" in text
    assert "--beams full_jones" in text
    assert "--stage full" not in text
    assert "full_round1" not in text
    assert "jones_imaging" in text


def test_continue_script_starts_from_checkpoint_and_keeps_fold_4_sealed() -> None:
    module = _module(CONTINUE_PY, "continue_3c391_diagnostic_full_jones")
    arguments = module.parse_args([])
    assert arguments.source.name == "full_jones"
    assert arguments.output.name == "jones_imaging_continue"
    assert arguments.steps == 400
    assert arguments.validation_interval == 5
    text = CONTINUE.read_text(encoding="utf-8")
    assert "jones_imaging/baseline/full_jones" in text
    assert "jones_imaging_continue" in text
    assert "must not be the live checkout" in text
    source = CONTINUE_PY.read_text(encoding="utf-8")
    assert "max_rounds=0" in source
    assert "do_not_freeze_full_jones" in source
