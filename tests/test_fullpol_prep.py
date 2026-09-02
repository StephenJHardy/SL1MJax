from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.calibration import identity_solution, import_casa_polarization_solution
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.fullpol_prep import (
    CASA_FULLPOL_STATE,
    FULLPOL_SCIENCE_FIELDS,
    GKB_ONLY_STATE,
    NATIVE_CHANNEL_COUNT,
    POLCAL_CHANNEL_START,
    POLCAL_CHANNEL_STOP,
    POLCAL_UNSUPPORTED_ANTENNA_REASON,
    POLCAL_UNSUPPORTED_REASON,
    REQUIRED_CORRELATIONS,
    THREE_C286_CASAGUIDE_ANGLE_DEG,
    THREE_C286_FRACTIONAL_LINEAR,
    apply_polarization_before_averaging,
    attach_fullpol_contract,
    beam_comparison_mask,
    casa_comparison_mask,
    common_active_mask,
    compare_dterm_crosshand_flags,
    evaluate_three_c286_gate,
    fold3_mosaic_score,
    freeze_polarisation_stokes_i_ancestor,
    fullpol_phase6_folds,
    leakage_modelling_evidence,
    paired_fold3_delta,
    parse_flag_version_list,
    polarization_terms_only,
    abort_fullpol_diagnostic_failures,
    diagnostic_run_metadata,
    require_calibrator_gate_report,
    require_frozen_diagonal_checkpoint,
    require_polarisation_ancestor,
    stamp_diagnostic_interpretation,
    unsupported_dterm_antennas,
    validate_fullpol_ms_inventory,
)
from sl1mjax.polarization import ReceptorBasis

POL_FIXTURE = Path(__file__).parent / "fixtures" / "3c391_polarization_golden.npz"
KBG_FIXTURE = Path(__file__).parent / "fixtures" / "3c391_calibration_golden.npz"
CREATE_SCRIPT = Path(__file__).parents[1] / "scripts" / "create_3c391_fullpol_ms_products.py"


def _four_corr_block(
    *,
    rows: int = 4,
    channels: int = NATIVE_CHANNEL_COUNT,
    antenna1: int = 0,
    antenna2: int = 1,
) -> VisibilityBlock:
    visibility = np.ones((rows, channels, 4), dtype=np.complex128)
    visibility[..., 1] = 0.12 + 0.04j
    visibility[..., 2] = 0.12 - 0.04j
    return VisibilityBlock(
        uvw_m=np.zeros((rows, 3), dtype=np.float64),
        frequency_hz=4.536e9 + 2.0e6 * np.arange(channels),
        visibility=visibility,
        weight=np.full(visibility.shape, 2.5),
        flag=np.zeros(visibility.shape, dtype=bool),
        time_s=10.0 * np.arange(rows, dtype=np.float64),
        antenna1=np.full(rows, antenna1, dtype=np.int32),
        antenna2=np.full(rows, antenna2, dtype=np.int32),
        correlations=REQUIRED_CORRELATIONS,
        receptor_basis=ReceptorBasis.CIRCULAR,
        interval_s=np.full(rows, 10.0),
        field_id=np.full(rows, 2, dtype=np.int32),
    )


def _inventory(**overrides):
    payload = {
        "polarizations": [{"correlations": ["RR", "RL", "LR", "LL"]}],
        "spectral_windows": [
            {
                "channel_count": 64,
                "channel_width_hz": [2.0e6] * 64,
            }
        ],
        "field_ids": list(FULLPOL_SCIENCE_FIELDS),
        "visibility_columns": ["DATA", "CORRECTED_DATA"],
        "active_samples": {
            "by_correlation": {"RR": 10, "RL": 8, "LR": 7, "LL": 10},
            "by_field": {
                str(field): {"RR": 2, "RL": 1, "LR": 1, "LL": 2}
                for field in FULLPOL_SCIENCE_FIELDS
            },
        },
    }
    payload.update(overrides)
    return payload


def test_validate_inventory_requires_rl_lr_and_science_fields() -> None:
    assert validate_fullpol_ms_inventory(_inventory()) == ()
    missing_rl = _inventory(
        polarizations=[{"correlations": ["RR", "LL"]}],
        active_samples={"by_correlation": {"RR": 1, "LL": 1}, "by_field": {}},
    )
    failures = validate_fullpol_ms_inventory(missing_rl)
    assert any("RL" in item for item in failures)
    assert any("LR" in item for item in failures)

    empty_cross = _inventory()
    empty_cross["active_samples"]["by_correlation"]["RL"] = 0
    empty_cross["active_samples"]["by_field"]["2"]["RL"] = 0
    failures = validate_fullpol_ms_inventory(empty_cross)
    assert any("RL has no active samples" in item for item in failures)
    assert any("field 2" in item for item in failures)


def test_attach_contract_records_calibration_state() -> None:
    attached = attach_fullpol_contract(
        _inventory(),
        calibration_state={"calibration_state": GKB_ONLY_STATE},
    )
    assert attached["contract_passed"] is True
    assert attached["calibration_state"]["calibration_state"] == GKB_ONLY_STATE


def test_parse_flag_version_list() -> None:
    versions = parse_flag_version_list(
        "sl1mjax_calibration_input : Tutorial\n"
        "sl1mjax_post_polcal : After K/B/G plus Kcross/Df/Xf apply\n"
    )
    assert [item["name"] for item in versions] == [
        "sl1mjax_calibration_input",
        "sl1mjax_post_polcal",
    ]


def test_polarization_terms_only_keeps_casa_parallel_preserving() -> None:
    if not POL_FIXTURE.is_file() or not KBG_FIXTURE.is_file():
        pytest.skip("polarisation golden is missing")
    imported = import_casa_polarization_solution(
        POL_FIXTURE, KBG_FIXTURE, label="flux_angle"
    )
    pol_only = polarization_terms_only(imported)
    assert pol_only.leakage_application == "casa_parallel_preserving"
    assert pol_only.apply_parallactic_angle is True
    np.testing.assert_allclose(np.abs(pol_only.gains), 1.0)
    assert pol_only.leakage is not None


def test_apply_polarization_before_averaging_masks_edges_and_keeps_weights() -> None:
    if not POL_FIXTURE.is_file() or not KBG_FIXTURE.is_file():
        pytest.skip("polarisation golden is missing")
    imported = import_casa_polarization_solution(
        POL_FIXTURE, KBG_FIXTURE, label="flux_angle"
    )
    block = _four_corr_block()
    prepared = apply_polarization_before_averaging(
        block,
        imported,
        frequency_bins=8,
        time_bin_seconds=20.0,
    )
    assert prepared.frequency_hz.size == 8
    assert prepared.provenance["polarisation_before_averaging"] is True
    assert prepared.provenance["polarisation_applied"] == 1
    assert prepared.provenance["weight_policy"] == "preserve_input_weights"
    assert prepared.provenance["unsupported_channel_reason"] == POLCAL_UNSUPPORTED_REASON
    assert prepared.provenance["unsupported_crosshand_reason"] == (
        POLCAL_UNSUPPORTED_ANTENNA_REASON
    )
    native = apply_polarization_before_averaging(
        block,
        imported,
        frequency_bins=None,
        time_bin_seconds=None,
    )
    assert native.frequency_hz.size == NATIVE_CHANNEL_COUNT
    assert np.all(native.flag[:, :POLCAL_CHANNEL_START, :])
    assert np.all(native.flag[:, POLCAL_CHANNEL_STOP:, :])
    np.testing.assert_allclose(native.weight, block.weight)
    with pytest.raises(ValueError, match="second polarisation apply"):
        apply_polarization_before_averaging(native, imported)


def test_unsupported_dterm_antennas_from_validity_mask() -> None:
    solution = identity_solution(
        antenna_count=3,
        correlations=REQUIRED_CORRELATIONS,
        frequency_hz=np.linspace(4.5e9, 4.6e9, 4),
    )
    leakage = np.zeros((3, 4, 2), dtype=np.complex128)
    valid = np.ones((3, 4, 2), dtype=bool)
    valid[2] = False
    solution = replace(
        solution,
        leakage=leakage,
        leakage_frequency_hz=solution.bandpass_frequency_hz,
        leakage_valid=valid,
    )
    assert unsupported_dterm_antennas(solution) == (2,)


def test_common_active_mask_is_intersection() -> None:
    first = _four_corr_block(channels=8)
    second_flag = np.zeros(first.shape, dtype=bool)
    second_flag[0, 0, 1] = True
    first_flag = np.zeros(first.shape, dtype=bool)
    first_flag[1, 0, 2] = True
    first = replace(first, flag=first_flag)
    second = replace(first, flag=second_flag)
    mask = common_active_mask(first, second)
    assert not mask[0, 0, 1]
    assert not mask[1, 0, 2]
    assert mask[0, 1, 0]


def test_create_script_names_two_immutable_products() -> None:
    spec = importlib.util.spec_from_file_location(
        "create_3c391_fullpol_ms_products", CREATE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.GKB_ONLY_NAME == "3c391_gkb_only_4corr.ms"
    assert module.CASA_FULLPOL_NAME == "3c391_casa_fullpol_4corr.ms"
    assert module.GKB_ONLY_FLAG == "sl1mjax_gkb_only"
    assert module.POST_POLCAL_FLAG == "sl1mjax_post_polcal"
    arguments = module.parse_args(["--skip-gkb-apply"])
    assert arguments.skip_gkb_apply is True
    assert CASA_FULLPOL_STATE == "casa_fullpol"


def test_beam_mask_keeps_casa_only_flags_out_of_internal_comparison() -> None:
    jax = _four_corr_block(channels=8)
    casa_flag = np.zeros(jax.shape, dtype=bool)
    casa_flag[0, 0, 1] = True
    casa = replace(jax, flag=casa_flag)
    beam = beam_comparison_mask(jax)
    casa_mask = casa_comparison_mask(jax, casa)
    assert bool(beam[0, 0, 1]) is True
    assert bool(casa_mask[0, 0, 1]) is False
    assert np.count_nonzero(beam) > np.count_nonzero(casa_mask)


def test_dterm_crosshand_agreement_covers_casa_extra_flags() -> None:
    rows = 3
    channels = NATIVE_CHANNEL_COUNT
    antenna1 = np.array([0, 0, 2], dtype=np.int32)
    antenna2 = np.array([1, 1, 1], dtype=np.int32)
    gkb = replace(
        _four_corr_block(rows=rows, channels=channels),
        antenna1=antenna1,
        antenna2=antenna2,
    )
    solution = identity_solution(
        antenna_count=3,
        correlations=REQUIRED_CORRELATIONS,
        frequency_hz=gkb.frequency_hz,
    )
    valid = np.ones((3, channels, 2), dtype=bool)
    valid[2] = False
    solution = replace(
        solution,
        leakage=np.zeros((3, channels, 2), dtype=np.complex128),
        leakage_frequency_hz=gkb.frequency_hz,
        leakage_valid=valid,
    )
    casa_flag = np.zeros(gkb.shape, dtype=bool)
    casa_flag[2, POLCAL_CHANNEL_START:POLCAL_CHANNEL_STOP, 1] = True
    casa_flag[2, POLCAL_CHANNEL_START:POLCAL_CHANNEL_STOP, 2] = True
    jax_flag = casa_flag.copy()
    agreed = compare_dterm_crosshand_flags(
        gkb, replace(gkb, flag=jax_flag), replace(gkb, flag=casa_flag), solution
    )
    assert agreed["agreed"] is True
    assert agreed["casa_only"] == 0
    missed = compare_dterm_crosshand_flags(gkb, gkb, replace(gkb, flag=casa_flag), solution)
    assert missed["agreed"] is False
    assert missed["casa_only"] > 0


def test_three_c286_gate_accepts_casaguide_fraction_and_angle() -> None:
    good = {
        "q": THREE_C286_FRACTIONAL_LINEAR * np.cos(np.deg2rad(66.0)),
        "u": THREE_C286_FRACTIONAL_LINEAR * np.sin(np.deg2rad(66.0)),
        "v": 0.001,
        "fractional_linear": THREE_C286_FRACTIONAL_LINEAR,
        "casaguide_angle_deg": THREE_C286_CASAGUIDE_ANGLE_DEG,
    }
    assert evaluate_three_c286_gate(good)["passed"] is True
    bad = dict(good, v=0.2)
    assert evaluate_three_c286_gate(bad)["passed"] is False


def test_fullpol_folds_keep_four_correlations_together_and_poison_fold_4() -> None:
    rows = 8
    block = replace(
        _four_corr_block(rows=rows, channels=8),
        time_s=60.0 * np.arange(rows, dtype=np.float64),
        interval_s=np.full(rows, 60.0),
    )
    folds, prepared = fullpol_phase6_folds((block,), poison_sealed=True)
    assert np.any(folds.sealed[0])
    assert not np.any(folds.train[0] & folds.sealed[0])
    assert not np.array_equal(prepared[0].visibility, block.visibility)
    assert np.array_equal(
        prepared[0].visibility[~folds.sealed[0]],
        block.visibility[~folds.sealed[0]],
    )


def test_require_frozen_diagonal_checkpoint_fails_when_absent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not ready to freeze"):
        require_frozen_diagonal_checkpoint(tmp_path)


def _write_diagonal_product(directory: Path, *, holdout_loss: float = 0.01440) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "beam_mode": "diagonal_copolar",
        "train_loss": 0.01363,
        "holdout_loss": holdout_loss,
        "kkt_residual": 0.0063,
        "audit_under_resolved": True,
        "sealed_fold_unused": True,
        "config": {"sky_max_depth": 2, "integration_max_depth": 3},
        "manifest": {
            "train_folds": [0, 1, 2],
            "holdout_fold": 3,
            "sealed_fold": 4,
        },
    }
    (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    for name in ("checkpoint.json", "component_table.json", "integration_plan.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    return directory


def test_freeze_labels_polarisation_test_ancestor_not_final_stokes_i(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    _write_diagonal_product(baseline / "diagonal_copolar")
    (baseline / "static_scalar").mkdir(parents=True)
    (baseline / "static_scalar" / "summary.json").write_text(
        json.dumps({"holdout_loss": 0.01423, "train_loss": 0.01328, "kkt_residual": 0.0093}),
        encoding="utf-8",
    )
    dest = tmp_path / "frozen_diagonal_ancestor"
    payload = freeze_polarisation_stokes_i_ancestor(baseline / "diagonal_copolar", dest)
    assert payload["role"] == "polarisation_test_ancestor"
    assert payload["not_final_cband_stokes_i"] is True
    assert payload["qualification"]["diagonal_is_not_winning_stokes_i"] is True
    assert payload["qualification"]["winning_stokes_i_holdout"] == "static_scalar"
    assert payload["hashes"]["checkpoint.json"]
    product, freeze = require_polarisation_ancestor(dest)
    assert product == dest
    assert freeze["role"] == "polarisation_test_ancestor"
    with pytest.raises(ValueError, match="polarisation-test ancestor"):
        require_polarisation_ancestor(baseline / "diagonal_copolar")


def test_calibrator_gate_report_requires_pass_and_single_apply(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "polarisation_applies": {"exactly_one_each": True},
            }
        ),
        encoding="utf-8",
    )
    assert require_calibrator_gate_report(path)["passed"] is True
    path.write_text(json.dumps({"passed": False, "failures": ["V"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="has not passed"):
        require_calibrator_gate_report(path)


def test_fold3_rl_lr_is_required_leakage_evidence() -> None:
    block = _four_corr_block(rows=4, channels=4)
    diagonal_pred = block.visibility.copy()
    full_pred = block.visibility.copy()
    full_pred[..., 1] += 0.05
    full_pred[..., 2] += 0.05
    mask = np.ones(block.shape, dtype=bool)
    antennas = np.array([[0.0, 0.0, 0.0], [25.0, 0.0, 0.0]], dtype=np.float64)
    diagonal = fold3_mosaic_score(
        (block,),
        (diagonal_pred,),
        (mask,),
        antenna_position_m=antennas,
        pointing_ids=("C1",),
    )
    worse = fold3_mosaic_score(
        (block,),
        (full_pred,),
        (mask,),
        antenna_position_m=antennas,
        pointing_ids=("C1",),
    )
    delta = paired_fold3_delta(diagonal, worse)
    evidence = leakage_modelling_evidence(delta)
    assert worse["RL_LR"]["mse"] > diagonal["RL_LR"]["mse"]
    assert evidence["sufficient_for_off_diagonal_jones"] is False
    better_pred = block.visibility.copy()
    better = fold3_mosaic_score(
        (block,),
        (better_pred,),
        (mask,),
        antenna_position_m=antennas,
        pointing_ids=("C1",),
    )
    rr_only = {
        "RL_LR": {"mse": 0.0},
        "RR_LL": {"mse": -0.01},
    }
    rr_only_evidence = leakage_modelling_evidence(rr_only)
    assert rr_only_evidence["rr_ll_only_is_not_evidence"] is True
    assert rr_only_evidence["sufficient_for_off_diagonal_jones"] is False
    improved = leakage_modelling_evidence({"RL_LR": {"mse": -0.02}, "RR_LL": {"mse": 0.0}})
    assert improved["sufficient_for_off_diagonal_jones"] is True
    assert better["total"]["samples"] == mask.size
    assert delta["by_pointing"][0]["pointing_id"] == "C1"


def test_diagnostic_run_is_not_beam_selection_or_validation() -> None:
    meta = diagnostic_run_metadata()
    assert meta["kind"] == "fullpol_heldout_residual_diagnostic"
    assert meta["scientific_validation"] is False
    assert meta["beam_selection"] is False
    assert meta["do_not_freeze_full_jones"] is True
    assert meta["fold_4_sealed"] is True
    stamped = stamp_diagnostic_interpretation({"beams": {}})
    assert stamped["do_not_freeze_full_jones"] is True
    assert stamped["not_evidence_about_the_sky"] is True


def test_abort_fullpol_diagnostic_on_wrong_ms_and_nonfinite_loss(tmp_path: Path) -> None:
    wrong = tmp_path / "3c391_casa_fullpol_4corr.ms"
    wrong.mkdir()
    (tmp_path / "3c391_casa_fullpol_4corr.ms.calibration_state.json").write_text(
        json.dumps(
            {
                "calibration_state": "casa_fullpol",
                "jax_polarisation_input": False,
                "do_not_apply_jax_polarisation": True,
                "applied": ["Kcross", "Df"],
            }
        ),
        encoding="utf-8",
    )
    failures = abort_fullpol_diagnostic_failures(measurement_set=wrong)
    assert any("wrong MeasurementSet" in item for item in failures)
    nan_summary = {
        "regional_polarization": "not_started",
        "folds": {"sealed": 4, "poisoned": True},
        "beams": {
            "diagonal_copolar": {"holdout_loss": float("nan"), "fold3": {"total": {"mse": 0.1}}},
            "full_jones_unfrozen": {"holdout_loss": 0.1, "fold3": {"total": {"mse": 0.1}}},
        },
    }
    assert any("not finite" in item for item in abort_fullpol_diagnostic_failures(summary=nan_summary))
    regional = abort_fullpol_diagnostic_failures(regional_started=True, fold4_opened=True)
    assert any("regional Q/U" in item for item in regional)
    assert any("fold 4" in item for item in regional)
