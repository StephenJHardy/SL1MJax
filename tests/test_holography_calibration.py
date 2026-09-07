from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sl1mjax.holography_calibration import (
    CASA_DF_JONES_APPLYCAL_V1,
    CASA_PARALLEL_PRESERVING_LEGACY_V1,
    COMPATIBILITY_TABLES_NOT_SCIENTIFIC_NOTE,
    CONSISTENT_3C147_FLUX_GAUGE_NOTE,
    G1_HOLD_SCAN51_KIND,
    IDENTITY_INHERITANCE_FORBIDDEN,
    MOST_IMPORTANT_NEXT_ARTIFACT,
    SCAN51_G_HOLDOUT_NOTE,
    SCIENTIFIC_RECOVERY_ORDER,
    THREE_C286_CASAGUIDE_TWO_CHI_DEG,
    THREE_C286_EVPA_DEG,
    THREE_C286_EVPA_IS_TWO_CHI_NOTE,
    THREE_C286_IAU_EVPA_DEG,
    VALIDATION_ORDER,
    apply_contract_for_casa_viscal,
    apply_contract_settings,
    calibration_field_policy,
    casa_table_is_global,
    classify_holdout,
    classify_parameter_rows,
    corrected_over_model_ratio_is_stable,
    df_variant_plan,
    flux_gauge_is_consistent,
    fullpol_antenna_support,
    jones_recovery_gate,
    leakage_systematic_floor,
    log_closure_amplitudes,
    product_manifest,
    reference_reference_scale_is_constant,
    scientific_calibration_policy,
    structure_audit_from_visibilities,
    term_coverage_from_rows,
    unwrapped_chi_span_rad,
)
from sl1mjax.holography_calibration_golden import (
    GOLDEN_CONVENTIONS,
    compare_casa_jax_visibilities,
    golden_convention_lock,
)
from sl1mjax.holography_calibration_oracle import (
    classify_bisection_residual,
    classify_chi_residual,
    classify_kcross_phase_fit,
    compare_correction_operators,
    correlation_correction_operator,
    effective_chi_from_p_operators,
    factor_kronecker_antenna_jones,
    fit_residual_phase_line,
    injected_basis_visibilities,
    pin_jones_gauge,
    score_df_convention_ladder,
    score_interpolation_methods,
    solution_for_stage,
    stratify_df_delta,
    xf_operator_effect,
)


def test_scan51_holdout_is_g_only() -> None:
    holdout = classify_holdout("G1_hold_scan51")
    assert holdout["kind"] == G1_HOLD_SCAN51_KIND
    assert holdout["tests"] == ("gain_interpolation",)
    assert "source_model" in holdout["does_not_test"]
    assert "end_to_end_calibration" in holdout["does_not_test"]
    assert "bandpass" in holdout["does_not_test"]
    assert "used in B0" in holdout["reason"]


def test_scan51_bandpass_ablation_is_the_source_holdout() -> None:
    holdout = classify_holdout("B2_scan2")
    assert holdout["kind"] == "bandpass_ablation"
    assert "source_model_on_scan51" in holdout["tests"]


def test_unknown_holdout_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown holography holdout"):
        classify_holdout("scan51_general")


def test_jones_recovery_stays_blocked_until_every_gate_passes() -> None:
    gate = jones_recovery_gate()
    assert gate["blocked"] is True
    assert gate["next_in_order"] == "diagonal_apply_back"
    assert gate["most_important_next_artifact"] == "cassbeam_diagonal_low_order_correction"
    assert gate["most_important_next_artifact"] == MOST_IMPORTANT_NEXT_ARTIFACT
    assert gate["pending"] == list(VALIDATION_ORDER)
    assert gate["scientific_recovery_order"][0] == "consistent_3c147_flux_gauge"
    done = jones_recovery_gate({step: True for step in VALIDATION_ORDER})
    assert done["blocked"] is False
    assert done["pending"] == []


def test_policy_records_g_holdout_and_identity_ban() -> None:
    notes = calibration_field_policy().notes
    assert SCAN51_G_HOLDOUT_NOTE in notes
    assert IDENTITY_INHERITANCE_FORBIDDEN in notes
    assert CONSISTENT_3C147_FLUX_GAUGE_NOTE in notes
    assert COMPATIBILITY_TABLES_NOT_SCIENTIFIC_NOTE in notes


def test_three_c286_casaguide_66_is_two_chi_not_iau_evpa() -> None:
    assert THREE_C286_CASAGUIDE_TWO_CHI_DEG == pytest.approx(66.0)
    assert THREE_C286_IAU_EVPA_DEG == pytest.approx(33.0)
    assert THREE_C286_EVPA_DEG == pytest.approx(66.0)
    q = 0.112 * np.cos(np.deg2rad(THREE_C286_EVPA_DEG))
    u = 0.112 * np.sin(np.deg2rad(THREE_C286_EVPA_DEG))
    two_chi = np.rad2deg(np.arctan2(u, q))
    evpa = 0.5 * two_chi
    assert two_chi == pytest.approx(66.0, abs=1e-6)
    assert evpa == pytest.approx(33.0, abs=1e-6)
    assert "not IAU EVPA" in THREE_C286_EVPA_IS_TWO_CHI_NOTE
    script = Path(__file__).parents[1] / "scripts" / "create_thol0001_scientific_calibration.py"
    text = script.read_text()
    assert "casaguide_two_chi_deg" in text
    assert "iau_evpa_deg" in text


def test_mixed_field0_field9_models_block_scientific_recovery() -> None:
    mixed = flux_gauge_is_consistent({0: 8.03, 9: 1.0})
    assert mixed["consistent"] is False
    assert mixed["scientific_recovery_blocked"] is True
    same = flux_gauge_is_consistent({0: 8.03, 9: 8.03})
    assert same["consistent"] is True
    policy = scientific_calibration_policy()
    assert policy["compatibility_tables_are_not_scientific"] is True
    assert policy["same_3c147_model_fields"] == [0, 9]
    assert policy["holoraster_used_for_moving_gains"] is False
    assert policy["holoraster_source"] == "field_10_model_data_per_row"
    assert policy["setjy_fluxd_is_integrated_flux_provenance_only"] is True
    assert SCIENTIFIC_RECOVERY_ORDER[0] == "consistent_3c147_flux_gauge"
    assert "field9_all_antenna_residual_jones" in SCIENTIFIC_RECOVERY_ORDER
    assert "field9_time_smooth_residual_jones" in SCIENTIFIC_RECOVERY_ORDER
    assert "scan53_56_chain_jump" in SCIENTIFIC_RECOVERY_ORDER
    assert "field9_estimator_identifiability" in SCIENTIFIC_RECOVERY_ORDER
    assert "field9_stabilized_residual_jones" in SCIENTIFIC_RECOVERY_ORDER
    assert "prediction_equivalent_beam_maps" in SCIENTIFIC_RECOVERY_ORDER
    assert "one_axis_visibility_holdouts" in SCIENTIFIC_RECOVERY_ORDER
    assert "forward_closure" in SCIENTIFIC_RECOVERY_ORDER
    assert "reference_visit_aligned_copolar_transfer" in SCIENTIFIC_RECOVERY_ORDER
    assert "loro_full_versus_diagonal" in SCIENTIFIC_RECOVERY_ORDER
    assert "loro_leakage_sensitivity" in SCIENTIFIC_RECOVERY_ORDER
    assert "highres_cassbeam_direct_visibility_validation" in SCIENTIFIC_RECOVERY_ORDER
    assert "spw4_multichannel_beam_prior" in SCIENTIFIC_RECOVERY_ORDER
    assert "c147_offset_ring_highres_cassbeam" in SCIENTIFIC_RECOVERY_ORDER
    assert "holoraster_cassbeam_comparison_report" in SCIENTIFIC_RECOVERY_ORDER
    assert "cassbeam_diagonal_cband_reference" in SCIENTIFIC_RECOVERY_ORDER
    assert "cassbeam_diagonal_low_order_correction" in SCIENTIFIC_RECOVERY_ORDER
    assert "interleaved_onaxis_beam_transfer" in SCIENTIFIC_RECOVERY_ORDER
    assert SCIENTIFIC_RECOVERY_ORDER.index("field9_estimator_identifiability") < (
        SCIENTIFIC_RECOVERY_ORDER.index("field9_stabilized_residual_jones")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("field9_stabilized_residual_jones") < (
        SCIENTIFIC_RECOVERY_ORDER.index("prediction_equivalent_beam_maps")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("prediction_equivalent_beam_maps") < (
        SCIENTIFIC_RECOVERY_ORDER.index("one_axis_visibility_holdouts")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("one_axis_visibility_holdouts") < (
        SCIENTIFIC_RECOVERY_ORDER.index("forward_closure")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("forward_closure") < (
        SCIENTIFIC_RECOVERY_ORDER.index("reference_visit_aligned_copolar_transfer")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("reference_visit_aligned_copolar_transfer") < (
        SCIENTIFIC_RECOVERY_ORDER.index("loro_full_versus_diagonal")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("loro_full_versus_diagonal") < (
        SCIENTIFIC_RECOVERY_ORDER.index("loro_leakage_sensitivity")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("loro_leakage_sensitivity") < (
        SCIENTIFIC_RECOVERY_ORDER.index("highres_cassbeam_direct_visibility_validation")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("highres_cassbeam_direct_visibility_validation") < (
        SCIENTIFIC_RECOVERY_ORDER.index("spw4_multichannel_beam_prior")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("spw4_multichannel_beam_prior") < (
        SCIENTIFIC_RECOVERY_ORDER.index("c147_offset_ring_highres_cassbeam")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("c147_offset_ring_highres_cassbeam") < (
        SCIENTIFIC_RECOVERY_ORDER.index("holoraster_cassbeam_comparison_report")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("holoraster_cassbeam_comparison_report") < (
        SCIENTIFIC_RECOVERY_ORDER.index("cassbeam_diagonal_cband_reference")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index("cassbeam_diagonal_cband_reference") < (
        SCIENTIFIC_RECOVERY_ORDER.index("cassbeam_diagonal_low_order_correction")
    )
    assert SCIENTIFIC_RECOVERY_ORDER.index(
        "field9_all_antenna_residual_jones"
    ) < SCIENTIFIC_RECOVERY_ORDER.index("independent_spw5_recovery")
    assert SCIENTIFIC_RECOVERY_ORDER.index(
        "field9_time_smooth_residual_jones"
    ) < SCIENTIFIC_RECOVERY_ORDER.index("independent_spw5_recovery")
    assert SCIENTIFIC_RECOVERY_ORDER.index(
        "scan53_56_chain_jump"
    ) < SCIENTIFIC_RECOVERY_ORDER.index("independent_spw5_recovery")


def test_corrected_over_model_ratio_ignores_resolved_amplitude() -> None:
    times = np.linspace(0.0, 1000.0, 20)
    model = 7.0 + 1.0 * np.linspace(0.0, 1.0, 20) + 0.2j * np.sin(times / 200.0)
    tracked = corrected_over_model_ratio_is_stable(model, model, times)
    assert tracked["passed"] is True
    assert tracked["median_abs"] == pytest.approx(1.0)
    drifted = model.copy()
    drifted[10:] *= 0.5
    failed = corrected_over_model_ratio_is_stable(drifted, model, times)
    assert failed["constant_amplitude"] is False
    assert failed["passed"] is False
    phased = model * np.exp(1j * 0.4 * (times / times[-1]))
    phase_fail = corrected_over_model_ratio_is_stable(phased, model, times)
    assert phase_fail["constant_phase"] is False
    assert phase_fail["passed"] is False
    wrapped = model * np.exp(1j * (0.01 + 2.0 * np.pi * (np.arange(20) > 10)))
    wrap_ok = corrected_over_model_ratio_is_stable(wrapped, model, times)
    assert wrap_ok["passed"] is True
    assert abs(wrap_ok["phase_drift_rad"]) < 0.05
    antenna1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    antenna2 = np.array([2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3])
    one_bad = model.copy()
    one_bad[antenna1 == 1] *= 0.4
    baseline = corrected_over_model_ratio_is_stable(
        one_bad, model, times, antenna1=antenna1, antenna2=antenna2
    )
    assert baseline["passed"] is False
    assert any(not item["passed"] for item in baseline["by_baseline"].values())


def test_reference_reference_flux_must_stay_on_one_scale() -> None:
    times = np.linspace(0.0, 1000.0, 20)
    stable = reference_reference_scale_is_constant(np.full(20, 8.03), times, model_jy=8.03)
    assert stable["passed"] is True
    drifted = reference_reference_scale_is_constant(
        np.concatenate([np.full(10, 8.03), np.full(10, 1.0)]),
        times,
        model_jy=8.03,
    )
    assert drifted["constant"] is False
    assert drifted["passed"] is False


def test_product_manifest_keeps_jones_blocked(tmp_path) -> None:
    table = tmp_path / "K0.cal"
    table.write_text("k\n")
    payload = product_manifest(
        name="diagonal",
        tables={"K0": table},
        model={"3C147": "Perley-Butler 2017"},
        parang=False,
        calwt=False,
        flag_version="test",
        notes=("G1_hold_scan51 is a G holdout only",),
    )
    assert payload["jones_recovery_blocked"] is True
    assert payload["most_important_next_artifact"] == MOST_IMPORTANT_NEXT_ARTIFACT
    assert payload["apply_contract"] == CASA_DF_JONES_APPLYCAL_V1
    settings = payload["apply_contract_settings"]
    assert settings["leakage_application"] == "casa_first_order"
    assert settings["casa_version"] == "6.7.6.14"
    assert settings["cparam_interpolation"] == "casa_float32_amp_phase"
    assert settings["gainfield_mapping"] == "all_fields_in_table"
    assert settings["compatibility_only"] is False
    assert apply_contract_settings(CASA_PARALLEL_PRESERVING_LEGACY_V1)["compatibility_only"] is True


def test_apply_contract_comes_from_viscal_or_explicit_policy() -> None:
    assert apply_contract_for_casa_viscal("Df Jones") == CASA_DF_JONES_APPLYCAL_V1
    assert (
        apply_contract_for_casa_viscal(
            "Df Jones", product_policy=CASA_PARALLEL_PRESERVING_LEGACY_V1
        )
        == CASA_PARALLEL_PRESERVING_LEGACY_V1
    )
    with pytest.raises(ValueError, match="no apply contract"):
        apply_contract_for_casa_viscal("G Jones")


def test_unflagged_identity_is_not_a_solution() -> None:
    flag = np.array([False, False], dtype=bool)
    assert classify_parameter_rows(flag, np.ones(2), identity=1.0) == "unflagged_identity"
    assert classify_parameter_rows(np.ones(2, dtype=bool), np.ones(2), identity=1.0) == "flagged"
    assert classify_parameter_rows(flag, np.array([1.2, 0.8]), identity=1.0) == "solved"
    assert classify_parameter_rows(np.zeros(0, dtype=bool), np.zeros(0), identity=1.0) == "absent"


def test_term_coverage_and_silent_identity_are_invalid() -> None:
    names = ("ea02", "ea04", "ea26")
    antenna = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    spw = np.array([4, 4, 4, 5, 5, 5], dtype=np.int32)
    xf = term_coverage_from_rows(
        term="Xf",
        antenna_names=names,
        antenna=antenna,
        spectral_window_id=spw,
        flag=np.array([False, True, False, False, True, False]),
        values=np.array([0.7 + 0.1j, 1.0, 1.0, 0.6, 1.0, 1.0]),
    )
    kcross = term_coverage_from_rows(
        term="Kcross",
        antenna_names=names,
        antenna=antenna,
        spectral_window_id=spw,
        flag=np.array([False, True, False, False, True, False]),
        values=np.array([3.0, 0.0, 0.0, 3.1, 0.0, 0.0]),
    )
    assert xf["ea02"]["4"] == "solved"
    assert xf["ea04"]["4"] == "flagged"
    assert xf["ea26"]["4"] == "unflagged_identity"
    support = fullpol_antenna_support(
        {"Kcross": kcross, "Xf": xf},
        three_c286_active=("ea02",),
        three_c286_absent=("ea04", "ea26"),
    )
    assert "ea04" in support["unsupported_full_pol"]
    assert "ea26" in support["invalid_identity_inheritance"]
    assert support["passed"] is False


def test_absent_3c286_antenna_with_solved_x_needs_justification() -> None:
    names = ("ea02", "ea28")
    antenna = np.array([0, 1], dtype=np.int32)
    spw = np.array([4, 4], dtype=np.int32)
    values = np.array([0.5 + 0.2j, 0.4 + 0.1j])
    xf = term_coverage_from_rows(
        term="Xf",
        antenna_names=names,
        antenna=antenna,
        spectral_window_id=spw,
        flag=np.zeros(2, dtype=bool),
        values=values,
    )
    kcross = term_coverage_from_rows(
        term="Kcross",
        antenna_names=names,
        antenna=antenna,
        spectral_window_id=spw,
        flag=np.zeros(2, dtype=bool),
        values=np.array([2.0, 1.5]),
    )
    support = fullpol_antenna_support(
        {"Kcross": kcross, "Xf": xf},
        three_c286_active=("ea02",),
        three_c286_absent=("ea28",),
    )
    assert support["by_antenna"]["ea28"]["support"] == "needs_justified_global_x"
    assert support["passed"] is False
    shared = fullpol_antenna_support(
        {"Kcross": kcross, "Xf": xf},
        three_c286_active=("ea02",),
        three_c286_absent=("ea28",),
        global_x=True,
    )
    assert shared["by_antenna"]["ea28"]["support"] == "full_pol_via_global_x"
    assert shared["passed"] is True


def test_identical_unflagged_rows_are_a_global_x() -> None:
    antenna = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    spw = np.array([4, 4, 4, 5, 5, 5], dtype=np.int32)
    values = np.array([3.44, 3.44, 3.44, 3.39, 3.39, 3.39])
    flag = np.zeros(6, dtype=bool)
    assert casa_table_is_global(antenna, flag, values, spectral_window_id=spw) is True
    mixed = values.copy()
    mixed[2] = 0.0
    assert casa_table_is_global(antenna, flag, mixed, spectral_window_id=spw) is False
    assert casa_table_is_global(antenna[:3], flag[:3], np.array([3.44, 3.44, 0.0])) is False


def test_chi_span_uses_time_unwrap_not_wrapped_minmax() -> None:
    times = np.array([0.0, 1.0, 2.0])
    chi = np.array([3.0, -3.0, -2.5])
    unwrapped = np.unwrap(chi)
    assert unwrapped_chi_span_rad(times, chi) == pytest.approx(
        float(np.max(unwrapped) - np.min(unwrapped))
    )
    assert unwrapped_chi_span_rad(times, chi) < np.pi
    assert (np.max(chi) - np.min(chi)) > np.pi


def test_df_variant_plan_excludes_offset_ring_and_requires_field_9() -> None:
    plan = df_variant_plan()
    assert plan["variants"]["Df"]["source_qu"] == (0.0, 0.0)
    assert plan["variants"]["Df+QU"]["poltype"] == "Df+QU"
    assert plan["c147_offset_used_for_d"] is False
    assert plan["holdout"]["field_id"] == 9
    assert plan["jones_recovery_blocked"] is True


def test_leakage_floor_is_the_d_model_difference() -> None:
    first = np.array([0.02 + 0.01j, 0.03], dtype=np.complex128)
    second = np.array([0.025 + 0.01j, 0.08], dtype=np.complex128)
    report = leakage_systematic_floor(first, second, np.array([True, True]))
    assert report["n"] == 2
    assert report["median_abs_delta"] == pytest.approx(0.0275, rel=0, abs=0.01)
    assert "leakage systematic floor" in report["notes"][0]


def test_structure_audit_rejects_flattening_and_accepts_a_point() -> None:
    n_ant = 5
    n_time = 3
    rows = []
    antenna1 = []
    antenna2 = []
    time_s = []
    uvw = []
    for time in range(n_time):
        for first in range(n_ant):
            for second in range(first + 1, n_ant):
                rows.append(1.0 + 0.0j)
                antenna1.append(first)
                antenna2.append(second)
                time_s.append(10.0 * time)
                uvw.append(((second - first) * 20.0, 0.0, 0.0))
    visibility = np.full((len(rows), 3, 4), 8.0 + 0.0j, dtype=np.complex128)
    report = structure_audit_from_visibilities(
        uvw_m=np.asarray(uvw, dtype=np.float64),
        frequency_hz=np.array([4.564e9, 4.628e9, 4.692e9]),
        visibility=visibility,
        flag=np.zeros(visibility.shape, dtype=bool),
        antenna1=np.asarray(antenna1, dtype=np.int32),
        antenna2=np.asarray(antenna2, dtype=np.int32),
        time_s=np.asarray(time_s, dtype=np.float64),
        model_i_jy=np.array([8.0, 8.0, 8.0]),
        holdout_kind="bandpass_ablation",
    )
    assert report["accepted_as_point"] is True
    assert report["channels"]["RR"][0]["flattened"] is False
    assert report["closure_amplitudes"]["point_like"] is True
    assert "flattening" in " ".join(report["notes"]).lower()


def test_structure_audit_detects_resolved_structure() -> None:
    antenna1 = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
    antenna2 = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
    uv = np.array([10.0, 80.0, 200.0, 70.0, 190.0, 120.0])
    amp = np.exp(-uv / 80.0)
    visibility = np.zeros((6, 1, 4), dtype=np.complex128)
    visibility[:, 0, :] = amp[:, None]
    report = structure_audit_from_visibilities(
        uvw_m=np.stack((uv, np.zeros(6), np.zeros(6)), axis=1),
        frequency_hz=np.array([4.564e9]),
        visibility=visibility,
        flag=np.zeros_like(visibility, dtype=bool),
        antenna1=antenna1,
        antenna2=antenna2,
        time_s=np.zeros(6),
        model_i_jy=np.array([1.0]),
        holdout_kind=G1_HOLD_SCAN51_KIND,
    )
    assert report["holdout_kind"] == G1_HOLD_SCAN51_KIND
    assert report["accepted_as_point"] is False


def test_golden_locks_parang_split_and_conventions() -> None:
    lock = golden_convention_lock(parang=True)
    assert lock["fullpol_parang"] is True
    assert lock["diagonal_parang"] is False
    assert lock["jones_recovery_blocked"] is True
    assert set(GOLDEN_CONVENTIONS) <= set(lock["conventions"])
    assert "parallactic_angle_treatment" in lock["conventions"]
    assert "no_double_application" in lock["conventions"]


def test_casa_jax_golden_comparison_passes_on_agreement() -> None:
    casa = np.ones((4, 2, 4), dtype=np.complex128)
    jax_vis = casa.copy()
    jax_vis[0, 0, 1] += 1.0e-10
    report = compare_casa_jax_visibilities(casa, jax_vis, np.ones(casa.shape, dtype=bool))
    assert report["passed"] is True
    assert report["per_correlation"]["RL"]["n"] == 8
    assert report["per_correlation"]["RL"]["relative_l2"] < 1.0e-8


def test_golden_breakdowns_broadcast_row_masks() -> None:
    casa = np.ones((6, 3, 4), dtype=np.complex128)
    report = compare_casa_jax_visibilities(
        casa,
        casa.copy(),
        np.ones(casa.shape, dtype=bool),
        antenna1=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        antenna2=np.array([1, 2, 0, 2, 0, 1], dtype=np.int32),
        time_s=np.arange(6, dtype=np.float64),
        frequency_hz=np.array([1.0e9, 2.0e9, 3.0e9]),
    )
    assert report["breakdowns"]["baseline_orientation"]["antenna1_lt_antenna2"]["RR"]["n"] > 0
    assert "0" in report["breakdowns"]["channel"]


def test_near_zero_casa_crosshand_does_not_rely_on_max_rel() -> None:
    casa = np.ones((8, 1, 4), dtype=np.complex128)
    casa[..., 1] = 1.0e-8
    casa[..., 2] = 1.0e-8
    jax_vis = casa.copy()
    jax_vis[..., 1] = 2.0e-8
    report = compare_casa_jax_visibilities(
        casa, jax_vis, np.ones(casa.shape, dtype=bool), crosshand_amp_floor=0.05
    )
    assert report["per_correlation"]["RL"]["max_rel"] > 0.5
    assert report["per_correlation"]["RL"]["relative_l2"] < 1.1
    assert report["per_correlation"]["RL"]["n_above_floor"] == 0
    assert report["per_correlation"]["RL"]["median_resid_over_rr_ll"] < 1.0e-7
    assert report["per_correlation"]["RL"]["max_rel_unsuitable"] is True
    assert report["per_correlation"]["RL"]["historical_max_rel_failure"] is True
    assert report["per_correlation"]["RL"]["passed"] is True
    assert report["passed"] is True


def test_golden_jax_apply_refuses_corrected_data() -> None:
    from sl1mjax.data.canonical import VisibilityBlock
    from sl1mjax.holography_calibration_golden import apply_imported_solution
    from sl1mjax.polarization import Correlation, ReceptorBasis

    block = VisibilityBlock(
        uvw_m=np.zeros((1, 3)),
        frequency_hz=np.array([4.564e9]),
        visibility=np.ones((1, 1, 4), dtype=np.complex128),
        weight=np.ones((1, 1, 4)),
        flag=np.zeros((1, 1, 4), dtype=bool),
        time_s=np.array([0.0]),
        antenna1=np.array([0], dtype=np.int32),
        antenna2=np.array([1], dtype=np.int32),
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        provenance={"column": "CORRECTED_DATA"},
    )
    with pytest.raises(ValueError, match="prevent double application"):
        apply_imported_solution(block, object())  # type: ignore[arg-type]


def test_casa_jax_golden_comparison_fails_on_hand_swap() -> None:
    casa = np.zeros((2, 1, 4), dtype=np.complex128)
    casa[..., 0] = 1.0
    casa[..., 3] = 2.0
    swapped = casa[..., [3, 1, 2, 0]]
    report = compare_casa_jax_visibilities(casa, swapped, np.ones(casa.shape, dtype=bool))
    assert report["passed"] is False
    assert report["per_correlation"]["RR"]["passed"] is False


def test_identity_jones_operator_is_identity() -> None:
    jones = np.eye(2, dtype=np.complex128)
    operator = correlation_correction_operator(jones, jones)
    assert operator == pytest.approx(np.eye(4), abs=1.0e-12)
    assert injected_basis_visibilities().shape == (4, 4)


def test_delay_reference_uses_cal_channel_not_midband() -> None:
    from sl1mjax.holography_calibration_golden import _delay_reference_frequency_hz

    ms = np.linspace(4.500e9, 4.626e9, 64)
    assert _delay_reference_frequency_hz(ms, 4.563e9) == 4.563e9
    assert _delay_reference_frequency_hz(ms, None) == float(ms[32])


def test_kronecker_factor_recovers_antenna_jones() -> None:
    jones_p = np.array([[1.1, 0.04 + 0.01j], [0.03, 0.9]], dtype=np.complex128)
    jones_q = np.array([[0.95, 0.02], [0.05 - 0.02j, 1.05]], dtype=np.complex128)
    operator = correlation_correction_operator(jones_p, jones_q, invert=False)
    factored = factor_kronecker_antenna_jones(operator)
    assert factored["relative_l2"] < 1.0e-10
    recovered_p, recovered_q = pin_jones_gauge(
        factored["jones_p"], factored["jones_q"], reference_is_p=True
    )
    expected_p = jones_p / jones_p[0, 0]
    expected_q = jones_q * np.conjugate(jones_p[0, 0])
    assert recovered_p == pytest.approx(expected_p, abs=1.0e-8)
    assert recovered_q == pytest.approx(expected_q, abs=1.0e-8)


def test_effective_chi_from_p_operators_recovers_two_chi() -> None:
    prefix = np.eye(4, dtype=np.complex128)
    two_chi = 0.4
    with_p = prefix.copy()
    with_p[1, 1] *= np.exp(1j * two_chi)
    with_p[2, 2] *= np.exp(-1j * two_chi)
    recovered = effective_chi_from_p_operators(with_p, prefix)
    assert recovered["two_chi_rad"] == pytest.approx(two_chi, abs=1.0e-12)
    assert abs(recovered["rl_lr_sum_rad"]) < 1.0e-12


def test_classify_chi_residual_names_hour_angle_and_antenna() -> None:
    constant = np.full(8, 0.41)
    assert classify_chi_residual(residual_deg=constant) == "constant_feed_or_receptor_angle"
    hour_angle = np.linspace(-1.0, 1.0, 8)
    sloping = 0.8 * hour_angle
    assert (
        classify_chi_residual(residual_deg=sloping, hour_angle_rad=hour_angle)
        == "hour_angle_apparent_or_time_standard"
    )
    antenna = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    antenna_offset = np.where(antenna == 0, 0.0, 0.3)
    assert (
        classify_chi_residual(residual_deg=antenna_offset, antenna=antenna)
        == "observatory_centre_versus_antenna_itrf"
    )


def test_apply_contract_records_casacore_antpos_direction() -> None:
    settings = apply_contract_settings(CASA_DF_JONES_APPLYCAL_V1)
    assert settings["antenna_position_direction"] == "casacore_itrf"
    assert settings["cparam_interpolation"] == "casa_float32_amp_phase"


def test_itrf_direction_is_unit_length_when_casacore_is_present() -> None:
    pytest.importorskip("casacore.measures")
    from sl1mjax.holography_calibration_measures import itrf_direction_casacore

    vla = np.array([[-1601185.4, -5041977.5, 3554875.6]])
    direction = itrf_direction_casacore(
        np.array([4959462157.5]),
        (1.4948845339, 0.8700817014),
        vla,
    )
    assert direction.shape == (1, 3)
    assert float(np.linalg.norm(direction[0])) == pytest.approx(1.0, abs=1e-12)


def test_interpolation_oracle_prefers_generating_method() -> None:
    cal_times = np.array([0.0, 10.0])
    cal_values = np.array([1.0 + 0.2j, 1.4 - 0.1j])
    query = np.array([4.0])
    from sl1mjax.calibration import interpolate_complex_series

    casa = interpolate_complex_series(query, cal_times, cal_values, method="linear_complex")
    ranked = score_interpolation_methods(query, cal_times, cal_values, casa)
    assert ranked[0]["method"] == "linear_complex"
    assert ranked[0]["relative_l2"] < 1.0e-12


def test_xf_operator_effect_detects_identity_and_phase() -> None:
    prefix = np.eye(4, dtype=np.complex128)
    assert xf_operator_effect(prefix, prefix)["identity"] is True
    rotated = prefix.copy()
    rotated[1, 1] *= np.exp(1j * 0.2)
    report = xf_operator_effect(rotated, prefix)
    assert report["identity"] is False
    assert report["rl_phase_deg"] == pytest.approx(np.rad2deg(0.2), abs=1.0e-10)


def test_kcross_phase_line_distinguishes_delay_from_constant() -> None:
    frequency = np.linspace(4.56e9, 4.69e9, 32)
    delay = fit_residual_phase_line(frequency, 0.1 + 2.0 * frequency / 1.0e9)
    assert classify_kcross_phase_fit(delay) == "delay_sign_units_reference_frequency_or_offset"
    constant = fit_residual_phase_line(frequency, np.full(frequency.shape, 0.03))
    assert classify_kcross_phase_fit(constant) == "constant_crosshand_gauge_or_receptor_placement"


def test_df_ladder_recovers_generating_convention() -> None:
    d_p = np.array([0.05 + 0.01j, -0.04], dtype=np.complex128)
    d_q = np.array([0.03, 0.06 - 0.02j], dtype=np.complex128)
    from sl1mjax.holography_calibration_oracle import leakage_jones_for_convention

    jones_p = leakage_jones_for_convention(d_p, invert=False)
    jones_q = leakage_jones_for_convention(d_q, invert=False)
    casa = correlation_correction_operator(jones_p, jones_q, invert=True)
    ranked = score_df_convention_ladder(casa, d_p, d_q)
    winner = ranked[0]
    assert winner["relative_l2"] < 1.0e-8
    assert winner["invert"] is True
    assert winner["q_hermitian"] is True
    assert winner["cparam_swapped"] is False
    assert winner["conjugate"] is False
    assert winner["d_before_prefix"] is False


def test_operator_detects_conjugation_swap() -> None:
    jones_p = np.array([[1.0, 0.1], [0.0, 1.0]], dtype=np.complex128)
    jones_q = np.array([[1.0, 0.0], [0.2, 1.0]], dtype=np.complex128)
    casa = correlation_correction_operator(jones_p, jones_q)
    swapped = correlation_correction_operator(jones_q, jones_p)
    report = compare_correction_operators(casa, swapped)
    assert report["passed"] is False
    assert report["relative_l2"] > 1.0e-3


def test_bisection_stage_strips_later_terms() -> None:
    from sl1mjax.calibration import identity_solution
    from sl1mjax.polarization import Correlation

    base = identity_solution(
        antenna_count=3,
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        frequency_hz=np.array([4.564e9, 4.692e9]),
        time_s=np.array([0.0, 10.0]),
    )
    delayed = replace_delays(base, np.array([[3.4e-9, 0.0], [3.4e-9, 0.0], [3.4e-9, 0.0]]))
    from dataclasses import replace as _replace

    delayed = _replace(delayed, antenna_position_m=np.zeros((3, 3)))
    only_k = solution_for_stage(delayed, "K")
    assert np.allclose(only_k.delays_s, delayed.delays_s)
    assert np.allclose(only_k.gains, 1)
    assert only_k.cross_hand_delay_s is None
    assert only_k.apply_parallactic_angle is False
    with_p = solution_for_stage(delayed, "K+B+G+Kcross+Df+Xf+P")
    assert with_p.apply_parallactic_angle is True
    leakage = np.zeros((2, 3, 2, 2), dtype=np.complex128)
    delayed = _replace(
        delayed,
        leakage=leakage,
        leakage_frequency_hz=np.array([4.564e9, 4.692e9]),
        leakage_valid=np.ones(leakage.shape, dtype=bool),
        leakage_time_s=np.array([0.0, 10.0]),
    )
    stripped = solution_for_stage(delayed, "K+B+G+Kcross")
    assert stripped.leakage is None
    assert stripped.leakage_time_s is None
    kept = solution_for_stage(delayed, "K+B+G+Kcross+Df")
    assert kept.leakage_time_s is not None
    np.testing.assert_array_equal(kept.leakage_time_s, delayed.leakage_time_s)


def replace_delays(solution, delays):
    from dataclasses import replace

    return replace(solution, delays_s=np.asarray(delays, dtype=np.float64))


def test_bisection_classifier_names_kcross_and_xf() -> None:
    assert (
        classify_bisection_residual(
            relative_l2_by_corr={"RR": 0.0, "LL": 0.0, "RL": 0.4, "LR": 0.4},
            phase_vs_frequency_slope_rad=1.2,
        )
        == "kcross_sign_units_ref_frequency_or_antenna_side"
    )
    assert (
        classify_bisection_residual(
            relative_l2_by_corr={"RR": 0.0, "LL": 0.0, "RL": 0.4, "LR": 0.4},
            mean_rl_phase_deg=80.0,
            mean_lr_phase_deg=-78.0,
            phase_vs_frequency_slope_rad=0.01,
        )
        == "xf_sign_receptor_order_or_conjugation"
    )
    assert (
        classify_bisection_residual(
            relative_l2_by_corr={"RR": 0.002, "LL": 0.002, "RL": 0.002, "LR": 0.002}
        )
        == "diagonal_phase_floor"
    )


def test_df_tail_is_not_adopted_when_concentrated() -> None:
    first = np.zeros((4, 8, 2), dtype=np.complex128)
    second = first.copy()
    second[0, :2] = 0.3
    valid = np.ones(first.shape, dtype=bool)
    report = stratify_df_delta(
        first,
        second,
        valid,
        antenna=np.array([0, 1, 2, 3], dtype=np.int32),
        spectral_window_id=np.array([4, 4, 5, 5], dtype=np.int32),
        antenna_names=("ea02", "ea04", "ea26", "ea03"),
        reference_antennas=("ea02", "ea26"),
        moving_antennas=("ea04",),
    )
    assert report["adopt_as_floor"] is False
    assert report["concentrated_in_failed_subset"] is True
    assert report["by_antenna"][0]["n_tail"] > 0
    assert "p90_abs_delta" in report["gauge_aligned"]


def test_log_closures_drop_low_amplitude_quads() -> None:
    closures = np.array([1.0, 1.02, 0.98, 1.01, 3.0, 0.2, 1.0, 0.99, 1.03])
    amps = np.array([2.0, 2.0, 2.0, 2.0, 0.05, 0.04, 2.0, 2.0, 2.0])
    report = log_closure_amplitudes(closures, min_amp=amps, amp_cut=0.2)
    assert report["n_excluded"] == 2
    assert report["log10_mad"] < 0.05


def test_recovery_script_uses_model_data_not_a_scalar() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_thol0001_diagonal_recovery.py"
    text = script.read_text()
    assert "MODEL_DATA" in text
    assert "source_coherency_visibility" in text
    assert "circular_visibility_to_source_coherency" in text
    assert "three_c147_casa_setjy_point_source_model" not in text
    assert "CASA_SETJY_3C147_C_IM_MODEL_DATA_JY_SPW4_CH32" not in text
    assert "stokes_i_is_not_s_pq" in text
    gauge = Path(__file__).parents[1] / "scripts" / "report_thol0001_flux_gauge.py"
    gauge_text = gauge.read_text()
    assert "corrected_over_model_ratio_is_stable" in gauge_text
    assert "reference_reference_scale_is_constant" not in gauge_text


def test_scientific_calibration_script_does_not_overwrite_field9_with_one_jy() -> None:
    script = Path(__file__).parents[1] / "scripts" / "create_thol0001_scientific_calibration.py"
    text = script.read_text()
    assert "fluxdensity=[1.0, 0.0, 0.0, 0.0]" not in text
    assert "3C147_C.im" in text
    assert "HOLORASTER_FIELD" in text
    assert "holoraster_used_for_moving_gains" in text
    assert "products/scientific" in text
    assert "field_9_manual_i1_overwrite" in text
    compatibility = (
        Path(__file__).parents[1] / "scripts" / "create_thol0001_calibration_products.py"
    )
    assert "fluxdensity=[1.0, 0.0, 0.0, 0.0]" in compatibility.read_text()
