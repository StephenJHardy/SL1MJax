"""THOL0001 lower-C calibration policy, source-model and 3C286 graph audits.

Direction-independent D is solved from on-axis 3C147 only. The C147-*
offset ring is a beam-leakage validation set, not a D-solver field.
Pass-2 raster cells stay on measured coordinates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sl1mjax.holography import (
    MODEL_DATA_SOURCE_COHERENCY_NOTE,
    THREE_C147_CASA_SETJY_VS_TABLE5_NOTE,
    perley_butler_2017_3c147_stokes_i_jy,
    three_c147_flux_scale_report,
    three_c147_point_source_model,
)
from sl1mjax.holography_commissioning import D_SCAN_PREFIX, EVPA_FIELD_NAMES, ON_AXIS_3C147_NAMES
from sl1mjax.holography_ms import (
    _read_antennas,
    _read_field_names,
    _read_field_phase_centre,
    _tables,
)
from sl1mjax.holography_pointing_maps import PASS2_MEASURED_RASTER_NOTE
from sl1mjax.holography_reference_jones import (
    ALL_ANTENNA_RESIDUAL_JONES_NOTE,
    CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE,
    COMPLETE_FIELD9_TRACK_NOTE,
    CONNECTED_HOLDOUT_NOTE,
    CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE,
    ESTIMATOR_IDENTIFIABILITY_NOTE,
    INTERLEAVED_BEAM_TRANSFER_GATE_NOTE,
    INTERLEAVED_ONAXIS_TRANSFER_NOTE,
    JJH_PROXY_IS_NOT_DECISIVE_NOTE,
    NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
    ONE_AXIS_VISIBILITY_HOLDOUT_NOTE,
    PER_SCAN_EPSILON_NOT_OBSERVABLE_NOTE,
    PREDICTION_EQUIVALENT_BEAM_NOTE,
    RESIDUAL_JONES_VISIBILITY_GATES_NOTE,
    SMOOTH_GP_REJECTED_NOTE,
    THREE_C147_QU_IS_NUISANCE_NOTE,
    THREE_C286_CIRCULAR_FLOOR_NOTE,
    ZERO_QU_IS_ABLATION_NOTE,
)
from sl1mjax.rime import SPEED_OF_LIGHT_M_S

HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_CASA_EXECUTABLE = Path("/home/stephen/casa/casa-6.7.6-14-py3.12.el9/bin/casa")
FLUX_BANDPASS_SCANS = (2, 51)
ON_AXIS_3C147_FIELD_IDS = (0, 9)
D_CODE_FIELD_ID = 9
C147_OFFSET_FIELD_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
REFERENCE_ANTENNA = "ea02"
HELD_OUT_REFERENCE_ANTENNA = "ea26"
CALWT = False
DIAGONAL_PARANG = False
FULLPOL_PARANG = True
THREE_C286_FRACTIONAL_POLARISATION = 0.112
THREE_C286_CASAGUIDE_TWO_CHI_DEG = 66.0
THREE_C286_IAU_EVPA_DEG = 0.5 * THREE_C286_CASAGUIDE_TWO_CHI_DEG
THREE_C286_EVPA_DEG = THREE_C286_CASAGUIDE_TWO_CHI_DEG
THREE_C286_EVPA_IS_TWO_CHI_NOTE = (
    "cos(66°) and sin(66°) set Q and U so arg(Q+iU)=66°. That angle is "
    "the casaguide 2χ / cross-hand phase, not IAU EVPA. IAU χ is "
    "(1/2) arg(Q+iU) ≈ 33°. THREE_C286_EVPA_DEG remains 66.0 so existing "
    "Q/U numbers are unchanged. Record both angles in provenance."
)
NOMINAL_WEIGHT = 4_000_000.0
G1_HOLD_SCAN51_KIND = "g_holdout"
SCAN51_G_HOLDOUT_NOTE = (
    "G1_hold_scan51 is a G holdout only: scan 51 is excluded from G but was "
    "used in B0. It is not an independent end-to-end calibration or "
    "source-model holdout."
)
SCAN51_B_ABLATION_NOTE = (
    "A genuine scan-51 source-model or end-to-end test requires a bandpass "
    "solved from scan 2 only (B2_scan2) applied to scan 51. Alternatively "
    "split scan 51 by baselines or channels."
)
IDENTITY_INHERITANCE_FORBIDDEN = (
    "Antennas absent from usable 3C286 data must receive a mathematically "
    "justified global/reference-frame X solution or be marked unsupported "
    "for full-polarisation holography. They must not silently inherit "
    "identity Kcross/Xf corrections."
)
JONES_RECOVERY_BLOCKED_NOTE = (
    "The on-axis CASA-compatible full-polarisation operator is largely "
    "working. Jones application conventions reproduce CASA. Kcross and X "
    "are understood. Diagonal holography is credible. 3C286 gives "
    "approximately the expected EVPA. Full-Jones recovery does not corrupt "
    "RR/LL. The scientifically usable direction-dependent full-Jones beam "
    "remains blocked: the off-diagonal signal does not survive the current "
    "per-reference identity-gauge estimator. That failure applies to the "
    "identity-reference estimator, not to full-Jones holography in "
    "principle. The 2.7% median |RL|/I floor blocks the identity-gauge "
    "product. The next estimator solves residual on-axis R_p for every "
    "antenna from field 9, with 3C147 Q/U as a nuisance, and is decided "
    "by its own held-out visibility gates."
)
FIRST_BEAM_RECOVERY_UNFROZEN_NOTE = (
    "first_beam_recovery is an unfrozen diagnostic. Do not freeze a beam "
    "from it until the source and calibration gates finish."
)
COMPATIBILITY_TABLES_NOT_SCIENTIFIC_NOTE = (
    "products/diagonal and products/fullpol are compatibility fixtures. "
    "They mix an 8 Jy field-0 model with a 1 Jy field-9 model inside G. "
    "They must not produce the scientific holography beam."
)
CONSISTENT_3C147_FLUX_GAUGE_NOTE = (
    "Fields 0 and 9 must carry the same 3C147 intrinsic model. Field 10 "
    "may receive that model for prediction only. Do not solve G from "
    "moving HOLORASTER baselines. Downstream Kcross/D/X must be re-solved "
    "from the replacement G table."
)
CORRECTED_OVER_MODEL_RATIO_NOTE = (
    "The residual-calibration gate is the per-row complex ratio "
    "CORRECTED_DATA / MODEL_DATA. Amplitude versus a scalar median MODEL "
    "confuses UV structure with gain drift. Restrict HOLORASTER to "
    "reference-reference baselines; moving-reference rows contain the beam."
)
VALIDATION_ORDER = (
    "diagonal_apply_back",
    "post_b_source_test",
    "polarisation_calibration",
    "three_c286_apply_back",
    "casa_jax_calibration_golden",
    "first_beam_recovery",
    "full_jones",
)
SCIENTIFIC_RECOVERY_ORDER = (
    "consistent_3c147_flux_gauge",
    "scientific_calibration_products",
    "reference_reference_flux_scale",
    "scientific_casa_jax_golden",
    "holoraster_apply_from_data",
    "absolute_and_relative_spw4_recovery",
    "reference_and_repeatability_holdouts",
    "crosshand_floor_coherence",
    "field9_all_antenna_residual_jones",
    "residual_jones_p_convention",
    "field9_channel_support",
    "field9_scan_state",
    "on_axis_beam_identity",
    "field9_time_smooth_residual_jones",
    "scan53_56_chain_jump",
    "field9_estimator_identifiability",
    "field9_stabilized_residual_jones",
    "prediction_equivalent_beam_maps",
    "one_axis_visibility_holdouts",
    "forward_closure",
    "reference_visit_aligned_copolar_transfer",
    "loro_full_versus_diagonal",
    "loro_leakage_sensitivity",
    "highres_cassbeam_direct_visibility_validation",
    "spw4_multichannel_beam_prior",
    "c147_offset_ring_highres_cassbeam",
    "holoraster_cassbeam_comparison_report",
    "cassbeam_diagonal_cband_reference",
    "cassbeam_diagonal_low_order_correction",
    "field9_acquisition_state",
    "hierarchical_scan_state_residual",
    "interleaved_onaxis_beam_transfer",
    "three_c147_qu_nuisance",
    "connected_residual_holdouts",
    "identity_prior_antenna_holdout",
    "residual_jones_holography_recovery",
    "training_only_offdiag_mask",
    "clustered_moving_reference_comparison",
    "independent_spw5_recovery",
    "spatial_or_array_average_beam",
)
MOST_IMPORTANT_NEXT_ARTIFACT = "cassbeam_diagonal_low_order_correction"
SCIENTIFIC_CALIBRATION_PRODUCT = "thol0001_lower_c_scientific"
COMPATIBILITY_CALIBRATION_PRODUCT = "thol0001_lower_c"
CASA_DF_JONES_APPLYCAL_V1 = "casa_df_jones_applycal_v1"
CASA_PARALLEL_PRESERVING_LEGACY_V1 = "casa_parallel_preserving_legacy_v1"
CASA_APPLYCAL_VERSION = "6.7.6.14"
CASA_CAL_TABLE_SCHEMA = "NewCalTable"
APPLY_CONTRACTS = {
    CASA_DF_JONES_APPLYCAL_V1: {
        "casa_version": CASA_APPLYCAL_VERSION,
        "viscal": "Df Jones",
        "requested_interpolation": {
            "antpos": "",
            "G": "linear",
            "K": "",
            "B": "nearest",
            "Kcross": "",
            "Df": "",
            "Xf": "",
        },
        "gainfield": "",
        "gainfield_mapping": "all_fields_in_table",
        "scan_bounding": False,
        "perscan": False,
        "interpolation": "linear",
        "cparam_interpolation": "casa_float32_amp_phase",
        "leakage_application": "casa_first_order",
        "parallactic_model": "casacore_hadec_geocentric",
        "jones_invert": "first_order_negate_offdiag",
        "antenna_position_direction": "casacore_itrf",
        "cal_table_schema": CASA_CAL_TABLE_SCHEMA,
        "compatibility_only": False,
        "notes": (
            "Selected from VisCal='Df Jones' and CASA release/6.7.6 source. "
            "G and D share CTTimeInterp1: float32 amplitude and sequentially "
            "unwrapped phase. JonesGenLin::invert only negates D off-diagonals. "
            "Antenna-position phase uses the casacore ITRF field direction. "
            "Empty gainfield keeps every table field; perscan was not requested. "
            "Not a source-name branch."
        ),
    },
    CASA_PARALLEL_PRESERVING_LEGACY_V1: {
        "casa_version": CASA_APPLYCAL_VERSION,
        "viscal": "explicit_product_policy",
        "requested_interpolation": {"G": "nearest", "D": "nearest"},
        "gainfield": "",
        "interpolation": "nearest",
        "cparam_interpolation": "nearest",
        "leakage_application": "casa_parallel_preserving",
        "parallactic_model": "gmst_geodetic",
        "jones_invert": "hybrid_parallel_preserving",
        "cal_table_schema": CASA_CAL_TABLE_SCHEMA,
        "compatibility_only": True,
        "notes": (
            "Compatibility-only 3C391 golden product policy. Hybrid D apply "
            "and GMST geodetic P. That parallactic model does not represent "
            "CASA accurately enough for new scientific polarisation work and "
            "must not be promoted as an alternative convention. Regenerate "
            "3C391 under the corrected contract rather than extending this one."
        ),
    },
}


def apply_contract_for_casa_viscal(
    viscal: str,
    *,
    product_policy: str | None = None,
) -> str:
    """Choose a versioned apply contract from table VisCal or an explicit policy."""

    if product_policy is not None:
        if product_policy not in APPLY_CONTRACTS:
            raise ValueError(f"unknown apply contract {product_policy!r}")
        return product_policy
    if viscal == "Df Jones":
        return CASA_DF_JONES_APPLYCAL_V1
    raise ValueError(f"no apply contract for VisCal={viscal!r}")


def apply_contract_settings(contract: str) -> dict[str, object]:
    if contract not in APPLY_CONTRACTS:
        raise ValueError(f"unknown apply contract {contract!r}")
    return dict(APPLY_CONTRACTS[contract])


COVERAGE_TERMS = ("K", "B", "G", "Kcross", "Df", "Xf")
DF_VARIANTS = ("Df", "Df+QU", "D_fixed_pol")
CAL_TABLE_IDENTITY = {
    "K": 0.0,
    "B": 1.0 + 0.0j,
    "G": 1.0 + 0.0j,
    "Kcross": 0.0,
    "Df": 0.0 + 0.0j,
    "Xf": 1.0 + 0.0j,
}


@dataclass(frozen=True)
class CalibrationFieldPolicy:
    """Which fields may enter each calibration term."""

    flux_bandpass_field_ids: tuple[int, ...]
    leakage_field_ids: tuple[int, ...]
    evpa_field_ids: tuple[int, ...]
    offset_validation_field_ids: tuple[int, ...]
    holoraster_gain_baselines: str
    notes: tuple[str, ...]


def calibration_field_policy() -> CalibrationFieldPolicy:
    return CalibrationFieldPolicy(
        flux_bandpass_field_ids=(0,),
        leakage_field_ids=ON_AXIS_3C147_FIELD_IDS,
        evpa_field_ids=(11,),
        offset_validation_field_ids=C147_OFFSET_FIELD_IDS,
        holoraster_gain_baselines="reference_reference",
        notes=(
            "C147-* offset ring is validation only; it must not enter a DI D solve",
            "field 9 is on-axis J0542+4951 with FIELD.CODE=D",
            "HOLORASTER moving baselines must not redefine the source or moving gains",
            CONSISTENT_3C147_FLUX_GAUGE_NOTE,
            COMPATIBILITY_TABLES_NOT_SCIENTIFIC_NOTE,
            THREE_C147_CASA_SETJY_VS_TABLE5_NOTE,
            PASS2_MEASURED_RASTER_NOTE,
            SCAN51_G_HOLDOUT_NOTE,
            SCAN51_B_ABLATION_NOTE,
            IDENTITY_INHERITANCE_FORBIDDEN,
            JONES_RECOVERY_BLOCKED_NOTE,
        ),
    )


def flux_gauge_is_consistent(
    field_model_jy: dict[int, float],
    *,
    rtol: float = 0.02,
) -> dict[str, object]:
    """Fields 0 and 9 must sit on one 3C147 flux model."""

    missing = [field for field in ON_AXIS_3C147_FIELD_IDS if field not in field_model_jy]
    if missing:
        raise ValueError(f"flux gauge needs model amplitudes for fields {missing}")
    field0 = float(field_model_jy[0])
    field9 = float(field_model_jy[9])
    if field0 <= 0.0 or field9 <= 0.0:
        raise ValueError("3C147 model amplitudes must be positive")
    ratio = field9 / field0
    consistent = abs(ratio - 1.0) <= float(rtol)
    return {
        "field0_model_jy": field0,
        "field9_model_jy": field9,
        "field9_over_field0": ratio,
        "rtol": float(rtol),
        "consistent": consistent,
        "scientific_recovery_blocked": not consistent,
        "notes": (
            CONSISTENT_3C147_FLUX_GAUGE_NOTE,
            COMPATIBILITY_TABLES_NOT_SCIENTIFIC_NOTE,
        ),
    }


def reference_reference_scale_is_constant(
    amplitudes_jy: np.ndarray,
    times_s: np.ndarray,
    *,
    model_jy: float,
    max_relative_drift: float = 0.05,
    max_model_offset: float = 0.08,
) -> dict[str, object]:
    """Corrected reference--reference amplitudes must stay on one flux scale."""

    values = np.asarray(amplitudes_jy, dtype=np.float64).reshape(-1)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if values.shape != times.shape:
        raise ValueError("amplitudes_jy must match times_s")
    usable = np.isfinite(values) & np.isfinite(times) & (values > 0.0)
    if int(np.sum(usable)) < 4:
        raise ValueError("flux-scale series needs at least four unflagged samples")
    values = values[usable]
    times = times[usable]
    order = np.argsort(times)
    values = values[order]
    times = times[order]
    median = float(np.median(values))
    early = float(np.median(values[: max(values.size // 4, 1)]))
    late = float(np.median(values[max(values.size - values.size // 4, 1) :]))
    drift = abs(late - early) / median if median else float("nan")
    model_offset = abs(median - float(model_jy)) / float(model_jy) if model_jy else float("nan")
    return {
        "n": int(values.size),
        "median_jy": median,
        "early_median_jy": early,
        "late_median_jy": late,
        "relative_time_drift": drift,
        "relative_model_offset": model_offset,
        "model_jy": float(model_jy),
        "constant": bool(drift <= float(max_relative_drift)),
        "on_model": bool(model_offset <= float(max_model_offset)),
        "passed": bool(
            drift <= float(max_relative_drift) and model_offset <= float(max_model_offset)
        ),
    }


def corrected_over_model_ratio_is_stable(
    corrected: np.ndarray,
    model: np.ndarray,
    times_s: np.ndarray,
    *,
    antenna1: np.ndarray | None = None,
    antenna2: np.ndarray | None = None,
    model_amp_floor: float = 0.1,
    max_amp_offset: float = 0.08,
    max_amp_drift: float = 0.05,
    max_phase_offset_rad: float = 0.15,
    max_phase_drift_rad: float = 0.15,
) -> dict[str, object]:
    """Test residual calibration with ``CORRECTED_DATA / MODEL_DATA``.

    A resolved model can make ``|CORRECTED|`` vary with UV even when the
    gains are stable. The complex ratio removes that source structure.
    """

    measured = np.asarray(corrected, dtype=np.complex128).reshape(-1)
    predicted = np.asarray(model, dtype=np.complex128).reshape(-1)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if measured.shape != predicted.shape or measured.shape != times.shape:
        raise ValueError("corrected, model, and times_s must share one shape")
    usable = (
        np.isfinite(measured)
        & np.isfinite(predicted)
        & np.isfinite(times)
        & (np.abs(predicted) >= float(model_amp_floor))
    )
    if int(np.sum(usable)) < 4:
        raise ValueError("ratio series needs at least four unflagged samples")
    first = None if antenna1 is None else np.asarray(antenna1).reshape(-1)
    second = None if antenna2 is None else np.asarray(antenna2).reshape(-1)
    if first is not None or second is not None:
        if first is None or second is None or first.shape != measured.shape:
            raise ValueError("antenna1 and antenna2 must match the visibility series")
        first = first[usable]
        second = second[usable]
    report = _ratio_series_stats(
        measured[usable],
        predicted[usable],
        times[usable],
        max_amp_offset=max_amp_offset,
        max_amp_drift=max_amp_drift,
        max_phase_offset_rad=max_phase_offset_rad,
        max_phase_drift_rad=max_phase_drift_rad,
    )
    by_baseline: dict[str, dict[str, object]] = {}
    if first is not None and second is not None:
        pairs = np.stack((np.minimum(first, second), np.maximum(first, second)), axis=1)
        for left, right in np.unique(pairs, axis=0):
            members = (pairs[:, 0] == left) & (pairs[:, 1] == right)
            if int(np.sum(members)) < 4:
                continue
            by_baseline[f"{int(left)}-{int(right)}"] = _ratio_series_stats(
                measured[usable][members],
                predicted[usable][members],
                times[usable][members],
                max_amp_offset=max_amp_offset,
                max_amp_drift=max_amp_drift,
                max_phase_offset_rad=max_phase_offset_rad,
                max_phase_drift_rad=max_phase_drift_rad,
            )
        report["by_baseline"] = by_baseline
        report["passed"] = bool(report["passed"]) and all(
            item["passed"] for item in by_baseline.values()
        )
    report["notes"] = (CORRECTED_OVER_MODEL_RATIO_NOTE,)
    return report


def _ratio_series_stats(
    corrected: np.ndarray,
    model: np.ndarray,
    times_s: np.ndarray,
    *,
    max_amp_offset: float,
    max_amp_drift: float,
    max_phase_offset_rad: float,
    max_phase_drift_rad: float,
) -> dict[str, object]:
    order = np.argsort(times_s)
    ratio = corrected[order] / model[order]
    amplitude = np.abs(ratio)
    median_amp = float(np.median(amplitude))
    split = max(amplitude.size // 4, 1)
    early = ratio[:split]
    late = ratio[max(ratio.size - split, 1) :]
    amp_drift = (
        abs(float(np.median(np.abs(late))) - float(np.median(np.abs(early)))) / median_amp
        if median_amp
        else float("nan")
    )
    phase_offset = float(np.angle(_complex_median(ratio)))
    phase_drift = float(np.angle(_complex_median(late) * np.conjugate(_complex_median(early))))
    amp_offset = abs(median_amp - 1.0)
    passed = bool(
        amp_offset <= float(max_amp_offset)
        and amp_drift <= float(max_amp_drift)
        and abs(phase_offset) <= float(max_phase_offset_rad)
        and abs(phase_drift) <= float(max_phase_drift_rad)
    )
    return {
        "n": int(ratio.size),
        "median_abs": median_amp,
        "early_median_abs": float(np.median(np.abs(early))),
        "late_median_abs": float(np.median(np.abs(late))),
        "relative_amp_drift": amp_drift,
        "amp_offset_from_one": amp_offset,
        "median_phase_rad": phase_offset,
        "early_median_phase_rad": float(np.angle(_complex_median(early))),
        "late_median_phase_rad": float(np.angle(_complex_median(late))),
        "phase_drift_rad": phase_drift,
        "phase_rms_rad": float(np.sqrt(np.mean(np.angle(ratio) ** 2))),
        "constant_amplitude": bool(amp_drift <= float(max_amp_drift)),
        "on_model": bool(amp_offset <= float(max_amp_offset)),
        "constant_phase": bool(abs(phase_drift) <= float(max_phase_drift_rad)),
        "phase_on_model": bool(abs(phase_offset) <= float(max_phase_offset_rad)),
        "passed": passed,
    }


def _complex_median(values: np.ndarray) -> complex:
    return complex(np.median(np.real(values)), np.median(np.imag(values)))


def scientific_calibration_policy() -> dict[str, object]:
    """Replacement calibration that puts fields 0 and 9 on one 3C147 model."""

    return {
        "schema_version": HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION,
        "product": SCIENTIFIC_CALIBRATION_PRODUCT,
        "compatibility_product": COMPATIBILITY_CALIBRATION_PRODUCT,
        "compatibility_tables_are_not_scientific": True,
        "same_3c147_model_fields": list(ON_AXIS_3C147_FIELD_IDS),
        "prediction_only_fields": (10,),
        "holoraster_used_for_moving_gains": False,
        "holoraster_source": "field_10_model_data_per_row",
        "setjy_fluxd_is_integrated_flux_provenance_only": True,
        "casa_setjy": {
            "standard": "Perley-Butler 2017",
            "model": "3C147_C.im",
            "scalebychan": True,
            "usescratch": True,
        },
        "flux_scale": three_c147_flux_scale_report(4.564e9),
        "scientific_recovery_order": list(SCIENTIFIC_RECOVERY_ORDER),
        "most_important_next_artifact": MOST_IMPORTANT_NEXT_ARTIFACT,
        "relative_and_absolute_products_are_distinct": True,
        "notes": (
            CONSISTENT_3C147_FLUX_GAUGE_NOTE,
            CORRECTED_OVER_MODEL_RATIO_NOTE,
            MODEL_DATA_SOURCE_COHERENCY_NOTE,
            COMPATIBILITY_TABLES_NOT_SCIENTIFIC_NOTE,
            THREE_C147_CASA_SETJY_VS_TABLE5_NOTE,
            FIRST_BEAM_RECOVERY_UNFROZEN_NOTE,
        ),
    }


def field_may_enter_d_solve(field_id: int, field_name: str) -> bool:
    if field_name.startswith(D_SCAN_PREFIX):
        return False
    return int(field_id) in ON_AXIS_3C147_FIELD_IDS or field_name in ON_AXIS_3C147_NAMES


def read_field_codes(path: Path) -> tuple[tuple[int, str, str], ...]:
    tables = _tables()
    names = _read_field_names(tables, Path(path))
    with tables.table(str(Path(path) / "FIELD"), readonly=True, ack=False) as table:
        codes = [
            str(table.getcell("CODE", index)) if "CODE" in table.colnames() else ""
            for index in range(table.nrows())
        ]
    return tuple((index, names[index], codes[index]) for index in range(len(names)))


def three_c286_surviving_graph(
    path: Path,
    *,
    chunk_rows: int = 16384,
) -> dict[str, object]:
    """Connectivity of unflagged 3C286 samples. Does not load DATA."""

    tables = _tables()
    measurement_set = Path(path)
    field_names = _read_field_names(tables, measurement_set)
    field_ids = [index for index, name in enumerate(field_names) if name in EVPA_FIELD_NAMES]
    if not field_ids:
        return {"present": False, "usable": False, "notes": "3C286 field is absent"}
    antenna_names = _antenna_names(tables, measurement_set)
    n_ant = len(antenna_names)
    names = ("RR", "RL", "LR", "LL")
    ids = ",".join(str(index) for index in field_ids)
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        selected = main.query(f"FIELD_ID IN [{ids}]")
        try:
            report = _accumulate_surviving_graph(
                selected,
                n_ant,
                names,
                antenna_names,
                chunk_rows=chunk_rows,
            )
        finally:
            selected.close()
    report["field_ids"] = field_ids
    report["antenna_names"] = list(antenna_names)
    report["notes"] = (
        "3C286 chi span is too small to separate source polarization from D",
        "surviving samples may still constrain Kcross, R-L phase, EVPA, and V floor",
        "A connected 3C286 graph proves an X solution is possible; it does not "
        "prove every antenna received a valid Kcross/Xf solution",
        IDENTITY_INHERITANCE_FORBIDDEN,
    )
    return report


def _accumulate_surviving_graph(
    selected: Any,
    n_ant: int,
    names: tuple[str, ...],
    antenna_names: tuple[str, ...],
    *,
    chunk_rows: int,
) -> dict[str, object]:
    n_row = int(selected.nrows())
    if n_row == 0:
        return {
            "present": True,
            "usable_for_x_evpa": False,
            "n_rows": 0,
            "active_antennas": [],
            "absent_antennas": list(antenna_names),
            "connectivity_is_not_term_coverage": True,
            "notes": "no 3C286 rows",
        }
    antenna1 = np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32)
    antenna2 = np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32)
    probe = np.asarray(selected.getcol("FLAG", 0, 1), dtype=bool)
    n_chan = int(probe.shape[-2] if probe.ndim == 3 else probe.shape[0])
    n_corr = int(probe.shape[-1])
    vis = np.zeros((n_chan, n_corr), dtype=np.int64)
    flagged = np.zeros((n_chan, n_corr), dtype=np.int64)
    parent = np.arange(n_ant * n_corr * (1 + n_chan), dtype=np.int32)
    active = np.zeros(n_ant * n_corr * (1 + n_chan), dtype=bool)
    for start in range(0, n_row, chunk_rows):
        stop = min(start + chunk_rows, n_row)
        flag = np.asarray(selected.getcol("FLAG", start, stop - start), dtype=bool)
        if flag.ndim == 2:
            flag = flag[:, None, :]
        usable = ~flag
        vis += np.sum(usable, axis=0)
        flagged += np.sum(flag, axis=0)
        for corr in range(n_corr):
            any_chan = np.any(usable[..., corr], axis=1)
            if np.any(any_chan):
                _union_baselines(
                    parent,
                    active,
                    antenna1[start:stop][any_chan],
                    antenna2[start:stop][any_chan],
                    n_ant=n_ant,
                    offset=n_ant * corr,
                )
            for chan in range(n_chan):
                mask = usable[:, chan, corr]
                if not np.any(mask):
                    continue
                _union_baselines(
                    parent,
                    active,
                    antenna1[start:stop][mask],
                    antenna2[start:stop][mask],
                    n_ant=n_ant,
                    offset=n_ant * (n_corr + chan * n_corr + corr),
                )
    per_corr = {}
    connected_channels = {}
    broadband = {}
    for corr in range(n_corr):
        used = [ant for ant in range(n_ant) if active[n_ant * corr + ant]]
        roots = {_find(parent, n_ant * corr + ant) for ant in used} if used else set()
        broadband[names[corr]] = {
            "n_active_antennas": len(used),
            "n_components": len(roots),
            "connected": len(roots) == 1,
        }
        channel_ok = []
        for chan in range(n_chan):
            offset = n_ant * (n_corr + chan * n_corr + corr)
            used_ch = [ant for ant in range(n_ant) if active[offset + ant]]
            if not used_ch:
                channel_ok.append(False)
                continue
            channel_ok.append(len({_find(parent, offset + ant) for ant in used_ch}) == 1)
        connected_channels[names[corr]] = {
            "n_connected_channels": int(np.sum(channel_ok)),
            "n_channels": n_chan,
            "fraction_connected": float(np.mean(channel_ok)) if channel_ok else 0.0,
        }
        total = int(np.sum(vis[:, corr]) + np.sum(flagged[:, corr]))
        per_corr[names[corr]] = {
            "n_unflagged": int(np.sum(vis[:, corr])),
            "flag_fraction": float(np.sum(flagged[:, corr]) / total) if total else 1.0,
        }
    usable = all(item["connected"] for item in broadband.values()) and all(
        item["n_connected_channels"] >= 8 for item in connected_channels.values()
    )
    used_any = {ant for corr in range(n_corr) for ant in range(n_ant) if active[n_ant * corr + ant]}
    return {
        "schema_version": HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION,
        "present": True,
        "n_rows": n_row,
        "n_channels": n_chan,
        "n_correlations": n_corr,
        "per_correlation": per_corr,
        "broadband_graph": broadband,
        "connectivity": connected_channels,
        "usable_for_x_evpa": usable,
        "disconnected_or_narrowband": not usable,
        "active_antennas": [antenna_names[ant] for ant in sorted(used_any)],
        "absent_antennas": [antenna_names[ant] for ant in range(n_ant) if ant not in used_any],
        "connectivity_is_not_term_coverage": True,
    }


def three_c147_point_model_audit(
    path: Path,
    *,
    scans: tuple[int, ...] = FLUX_BANDPASS_SCANS,
    field_id: int = 0,
    holdout_antenna: str = HELD_OUT_REFERENCE_ANTENNA,
    n_uv_bins: int = 8,
) -> dict[str, object]:
    """Test the Perley-Butler point model on flux/bandpass scans.

    Per-baseline flattening is recorded as a diagnostic only. Acceptance
    for Jones work requires ``three_c147_structure_audit`` model ratios and
    closure amplitudes. Scan 51 on B0 is a G holdout, not a source holdout.
    """

    tables = _tables()
    measurement_set = Path(path)
    antenna_names = _antenna_names(tables, measurement_set)
    holdout_id = antenna_names.index(holdout_antenna) if holdout_antenna in antenna_names else -1
    scan_list = ",".join(str(scan) for scan in scans)
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        selected = main.query(f"FIELD_ID=={int(field_id)} && SCAN_NUMBER IN [{scan_list}]")
        try:
            if selected.nrows() == 0:
                raise ValueError(f"no rows for field {field_id} scans {scans}")
            uvw = np.asarray(selected.getcol("UVW"), dtype=np.float64)
            antenna1 = np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32)
            antenna2 = np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32)
            data = np.asarray(selected.getcol("DATA"))
            flag = np.asarray(selected.getcol("FLAG"), dtype=bool)
            n_selected = int(selected.nrows())
        finally:
            selected.close()
    frequencies = _channel_frequencies(tables, measurement_set, 4)
    model = three_c147_point_source_model(frequencies)
    model_i = perley_butler_2017_3c147_stokes_i_jy(frequencies)
    usable = ~flag
    amp = np.abs(data)
    uv_m = np.hypot(uvw[:, 0], uvw[:, 1])
    wavelength = SPEED_OF_LIGHT_M_S / frequencies
    report: dict[str, object] = {
        "schema_version": HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION,
        "scans": list(scans),
        "field_id": field_id,
        "n_rows": n_selected,
        "model": {
            "name": model.name,
            "kind": model.kind,
            "standard": model.standard,
            "stokes_i_jy": model_i.tolist(),
            "stokes_q": 0.0,
            "stokes_u": 0.0,
            "stokes_v": 0.0,
            "polarization_uncertainty_floor": (
                "3C147 treated as unpolarized pending a measured Q/U bound"
            ),
        },
        "nominal_weight": NOMINAL_WEIGHT,
        "weight_is_empirical_variance": False,
        "channels": {},
    }
    for label, corr in (("RR", 0), ("LL", 3)):
        channel_report = []
        for chan in (0, frequencies.size // 2, frequencies.size - 1):
            mask = usable[:, chan, corr]
            if holdout_id >= 0:
                mask = mask & (antenna1 != holdout_id) & (antenna2 != holdout_id)
            if int(np.sum(mask)) < 16:
                continue
            baseline = np.hypot(
                np.minimum(antenna1[mask], antenna2[mask]),
                np.maximum(antenna1[mask], antenna2[mask]) * 1000,
            )
            values = amp[mask, chan, corr]
            flattened = np.empty_like(values)
            for ident in np.unique(baseline):
                rows = baseline == ident
                scale = float(np.median(values[rows]))
                flattened[rows] = values[rows] / scale if scale > 0 else values[rows]
            uv_lambda = uv_m[mask] / float(wavelength[chan])
            slope = _binned_slope(uv_lambda, flattened, n_uv_bins)
            median_amp = float(np.median(values))
            channel_report.append(
                {
                    "channel": int(chan),
                    "frequency_hz": float(frequencies[chan]),
                    "n": int(np.sum(mask)),
                    "median_amp": median_amp,
                    "model_i_jy": float(model_i[chan]),
                    "median_amp_over_model": median_amp / float(model_i[chan]),
                    "flattened_uv_slope": slope,
                    "point_like": abs(slope) < 0.15,
                }
            )
        report["channels"][label] = channel_report
    slopes = [item["flattened_uv_slope"] for corr in report["channels"].values() for item in corr]
    report["accepted_as_point"] = bool(slopes) and all(abs(slope) < 0.15 for slope in slopes)
    report["holdout_antenna"] = holdout_antenna
    report["scan51_kind"] = G1_HOLD_SCAN51_KIND if 51 in scans else None
    report["notes"] = (
        "A calibrator name does not make a point model acceptable",
        "Flattened UV slope is a diagnostic; do not accept a point model from "
        "per-baseline flattening alone",
        SCAN51_G_HOLDOUT_NOTE,
        SCAN51_B_ABLATION_NOTE,
    )
    return report


def empirical_variance_from_repeat_scatter(
    values: NDArray[np.complex128],
    groups: NDArray[np.int32],
) -> float:
    """Robust variance from repeated samples of the same cell or baseline."""

    scatter: list[float] = []
    for ident in np.unique(groups):
        selected = values[groups == ident]
        selected = selected[np.isfinite(selected)]
        if selected.size < 2:
            continue
        scatter.append(float(np.var(selected - np.median(selected))))
    if not scatter:
        return float("nan")
    return float(np.median(scatter))


def hash_path(path: Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    if source.is_file():
        digest.update(source.read_bytes())
        return digest.hexdigest()
    for file in sorted(p for p in source.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(source)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def product_manifest(
    *,
    name: str,
    tables: dict[str, Path],
    model: dict[str, object],
    parang: bool,
    calwt: bool,
    flag_version: str,
    notes: tuple[str, ...],
    apply_contract: str = CASA_DF_JONES_APPLYCAL_V1,
) -> dict[str, object]:
    settings = apply_contract_settings(apply_contract)
    return {
        "schema_version": HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION,
        "product": name,
        "parang": parang,
        "calwt": calwt,
        "flag_version": flag_version,
        "model": model,
        "tables": {
            key: {"path": str(path), "sha256": hash_path(path) if Path(path).exists() else ""}
            for key, path in tables.items()
        },
        "apply_contract": apply_contract,
        "apply_contract_settings": settings,
        "snap_pass2_to_memo_lattice": False,
        "c147_offset_used_for_d": False,
        "weight_model": "preserve nominal WEIGHT; do not treat 4e6 as reduced chi-squared",
        "jones_recovery_blocked": True,
        "most_important_next_artifact": MOST_IMPORTANT_NEXT_ARTIFACT,
        "notes": list(notes),
    }


def _antenna_names(tables: Any, measurement_set: Path) -> tuple[str, ...]:
    with tables.table(str(measurement_set / "ANTENNA"), readonly=True, ack=False) as table:
        return tuple(str(name) for name in table.getcol("NAME"))


def _channel_frequencies(tables: Any, measurement_set: Path, spw: int) -> NDArray[np.float64]:
    with tables.table(str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False) as table:
        return np.asarray(table.getcell("CHAN_FREQ", int(spw)), dtype=np.float64)


def _find(parent: NDArray[np.int32], item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = int(parent[item])
    return int(item)


def _union_baselines(
    parent: NDArray[np.int32],
    active: NDArray[np.bool_],
    antenna1: NDArray[np.int32],
    antenna2: NDArray[np.int32],
    *,
    n_ant: int,
    offset: int,
) -> None:
    for first, second in zip(antenna1, antenna2, strict=True):
        left = offset + int(first)
        right = offset + int(second)
        if not (0 <= int(first) < n_ant and 0 <= int(second) < n_ant):
            continue
        active[left] = True
        active[right] = True
        root_left = _find(parent, left)
        root_right = _find(parent, right)
        if root_left != root_right:
            parent[root_right] = root_left


def _binned_slope(x: NDArray[np.float64], y: NDArray[np.float64], n_bins: int) -> float:
    if x.size < 8:
        return float("nan")
    edges = np.quantile(x, np.linspace(0.0, 1.0, int(n_bins) + 1))
    centres: list[float] = []
    means: list[float] = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        mask = (x >= start) & (x <= stop)
        if int(np.sum(mask)) < 4:
            continue
        centres.append(float(np.median(x[mask])))
        means.append(float(np.median(y[mask])))
    if len(centres) < 2:
        return float("nan")
    return float(np.polyfit(centres, means, 1)[0] * (centres[-1] - centres[0]))


def write_json(payload: dict[str, object], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


def classify_holdout(name: str) -> dict[str, object]:
    """Record what a named holdout actually tests."""

    key = str(name).strip()
    if key in {"G1_hold_scan51", "g_holdout_scan51", G1_HOLD_SCAN51_KIND}:
        return {
            "name": "G1_hold_scan51",
            "kind": G1_HOLD_SCAN51_KIND,
            "tests": ("gain_interpolation",),
            "does_not_test": (
                "end_to_end_calibration",
                "source_model",
                "bandpass",
            ),
            "reason": SCAN51_G_HOLDOUT_NOTE,
        }
    if key in {"B2_scan2", "bandpass_ablation_scan51", "scan51_b_ablation"}:
        return {
            "name": "B2_scan2_on_scan51",
            "kind": "bandpass_ablation",
            "tests": ("bandpass_transfer", "source_model_on_scan51"),
            "does_not_test": (),
            "reason": SCAN51_B_ABLATION_NOTE,
        }
    raise ValueError(f"unknown holography holdout {name!r}")


def jones_recovery_gate(
    completed: dict[str, bool] | None = None,
) -> dict[str, object]:
    """Jones recovery stays blocked until every validation step passes."""

    status = {step: bool((completed or {}).get(step, False)) for step in VALIDATION_ORDER}
    pending = [step for step, done in status.items() if not done]
    return {
        "schema_version": HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION,
        "blocked": True if pending else False,
        "completed": status,
        "pending": pending,
        "next_in_order": pending[0] if pending else None,
        "most_important_next_artifact": MOST_IMPORTANT_NEXT_ARTIFACT,
        "notes": (
            JONES_RECOVERY_BLOCKED_NOTE,
            FIRST_BEAM_RECOVERY_UNFROZEN_NOTE,
            CONSISTENT_3C147_FLUX_GAUGE_NOTE,
            COMPATIBILITY_TABLES_NOT_SCIENTIFIC_NOTE,
            "Diagonal product uses parang=false; full-pol uses parang=true",
            "CASA/JAX golden locked CASA 6.7.6 apply conventions",
            CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE,
            ALL_ANTENNA_RESIDUAL_JONES_NOTE,
            CONNECTED_HOLDOUT_NOTE,
            RESIDUAL_JONES_VISIBILITY_GATES_NOTE,
            THREE_C147_QU_IS_NUISANCE_NOTE,
            JJH_PROXY_IS_NOT_DECISIVE_NOTE,
            THREE_C286_CIRCULAR_FLOOR_NOTE,
            SMOOTH_GP_REJECTED_NOTE,
            PER_SCAN_EPSILON_NOT_OBSERVABLE_NOTE,
            ESTIMATOR_IDENTIFIABILITY_NOTE,
            COMPLETE_FIELD9_TRACK_NOTE,
            PREDICTION_EQUIVALENT_BEAM_NOTE,
            ONE_AXIS_VISIBILITY_HOLDOUT_NOTE,
            CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE,
            ZERO_QU_IS_ABLATION_NOTE,
            NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
            INTERLEAVED_ONAXIS_TRANSFER_NOTE,
            INTERLEAVED_BEAM_TRANSFER_GATE_NOTE,
        ),
        "scientific_recovery_order": list(SCIENTIFIC_RECOVERY_ORDER),
    }


def classify_parameter_rows(
    flag: NDArray[np.bool_],
    values: NDArray[Any],
    *,
    identity: complex | float,
) -> str:
    """solved / flagged / absent / unflagged_identity for one antenna-SPW term."""

    mask = np.asarray(flag, dtype=bool)
    if mask.size == 0:
        return "absent"
    if np.all(mask):
        return "flagged"
    usable = np.asarray(values)[~mask]
    if np.allclose(np.asarray(usable, dtype=np.complex128), identity, atol=1e-8):
        return "unflagged_identity"
    return "solved"


def term_coverage_from_rows(
    *,
    term: str,
    antenna_names: tuple[str, ...],
    antenna: NDArray[np.int32],
    spectral_window_id: NDArray[np.int32],
    flag: NDArray[np.bool_],
    values: NDArray[Any],
    spectral_window_ids: tuple[int, ...] = (4, 5),
) -> dict[str, dict[str, str]]:
    """Coverage matrix for one CASA term: antenna → SPW → status."""

    if term not in COVERAGE_TERMS:
        raise ValueError(f"unknown coverage term {term!r}")
    identity = CAL_TABLE_IDENTITY[term]
    matrix = {name: {str(spw): "absent" for spw in spectral_window_ids} for name in antenna_names}
    flags = np.asarray(flag, dtype=bool)
    if flags.ndim == 1:
        flags = flags[:, None]
    payload = np.asarray(values)
    if payload.ndim == 1:
        payload = payload[:, None]
    for name, spw in ((name, spw) for name in antenna_names for spw in spectral_window_ids):
        index = antenna_names.index(name)
        rows = (np.asarray(antenna, dtype=np.int32) == index) & (
            np.asarray(spectral_window_id, dtype=np.int32) == int(spw)
        )
        if not np.any(rows):
            continue
        matrix[name][str(spw)] = classify_parameter_rows(
            flags[rows], payload[rows], identity=identity
        )
    return matrix


def fullpol_antenna_support(
    coverage: dict[str, dict[str, dict[str, str]]],
    *,
    three_c286_active: tuple[str, ...],
    three_c286_absent: tuple[str, ...],
    global_x: bool = False,
) -> dict[str, object]:
    """Kcross/Xf coverage is not implied by a connected 3C286 graph."""

    antennas = sorted(
        {antenna for term in ("Kcross", "Xf") for antenna in coverage.get(term, {})}
        | set(three_c286_active)
        | set(three_c286_absent)
    )
    by_antenna: dict[str, dict[str, object]] = {}
    unsupported: list[str] = []
    silent_identity: list[str] = []
    justified_global: list[str] = []
    unjustified: list[str] = []
    for antenna in antennas:
        statuses = []
        for term in ("Kcross", "Xf"):
            spw_status = coverage.get(term, {}).get(antenna, {})
            statuses.extend(spw_status.values())
        if any(status == "unflagged_identity" for status in statuses):
            support = "invalid_identity_inheritance"
            silent_identity.append(antenna)
        elif all(status in {"flagged", "absent"} for status in statuses) or not statuses:
            support = "unsupported_full_pol"
            unsupported.append(antenna)
        elif antenna in three_c286_absent:
            if global_x:
                support = "full_pol_via_global_x"
                justified_global.append(antenna)
            else:
                support = "needs_justified_global_x"
                unjustified.append(antenna)
        else:
            support = "full_pol"
        by_antenna[antenna] = {
            "Kcross": coverage.get("Kcross", {}).get(antenna, {}),
            "Xf": coverage.get("Xf", {}).get(antenna, {}),
            "in_usable_3c286": antenna in three_c286_active,
            "support": support,
        }
    return {
        "schema_version": HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION,
        "by_antenna": by_antenna,
        "unsupported_full_pol": unsupported,
        "invalid_identity_inheritance": silent_identity,
        "full_pol_via_global_x": justified_global,
        "needs_justified_global_x": unjustified,
        "global_x": global_x,
        "passed": not silent_identity and not unjustified,
        "identity_inheritance_forbidden": True,
        "notes": (
            IDENTITY_INHERITANCE_FORBIDDEN,
            "A connected graph is necessary but not sufficient for term coverage",
            "Identical unflagged Kcross/Xf on every antenna is a global reference-frame X",
        ),
    }


def term_coverage_from_casa_tables(
    tables: dict[str, Path],
    *,
    measurement_set: Path,
    three_c286_active: tuple[str, ...] | None = None,
    three_c286_absent: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Read CASA FLAG/CPARAM/FPARAM and build the antenna×SPW coverage matrix."""

    casa = _tables()
    antenna_names = _antenna_names(casa, Path(measurement_set))
    coverage: dict[str, dict[str, dict[str, str]]] = {}
    for term, path in tables.items():
        if term not in COVERAGE_TERMS:
            continue
        coverage[term] = _coverage_from_one_table(casa, Path(path), term, antenna_names)
    active = three_c286_active
    absent = three_c286_absent
    if active is None or absent is None:
        graph = three_c286_surviving_graph(Path(measurement_set))
        active = tuple(str(name) for name in graph.get("active_antennas", ()))
        absent = tuple(str(name) for name in graph.get("absent_antennas", ()))
    global_x = _casa_tables_are_global(
        casa, {key: tables[key] for key in ("Kcross", "Xf") if key in tables}
    )
    support = fullpol_antenna_support(
        coverage,
        three_c286_active=active or (),
        three_c286_absent=absent or (),
        global_x=global_x,
    )
    return {
        "schema_version": HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION,
        "antenna_names": list(antenna_names),
        "spectral_window_ids": [4, 5],
        "terms": coverage,
        "three_c286_active": list(active or ()),
        "three_c286_absent": list(absent or ()),
        "global_x": global_x,
        "fullpol_support": support,
        "jones_recovery_blocked": True,
        "notes": (
            IDENTITY_INHERITANCE_FORBIDDEN,
            JONES_RECOVERY_BLOCKED_NOTE,
        ),
    }


def casa_table_is_global(
    antenna: NDArray[np.int32],
    flag: NDArray[np.bool_],
    values: NDArray[Any],
    spectral_window_id: NDArray[np.int32] | None = None,
) -> bool:
    """True when unflagged antenna rows are identical within each SPW."""

    mask = np.asarray(flag, dtype=bool)
    if mask.ndim == 1:
        usable = ~mask
    else:
        usable = ~np.all(mask.reshape(mask.shape[0], -1), axis=1)
    if not np.any(usable) or np.unique(np.asarray(antenna)[usable]).size < 2:
        return False
    groups = (
        np.asarray(spectral_window_id, dtype=np.int32)
        if spectral_window_id is not None
        else np.zeros(np.asarray(antenna).size, dtype=np.int32)
    )
    payload = np.asarray(values)
    for spw in np.unique(groups[usable]):
        rows = payload[usable & (groups == spw)]
        if rows.size == 0:
            continue
        if not all(np.allclose(row, rows[0]) for row in rows[1:]):
            return False
    return True


def _casa_tables_are_global(tables: Any, paths: dict[str, Path]) -> bool:
    if not paths:
        return False
    for path in paths.values():
        source = Path(path)
        if not source.exists():
            return False
        with tables.table(str(source), readonly=True, ack=False) as table:
            if table.nrows() == 0 or "ANTENNA1" not in table.colnames():
                return False
            flag = (
                np.asarray(table.getcol("FLAG"), dtype=bool)
                if "FLAG" in table.colnames()
                else np.zeros(table.nrows(), dtype=bool)
            )
            if "CPARAM" in table.colnames() and table.iscelldefined("CPARAM", 0):
                values = np.asarray(table.getcol("CPARAM"))
            elif "FPARAM" in table.colnames() and table.iscelldefined("FPARAM", 0):
                values = np.asarray(table.getcol("FPARAM"))
            else:
                return False
            spw = (
                np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
                if "SPECTRAL_WINDOW_ID" in table.colnames()
                else None
            )
            if not casa_table_is_global(
                np.asarray(table.getcol("ANTENNA1"), dtype=np.int32),
                flag,
                values,
                spectral_window_id=spw,
            ):
                return False
    return True


def _coverage_from_one_table(
    tables: Any,
    path: Path,
    term: str,
    antenna_names: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return {name: {str(spw): "absent" for spw in (4, 5)} for name in antenna_names}
    with tables.table(str(source), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        n_row = int(table.nrows())
        if n_row == 0 or "ANTENNA1" not in columns:
            antenna = np.zeros(0, dtype=np.int32)
            spw = np.zeros(0, dtype=np.int32)
            flag = np.zeros((0, 1), dtype=bool)
            values = np.zeros((0, 1), dtype=np.complex128)
        else:
            antenna = np.asarray(table.getcol("ANTENNA1"), dtype=np.int32)
            spw = (
                np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
                if "SPECTRAL_WINDOW_ID" in columns
                else np.zeros(n_row, dtype=np.int32)
            )
            flag = (
                np.asarray(table.getcol("FLAG"), dtype=bool)
                if "FLAG" in columns
                else np.zeros((n_row, 1), dtype=bool)
            )
            if "CPARAM" in columns and table.iscelldefined("CPARAM", 0):
                values = np.asarray(table.getcol("CPARAM"))
            elif "FPARAM" in columns and table.iscelldefined("FPARAM", 0):
                values = np.asarray(table.getcol("FPARAM"))
            else:
                values = np.zeros(flag.shape, dtype=np.complex128)
    return term_coverage_from_rows(
        term=term,
        antenna_names=antenna_names,
        antenna=antenna,
        spectral_window_id=spw,
        flag=flag,
        values=values,
    )


def field_parallactic_report(
    path: Path,
    field_id: int,
    *,
    antenna_index: int = 0,
) -> dict[str, object]:
    """Parallactic-angle span for one field. Field 9 is the D-code 3C147 track."""

    tables = _tables()
    measurement_set = Path(path)
    names = _read_field_names(tables, measurement_set)
    if not (0 <= int(field_id) < len(names)):
        raise ValueError(f"field {field_id} is outside FIELD")
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        field = np.asarray(main.getcol("FIELD_ID"), dtype=np.int32)
        time_s = np.asarray(main.getcol("TIME"), dtype=np.float64)
    times = np.unique(time_s[field == int(field_id)])
    if times.size == 0:
        return {
            "present": True,
            "field_id": int(field_id),
            "field_name": names[int(field_id)],
            "n_times": 0,
            "notes": "field has no MAIN rows",
        }
    phase = _read_field_phase_centre(tables, measurement_set, names[int(field_id)])
    _ids, _names, positions = _read_antennas(tables, measurement_set)
    from sl1mjax.calibration_terms import parallactic_angle_rad

    position = np.asarray(positions, dtype=np.float64)[int(antenna_index) : int(antenna_index) + 1]
    chi = parallactic_angle_rad(times, phase, position)[:, 0]
    span = unwrapped_chi_span_rad(times, chi)
    return {
        "schema_version": HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION,
        "present": True,
        "field_id": int(field_id),
        "field_name": names[int(field_id)],
        "n_times": int(times.size),
        "chi_rad_min": float(np.min(chi)),
        "chi_rad_max": float(np.max(chi)),
        "chi_rad_span_wrapped": float(np.max(chi) - np.min(chi)),
        "chi_rad_span": span,
        "chi_deg_span": float(np.rad2deg(span)),
        "notes": (
            "Field 9 spans much more of the observation than 3C286; use it for Df vs Df+QU",
            "3C286 is a single short visit and is not a parallactic-angle track",
            "chi_deg_span is the time-ordered unwrapped track, not wrapped min/max",
        ),
    }


def unwrapped_chi_span_rad(times: NDArray[np.float64], chi: NDArray[np.float64]) -> float:
    """Time-ordered unwrapped χ span. Wrapped min/max is not a track length."""

    order = np.argsort(np.asarray(times, dtype=np.float64))
    unwrapped = np.unwrap(np.asarray(chi, dtype=np.float64)[order])
    return float(np.max(unwrapped) - np.min(unwrapped))


def leakage_systematic_floor(
    first: NDArray[np.complex128],
    second: NDArray[np.complex128],
    valid: NDArray[np.bool_],
) -> dict[str, object]:
    """Difference between two reasonable on-axis D models."""

    mask = np.asarray(valid, dtype=bool) & np.isfinite(first) & np.isfinite(second)
    if not np.any(mask):
        return {
            "n": 0,
            "median_abs_delta": float("nan"),
            "p90_abs_delta": float("nan"),
            "notes": "no overlapping valid D samples",
        }
    delta = np.abs(np.asarray(first)[mask] - np.asarray(second)[mask])
    return {
        "n": int(np.sum(mask)),
        "median_abs_delta": float(np.median(delta)),
        "p90_abs_delta": float(np.quantile(delta, 0.9)),
        "notes": (
            "Any material difference is the leakage systematic floor",
            "Recovered off-axis Jones must not depend on which reasonable D was used",
        ),
    }


def df_variant_plan() -> dict[str, object]:
    """Required Df / Df+QU / fixed-pol comparison. Not a solved product."""

    return {
        "schema_version": HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION,
        "variants": {
            "Df": {
                "poltype": "Df",
                "source_qu": (0.0, 0.0),
                "assumption": "on-axis 3C147 polarization is zero",
            },
            "Df+QU": {
                "poltype": "Df+QU",
                "source_qu": "solved_global_3C147",
                "assumption": "common Q/U is not absorbed into a common-mode D-term",
            },
            "D_fixed_pol": {
                "poltype": "Df",
                "source_qu": "independently_fixed_if_available",
                "assumption": "external 3C147 polarization model, if one exists",
            },
        },
        "holdout": {
            "field_id": D_CODE_FIELD_ID,
            "axes": ("time", "antenna", "channel"),
            "held_out_reference": HELD_OUT_REFERENCE_ANTENNA,
        },
        "c147_offset_used_for_d": False,
        "jones_recovery_blocked": True,
        "notes": (
            "A Df solve can turn real common Q/U into a common-mode D-term",
            "Compare variants on held-out field-9 times, antennas, and channels",
        ),
    }


def structure_audit_from_visibilities(
    *,
    uvw_m: NDArray[np.float64],
    frequency_hz: NDArray[np.float64],
    visibility: NDArray[np.complex128],
    flag: NDArray[np.bool_],
    antenna1: NDArray[np.int32],
    antenna2: NDArray[np.int32],
    time_s: NDArray[np.float64],
    model_i_jy: NDArray[np.float64],
    n_uv_bins: int = 8,
    holdout_kind: str = "none",
) -> dict[str, object]:
    """Model ratios and closure amplitudes versus UV. No per-baseline flattening."""

    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    model_i = np.asarray(model_i_jy, dtype=np.float64)
    amp = np.abs(np.asarray(visibility))
    usable = ~np.asarray(flag, dtype=bool)
    uv_m = np.hypot(np.asarray(uvw_m)[:, 0], np.asarray(uvw_m)[:, 1])
    wavelength = SPEED_OF_LIGHT_M_S / frequencies
    channels: dict[str, list[dict[str, object]]] = {}
    for label, corr in (("RR", 0), ("LL", min(3, amp.shape[-1] - 1))):
        channel_report = []
        for chan in (0, frequencies.size // 2, frequencies.size - 1):
            mask = usable[:, chan, corr]
            if int(np.sum(mask)) < 16:
                continue
            ratio = amp[mask, chan, corr] / float(model_i[chan])
            uv_lambda = uv_m[mask] / float(wavelength[chan])
            slope = _binned_slope(uv_lambda, ratio, n_uv_bins)
            channel_report.append(
                {
                    "channel": int(chan),
                    "frequency_hz": float(frequencies[chan]),
                    "n": int(np.sum(mask)),
                    "median_amp_over_model": float(np.median(ratio)),
                    "model_ratio_uv_slope": slope,
                    "flattened": False,
                }
            )
        channels[label] = channel_report
    closures = _closure_amplitudes_vs_uv(
        uv_m=uv_m,
        frequency_hz=frequencies,
        amp=amp,
        usable=usable,
        antenna1=np.asarray(antenna1, dtype=np.int32),
        antenna2=np.asarray(antenna2, dtype=np.int32),
        time_s=np.asarray(time_s, dtype=np.float64),
        channel=int(frequencies.size // 2),
        corr=0,
        n_uv_bins=n_uv_bins,
    )
    ratio_slopes = [
        abs(float(item["model_ratio_uv_slope"]))
        for items in channels.values()
        for item in items
        if np.isfinite(item["model_ratio_uv_slope"])
    ]
    return {
        "schema_version": HOLOGRAPHY_CALIBRATION_SCHEMA_VERSION,
        "holdout_kind": holdout_kind,
        "channels": channels,
        "closure_amplitudes": closures,
        "accepted_as_point": bool(ratio_slopes)
        and all(slope < 0.15 for slope in ratio_slopes)
        and bool(closures.get("point_like")),
        "notes": (
            "Model ratios and closure amplitudes are the source-structure test",
            "Per-baseline flattening is omitted because it can hide resolved structure",
            SCAN51_G_HOLDOUT_NOTE
            if holdout_kind == G1_HOLD_SCAN51_KIND
            else SCAN51_B_ABLATION_NOTE,
        ),
    }


def _closure_amplitudes_vs_uv(
    *,
    uv_m: NDArray[np.float64],
    frequency_hz: NDArray[np.float64],
    amp: NDArray[np.floating],
    usable: NDArray[np.bool_],
    antenna1: NDArray[np.int32],
    antenna2: NDArray[np.int32],
    time_s: NDArray[np.float64],
    channel: int,
    corr: int,
    n_uv_bins: int,
) -> dict[str, object]:
    from itertools import combinations

    closures: list[float] = []
    uv_max: list[float] = []
    min_amp: list[float] = []
    wavelength = SPEED_OF_LIGHT_M_S / float(frequency_hz[channel])
    for time in np.unique(time_s):
        rows = np.flatnonzero((time_s == time) & usable[:, channel, corr])
        if rows.size < 6:
            continue
        baselines: dict[tuple[int, int], tuple[float, float]] = {}
        for row in rows:
            pair = (
                int(min(antenna1[row], antenna2[row])),
                int(max(antenna1[row], antenna2[row])),
            )
            if pair[0] == pair[1] or pair in baselines:
                continue
            baselines[pair] = (float(amp[row, channel, corr]), float(uv_m[row]))
        antennas = sorted({ant for pair in baselines for ant in pair})
        for quad in combinations(antennas, 4):
            first, second, third, fourth = quad
            ab = baselines.get((first, second))
            cd = baselines.get((third, fourth))
            ac = baselines.get((first, third))
            bd = baselines.get((second, fourth))
            if ab is None or cd is None or ac is None or bd is None:
                continue
            denom = ac[0] * bd[0]
            if denom <= 0.0:
                continue
            closures.append(ab[0] * cd[0] / denom)
            uv_max.append(max(ab[1], cd[1], ac[1], bd[1]) / wavelength)
            min_amp.append(min(ab[0], cd[0], ac[0], bd[0]))
        if len(closures) >= 4096:
            break
    if len(closures) < 8:
        return {
            "n": len(closures),
            "median": float("nan"),
            "uv_slope": float("nan"),
            "point_like": False,
        }
    values = np.asarray(closures, dtype=np.float64)
    uv_values = np.asarray(uv_max, dtype=np.float64)
    min_amps = np.asarray(min_amp, dtype=np.float64) if min_amp else np.ones(values.size)
    slope = _binned_slope(uv_values, values, n_uv_bins)
    median = float(np.median(values))
    amp_cut = float(np.median(min_amps)) * 0.1 if min_amps.size else None
    return {
        "n": int(values.size),
        "median": median,
        "mad": float(np.median(np.abs(values - median))),
        "uv_slope": slope,
        "point_like": abs(median - 1.0) < 0.05 and (not np.isfinite(slope) or abs(slope) < 0.15),
        "log_closures": log_closure_amplitudes(values, min_amp=min_amps, amp_cut=amp_cut),
    }


def log_closure_amplitudes(
    closures: NDArray[np.float64],
    *,
    min_amp: NDArray[np.float64] | None = None,
    amp_cut: float | None = None,
) -> dict[str, object]:
    """Log closure amplitudes after excluding low-amplitude quads."""

    values = np.asarray(closures, dtype=np.float64)
    keep = np.isfinite(values) & (values > 0)
    if min_amp is not None and amp_cut is not None:
        keep = keep & (np.asarray(min_amp) >= float(amp_cut))
    selected = values[keep]
    if selected.size < 4:
        return {
            "n": int(selected.size),
            "n_excluded": int(values.size - selected.size),
            "log10_mad": float("nan"),
            "median": float("nan"),
        }
    log = np.log10(selected)
    median = float(np.median(selected))
    return {
        "n": int(selected.size),
        "n_excluded": int(values.size - selected.size),
        "median": median,
        "mad": float(np.median(np.abs(selected - median))),
        "log10_median": float(np.median(log)),
        "log10_mad": float(np.median(np.abs(log - np.median(log)))),
        "amp_cut": amp_cut,
    }


def three_c147_structure_audit(
    path: Path,
    *,
    scans: tuple[int, ...] = (51,),
    field_id: int = 0,
    holdout_kind: str = "bandpass_ablation",
) -> dict[str, object]:
    """Post-B source test on selected 3C147 scans. Does not flatten per baseline."""

    tables = _tables()
    measurement_set = Path(path)
    scan_list = ",".join(str(scan) for scan in scans)
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        selected = main.query(f"FIELD_ID=={int(field_id)} && SCAN_NUMBER IN [{scan_list}]")
        try:
            if selected.nrows() == 0:
                raise ValueError(f"no rows for field {field_id} scans {scans}")
            uvw = np.asarray(selected.getcol("UVW"), dtype=np.float64)
            antenna1 = np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32)
            antenna2 = np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32)
            time_s = np.asarray(selected.getcol("TIME"), dtype=np.float64)
            column = "CORRECTED_DATA" if "CORRECTED_DATA" in selected.colnames() else "DATA"
            data = np.asarray(selected.getcol(column))
            flag = np.asarray(selected.getcol("FLAG"), dtype=bool)
        finally:
            selected.close()
    frequencies = _channel_frequencies(tables, measurement_set, 4)
    report = structure_audit_from_visibilities(
        uvw_m=uvw,
        frequency_hz=frequencies,
        visibility=data,
        flag=flag,
        antenna1=antenna1,
        antenna2=antenna2,
        time_s=time_s,
        model_i_jy=perley_butler_2017_3c147_stokes_i_jy(frequencies),
        holdout_kind=holdout_kind,
    )
    report["scans"] = list(scans)
    report["field_id"] = int(field_id)
    report["visibility_column"] = column
    return report
