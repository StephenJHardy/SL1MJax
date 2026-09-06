"""CASA/JAX calibration golden for THOL0001 lower-C.

The golden applies the same CASA tables from raw DATA in JAX and compares
RR/RL/LR/LL on a small native-resolution HOLORASTER copy. It locks
calibration conventions before holography-coordinate and beam-recovery
errors become entangled.

Jones recovery stays blocked until this comparison passes.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sl1mjax.calibration import (
    CalibrationSolution,
    _embedded_cal_frequencies,
    _pivot_casa_table,
    apply_calibration,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.holography_calibration import (
    CASA_DF_JONES_APPLYCAL_V1,
    FULLPOL_PARANG,
    HELD_OUT_REFERENCE_ANTENNA,
    JONES_RECOVERY_BLOCKED_NOTE,
    MOST_IMPORTANT_NEXT_ARTIFACT,
    REFERENCE_ANTENNA,
    apply_contract_for_casa_viscal,
    apply_contract_settings,
    hash_path,
    write_json,
)
from sl1mjax.holography_ms import (
    _read_field_names,
    _read_field_phase_centre,
    _tables,
    is_casacore_measurement_set,
)
from sl1mjax.polarization import Correlation, ReceptorBasis

HOLOGRAPHY_GOLDEN_SCHEMA_VERSION = 1
GOLDEN_CONVENTIONS = (
    "table_order",
    "conjugation_and_baseline_orientation",
    "correlation_packing",
    "rl_handedness",
    "parallactic_angle_treatment",
    "unsupported_antenna_channel_masking",
    "no_double_application",
    "df_time_interpolation",
    "df_first_order",
    "cparam_casa_float32_amp_phase",
    "apply_contract",
    "parallactic_casacore_hadec_geocentric",
)
DIAGONAL_TABLE_ORDER = ("antpos", "G1", "K0", "B0")
FULLPOL_TABLE_ORDER = ("antpos", "G1", "K0", "B0", "Kcross", "Df", "Xf")
ProductName = Literal["diagonal", "fullpol"]


def golden_convention_lock(*, parang: bool) -> dict[str, object]:
    """What the CASA/JAX golden must lock before any Jones recovery."""

    return {
        "schema_version": HOLOGRAPHY_GOLDEN_SCHEMA_VERSION,
        "conventions": list(GOLDEN_CONVENTIONS),
        "diagonal_parang": False,
        "fullpol_parang": True,
        "requested_parang": bool(parang),
        "diagonal_table_order": list(DIAGONAL_TABLE_ORDER),
        "fullpol_table_order": list(FULLPOL_TABLE_ORDER),
        "calwt": False,
        "apply_from": "raw DATA",
        "compare": ("RR", "RL", "LR", "LL"),
        "jones_recovery_blocked": True,
        "most_important_next_artifact": MOST_IMPORTANT_NEXT_ARTIFACT,
        "notes": (
            JONES_RECOVERY_BLOCKED_NOTE,
            "Full-pol uses parang=true; diagonal uses parang=false",
            "Do not apply tables twice; JAX starts from DATA, not CORRECTED_DATA",
            "G and D CPARAM use CASA 6.7.6 float32 amplitude/unwrapped phase",
            "Apply contract comes from VisCal='Df Jones', not the source name",
            "P uses casacore HADEC and geocentric latitude; not POINTING_OFFSET",
        ),
    }


def select_golden_holoraster_rows(
    measurement_set: Path,
    *,
    field_name: str = "HOLORASTER",
    scans: tuple[int, ...] = (18, 58),
    times_per_scan: int = 2,
    data_desc_ids: tuple[int, ...] = (4, 5),
) -> NDArray[np.int64]:
    """Few native-resolution HOLORASTER times from both raster passes."""

    tables = _tables()
    names = _read_field_names(tables, Path(measurement_set))
    if field_name not in names:
        raise ValueError(f"{field_name!r} is not in FIELD")
    field_id = names.index(field_name)
    with tables.table(str(Path(measurement_set)), readonly=True, ack=False) as main:
        field = np.asarray(main.getcol("FIELD_ID"), dtype=np.int32)
        scan = np.asarray(main.getcol("SCAN_NUMBER"), dtype=np.int32)
        time_s = np.asarray(main.getcol("TIME"), dtype=np.float64)
        ddid = np.asarray(main.getcol("DATA_DESC_ID"), dtype=np.int32)
    keep = (field == field_id) & np.isin(ddid, np.asarray(data_desc_ids, dtype=np.int32))
    chosen_times: list[float] = []
    for scan_id in scans:
        times = np.unique(time_s[keep & (scan == int(scan_id))])
        if times.size == 0:
            continue
        index = np.linspace(0, times.size - 1, int(times_per_scan), dtype=np.int64)
        chosen_times.extend(float(value) for value in times[index])
    if not chosen_times:
        raise ValueError(f"no HOLORASTER rows for scans {scans}")
    selected = np.flatnonzero(keep & np.isin(time_s, np.asarray(chosen_times)))
    return selected.astype(np.int64)


def copy_golden_measurement_set(
    source: Path,
    destination: Path,
    rows: NDArray[np.int64],
) -> Path:
    """Deep-copy selected MAIN rows. The source archive is not modified."""

    dest = Path(destination)
    if dest.exists():
        raise FileExistsError(f"golden MS already exists: {dest}")
    if not is_casacore_measurement_set(Path(source)):
        raise ValueError(f"not a casacore Measurement Set: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tables = _tables()
    with tables.table(str(Path(source)), readonly=True, ack=False) as main:
        selected = main.selectrows(np.asarray(rows, dtype=np.int64))
        try:
            selected.copy(str(dest), deep=True)
        finally:
            selected.close()
    if not is_casacore_measurement_set(dest):
        raise RuntimeError(f"golden copy did not produce an MS: {dest}")
    return dest


def compare_casa_jax_visibilities(
    casa: NDArray[np.complex128],
    jax_vis: NDArray[np.complex128],
    valid: NDArray[np.bool_],
    *,
    correlations: tuple[str, ...] = ("RR", "RL", "LR", "LL"),
    relative_tolerance: float = 1.0e-5,
    absolute_tolerance: float = 1.0e-8,
    crosshand_amp_floor: float = 0.05,
    antenna1: NDArray[np.int32] | None = None,
    antenna2: NDArray[np.int32] | None = None,
    time_s: NDArray[np.float64] | None = None,
    frequency_hz: NDArray[np.float64] | None = None,
    antenna_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Per-correlation CASA vs JAX agreement on unflagged golden samples.

    ``max_rel`` is kept, but it is inflated when CASA RL/LR is near zero.
    Relative L2 and residual / RR-LL amplitude are the structure metrics.
    """

    casa_vis = np.asarray(casa)
    jax_values = np.asarray(jax_vis)
    mask = np.asarray(valid, dtype=bool)
    if casa_vis.shape != jax_values.shape or casa_vis.shape != mask.shape:
        raise ValueError("CASA, JAX, and validity arrays must share one shape")
    names = tuple(correlations)
    index = {name: slot for slot, name in enumerate(names)}
    parallel = np.zeros(casa_vis.shape[:2], dtype=np.float64)
    if "RR" in index and "LL" in index:
        parallel = 0.5 * (np.abs(casa_vis[..., index["RR"]]) + np.abs(casa_vis[..., index["LL"]]))
    per_corr: dict[str, dict[str, object]] = {}
    passed = True
    residual = casa_vis - jax_values
    for slot, name in enumerate(names):
        usable = (
            mask[..., slot] & np.isfinite(casa_vis[..., slot]) & np.isfinite(jax_values[..., slot])
        )
        if not np.any(usable):
            per_corr[name] = {"n": 0, "max_abs": float("nan"), "rms": float("nan"), "passed": False}
            passed = False
            continue
        values = residual[..., slot][usable]
        casa_amp = np.abs(casa_vis[..., slot][usable])
        scale = np.maximum(casa_amp, absolute_tolerance)
        max_abs = float(np.max(np.abs(values)))
        rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
        casa_l2 = float(np.sqrt(np.sum(casa_amp**2)))
        rel_l2 = (
            float(np.sqrt(np.sum(np.abs(values) ** 2)) / casa_l2) if casa_l2 > 0 else float("nan")
        )
        parallel_amp = parallel[usable] if parallel.shape == casa_vis.shape[:2] else casa_amp
        over_parallel = np.abs(values) / np.maximum(parallel_amp, absolute_tolerance)
        above = casa_amp >= float(crosshand_amp_floor)
        if name in {"RL", "LR"}:
            above = above & (parallel_amp >= float(crosshand_amp_floor))
        phase = np.angle(casa_vis[..., slot][usable] * np.conj(jax_values[..., slot][usable]))
        amp_err = np.abs(jax_values[..., slot][usable]) - casa_amp
        max_rel = float(np.max(np.abs(values) / scale))
        near_zero_crosshand = name in {"RL", "LR"} and int(np.sum(above)) == 0
        if near_zero_crosshand:
            ok = (
                max_abs <= max(absolute_tolerance, 1.0e-6)
                or rel_l2 <= relative_tolerance
                or float(np.median(over_parallel)) <= relative_tolerance
            )
            pass_metric = "absolute_or_resid_over_rr_ll"
        else:
            ok = max_abs <= absolute_tolerance or max_rel <= relative_tolerance
            pass_metric = "max_abs_or_max_rel"
        per_corr[name] = {
            "n": int(np.sum(usable)),
            "max_abs": max_abs,
            "rms": rms,
            "max_rel": max_rel,
            "relative_l2": rel_l2,
            "median_resid_over_rr_ll": float(np.median(over_parallel)),
            "p90_resid_over_rr_ll": float(np.quantile(over_parallel, 0.9)),
            "crosshand_amp_floor": float(crosshand_amp_floor),
            "n_above_floor": int(np.sum(above)),
            "amp_err_above_floor": _finite_median(amp_err[above]),
            "phase_err_deg_above_floor": _finite_median(np.rad2deg(phase[above])),
            "phase_rms_deg_above_floor": _finite_rms_deg(phase[above]),
            "pass_metric": pass_metric,
            "max_rel_unsuitable": bool(near_zero_crosshand),
            "historical_max_rel_failure": bool(
                near_zero_crosshand and max_rel > relative_tolerance
            ),
            "passed": ok,
        }
        passed = passed and ok
    breakdowns = _golden_breakdowns(
        residual,
        casa_vis,
        mask,
        names,
        parallel,
        antenna1=antenna1,
        antenna2=antenna2,
        time_s=time_s,
        frequency_hz=frequency_hz,
        antenna_names=antenna_names,
    )
    return {
        "schema_version": HOLOGRAPHY_GOLDEN_SCHEMA_VERSION,
        "passed": passed,
        "per_correlation": per_corr,
        "breakdowns": breakdowns,
        "conventions": list(GOLDEN_CONVENTIONS),
        "jones_recovery_blocked": not passed,
        "notes": (
            "max_rel is inflated when CASA RL/LR is near zero; that historical "
            "failure is an unsuitable metric, not a convention failure",
            "Near-zero RL/LR uses absolute residual or resid/RR-LL, not max_rel",
            "This comparison isolates calibration conventions from holography coordinates",
            JONES_RECOVERY_BLOCKED_NOTE,
        ),
    }


def golden_visibility_block(
    measurement_set: Path,
    *,
    data_column: str = "DATA",
    data_desc_id: int,
) -> tuple[VisibilityBlock, NDArray[np.complex128] | None, NDArray[np.bool_]]:
    """One SPW of the golden MS as a VisibilityBlock, plus CASA CORRECTED_DATA."""

    tables = _tables()
    source = Path(measurement_set)
    names = _read_field_names(tables, source)
    with tables.table(str(source), readonly=True, ack=False) as main:
        ddid = np.asarray(main.getcol("DATA_DESC_ID"), dtype=np.int32)
        rows = np.flatnonzero(ddid == int(data_desc_id))
        if rows.size == 0:
            raise ValueError(f"no rows for DATA_DESC_ID {data_desc_id}")
        selected = main.selectrows(rows)
        try:
            columns = set(selected.colnames())
            visibility = np.asarray(selected.getcol(data_column))
            corrected = (
                np.asarray(selected.getcol("CORRECTED_DATA"))
                if "CORRECTED_DATA" in columns
                else None
            )
            flag = np.asarray(selected.getcol("FLAG"), dtype=bool)
            block = VisibilityBlock(
                uvw_m=np.asarray(selected.getcol("UVW"), dtype=np.float64),
                frequency_hz=_spectral_window_frequencies(tables, source, int(data_desc_id)),
                visibility=visibility,
                weight=_row_weights(selected, visibility.shape),
                flag=flag,
                time_s=np.asarray(selected.getcol("TIME"), dtype=np.float64),
                antenna1=np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32),
                antenna2=np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32),
                field_id=np.asarray(selected.getcol("FIELD_ID"), dtype=np.int32),
                scan_id=np.asarray(selected.getcol("SCAN_NUMBER"), dtype=np.int32),
                correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
                receptor_basis=ReceptorBasis.CIRCULAR,
                phase_centre_rad=_read_field_phase_centre(
                    tables, source, names[int(selected.getcell("FIELD_ID", 0))]
                ),
                data_description_id=int(data_desc_id),
                spectral_window_id=_spectral_window_id(tables, source, int(data_desc_id)),
                provenance={"source": str(source), "column": data_column},
            )
        finally:
            selected.close()
    return block, corrected, flag


def import_thol0001_casa_tables(
    tables: dict[str, Path],
    *,
    measurement_set: Path,
    spectral_window_id: int,
    product: ProductName,
    interpolation: str = "linear",
) -> CalibrationSolution:
    """Import live CASA tables for one SPW. Flagged rows stay invalid."""

    casa = _tables()
    source = Path(measurement_set)
    with casa.table(str(source / "ANTENNA"), readonly=True, ack=False) as antenna:
        antenna_names = np.asarray(antenna.getcol("NAME")).astype(str)
        antenna_position_m = np.asarray(antenna.getcol("POSITION"), dtype=np.float64)
    frequencies = _spectral_window_frequencies(
        casa, source, _ddid_for_spw(casa, source, spectral_window_id)
    )
    arrays: dict[str, np.ndarray] = {}
    _add_cal_arrays(casa, arrays, "gain", tables["G1"], "CPARAM", spectral_window_id)
    _add_cal_arrays(casa, arrays, "delay", tables["K0"], "FPARAM", spectral_window_id)
    _add_cal_arrays(casa, arrays, "bandpass", tables["B0"], "CPARAM", spectral_window_id)
    gain_times, gains, gain_valid = _pivot_casa_table(arrays, "gain", "cparam")
    interval_times, gain_interval, _ = _pivot_casa_table(arrays, "gain", "interval")
    if not np.array_equal(interval_times, gain_times):
        raise ValueError("CASA gain interval and parameter coordinates differ")
    gains = gains[:, :, 0, :]
    gain_valid = gain_valid[:, :, 0, :]
    gain_interval = np.broadcast_to(gain_interval[:, :, None], gains.shape).copy()
    gain_interval = np.where(gain_interval <= 0.0, np.inf, gain_interval)
    _, delay, delay_valid = _pivot_casa_table(arrays, "delay", "fparam")
    delay = -delay[0, :, 0, :] * 1e-9
    delay_valid = delay_valid[0, :, 0, :]
    _, bandpass, bandpass_valid = _pivot_casa_table(arrays, "bandpass", "cparam")
    bandpass = bandpass[0]
    bandpass_valid = bandpass_valid[0]
    offsets = np.zeros((gains.shape[1], 3), dtype=np.float64)
    if "antpos" in tables and Path(tables["antpos"]).exists():
        _add_cal_arrays(casa, arrays, "antpos", tables["antpos"], "FPARAM", None)
        antpos = np.asarray(arrays["antpos_fparam"])[..., 0, :]
        antpos_antenna = np.asarray(arrays["antpos_antenna1"], dtype=np.int32)
        offsets[antpos_antenna] = antpos
    reference_matches = np.flatnonzero(antenna_names == REFERENCE_ANTENNA)
    if reference_matches.size != 1:
        raise ValueError(f"reference antenna {REFERENCE_ANTENNA!r} not found")
    solution = CalibrationSolution(
        gains=gains,
        gain_time_s=gain_times,
        gain_valid=gain_valid,
        gain_interval_s=gain_interval,
        delays_s=delay,
        delay_valid=delay_valid,
        bandpass=bandpass,
        bandpass_frequency_hz=frequencies[: bandpass.shape[1]]
        if bandpass.shape[1] != frequencies.size
        else frequencies,
        bandpass_valid=bandpass_valid,
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        reference_antenna=int(reference_matches[0]),
        reference_frequency_hz=_delay_reference_frequency_hz(
            frequencies, _read_cal_channel_frequency(casa, tables["K0"], spectral_window_id)
        ),
        interpolation=interpolation,
        antenna_position_offset_m=offsets,
        antenna_position_m=antenna_position_m,
        apply_parallactic_angle=product == "fullpol",
        provenance={
            "product": product,
            "spectral_window_id": int(spectral_window_id),
            "calwt": False,
            "parang": product == "fullpol",
            "kcross_convention": "negated_ns_like_K",
            "delay_reference": "casa_cal_table_chan_freq",
            "jones_order": "GKB Kcross D X P",
            "df_time": "all_casa_solution_times",
            "cparam_interpolation": "casa_float32_amp_phase",
            "casa_version": "6.7.6.14",
            "time_reference": "UTC",
            "direction_frame": "J2000",
        },
    )
    if product == "diagonal":
        return replace(
            solution,
            parallactic_model="casacore_hadec_geocentric",
            provenance={
                **dict(solution.provenance),
                "antenna_position_direction": "casacore_itrf",
            },
        )
    _add_cal_arrays(casa, arrays, "kcross", tables["Kcross"], "FPARAM", spectral_window_id)
    _add_cal_arrays(casa, arrays, "dterms", tables["Df"], "CPARAM", spectral_window_id)
    _add_cal_arrays(casa, arrays, "angle", tables["Xf"], "CPARAM", spectral_window_id)
    _, kcross, kcross_valid = _pivot_casa_table(arrays, "kcross", "fparam")
    leakage_times, leakage, leakage_valid = _pivot_casa_table(arrays, "dterms", "cparam")
    _, rl_phase, rl_phase_valid = _pivot_casa_table(arrays, "angle", "cparam")
    rl_phase = rl_phase[0, :, :, 0]
    rl_phase_valid = rl_phase_valid[0, :, :, 0]
    contract = apply_contract_for_casa_viscal(_read_viscal(casa, tables["Df"]))
    settings = apply_contract_settings(contract)
    return replace_polarization(
        solution,
        apply_contract=contract,
        interpolation=str(settings["interpolation"]),
        provenance={
            **dict(solution.provenance),
            "apply_contract": contract,
            "viscal": _read_viscal(casa, tables["Df"]),
            "leakage_application": settings["leakage_application"],
            "cparam_interpolation": settings.get("cparam_interpolation"),
            "casa_version": settings.get("casa_version"),
            "gainfield_mapping": settings.get("gainfield_mapping"),
            "jones_invert": settings.get("jones_invert"),
        },
        cross_hand_delay_s=-kcross[0, :, 0, :] * 1e-9,
        cross_hand_delay_valid=kcross_valid[0, :, 0, :],
        leakage=leakage,
        leakage_frequency_hz=_embedded_cal_frequencies(frequencies, leakage.shape[2]),
        leakage_valid=leakage_valid,
        leakage_time_s=leakage_times,
        rl_phase=rl_phase,
        rl_phase_frequency_hz=_embedded_cal_frequencies(frequencies, rl_phase.shape[1]),
        rl_phase_valid=rl_phase_valid,
    )


def replace_polarization(solution: CalibrationSolution, **kwargs: Any) -> CalibrationSolution:
    from dataclasses import replace

    contract = str(kwargs.pop("apply_contract", None) or CASA_DF_JONES_APPLYCAL_V1)
    settings = apply_contract_settings(contract)
    return replace(
        solution,
        apply_parallactic_angle=True,
        leakage_application=str(settings["leakage_application"]),
        parallactic_model=str(settings["parallactic_model"]),
        **kwargs,
    )


def compare_golden_measurement_set(
    measurement_set: Path,
    tables: dict[str, Path],
    *,
    product: ProductName,
    data_desc_ids: tuple[int, ...] = (4, 5),
) -> dict[str, object]:
    """Apply imported tables from DATA and compare to CASA CORRECTED_DATA."""

    comparisons = {}
    passed = True
    for ddid in data_desc_ids:
        block, casa_corrected, _flag = golden_visibility_block(
            measurement_set, data_column="DATA", data_desc_id=ddid
        )
        if casa_corrected is None:
            raise ValueError(f"{measurement_set} has no CORRECTED_DATA for DATA_DESC_ID {ddid}")
        solution = import_thol0001_casa_tables(
            tables,
            measurement_set=measurement_set,
            spectral_window_id=block.spectral_window_id,
            product=product,
        )
        jax_block = apply_imported_solution(block, solution)
        usable = ~block.flag & ~jax_block.flag
        report = compare_casa_jax_visibilities(
            casa_corrected,
            jax_block.visibility,
            usable,
            antenna1=block.antenna1,
            antenna2=block.antenna2,
            time_s=block.time_s,
            frequency_hz=block.frequency_hz,
        )
        comparisons[f"ddid_{ddid}"] = report
        passed = passed and bool(report["passed"])
    payload = golden_report(
        product=product,
        comparison={"passed": passed, "by_data_desc": comparisons},
        measurement_set=measurement_set,
        tables=tables,
    )
    payload["comparison"]["passed"] = passed
    return payload


def apply_imported_solution(
    block: VisibilityBlock,
    solution: CalibrationSolution,
) -> VisibilityBlock:
    """Apply imported CASA tables from DATA. Never from CORRECTED_DATA."""

    if block.provenance.get("column") == "CORRECTED_DATA":
        raise ValueError("golden JAX apply must start from DATA to prevent double application")
    return apply_calibration(block, solution, extrapolate=True)


def calibration_solution_up_to(
    solution: CalibrationSolution,
    stage: str,
) -> CalibrationSolution | None:
    """Return a prefix of ``GKB Kcross D X P``. ``DATA`` is uncalibrated."""

    from sl1mjax.holography_reference_jones import CALIBRATION_CHAIN_STAGES

    if stage not in CALIBRATION_CHAIN_STAGES:
        raise ValueError(f"unknown calibration chain stage {stage!r}")
    if stage == "DATA":
        return None
    stripped = solution
    if stage == "K_B_G":
        stripped = replace(
            solution,
            cross_hand_delay_s=None,
            cross_hand_delay_valid=None,
            leakage=None,
            leakage_frequency_hz=None,
            leakage_valid=None,
            leakage_time_s=None,
            rl_phase=None,
            rl_phase_frequency_hz=None,
            rl_phase_valid=None,
            apply_parallactic_angle=False,
        )
    elif stage == "K_B_G_Kcross":
        stripped = replace(
            solution,
            leakage=None,
            leakage_frequency_hz=None,
            leakage_valid=None,
            leakage_time_s=None,
            rl_phase=None,
            rl_phase_frequency_hz=None,
            rl_phase_valid=None,
            apply_parallactic_angle=False,
        )
    elif stage == "K_B_G_Kcross_Df":
        stripped = replace(
            solution,
            rl_phase=None,
            rl_phase_frequency_hz=None,
            rl_phase_valid=None,
            apply_parallactic_angle=False,
        )
    elif stage == "K_B_G_Kcross_Df_Xf":
        stripped = replace(solution, apply_parallactic_angle=False)
    return stripped


def golden_report(
    *,
    product: ProductName,
    comparison: dict[str, object],
    measurement_set: Path,
    tables: dict[str, Path],
) -> dict[str, object]:
    parang = FULLPOL_PARANG if product == "fullpol" else False
    return {
        "schema_version": HOLOGRAPHY_GOLDEN_SCHEMA_VERSION,
        "product": product,
        "measurement_set": str(measurement_set),
        "tables": {
            key: {"path": str(path), "sha256": hash_path(path)} for key, path in tables.items()
        },
        "conventions": golden_convention_lock(parang=parang),
        "comparison": comparison,
        "held_out_reference": HELD_OUT_REFERENCE_ANTENNA,
        "reference_antenna": REFERENCE_ANTENNA,
        "jones_recovery_blocked": True,
        "most_important_next_artifact": MOST_IMPORTANT_NEXT_ARTIFACT,
    }


def write_golden_report(payload: dict[str, object], path: Path) -> Path:
    return write_json(payload, path)


def _finite_median(values: NDArray[np.floating]) -> float:
    selected = np.asarray(values)
    selected = selected[np.isfinite(selected)]
    return float(np.median(selected)) if selected.size else float("nan")


def _finite_rms_deg(phase_rad: NDArray[np.floating]) -> float:
    selected = np.asarray(phase_rad)
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        return float("nan")
    return float(np.rad2deg(np.sqrt(np.mean(selected**2))))


def _golden_breakdowns(
    residual: NDArray[np.complex128],
    casa: NDArray[np.complex128],
    mask: NDArray[np.bool_],
    correlations: tuple[str, ...],
    parallel_amp: NDArray[np.float64],
    *,
    antenna1: NDArray[np.int32] | None,
    antenna2: NDArray[np.int32] | None,
    time_s: NDArray[np.float64] | None,
    frequency_hz: NDArray[np.float64] | None,
    antenna_names: tuple[str, ...] | None,
) -> dict[str, object]:
    """Residual / RR-LL by antenna, baseline orientation, channel, and time."""

    payload: dict[str, object] = {}
    if antenna1 is None or antenna2 is None:
        return payload
    first = np.asarray(antenna1, dtype=np.int32)
    second = np.asarray(antenna2, dtype=np.int32)
    orientation = np.where(first < second, "antenna1_lt_antenna2", "antenna1_gt_antenna2")
    by_orientation = {}
    for label in ("antenna1_lt_antenna2", "antenna1_gt_antenna2"):
        by_orientation[label] = _group_resid_over_parallel(
            residual, casa, mask, correlations, parallel_amp, orientation == label
        )
    payload["baseline_orientation"] = by_orientation
    antennas = np.unique(np.concatenate((first, second)))
    by_antenna = {}
    for ant in antennas:
        name = (
            antenna_names[int(ant)]
            if antenna_names and int(ant) < len(antenna_names)
            else str(int(ant))
        )
        by_antenna[name] = _group_resid_over_parallel(
            residual, casa, mask, correlations, parallel_amp, (first == ant) | (second == ant)
        )
    payload["antenna"] = by_antenna
    if frequency_hz is not None and residual.shape[1] == np.asarray(frequency_hz).size:
        channels = {}
        for chan in (0, residual.shape[1] // 2, residual.shape[1] - 1):
            row_mask = np.ones(residual.shape[0], dtype=bool)
            channels[str(chan)] = _group_resid_over_parallel(
                residual[:, chan : chan + 1],
                casa[:, chan : chan + 1],
                mask[:, chan : chan + 1],
                correlations,
                parallel_amp[:, chan : chan + 1],
                row_mask,
            )
        payload["channel"] = channels
    if time_s is not None:
        times = np.unique(time_s)
        payload["n_times"] = int(times.size)
        if times.size:
            payload["time"] = {
                "first": _group_resid_over_parallel(
                    residual, casa, mask, correlations, parallel_amp, time_s == times[0]
                ),
                "last": _group_resid_over_parallel(
                    residual, casa, mask, correlations, parallel_amp, time_s == times[-1]
                ),
            }
    return payload


def _group_resid_over_parallel(
    residual: NDArray[np.complex128],
    casa: NDArray[np.complex128],
    mask: NDArray[np.bool_],
    correlations: tuple[str, ...],
    parallel_amp: NDArray[np.float64],
    row_mask: NDArray[np.bool_],
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    rows = np.asarray(row_mask, dtype=bool)
    if rows.ndim == 1:
        rows = rows[:, None]
    for slot, name in enumerate(correlations):
        usable = rows & mask[..., slot] & np.isfinite(residual[..., slot])
        if not np.any(usable):
            out[name] = {
                "n": 0,
                "relative_l2": float("nan"),
                "median_resid_over_rr_ll": float("nan"),
            }
            continue
        values = residual[..., slot][usable]
        casa_l2 = float(np.sqrt(np.sum(np.abs(casa[..., slot][usable]) ** 2)))
        scale = (
            parallel_amp[usable]
            if parallel_amp.shape[:2] == residual.shape[:2]
            else np.abs(casa[..., slot][usable])
        )
        out[name] = {
            "n": int(np.sum(usable)),
            "relative_l2": float(np.sqrt(np.sum(np.abs(values) ** 2)) / casa_l2)
            if casa_l2
            else float("nan"),
            "median_resid_over_rr_ll": float(
                np.median(np.abs(values) / np.maximum(scale, 1.0e-12))
            ),
        }
    return out


def _ddid_for_spw(tables: Any, measurement_set: Path, spectral_window_id: int) -> int:
    with tables.table(
        str(Path(measurement_set) / "DATA_DESCRIPTION"), readonly=True, ack=False
    ) as table:
        mapping = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
    matches = np.flatnonzero(mapping == int(spectral_window_id))
    if matches.size != 1:
        raise ValueError(f"SPW {spectral_window_id} does not map to one DATA_DESC_ID")
    return int(matches[0])


def _add_cal_arrays(
    tables: Any,
    arrays: dict[str, np.ndarray],
    prefix: str,
    path: Path,
    parameter: str,
    spectral_window_id: int | None,
) -> None:
    with tables.table(str(Path(path)), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        keep = np.ones(table.nrows(), dtype=bool)
        if spectral_window_id is not None and "SPECTRAL_WINDOW_ID" in columns:
            keep &= np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32) == int(
                spectral_window_id
            )
        arrays[f"{prefix}_antenna1"] = np.asarray(table.getcol("ANTENNA1"), dtype=np.int32)[keep]
        arrays[f"{prefix}_time"] = np.asarray(table.getcol("TIME"), dtype=np.float64)[keep]
        arrays[f"{prefix}_flag"] = np.asarray(table.getcol("FLAG"), dtype=bool)[keep]
        arrays[f"{prefix}_{parameter.lower()}"] = np.asarray(table.getcol(parameter))[keep]
        if "INTERVAL" in columns:
            arrays[f"{prefix}_interval"] = np.asarray(table.getcol("INTERVAL"), dtype=np.float64)[
                keep
            ]
        else:
            arrays[f"{prefix}_interval"] = np.zeros(int(np.sum(keep)), dtype=np.float64)
        if "FIELD_ID" in columns:
            arrays[f"{prefix}_field_id"] = np.asarray(table.getcol("FIELD_ID"), dtype=np.int32)[
                keep
            ]


def _delay_reference_frequency_hz(
    ms_frequencies: NDArray[np.float64],
    cal_channel_frequency_hz: float | None,
) -> float:
    """CASA delay origin is the cal-table channel frequency, not the MS midpoint.

    A 1 MHz mismatch at 3.44 ns is a 1.4 deg Kcross residual and a 0.2 deg K floor.
    """

    if cal_channel_frequency_hz is not None and np.isfinite(cal_channel_frequency_hz):
        return float(cal_channel_frequency_hz)
    freqs = np.asarray(ms_frequencies, dtype=np.float64)
    return float(freqs[freqs.size // 2])


def _read_viscal(tables: Any, cal_table: Path) -> str:
    with tables.table(str(cal_table), readonly=True, ack=False) as table:
        return str(table.getkeyword("VisCal"))


def _read_cal_channel_frequency(
    tables: Any, cal_table: Path, spectral_window_id: int
) -> float | None:
    path = Path(cal_table) / "SPECTRAL_WINDOW"
    if not path.is_dir():
        return None
    with tables.table(str(path), readonly=True, ack=False) as table:
        n_row = int(table.nrows())
        if n_row <= 0:
            return None
        if "SPECTRAL_WINDOW_ID" in table.colnames():
            ids = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
            matches = np.flatnonzero(ids == int(spectral_window_id))
            row = int(matches[0]) if matches.size else min(int(spectral_window_id), n_row - 1)
        else:
            row = min(int(spectral_window_id), n_row - 1)
        channels = np.asarray(table.getcell("CHAN_FREQ", row), dtype=np.float64).reshape(-1)
    return float(channels[0]) if channels.size else None


def _spectral_window_id(tables: Any, measurement_set: Path, data_desc_id: int) -> int:
    with tables.table(
        str(Path(measurement_set) / "DATA_DESCRIPTION"), readonly=True, ack=False
    ) as table:
        return int(table.getcell("SPECTRAL_WINDOW_ID", int(data_desc_id)))


def _spectral_window_frequencies(
    tables: Any, measurement_set: Path, data_desc_id: int
) -> NDArray[np.float64]:
    spw = _spectral_window_id(tables, measurement_set, data_desc_id)
    with tables.table(
        str(Path(measurement_set) / "SPECTRAL_WINDOW"), readonly=True, ack=False
    ) as table:
        return np.asarray(table.getcell("CHAN_FREQ", spw), dtype=np.float64)


def _row_weights(selected: Any, shape: tuple[int, ...]) -> NDArray[np.float64]:
    columns = set(selected.colnames())
    if "WEIGHT" not in columns:
        return np.ones(shape, dtype=np.float64)
    weight = np.asarray(selected.getcol("WEIGHT"), dtype=np.float64)
    if weight.ndim == 2:
        return np.broadcast_to(weight[:, None, :], shape).copy()
    return np.asarray(weight, dtype=np.float64)


def golden_casa_script_payload(
    *,
    vis: Path,
    product: ProductName,
    tables: dict[str, Path],
) -> dict[str, object]:
    """Arguments for the standalone CASA applycal golden script."""

    order = FULLPOL_TABLE_ORDER if product == "fullpol" else DIAGONAL_TABLE_ORDER
    missing = [name for name in order if name not in tables]
    if missing:
        raise ValueError(f"{product} golden is missing tables: {missing}")
    return {
        "vis": str(vis),
        "product": product,
        "parang": product == "fullpol",
        "calwt": False,
        "applymode": "calflag",
        "gaintable": [str(tables[name]) for name in order],
        "interp": ["", "linear", "", "nearest"] + ([""] * (len(order) - 4)),
        "conventions": list(GOLDEN_CONVENTIONS),
    }


def write_golden_casa_request(payload: dict[str, object], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination
