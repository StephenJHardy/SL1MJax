"""Recover unfrozen per-reference diagonal holography on THOL0001 HOLORASTER.

Measured cells only. SPW 4 / channel 32 first, then SPW 5 / channel 32 as
the frequency-transfer companion. Does not interpolate, array-average, or
freeze a beam.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.beam_operator import unique_visibility_times
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.holography import (
    MODEL_DATA_SOURCE_COHERENCY_NOTE,
    HolographyHoldoutAxis,
    HolographyObservation,
    ResolvedAntennaPointing,
    casa_setjy_fluxd_jy,
    circular_visibility_to_source_coherency,
    three_c147_flux_scale_report,
)
from sl1mjax.holography_calibration import (
    COMPATIBILITY_TABLES_NOT_SCIENTIFIC_NOTE,
    FIRST_BEAM_RECOVERY_UNFROZEN_NOTE,
    write_json,
)
from sl1mjax.holography_diagonal import (
    ABSOLUTE_RECOVERY_NOTE,
    FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE,
    FIRST_RECOVERY_FREQUENCIES_HZ,
    RELATIVE_ONAXIS_NORMALIZATION_NOTE,
    artifact_from_samples,
    compare_absolute_and_restored_relative,
    normalize_holography_diagonal_on_axis,
    recover_holography_diagonal_per_reference,
    samples_from_records,
)
from sl1mjax.holography_diagonal_diagnostics import (
    bootstrap_empirical_over_cassbeam_squint,
    compare_models_at_measured_cells,
    first_recovery_artifact_dict,
    holdout_prediction_report,
    leave_one_reference_out_report,
    physical_diagonal_diagnostics,
    reference_reference_residual_report,
    reference_treatment_report,
    repeated_visit_holdout_report,
    stratified_sample_holdouts,
)
from sl1mjax.holography_ms import audit_holography_measurement_set
from sl1mjax.polarization import Correlation, ReceptorBasis

DEFAULT_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
DEFAULT_SETJY_RECORD = Path(
    "/media/stephen/astro/vla/extracted/commissioning/products/scientific/setjy_3c147.json"
)
CHANNEL_32 = 32
HOLORASTER_FIELD = 10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_set", nargs="?", type=Path, default=DEFAULT_MS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-column", default="CORRECTED_DATA")
    parser.add_argument(
        "--spw",
        type=int,
        nargs="+",
        default=[4, 5],
        help="Spectral windows. SPW 4 first, SPW 5 is the frequency-transfer test.",
    )
    parser.add_argument("--channel", type=int, default=CHANNEL_32)
    parser.add_argument("--from-samples", type=Path)
    parser.add_argument("--visibility-holdouts", action="store_true")
    parser.add_argument(
        "--holdouts-only",
        action="store_true",
        help="From an existing sample JSON, run visibility holdouts only.",
    )
    parser.add_argument("--squint-bootstrap", action="store_true")
    parser.add_argument(
        "--product",
        choices=("absolute", "relative", "both"),
        default="both",
    )
    parser.add_argument(
        "--source-standard",
        choices=("casa_setjy", "table5"),
        default="casa_setjy",
        help="Integrated-flux provenance label only. Recovery uses MODEL_DATA.",
    )
    parser.add_argument(
        "--stokes-i",
        type=float,
        help="Optional setjy fluxd override for provenance. Not S_pq.",
    )
    parser.add_argument(
        "--setjy-record",
        type=Path,
        default=DEFAULT_SETJY_RECORD,
        help="CASA setjy JSON whose fluxd is stored as provenance only.",
    )
    parser.add_argument(
        "--calibration-product",
        choices=("scientific", "compatibility"),
        default="scientific",
    )
    parser.add_argument("--allow-compatibility-tables", action="store_true")
    arguments = parser.parse_args()
    if (
        arguments.calibration_product == "compatibility"
        and not arguments.allow_compatibility_tables
    ):
        raise ValueError(COMPATIBILITY_TABLES_NOT_SCIENTIFIC_NOTE)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    if arguments.from_samples is not None:
        if arguments.holdouts_only:
            report = _holdouts_from_samples(arguments.from_samples)
            output = arguments.output_dir / "visibility_holdouts.json"
        else:
            report = _diagnostics_from_samples(
                arguments.from_samples,
                squint_bootstrap=arguments.squint_bootstrap,
                product=arguments.product,
            )
            output = arguments.output_dir / "first_beam_recovery_from_samples.json"
        write_json(report, output)
        print(output)
        return 0
    audit = audit_holography_measurement_set(arguments.measurement_set)
    summaries = []
    for spectral_window_id in arguments.spw:
        report = _recover_one_window(
            arguments.measurement_set,
            audit.resolved,
            spectral_window_id=spectral_window_id,
            channel=arguments.channel,
            data_column=arguments.data_column,
            moving_antenna_ids=audit.moving_antenna_ids,
            output_dir=arguments.output_dir,
            visibility_holdouts=arguments.visibility_holdouts,
            squint_bootstrap=arguments.squint_bootstrap,
            product=arguments.product,
            source_standard=arguments.source_standard,
            stokes_i=arguments.stokes_i,
            setjy_record=arguments.setjy_record,
            calibration_product=arguments.calibration_product,
        )
        output = arguments.output_dir / f"first_beam_recovery_spw{spectral_window_id}.json"
        write_json(report, output)
        print(output)
        summaries.append(
            {
                "spectral_window_id": spectral_window_id,
                "frequency_hz": report["frequency_hz"],
                "n_samples": int(report.get("n_samples", 0)),
                "n_valid": int(report.get("n_valid", 0)),
                "output": str(output),
            }
        )
    write_json(
        {
            "frozen": False,
            "first_beam_recovery_unfrozen": True,
            "windows": summaries,
            "notes": [
                FIRST_BEAM_RECOVERY_DIAGNOSTIC_NOTE,
                FIRST_BEAM_RECOVERY_UNFROZEN_NOTE,
                "SPW 5 is recovered independently; it is not predicted from SPW 4",
                ABSOLUTE_RECOVERY_NOTE,
                RELATIVE_ONAXIS_NORMALIZATION_NOTE,
                COMPATIBILITY_TABLES_NOT_SCIENTIFIC_NOTE,
            ],
        },
        arguments.output_dir / "first_beam_recovery_summary.json",
    )
    return 0


def _recover_one_window(
    measurement_set: Path,
    pointing: ResolvedAntennaPointing,
    *,
    spectral_window_id: int,
    channel: int,
    data_column: str,
    moving_antenna_ids: tuple[int, ...],
    output_dir: Path,
    visibility_holdouts: bool = False,
    squint_bootstrap: bool = False,
    product: str = "both",
    source_standard: str = "casa_setjy",
    stokes_i: float | None = None,
    setjy_record: Path | None = None,
    calibration_product: str = "scientific",
) -> dict:
    from sl1mjax.holography_ms import _tables

    tables = _tables()
    ddid = _ddid_for_spw(tables, measurement_set, spectral_window_id)
    block, source = _holoraster_channel_block(
        tables,
        measurement_set,
        data_desc_id=ddid,
        channel=channel,
        data_column=data_column,
        spectral_window_id=spectral_window_id,
    )
    frequency = float(block.frequency_hz[0])
    fluxd = _fluxd_provenance(
        setjy_record,
        spectral_window_id=spectral_window_id,
        stokes_i=stokes_i,
    )
    observation = HolographyObservation(
        block=block,
        pointing=_subset_pointing(pointing, block.time_s),
        antenna_position_m=_antenna_positions(tables, measurement_set),
        calibration_state="casa_parang_true",
        phase_centre_rad=block.phase_centre_rad,
        source_name="3C147",
        source_coherency_visibility=source,
        selected_spw_id=spectral_window_id,
        provenance={
            "measurement_set": str(measurement_set),
            "data_column": data_column,
            "model_column": "MODEL_DATA",
            "spectral_window_id": spectral_window_id,
            "channel": channel,
            "field": "HOLORASTER",
            "source": "field_10_model_data_per_row",
            "source_standard": source_standard,
            "casa_setjy_fluxd_jy": fluxd,
            "stokes_i_is_not_s_pq": True,
            "calibration_product": calibration_product,
            "beam_product": "absolute",
        },
    )
    artifact = recover_holography_diagonal_per_reference(observation)
    checkpoint = first_recovery_artifact_dict(artifact)
    checkpoint["frequency_hz"] = float(block.frequency_hz[0])
    checkpoint["spectral_window_id"] = spectral_window_id
    checkpoint["channel"] = channel
    checkpoint["stage"] = "per_reference_samples"
    write_json(checkpoint, output_dir / f"first_beam_recovery_spw{spectral_window_id}_samples.json")
    reference_report = reference_treatment_report(artifact)
    holdouts = {}
    if visibility_holdouts:
        holdouts = {
            "leave_one_reference_out": leave_one_reference_out_report(observation),
            "reference_reference": reference_reference_residual_report(observation),
            "repeated_visits": repeated_visit_holdout_report(observation),
            "spatial_raster_cells": holdout_prediction_report(
                observation, axis=HolographyHoldoutAxis.SPATIAL
            ),
        }
        if len(moving_antenna_ids) >= 2:
            holdouts["moving_antenna"] = holdout_prediction_report(
                observation,
                axis=HolographyHoldoutAxis.MOVING_ANTENNA,
                holdout_antenna_id=int(moving_antenna_ids[0]),
            )
    payload = first_recovery_artifact_dict(
        artifact,
        reference_report=reference_report,
        holdouts=holdouts,
        model_comparison=compare_models_at_measured_cells(artifact),
        physical=physical_diagonal_diagnostics(artifact),
        include_samples=False,
    )
    payload["frequency_hz"] = frequency
    payload["spectral_window_id"] = spectral_window_id
    payload["channel"] = channel
    payload["native_frequencies_hz"] = list(FIRST_RECOVERY_FREQUENCIES_HZ)
    payload["provenance"] = dict(observation.provenance)
    payload["flux_scale"] = three_c147_flux_scale_report(frequency)
    payload["source"] = "field_10_model_data_per_row"
    payload["source_standard"] = source_standard
    payload["casa_setjy_fluxd_jy"] = fluxd
    payload["model_data_rr"] = _model_rr_spread(source)
    payload["beam_product"] = "absolute"
    payload["notes"] = list(payload["notes"]) + [
        ABSOLUTE_RECOVERY_NOTE,
        MODEL_DATA_SOURCE_COHERENCY_NOTE,
    ]
    if squint_bootstrap:
        payload["squint_bootstrap"] = bootstrap_empirical_over_cassbeam_squint(artifact)
    if product in {"relative", "both"}:
        relative = normalize_holography_diagonal_on_axis(artifact)
        relative_payload = first_recovery_artifact_dict(
            relative,
            physical=physical_diagonal_diagnostics(relative),
            include_samples=False,
        )
        relative_payload["beam_product"] = "relative_onaxis"
        relative_payload["restore_agreement"] = compare_absolute_and_restored_relative(
            artifact, relative
        )
        write_json(
            relative_payload,
            output_dir / f"first_beam_recovery_spw{spectral_window_id}_relative.json",
        )
        payload["relative_product"] = str(
            output_dir / f"first_beam_recovery_spw{spectral_window_id}_relative.json"
        )
        payload["restore_agreement"] = relative_payload["restore_agreement"]
        payload["notes"] = list(payload["notes"]) + [RELATIVE_ONAXIS_NORMALIZATION_NOTE]
    return payload


def _holdouts_from_samples(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    samples = samples_from_records(payload["samples"])
    artifact = artifact_from_samples(
        samples,
        calibration_state=str(payload.get("calibration_state", "casa_parang_true")),
        source_name=str(payload.get("source_name", "3C147")),
        reference_combination=str(
            payload.get("reference_combination", "per_reference_identity_gauge")
        ),
        notes=tuple(payload.get("notes", ())),
    )
    return {
        "schema": "thol0001_spw4_visibility_holdouts_v1",
        "samples_path": str(path),
        "n_samples": len(samples),
        "n_valid": int(sum(1 for sample in samples if sample.valid)),
        "frequency_hz": payload.get("frequency_hz"),
        "spectral_window_id": payload.get("spectral_window_id"),
        "channel": payload.get("channel"),
        "source": payload.get("source"),
        "visibility_holdouts": stratified_sample_holdouts(artifact, include_model_gauges=False),
        "notes": [
            "training-only interpolation; holdout cells never enter the neighbor cloud",
            "complex voltage residuals equal visibility residuals up to already-divided S_pq",
            "reuses the completed SPW-4 sample product; does not overwrite it",
        ],
    }


def _diagnostics_from_samples(
    path: Path,
    *,
    squint_bootstrap: bool = False,
    product: str = "both",
) -> dict:
    payload = json.loads(Path(path).read_text())
    samples = samples_from_records(payload["samples"])
    artifact = artifact_from_samples(
        samples,
        calibration_state=str(payload.get("calibration_state", "casa_parang_true")),
        source_name=str(payload.get("source_name", "3C147")),
        reference_combination=str(
            payload.get("reference_combination", "per_reference_identity_gauge")
        ),
        notes=tuple(payload.get("notes", ())),
    )
    report = first_recovery_artifact_dict(
        artifact,
        reference_report=reference_treatment_report(artifact),
        model_comparison=compare_models_at_measured_cells(artifact),
        physical=physical_diagonal_diagnostics(artifact),
        include_samples=False,
    )
    report["samples_path"] = str(path)
    report["frequency_hz"] = payload.get("frequency_hz")
    report["spectral_window_id"] = payload.get("spectral_window_id")
    report["channel"] = payload.get("channel")
    report["stage"] = "sample_diagnostics"
    report["beam_product"] = "absolute"
    report["visibility_holdouts"] = stratified_sample_holdouts(artifact)
    report["notes"] = list(report["notes"]) + [
        "sample holdouts interpolate from training cells only",
        "complex voltage residuals equal visibility residuals up to the already-divided S_pq",
        "model comparison uses unique measured cells, not every per-reference time sample",
        ABSOLUTE_RECOVERY_NOTE,
        RELATIVE_ONAXIS_NORMALIZATION_NOTE,
    ]
    if squint_bootstrap:
        report["squint_bootstrap"] = bootstrap_empirical_over_cassbeam_squint(artifact)
    if product in {"relative", "both"}:
        relative = normalize_holography_diagonal_on_axis(artifact)
        report["restore_agreement"] = compare_absolute_and_restored_relative(artifact, relative)
        report["relative_on_axis_n_valid"] = int(
            sum(1 for sample in relative.samples if sample.valid)
        )
    return report


def _holoraster_channel_block(
    tables,
    measurement_set: Path,
    *,
    data_desc_id: int,
    channel: int,
    data_column: str,
    spectral_window_id: int,
    channel_stop: int | None = None,
) -> tuple[VisibilityBlock, np.ndarray]:
    from sl1mjax.holography_ms import _read_field_phase_centre

    with tables.table(str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False) as window:
        frequencies = np.asarray(
            window.getcell("CHAN_FREQ", spectral_window_id), dtype=np.float64
        ).reshape(-1)
    first = int(channel)
    last = first if channel_stop is None else int(channel_stop) - 1
    if last < first or first < 0 or last >= frequencies.size:
        raise ValueError(f"channel range [{first}, {last + 1}) is outside SPW {spectral_window_id}")
    n_chan = last - first + 1
    from sl1mjax.evla_c_diagonal_survey import survey_holoraster_row_cache

    cache_path = survey_holoraster_row_cache(measurement_set, int(data_desc_id))
    parent = None
    selected_cm = None
    if cache_path is not None and cache_path.is_file():
        row_ids = np.load(cache_path)
        parent = tables.table(str(measurement_set), readonly=True, ack=False)
        selected_cm = parent.selectrows(np.asarray(row_ids, dtype=np.int64))
    if selected_cm is None:
        query = (
            f"SELECT FROM '{measurement_set}' "
            f"WHERE FIELD_ID={HOLORASTER_FIELD} AND DATA_DESC_ID={int(data_desc_id)}"
        )
        selected_cm = tables.taql(query)
    try:
        with selected_cm as selected:
            if selected.nrows() == 0:
                raise ValueError(f"no HOLORASTER rows for DATA_DESC_ID {data_desc_id}")
            columns = set(selected.colnames())
            if data_column not in columns:
                raise ValueError(f"{data_column} is not in {measurement_set}")
            if "MODEL_DATA" not in columns:
                raise ValueError(
                    "HOLORASTER recovery requires MODEL_DATA on the same rows as "
                    f"{data_column}; a scalar Stokes I is not S_pq"
                )
            visibility = np.asarray(selected.getcolslice(data_column, [first, 0], [last, -1]))
            model = np.asarray(selected.getcolslice("MODEL_DATA", [first, 0], [last, -1]))
            flag = np.asarray(selected.getcolslice("FLAG", [first, 0], [last, -1]), dtype=bool)
            flag_row = (
                np.asarray(selected.getcol("FLAG_ROW"), dtype=bool)
                if "FLAG_ROW" in columns
                else None
            )
            from sl1mjax.evla_c_survey_ms_contract import (
                combine_channel_and_row_flags,
                require_circular_correlation_order,
                select_row_or_spectrum_weight,
            )

            flag = combine_channel_and_row_flags(flag, flag_row)
            spectrum = None
            spectrum_defined = False
            if "WEIGHT_SPECTRUM" in columns and selected.iscelldefined("WEIGHT_SPECTRUM", 0):
                spectrum = np.asarray(
                    selected.getcolslice("WEIGHT_SPECTRUM", [first, 0], [last, -1]),
                    dtype=np.float64,
                )
                spectrum_defined = True
            weight = select_row_or_spectrum_weight(
                selected.getcol("WEIGHT"),
                weight_spectrum=spectrum,
                spectrum_defined=spectrum_defined,
                n_chan=n_chan,
            )
            corr_type = None
            with tables.table(
                str(measurement_set / "DATA_DESCRIPTION"), readonly=True, ack=False
            ) as description:
                pol_id = int(description.getcell("POLARIZATION_ID", int(data_desc_id)))
            with tables.table(
                str(measurement_set / "POLARIZATION"), readonly=True, ack=False
            ) as polarization:
                corr_type = [
                    int(code)
                    for code in np.asarray(polarization.getcell("CORR_TYPE", pol_id)).ravel()
                ]
            correlations = require_circular_correlation_order(
                (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
                corr_type=corr_type,
            )
            source = circular_visibility_to_source_coherency(model, correlations)
            block = VisibilityBlock(
                uvw_m=np.asarray(selected.getcol("UVW"), dtype=np.float64),
                frequency_hz=frequencies[first : last + 1],
                visibility=visibility,
                weight=weight,
                flag=flag,
                time_s=np.asarray(selected.getcol("TIME"), dtype=np.float64),
                antenna1=np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32),
                antenna2=np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32),
                field_id=np.asarray(selected.getcol("FIELD_ID"), dtype=np.int32),
                scan_id=np.asarray(selected.getcol("SCAN_NUMBER"), dtype=np.int32),
                correlations=correlations,
                receptor_basis=ReceptorBasis.CIRCULAR,
                phase_centre_rad=_read_field_phase_centre(tables, measurement_set, "HOLORASTER"),
                data_description_id=int(data_desc_id),
                spectral_window_id=int(spectral_window_id),
                provenance={
                    "source": str(measurement_set),
                    "column": data_column,
                    "model_column": "MODEL_DATA",
                    "holoraster_row_cache": str(cache_path) if cache_path is not None else None,
                },
            )
            if cache_path is not None and not cache_path.is_file():
                np.save(cache_path, np.asarray(selected.rownumbers(), dtype=np.int64))
    finally:
        if parent is not None:
            parent.close()
    return block, source


def _fluxd_provenance(
    setjy_record: Path | None,
    *,
    spectral_window_id: int,
    stokes_i: float | None,
) -> float | None:
    if stokes_i is not None:
        return float(stokes_i)
    if setjy_record is None or not Path(setjy_record).is_file():
        return None
    record = json.loads(Path(setjy_record).read_text())
    return casa_setjy_fluxd_jy(record, field_id=0, spectral_window_id=spectral_window_id)


def _model_rr_spread(source: np.ndarray) -> dict[str, float]:
    rr = np.abs(np.asarray(source)[:, 0, 0, 0])
    usable = rr[np.isfinite(rr)]
    if usable.size == 0:
        return {"n": 0, "median": float("nan"), "p16": float("nan"), "p84": float("nan")}
    median = float(np.median(usable))
    p16 = float(np.percentile(usable, 16))
    p84 = float(np.percentile(usable, 84))
    return {
        "n": int(usable.size),
        "median": median,
        "p16": p16,
        "p84": p84,
        "relative_spread": (p84 - p16) / median if median else float("nan"),
    }


def _subset_pointing(
    pointing: ResolvedAntennaPointing, time_s: np.ndarray
) -> ResolvedAntennaPointing:
    unique_times, _ = unique_visibility_times(time_s)
    index = {float(time): i for i, time in enumerate(pointing.unique_time_s)}
    keep = np.asarray([index[float(time)] for time in unique_times], dtype=np.int64)
    return ResolvedAntennaPointing(
        unique_time_s=pointing.unique_time_s[keep],
        antenna_id=pointing.antenna_id,
        offset_lm_rad=pointing.offset_lm_rad[keep],
        valid=pointing.valid[keep],
        settled=pointing.settled[keep],
        role=pointing.role[keep],
        selected_column=pointing.selected_column,
        offset_sign=pointing.offset_sign,
        join_rule=pointing.join_rule,
        join_tolerance_s=pointing.join_tolerance_s,
        measure_ref=pointing.measure_ref,
        units=pointing.units,
        notes=pointing.notes,
    )


def _antenna_positions(tables, measurement_set: Path) -> np.ndarray:
    with tables.table(str(measurement_set / "ANTENNA"), readonly=True, ack=False) as antenna:
        return np.asarray(antenna.getcol("POSITION"), dtype=np.float64)


def _ddid_for_spw(tables, measurement_set: Path, spectral_window_id: int) -> int:
    with tables.table(str(measurement_set / "DATA_DESCRIPTION"), readonly=True, ack=False) as table:
        windows = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
    matches = np.flatnonzero(windows == int(spectral_window_id))
    if matches.size != 1:
        raise ValueError(f"expected one DATA_DESC_ID for SPW {spectral_window_id}")
    return int(matches[0])


if __name__ == "__main__":
    raise SystemExit(main())
