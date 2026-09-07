"""Sealed C147-* offset-ring validation of high-resolution CASSBEAM.

Starts from DATA, applies the locked scientific chain once in memory, and
never writes back into the scientific MS or calibration tables. Channel 32
must pass before all 64 SPW-4 channels are opened. SPW 5 stays sealed
unless a model is selected with held-out cross-hand improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np

from sl1mjax.beam_operator import unique_visibility_times
from sl1mjax.calibration_terms import parallactic_angle_rad as parallactic_angle
from sl1mjax.cassbeam_highres import DEFAULT_HIGHRES_ROOT, HighresCassbeamCatalog
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.holography import (
    CASA_SETJY_3C147_C_IM_FLUXD_JY_SPW4,
    THOL0001_SPW4_CHANNEL_32_HZ,
    perley_butler_2017_3c147_stokes_i_jy,
)
from sl1mjax.holography_beam_prior import (
    beams_from_native_unique,
    bootstrap_complex_from_moments,
    contiguous_channel_block_masks,
    fit_frequency_smooth_alpha,
    fit_unit_and_scalar,
    paired_power_improves,
    sky_frame_residual_numpy,
    unique_native_jones,
)
from sl1mjax.holography_c147_offset_ring import (
    C147_OFFSET_RING,
    CHANNEL_HOLD_WIDTH,
    FULL_JONES_EXPERIMENTAL_NOTE,
    LOCKED_CONVENTION_NOTE,
    NATIVE_SPW4_CHANNELS,
    NO_HOLORASTER_NOTE,
    OFFSET_RING_NOTE,
    SMOKE_CHANNEL,
    FieldSkyRecord,
    SourceSkyRecord,
    antenna_power_share,
    channel32_smoke_gates,
    classify_c147_offset_ring,
    apply_channel_holdout,
    diagnose_directional_disagreement,
    clustered_null_upper_limit,
    cluster_ids_scan_baseline,
    declare_field_partitions,
    diagonal_rr_ll_closure,
    field_partition_masks,
    geometry_to_dict,
    locked_convention,
    moments_for_increment,
    paired_loss_by_group,
    partition_to_dict,
    predict_offset_field_visibilities,
    reconstruct_field_offsets,
    refuse_convention_search,
    scale_compatible,
    select_3c147_sky_direction,
    select_training_qu,
    source_relative_offsets,
    upper_limit_to_dict,
    write_offset_ring_plots,
)
from sl1mjax.holography_calibration import (
    C147_OFFSET_FIELD_IDS,
    hash_path,
    write_json,
)
from sl1mjax.holography_calibration_golden import (
    apply_imported_solution,
    import_thol0001_casa_tables,
)
from sl1mjax.holography_full_jones import _antenna_jones_planes
from sl1mjax.holography_highres_cassbeam import artifact_checksum_report, run_software_gates
from sl1mjax.holography_ms import _read_antennas, _tables, audit_holography_measurement_set
from sl1mjax.holography_reference_jones import deserialize_reference_jones
from sl1mjax.polarization import Correlation, Receptor, ReceptorBasis, circular_stokes_to_coherency, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
SCIENTIFIC_CAL = Path("/media/stephen/astro/vla/extracted/commissioning/products/scientific")
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
PRODUCT_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/c147_offset_ring"
)
FROZEN_NAMES = (
    "one_axis_visibility_holdouts.json",
    "one_axis_visibility_holdouts_v2.json",
    "forward_closure",
    "reference_visit_alignment",
    "loro_full_versus_diagonal",
    "loro_leakage_sensitivity",
    "highres_cassbeam_direct",
    "spw4_beam_prior",
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        if not (np.isfinite(number.real) and np.isfinite(number.imag)):
            return None
        return [number.real, number.imag]
    return value


def _source_revision() -> dict[str, object]:
    isolated = Path("/tmp/sl1mjax-pointing-audit/source_revision.json")
    if isolated.is_file():
        return json.loads(isolated.read_text())
    root = Path("/tmp/sl1mjax-pointing-audit/sl1mjax")
    if not root.is_dir():
        return {"status": "not_on_bacchus"}
    files = sorted(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return {
        "tree": str(root),
        "n_py": len(files),
        "tree_sha256": digest.hexdigest(),
    }


def _scientific_tables(root: Path) -> dict[str, Path]:
    return {
        "antpos": root / "diagonal" / "antpos.cal",
        "G1": root / "diagonal" / "G1.cal",
        "K0": root / "diagonal" / "K0.cal",
        "B0": root / "diagonal" / "B0.cal",
        "Kcross": root / "fullpol" / "Kcross.cal",
        "Df": root / "fullpol" / "Df.cal",
        "Xf": root / "fullpol" / "Xf.cal",
    }


def _residual_jones(product_dir: Path) -> dict[int, np.ndarray]:
    family = product_dir / "prediction_equivalent_family.json"
    stabilized = product_dir / "field9_stabilized_residual_jones.json"
    if family.is_file():
        return deserialize_reference_jones(json.loads(family.read_text())["nominal"]["jones"])
    if stabilized.is_file():
        return deserialize_reference_jones(json.loads(stabilized.read_text())["jones"])
    raise FileNotFoundError("need field9_stabilized_residual_jones.json or the family cache")


def _read_fields_and_sources(tables, measurement_set: Path):
    fields: list[FieldSkyRecord] = []
    with tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        for row in range(table.nrows()):
            direction = np.asarray(table.getcell("PHASE_DIR", row), dtype=np.float64).reshape(-1, 2)
            fields.append(
                FieldSkyRecord(
                    field_id=int(row),
                    name=str(table.getcell("NAME", row)),
                    phase_centre_rad=(float(direction[0, 0]), float(direction[0, 1])),
                    code=str(table.getcell("CODE", row)) if "CODE" in columns else "",
                )
            )
    sources: list[SourceSkyRecord] = []
    source = measurement_set / "SOURCE"
    if source.exists():
        with tables.table(str(source), readonly=True, ack=False) as table:
            columns = set(table.colnames())
            for row in range(table.nrows()):
                if "DIRECTION" not in columns:
                    continue
                direction = np.asarray(table.getcell("DIRECTION", row), dtype=np.float64).reshape(
                    -1, 2
                )
                sources.append(
                    SourceSkyRecord(
                        name=str(table.getcell("NAME", row)) if "NAME" in columns else "",
                        direction_rad=(float(direction[0, 0]), float(direction[0, 1])),
                        field_id=int(table.getcell("SOURCE_ID", row))
                        if "SOURCE_ID" in columns
                        else None,
                    )
                )
    return fields, sources


def _ddid_for_spw(tables, measurement_set: Path, spectral_window_id: int) -> int:
    with tables.table(str(measurement_set / "DATA_DESCRIPTION"), readonly=True, ack=False) as table:
        windows = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
    matches = np.flatnonzero(windows == int(spectral_window_id))
    if matches.size != 1:
        raise ValueError(f"expected one DATA_DESC_ID for SPW {spectral_window_id}")
    return int(matches[0])


def _load_offset_block(
    tables,
    measurement_set: Path,
    *,
    field_id: int,
    phase_centre_rad: tuple[float, float],
    data_desc_id: int,
    spectral_window_id: int,
    channel: int,
    channel_stop: int,
) -> VisibilityBlock:
    with tables.table(str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False) as window:
        frequencies = np.asarray(
            window.getcell("CHAN_FREQ", spectral_window_id), dtype=np.float64
        ).reshape(-1)
    first = int(channel)
    last = int(channel_stop) - 1
    if last < first or first < 0 or last >= frequencies.size:
        raise ValueError(f"channel range [{first}, {last + 1}) is outside SPW {spectral_window_id}")
    n_chan = last - first + 1
    query = (
        f"SELECT FROM '{measurement_set}' "
        f"WHERE FIELD_ID={int(field_id)} AND DATA_DESC_ID={int(data_desc_id)}"
    )
    with tables.taql(query) as selected:
        if selected.nrows() == 0:
            raise ValueError(f"no rows for field {field_id} DATA_DESC_ID {data_desc_id}")
        columns = set(selected.colnames())
        if "DATA" not in columns:
            raise ValueError(f"DATA is not in {measurement_set}")
        visibility = np.asarray(selected.getcolslice("DATA", [first, 0], [last, -1]))
        flag = np.asarray(selected.getcolslice("FLAG", [first, 0], [last, -1]), dtype=bool)
        weight = np.repeat(
            np.asarray(selected.getcol("WEIGHT"), dtype=np.float64)[:, None, :],
            n_chan,
            axis=1,
        )
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
            correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
            receptor_basis=ReceptorBasis.CIRCULAR,
            phase_centre_rad=phase_centre_rad,
            data_description_id=int(data_desc_id),
            spectral_window_id=int(spectral_window_id),
            provenance={
                "source": str(measurement_set),
                "column": "DATA",
                "model_column": None,
            },
        )
    return block


def _setjy_source(frequencies: np.ndarray, q_over_i: float, u_over_i: float, n_row: int):
    table = perley_butler_2017_3c147_stokes_i_jy(frequencies)
    ref = float(perley_butler_2017_3c147_stokes_i_jy(np.array([THOL0001_SPW4_CHANNEL_32_HZ]))[0])
    intensity = table * (CASA_SETJY_3C147_C_IM_FLUXD_JY_SPW4 / ref)
    sky = circular_stokes_to_coherency(
        intensity,
        float(q_over_i) * intensity,
        float(u_over_i) * intensity,
        0.0 * intensity,
    )
    return np.broadcast_to(sky, (n_row, frequencies.size, 2, 2)).copy()


def _pointing_offsets(audit, time_s: np.ndarray, antenna: np.ndarray) -> np.ndarray:
    unique_times, inverse = unique_visibility_times(time_s)
    index = {float(time): i for i, time in enumerate(audit.resolved.unique_time_s)}
    keep = np.asarray([index[float(time)] for time in unique_times], dtype=np.int64)
    offsets = np.asarray(audit.resolved.offset_lm_rad, dtype=np.float64)[keep]
    valid = np.asarray(audit.resolved.valid, dtype=bool)[keep]
    chosen = offsets[inverse, np.asarray(antenna, dtype=np.int32)]
    ok = valid[inverse, np.asarray(antenna, dtype=np.int32)]
    return np.where(ok[:, None], chosen, 0.0)


def _chi(time_s, phase_centre, positions, antenna) -> np.ndarray:
    unique_times, inverse = unique_visibility_times(time_s)
    angles = parallactic_angle(unique_times, phase_centre, positions)
    return angles[inverse, np.asarray(antenna, dtype=np.int32)]


def _predict(
    catalog,
    frequencies,
    offset_p,
    offset_q,
    chi_p,
    chi_q,
    residual,
    ant1,
    ant2,
    source,
    offdiag,
    uvw_m,
    sky_lm,
):
    convention = locked_convention()
    refuse_convention_search(None)
    native_p, valid_p, inv_p = unique_native_jones(
        offset_p, chi_p, frequencies, catalog, convention
    )
    native_q, valid_q, inv_q = unique_native_jones(
        offset_q, chi_q, frequencies, catalog, convention
    )
    beam_p, ok_p = beams_from_native_unique(
        native_p,
        valid_p,
        inv_p,
        convention,
        catalog=catalog,
        frequencies_hz=frequencies,
        chi=chi_p,
        off_diagonal=offdiag,
        calibration_state="casa_parang_true",
    )
    beam_q, ok_q = beams_from_native_unique(
        native_q,
        valid_q,
        inv_q,
        convention,
        catalog=catalog,
        frequencies_hz=frequencies,
        chi=chi_q,
        off_diagonal=offdiag,
        calibration_state="casa_parang_true",
    )
    r_p = sky_frame_residual_numpy(_antenna_jones_planes(residual, ant1), chi_p)
    r_q = sky_frame_residual_numpy(_antenna_jones_planes(residual, ant2), chi_q)
    pred = predict_offset_field_visibilities(
        r_p,
        beam_p,
        source,
        beam_q,
        r_q,
        uvw_m=uvw_m,
        frequency_hz=frequencies,
        sky_lm_rad=sky_lm,
    )
    ok = ok_p & ok_q
    return np.where(ok[..., None, None], pred, np.nan)


def _hand_weight(flag, weight, n_row, n_chan):
    wgt = np.asarray(weight, dtype=np.float64)
    flagged = np.asarray(flag, dtype=bool)
    if wgt.ndim == 2:
        wgt = np.repeat(wgt[:, None, :], n_chan, axis=1)
        flagged = np.repeat(flagged[:, None, :], n_chan, axis=1)
    active = (~flagged) & np.isfinite(wgt) & (wgt > 0.0)
    out = np.zeros((n_row, n_chan, 2, 2), dtype=np.float64)
    out[:, :, 0, 0] = np.where(active[:, :, 0], wgt[:, :, 0], 0.0)
    out[:, :, 0, 1] = np.where(active[:, :, 1], wgt[:, :, 1], 0.0)
    out[:, :, 1, 0] = np.where(active[:, :, 2], wgt[:, :, 2], 0.0)
    out[:, :, 1, 1] = np.where(active[:, :, 3], wgt[:, :, 3], 0.0)
    return out


def _moments(template, residual, weight, field_id, scan, ant1, ant2, hands):
    return moments_for_increment(
        template,
        residual,
        weight,
        cluster_ids=cluster_ids_scan_baseline(scan, ant1, ant2),
        channel_ids=None,
        field_ids=field_id,
        antenna1=ant1,
        antenna2=ant2,
        hands=hands,
    )


def _readme(payload: dict) -> str:
    gate = payload.get("decision_gate")
    if not isinstance(gate, dict):
        gate = payload
    limit = payload.get("upper_limit") if isinstance(payload.get("upper_limit"), dict) else {}
    lines = [
        "# THOL0001 C147-* offset-ring high-resolution CASSBEAM validation",
        "",
        f"Status: **{gate.get('status', payload.get('status', 'unknown'))}**. "
        f"Decision: `{gate.get('decision', payload.get('decision', 'unknown'))}`.",
        "",
        OFFSET_RING_NOTE,
        "",
        LOCKED_CONVENTION_NOTE,
        "",
        FULL_JONES_EXPERIMENTAL_NOTE,
        "",
        NO_HOLORASTER_NOTE,
        "",
        f"Selected model: {gate.get('selected_model')}.",
        f"SPW 5 closed: {gate.get('spw5_closed', True)}.",
        "",
    ]
    if limit:
        lines.extend(
            [
                "## Upper limit",
                "",
                f"- status: `{limit.get('status')}`",
                f"- |α| hat: {limit.get('alpha_abs_hat')}",
                f"- 95% |α| UL: {limit.get('alpha_abs_ul95')}",
                f"- clustered null 95%: {limit.get('null_abs_95')}",
                f"- coherent voltage UL: {limit.get('coherent_voltage_ul95')}",
                f"- clusters: {limit.get('n_clusters')}",
                "",
            ]
        )
    lines.append("Full Jones remains unfrozen. The production factory is unmodified.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--cal-root", type=Path, default=SCIENTIFIC_CAL)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--output-dir", type=Path, default=PRODUCT_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument(
        "--stage",
        choices=("geometry", "channel32", "spw4", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    if int(arguments.spectral_window) not in {4, 5}:
        raise ValueError("only SPW 4 or a later sealed SPW-5 replica is defined")
    if int(arguments.spectral_window) == 5:
        raise ValueError("SPW 5 stays sealed until SPW 4 selects a held-out model")
    forbidden = {arguments.product_dir.resolve()}
    forbidden.update((arguments.product_dir / name).resolve() for name in FROZEN_NAMES)
    forbidden.add(arguments.measurement_set.resolve())
    forbidden.add(arguments.cal_root.resolve())
    if arguments.output_dir.resolve() in forbidden:
        raise ValueError("refusing to overwrite scientific MS, cal tables, or frozen products")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "thol0001_c147_offset_ring_v1",
        "gate_name": C147_OFFSET_RING,
        "status": "fail",
        "blocking": True,
        "spectral_window_id": 4,
        "full_jones_frozen": False,
        "production_factory_modified": False,
        "spw5_closed": True,
        "c147_offset_used_for_d": False,
        "applied_from": "DATA",
        "scientific_tables_altered": False,
        "convention": locked_convention().name,
        "convention_ladder_searched": False,
        "source_revision": _source_revision(),
        "notes": (
            OFFSET_RING_NOTE,
            LOCKED_CONVENTION_NOTE,
            FULL_JONES_EXPERIMENTAL_NOTE,
            NO_HOLORASTER_NOTE,
        ),
    }
    try:
        catalog = HighresCassbeamCatalog(arguments.artifact_root)
        plane = catalog.plane(THOL0001_SPW4_CHANNEL_32_HZ)
        software = run_software_gates(plane)
        checksums = artifact_checksum_report(catalog, THOL0001_SPW4_CHANNEL_32_HZ)
        write_json(_jsonable({"software": software, "checksums": checksums}), arguments.output_dir / "artifact.json")
        tables = _tables()
        _ids, names, positions = _read_antennas(tables, arguments.measurement_set)
        positions = np.asarray(positions, dtype=np.float64)
        fields, sources = _read_fields_and_sources(tables, arguments.measurement_set)
        source_radec, source_from = select_3c147_sky_direction(fields, sources)
        geometries = reconstruct_field_offsets(fields, source_radec, source_from=source_from)
        partition = declare_field_partitions(geometries)
        geometry_payload = {
            "source_radec_rad": list(source_radec),
            "source_from": source_from,
            "fields": [geometry_to_dict(item) for item in geometries],
            "field_names_not_used_for_offsets": True,
            "partition": partition_to_dict(partition),
            "channel_holdout": contiguous_channel_block_masks(
                NATIVE_SPW4_CHANNELS, hold_width=CHANNEL_HOLD_WIDTH
            ),
        }
        write_json(_jsonable(geometry_payload), arguments.output_dir / "geometry.json")
        write_json(_jsonable(partition_to_dict(partition)), arguments.output_dir / "partitions.json")
        if arguments.stage == "geometry":
            payload["status"] = "geometry"
            payload["geometry"] = geometry_payload
            write_json(_jsonable(payload), arguments.output_dir / "report.json")
            return 0

        cal_tables = _scientific_tables(arguments.cal_root)
        missing = [key for key, path in cal_tables.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"scientific tables missing: {missing}")
        cal_hashes = {key: hash_path(path) for key, path in cal_tables.items()}
        write_json(
            {
                "tables": {key: {"path": str(path), "sha256": cal_hashes[key]} for key, path in cal_tables.items()},
                "applied_from": "DATA",
                "apply_count": 1,
                "wrote_corrected_data": False,
                "scientific_ms_modified": False,
            },
            arguments.output_dir / "calibration_state.json",
        )
        residual = _residual_jones(arguments.product_dir)
        solution = import_thol0001_casa_tables(
            cal_tables,
            measurement_set=arguments.measurement_set,
            spectral_window_id=4,
            product="fullpol",
        )
        ddid = _ddid_for_spw(tables, arguments.measurement_set, 4)
        smoke_path = arguments.output_dir / "channel32_smoke.json"
        resume_smoke = False
        if arguments.stage in {"spw4", "all"} and smoke_path.is_file():
            prior_smoke = json.loads(smoke_path.read_text())
            if bool((prior_smoke.get("smoke") or {}).get("continue_to_spw4")):
                resume_smoke = True
                smoke = prior_smoke["smoke"]
                closure = prior_smoke["closure"]
                row_accounting = prior_smoke.get("row_accounting") or {}
        smoke_channel = SMOKE_CHANNEL
        field_rows = []
        for geom in geometries if not resume_smoke else ():
            try:
                audit = audit_holography_measurement_set(
                    arguments.measurement_set, field_name=geom.name
                )
            except Exception:
                audit = None
            raw = _load_offset_block(
                tables,
                arguments.measurement_set,
                field_id=geom.field_id,
                phase_centre_rad=geom.phase_centre_rad,
                data_desc_id=ddid,
                spectral_window_id=4,
                channel=smoke_channel,
                channel_stop=smoke_channel + 1,
            )
            if raw.provenance.get("column") != "DATA":
                raise ValueError("offset-ring apply must start from DATA")
            applied = apply_imported_solution(raw, solution)
            packed = pack_coherency(applied.visibility, applied.correlations, (Receptor.R, Receptor.L))
            sky_lm = np.broadcast_to(
                np.array([[geom.l_rad, geom.m_rad]], dtype=np.float64),
                (applied.time_s.size, 2),
            ).copy()
            if audit is None:
                delta_p = np.zeros((applied.time_s.size, 2), dtype=np.float64)
                delta_q = np.zeros((applied.time_s.size, 2), dtype=np.float64)
            else:
                delta_p = _pointing_offsets(audit, applied.time_s, applied.antenna1)
                delta_q = _pointing_offsets(audit, applied.time_s, applied.antenna2)
            field_rows.append(
                {
                    "field_id": np.full(applied.time_s.size, geom.field_id, dtype=np.int32),
                    "scan": np.asarray(applied.scan_id, dtype=np.int32),
                    "antenna1": np.asarray(applied.antenna1, dtype=np.int32),
                    "antenna2": np.asarray(applied.antenna2, dtype=np.int32),
                    "time_s": np.asarray(applied.time_s, dtype=np.float64),
                    "visibility": packed,
                    "weight": _hand_weight(
                        applied.flag, applied.weight, applied.time_s.size, packed.shape[1]
                    ),
                    "offset_p": source_relative_offsets(sky_lm, delta_p),
                    "offset_q": source_relative_offsets(sky_lm, delta_q),
                    "sky_lm": sky_lm,
                    "uvw_m": np.asarray(applied.uvw_m, dtype=np.float64),
                    "chi_p": _chi(applied.time_s, geom.phase_centre_rad, positions, applied.antenna1),
                    "chi_q": _chi(applied.time_s, geom.phase_centre_rad, positions, applied.antenna2),
                    "frequencies": np.asarray(applied.frequency_hz, dtype=np.float64),
                    "pointing_abs_p": float(np.nanmedian(np.hypot(delta_p[:, 0], delta_p[:, 1]))),
                    "pointing_abs_q": float(np.nanmedian(np.hypot(delta_q[:, 0], delta_q[:, 1]))),
                }
            )
        if not resume_smoke:
            stacked = {key: np.concatenate([item[key] for item in field_rows], axis=0) for key in (
            "field_id",
            "scan",
            "antenna1",
            "antenna2",
            "time_s",
            "visibility",
            "weight",
            "offset_p",
            "offset_q",
            "sky_lm",
            "uvw_m",
            "chi_p",
            "chi_q",
        )}
            frequencies = field_rows[0]["frequencies"]
            masks = field_partition_masks(stacked["field_id"], partition)
            source0 = _setjy_source(frequencies, 0.0, 0.0, stacked["visibility"].shape[0])
            pred_diag = _predict(
                catalog,
                frequencies,
                stacked["offset_p"],
                stacked["offset_q"],
                stacked["chi_p"],
                stacked["chi_q"],
                residual,
                stacked["antenna1"],
                stacked["antenna2"],
                source0,
                False,
                stacked["uvw_m"],
                stacked["sky_lm"],
            )
            pred_full = _predict(
                catalog,
                frequencies,
                stacked["offset_p"],
                stacked["offset_q"],
                stacked["chi_p"],
                stacked["chi_q"],
                residual,
                stacked["antenna1"],
                stacked["antenna2"],
                source0,
                True,
                stacked["uvw_m"],
                stacked["sky_lm"],
            )
            finite = bool(np.isfinite(pred_diag).any() and np.isfinite(pred_full).any())
            closure = diagonal_rr_ll_closure(
                stacked["visibility"][masks["training"]],
                pred_diag[masks["training"]],
                stacked["weight"][masks["training"]],
            )
            smoke = channel32_smoke_gates(
                geometry_ok=True,
                calibration_hashed=True,
                predictions_finite=finite,
                diagonal_closure_ok=bool(closure["passed"]),
            )
            row_accounting = {
                "n_rows": int(stacked["field_id"].size),
                "by_field": {
                    str(int(ident)): int(np.sum(stacked["field_id"] == ident))
                    for ident in np.unique(stacked["field_id"])
                },
                "training": int(np.sum(masks["training"])),
                "inner_holdout": int(np.sum(masks["inner_holdout"])),
                "sealed_holdout": int(np.sum(masks["sealed_holdout"])),
                "pointing_median_abs_rad": {
                    str(int(item["field_id"][0])): {
                        "p": item["pointing_abs_p"],
                        "q": item["pointing_abs_q"],
                    }
                    for item in field_rows
                },
            }
            write_json(
                _jsonable(
                    diagnose_directional_disagreement(
                        stacked["visibility"],
                        pred_full,
                        pred_diag,
                        stacked["weight"],
                        stacked["sky_lm"],
                    )
                ),
                arguments.output_dir / "directional_diagnosis.json",
            )
            write_json(
                _jsonable(
                    {
                        "smoke": smoke,
                        "closure": closure,
                        "row_accounting": row_accounting,
                        "median_abs": {
                            "measured_rr": float(np.nanmedian(np.abs(stacked["visibility"][:, 0, 0, 0]))),
                            "measured_ll": float(np.nanmedian(np.abs(stacked["visibility"][:, 0, 1, 1]))),
                            "pred_diag_rr": float(np.nanmedian(np.abs(pred_diag[:, 0, 0, 0]))),
                            "pred_diag_ll": float(np.nanmedian(np.abs(pred_diag[:, 0, 1, 1]))),
                        },
                    }
                ),
                arguments.output_dir / "channel32_smoke.json",
            )
        if not smoke["passed"]:
            gate = classify_c147_offset_ring(
                software_ok=bool(software.get("passed", True)),
                split_clean=True,
                smoke_ok=False,
                rr_ll_regression=False,
                inner_rl_improves=False,
                inner_lr_improves=False,
                sealed_rl_improves=False,
                sealed_lr_improves=False,
                scale_ok=False,
                antenna_ok=False,
                unit_better_than_diag=False,
                scaled_better_than_unit=False,
                upper_limit=None,
            )
            payload.update(gate)
            payload["decision_gate"] = gate
            payload["smoke"] = smoke
            payload["closure"] = closure
            write_json(_jsonable(payload), arguments.output_dir / "decision.json")
            write_json(_jsonable(payload), arguments.output_dir / "report.json")
            (arguments.output_dir / "README.md").write_text(_readme(payload), encoding="utf-8")
            return 1
        if arguments.stage == "channel32":
            payload["status"] = "channel32_passed"
            payload["smoke"] = smoke
            payload["row_accounting"] = row_accounting
            write_json(_jsonable(payload), arguments.output_dir / "report.json")
            return 0

        raw_fields = []
        for geom in geometries:
            try:
                audit = audit_holography_measurement_set(
                    arguments.measurement_set, field_name=geom.name
                )
            except Exception:
                audit = None
            raw = _load_offset_block(
                tables,
                arguments.measurement_set,
                field_id=geom.field_id,
                phase_centre_rad=geom.phase_centre_rad,
                data_desc_id=ddid,
                spectral_window_id=4,
                channel=0,
                channel_stop=NATIVE_SPW4_CHANNELS,
            )
            applied = apply_imported_solution(raw, solution)
            packed = pack_coherency(applied.visibility, applied.correlations, (Receptor.R, Receptor.L))
            sky_lm = np.broadcast_to(
                np.array([[geom.l_rad, geom.m_rad]], dtype=np.float64),
                (applied.time_s.size, 2),
            ).copy()
            if audit is None:
                delta_p = np.zeros((applied.time_s.size, 2), dtype=np.float64)
                delta_q = np.zeros((applied.time_s.size, 2), dtype=np.float64)
            else:
                delta_p = _pointing_offsets(audit, applied.time_s, applied.antenna1)
                delta_q = _pointing_offsets(audit, applied.time_s, applied.antenna2)
            raw_fields.append(
                {
                    "field_id": np.full(applied.time_s.size, geom.field_id, dtype=np.int32),
                    "scan": np.asarray(applied.scan_id, dtype=np.int32),
                    "antenna1": np.asarray(applied.antenna1, dtype=np.int32),
                    "antenna2": np.asarray(applied.antenna2, dtype=np.int32),
                    "visibility": packed,
                    "weight": _hand_weight(
                        applied.flag, applied.weight, applied.time_s.size, packed.shape[1]
                    ),
                    "offset_p": source_relative_offsets(sky_lm, delta_p),
                    "offset_q": source_relative_offsets(sky_lm, delta_q),
                    "sky_lm": sky_lm,
                    "uvw_m": np.asarray(applied.uvw_m, dtype=np.float64),
                    "chi_p": _chi(applied.time_s, geom.phase_centre_rad, positions, applied.antenna1),
                    "chi_q": _chi(applied.time_s, geom.phase_centre_rad, positions, applied.antenna2),
                    "frequencies": np.asarray(applied.frequency_hz, dtype=np.float64),
                }
            )
        stacked = {key: np.concatenate([item[key] for item in raw_fields], axis=0) for key in (
            "field_id",
            "scan",
            "antenna1",
            "antenna2",
            "visibility",
            "weight",
            "offset_p",
            "offset_q",
            "sky_lm",
            "uvw_m",
            "chi_p",
            "chi_q",
        )}
        frequencies = raw_fields[0]["frequencies"]
        if frequencies.size != NATIVE_SPW4_CHANNELS:
            raise ValueError(f"expected {NATIVE_SPW4_CHANNELS} native channels, got {frequencies.size}")
        masks = field_partition_masks(stacked["field_id"], partition)
        channel_masks = contiguous_channel_block_masks(
            frequencies.size, hold_width=CHANNEL_HOLD_WIDTH
        )
        qu_scores = {}
        train = masks["training"]
        for q_over_i, u_over_i in (
            (0.0, 0.0),
            (-0.001, 0.0),
            (0.001, 0.0),
            (0.0, -0.001),
            (0.0, 0.001),
        ):
            source = _setjy_source(frequencies, q_over_i, u_over_i, stacked["visibility"].shape[0])
            pred = _predict(
                catalog,
                frequencies,
                stacked["offset_p"],
                stacked["offset_q"],
                stacked["chi_p"],
                stacked["chi_q"],
                residual,
                stacked["antenna1"],
                stacked["antenna2"],
                source,
                False,
                stacked["uvw_m"],
                stacked["sky_lm"],
            )
            closure_q = diagonal_rr_ll_closure(
                apply_channel_holdout(stacked["visibility"][train], channel_masks["train"]),
                apply_channel_holdout(pred[train], channel_masks["train"]),
                apply_channel_holdout(stacked["weight"][train], channel_masks["train"]),
            )
            qu_scores[(q_over_i, u_over_i)] = float(
                closure_q["rr"]["relative_power"] + closure_q["ll"]["relative_power"]
            )
        q_over_i, u_over_i = select_training_qu(qu_scores)
        source = _setjy_source(frequencies, q_over_i, u_over_i, stacked["visibility"].shape[0])
        pred_diag = _predict(
            catalog,
            frequencies,
            stacked["offset_p"],
            stacked["offset_q"],
            stacked["chi_p"],
            stacked["chi_q"],
            residual,
            stacked["antenna1"],
            stacked["antenna2"],
            source,
            False,
            stacked["uvw_m"],
            stacked["sky_lm"],
        )
        pred_full = _predict(
            catalog,
            frequencies,
            stacked["offset_p"],
            stacked["offset_q"],
            stacked["chi_p"],
            stacked["chi_q"],
            residual,
            stacked["antenna1"],
            stacked["antenna2"],
            source,
            True,
            stacked["uvw_m"],
            stacked["sky_lm"],
        )
        increment = pred_full - pred_diag
        residual_vis = stacked["visibility"] - pred_diag
        train_moments = _moments(
            increment[train],
            residual_vis[train],
            stacked["weight"][train],
            stacked["field_id"][train],
            stacked["scan"][train],
            stacked["antenna1"][train],
            stacked["antenna2"][train],
            ("rl", "lr"),
        )
        inner = masks["inner_holdout"]
        sealed = masks["sealed_holdout"]
        inner_x = _moments(
            increment[inner],
            residual_vis[inner],
            stacked["weight"][inner],
            stacked["field_id"][inner],
            stacked["scan"][inner],
            stacked["antenna1"][inner],
            stacked["antenna2"][inner],
            ("rl", "lr"),
        )
        sealed_x = _moments(
            increment[sealed],
            residual_vis[sealed],
            stacked["weight"][sealed],
            stacked["field_id"][sealed],
            stacked["scan"][sealed],
            stacked["antenna1"][sealed],
            stacked["antenna2"][sealed],
            ("rl", "lr"),
        )
        inner_rrll = _moments(
            increment[inner],
            residual_vis[inner],
            stacked["weight"][inner],
            stacked["field_id"][inner],
            stacked["scan"][inner],
            stacked["antenna1"][inner],
            stacked["antenna2"][inner],
            ("rr", "ll"),
        )
        scalar = fit_unit_and_scalar(train_moments)
        smooth = fit_frequency_smooth_alpha(train_moments)
        alpha_unit = 1.0 + 0.0j
        alpha_hat = complex(scalar["alpha_hat"])
        alpha_smooth = np.asarray(smooth["alpha_channel"], dtype=np.complex128)
        alpha_smooth_mean = complex(np.nanmean(alpha_smooth)) if alpha_smooth.size else alpha_hat
        inner_unit = paired_power_improves(inner_x, alpha_unit)
        inner_hat = paired_power_improves(inner_x, alpha_hat)
        sealed_unit = paired_power_improves(sealed_x, alpha_unit)
        sealed_hat = paired_power_improves(sealed_x, alpha_smooth_mean)
        rrll_unit = paired_power_improves(inner_rrll, alpha_unit)
        rr_ll_regression = bool(rrll_unit["power_full"] > rrll_unit["power_diag"] * (1.0 + 0.02))
        field_alphas = []
        for field_id in partition.training:
            mask = train & (stacked["field_id"] == field_id)
            moments = _moments(
                increment[mask],
                residual_vis[mask],
                stacked["weight"][mask],
                stacked["field_id"][mask],
                stacked["scan"][mask],
                stacked["antenna1"][mask],
                stacked["antenna2"][mask],
                ("rl", "lr"),
            )
            field_alphas.append(complex(fit_unit_and_scalar(moments)["alpha_hat"]))
        # Channel compatibility uses training-field rows; reserved channels stay scored, not fit.
        train_chan_moments = [
            item
            for item in train_moments
            if 0 <= int(item.channel) < frequencies.size and not channel_masks["holdout"][int(item.channel)]
        ]
        hold_chan_moments = [
            item
            for item in train_moments
            if 0 <= int(item.channel) < frequencies.size and channel_masks["holdout"][int(item.channel)]
        ]
        chan_alphas = []
        if train_chan_moments:
            chan_alphas.append(complex(fit_unit_and_scalar(train_chan_moments)["alpha_hat"]))
        hold_chan_alpha = (
            complex(fit_unit_and_scalar(hold_chan_moments)["alpha_hat"])
            if hold_chan_moments
            else None
        )
        scale_ok = bool(scale_compatible(field_alphas)["compatible"])
        payload["channel_holdout"] = {
            "hold_start": int(channel_masks["hold_start"]),
            "hold_stop": int(channel_masks["hold_stop"]),
            "used_in_qu_or_scale_fit": False,
            "hold_channel_alpha_scored_only": (
                [hold_chan_alpha.real, hold_chan_alpha.imag] if hold_chan_alpha is not None else None
            ),
        }
        antenna = antenna_power_share(inner_x)
        template_abs = np.abs(increment[inner][:, :, 0, 1])
        template_abs = np.concatenate(
            [template_abs.reshape(-1), np.abs(increment[inner][:, :, 1, 0]).reshape(-1)]
        )
        template_median = float(np.nanmedian(template_abs[np.isfinite(template_abs)]))
        limit = clustered_null_upper_limit(
            inner_x + sealed_x, template_median_abs=template_median, n_perm=400, seed=17
        )
        holdout_alpha = bootstrap_complex_from_moments(inner_x + sealed_x, n_boot=400, seed=19)
        inner_rl_only = _moments(
            increment[inner],
            residual_vis[inner],
            stacked["weight"][inner],
            stacked["field_id"][inner],
            stacked["scan"][inner],
            stacked["antenna1"][inner],
            stacked["antenna2"][inner],
            ("rl",),
        )
        inner_lr_only = _moments(
            increment[inner],
            residual_vis[inner],
            stacked["weight"][inner],
            stacked["field_id"][inner],
            stacked["scan"][inner],
            stacked["antenna1"][inner],
            stacked["antenna2"][inner],
            ("lr",),
        )
        sealed_rl_only = _moments(
            increment[sealed],
            residual_vis[sealed],
            stacked["weight"][sealed],
            stacked["field_id"][sealed],
            stacked["scan"][sealed],
            stacked["antenna1"][sealed],
            stacked["antenna2"][sealed],
            ("rl",),
        )
        sealed_lr_only = _moments(
            increment[sealed],
            residual_vis[sealed],
            stacked["weight"][sealed],
            stacked["field_id"][sealed],
            stacked["scan"][sealed],
            stacked["antenna1"][sealed],
            stacked["antenna2"][sealed],
            ("lr",),
        )
        chosen_alpha = alpha_smooth_mean if inner_hat["improves"] and not inner_unit["improves"] else alpha_unit
        gate = classify_c147_offset_ring(
            software_ok=bool(software.get("passed", True)),
            split_clean=True,
            smoke_ok=True,
            rr_ll_regression=rr_ll_regression,
            inner_rl_improves=bool(paired_power_improves(inner_rl_only, chosen_alpha)["improves"]),
            inner_lr_improves=bool(paired_power_improves(inner_lr_only, chosen_alpha)["improves"]),
            sealed_rl_improves=bool(paired_power_improves(sealed_rl_only, chosen_alpha)["improves"]),
            sealed_lr_improves=bool(paired_power_improves(sealed_lr_only, chosen_alpha)["improves"]),
            scale_ok=scale_ok,
            antenna_ok=bool(antenna["ok"]),
            unit_better_than_diag=bool(inner_unit["improves"]),
            scaled_better_than_unit=bool(inner_hat["improves"] and not inner_unit["improves"]),
            upper_limit=limit,
        )
        losses = {
            "field": paired_loss_by_group(inner_x + sealed_x, chosen_alpha, group="field"),
            "channel": paired_loss_by_group(inner_x + sealed_x, chosen_alpha, group="channel"),
            "cluster": paired_loss_by_group(inner_x + sealed_x, chosen_alpha, group="cluster"),
        }
        payload.update(gate)
        payload["decision_gate"] = gate
        payload.update(
            {
                "geometry": geometry_payload,
                "smoke": smoke,
                "row_accounting": {
                    "n_rows": int(stacked["field_id"].size),
                    "n_channels": int(frequencies.size),
                    "training": int(np.sum(train)),
                    "inner_holdout": int(np.sum(inner)),
                    "sealed_holdout": int(np.sum(sealed)),
                },
                "nuisance": {
                    "q_over_i": q_over_i,
                    "u_over_i": u_over_i,
                    "qu_scores": {f"{q},{u}": score for (q, u), score in qu_scores.items()},
                    "residual_jones": "field9_locked_not_refit",
                    "alpha_unit": [1.0, 0.0],
                    "alpha_hat": [alpha_hat.real, alpha_hat.imag],
                    "alpha_smooth_mean": [alpha_smooth_mean.real, alpha_smooth_mean.imag],
                    "field_alphas": [[c.real, c.imag] for c in field_alphas],
                    "channel_block_alphas": [[c.real, c.imag] for c in chan_alphas],
                    "scale_compatible": scale_compatible(field_alphas),
                },
                "scores": {
                    "inner_unit": inner_unit,
                    "inner_hat": inner_hat,
                    "sealed_unit": sealed_unit,
                    "sealed_hat": sealed_hat,
                    "rrll_unit": rrll_unit,
                    "holdout_alpha": holdout_alpha,
                    "antenna": antenna,
                    "paired_loss": losses,
                },
                "upper_limit": upper_limit_to_dict(limit),
                "calibration_hashes": cal_hashes,
            }
        )
        write_json(_jsonable(payload), arguments.output_dir / "decision.json")
        write_json(_jsonable(payload), arguments.output_dir / "report.json")
        write_offset_ring_plots(
            arguments.output_dir / "plots",
            geometries=geometries,
            measured=stacked["visibility"],
            predicted_diag=pred_diag,
            predicted_full=pred_full,
            field_id=stacked["field_id"],
            frequencies_hz=frequencies,
            bootstrap=holdout_alpha,
        )
        (arguments.output_dir / "README.md").write_text(_readme(payload), encoding="utf-8")
        write_json(_source_revision(), arguments.output_dir / "source_revision.json")
        return 0 if gate.get("selected_model") else 0
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
        payload["decision"] = "software_gate_failed"
        payload["process_failure"] = True
        write_json(_jsonable(payload), arguments.output_dir / "decision.json")
        write_json(_jsonable(payload), arguments.output_dir / "report.json")
        (arguments.output_dir / "README.md").write_text(_readme(payload), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
