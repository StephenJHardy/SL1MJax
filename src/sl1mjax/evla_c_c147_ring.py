"""C147-* EVLA-C comparison at the nine publication frequencies.

Uses reconstructed source-to-pointing geometry and the geometric fringe.
Does not apply the HOLORASTER negative-commanded-offset shortcut. Field
partitions stay historical development diagnostics.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from sl1mjax.evla_c_metrics import hand_scores
from sl1mjax.evla_c_validation_refresh import (
    C147_OFFSET_FIELD_IDS,
    PUBLICATION_CHANNELS,
    RESIDUAL_JONES_PATH,
    refuse_frozen_write,
    write_json_atomic,
)
from sl1mjax.holography_c147_offset_ring import (
    OFFSET_RING_NOTE,
    declare_field_partitions,
    field_partition_masks,
    geometry_to_dict,
    partition_to_dict,
    reconstruct_field_offsets,
    select_3c147_sky_direction,
    source_relative_offsets,
)
from sl1mjax.holography_calibration_golden import (
    apply_imported_solution,
    import_thol0001_casa_tables,
)
from sl1mjax.holography_diagonal_correction import refuse_spw5
from sl1mjax.holography_ms import audit_holography_measurement_set
from sl1mjax.holography_reference_jones import deserialize_reference_jones
from sl1mjax.polarization import Receptor, pack_coherency


def load_field9_residual_jones(path: Path = RESIDUAL_JONES_PATH) -> dict[int, np.ndarray]:
    """Load the field-9 channel-32 residual Jones plane without the HOLORASTER script."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return deserialize_reference_jones(payload["jones"])


def _c147_script():
    candidates = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_thol0001_c147_offset_ring.py",
        Path("/tmp/sl1mjax-evla-c-refresh-scripts/run_thol0001_c147_offset_ring.py"),
    )
    path = next((item for item in candidates if item.is_file()), candidates[0])
    spec = importlib.util.spec_from_file_location("thol0001_c147_offset_ring", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_c147_channel(
    *,
    visibility,
    pred_diag,
    pred_full,
    weight,
    field_id,
    partition,
    source_i_jy: float,
) -> dict[str, object]:
    masks = field_partition_masks(field_id, partition)
    out: dict[str, object] = {}
    for name, mask in masks.items():
        if not bool(np.any(mask)):
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": int(np.sum(mask)),
            "diagonal": hand_scores(
                visibility[mask], pred_diag[mask], weight[mask], source_i_jy=source_i_jy
            ),
            "full_jones": hand_scores(
                visibility[mask], pred_full[mask], weight[mask], source_i_jy=source_i_jy
            ),
        }
    return out


def run_c147_publication_channels(
    *,
    measurement_set: Path,
    cal_root: Path,
    catalog,
    residual,
    output_dir: Path,
    channels: tuple[int, ...] = PUBLICATION_CHANNELS,
) -> dict[str, object]:
    refuse_spw5(spectral_window_id=4, opened=False)
    refuse_frozen_write(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("c147: load offset-ring helpers", flush=True)
    script = _c147_script()
    print("c147: read antennas/fields", flush=True)
    tables = script._tables()
    _ids, _names, positions = script._read_antennas(tables, measurement_set)
    positions = np.asarray(positions, dtype=np.float64)
    fields, sources = script._read_fields_and_sources(tables, measurement_set)
    source_radec, source_from = select_3c147_sky_direction(fields, sources)
    geometries = [
        item
        for item in reconstruct_field_offsets(fields, source_radec, source_from=source_from)
        if int(item.field_id) in set(C147_OFFSET_FIELD_IDS)
    ]
    if len(geometries) != 8:
        ids = [item.field_id for item in geometries]
        raise ValueError(f"expected eight C147-* fields, got {ids}")
    partition = declare_field_partitions(geometries)
    cal_tables = script._scientific_tables(cal_root)
    missing = [key for key, path in cal_tables.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"scientific tables missing: {missing}")
    print("c147: import scientific calibration", flush=True)
    solution = import_thol0001_casa_tables(
        cal_tables,
        measurement_set=measurement_set,
        spectral_window_id=4,
        product="fullpol",
    )
    ddid = script._ddid_for_spw(tables, measurement_set, 4)
    audits: dict[int, object] = {}
    print("c147: audit eight C147-* fields", flush=True)
    for geom in geometries:
        try:
            audits[int(geom.field_id)] = audit_holography_measurement_set(
                measurement_set, field_name=geom.name
            )
        except Exception:
            audits[int(geom.field_id)] = None
    field_rows = []
    band_frequencies = None
    for geom in geometries:
        print(f"c147: load field {geom.field_id} {geom.name} all SPW-4 channels", flush=True)
        audit = audits[int(geom.field_id)]
        raw = script._load_offset_block(
            tables,
            measurement_set,
            field_id=geom.field_id,
            phase_centre_rad=geom.phase_centre_rad,
            data_desc_id=ddid,
            spectral_window_id=4,
            channel=0,
            channel_stop=64,
        )
        if raw.provenance.get("column") != "DATA":
            raise ValueError("C147 refresh must start from DATA")
        applied = apply_imported_solution(raw, solution)
        packed = pack_coherency(
            applied.visibility, applied.correlations, (Receptor.R, Receptor.L)
        )
        sky_lm = np.broadcast_to(
            np.array([[geom.l_rad, geom.m_rad]], dtype=np.float64),
            (applied.time_s.size, 2),
        ).copy()
        if audit is None:
            delta_p = np.zeros((applied.time_s.size, 2), dtype=np.float64)
            delta_q = np.zeros((applied.time_s.size, 2), dtype=np.float64)
        else:
            delta_p = script._pointing_offsets(audit, applied.time_s, applied.antenna1)
            delta_q = script._pointing_offsets(audit, applied.time_s, applied.antenna2)
        frequencies = np.asarray(applied.frequency_hz, dtype=np.float64).reshape(-1)
        if band_frequencies is None:
            band_frequencies = frequencies
        field_rows.append(
            {
                "field_id": np.full(applied.time_s.size, geom.field_id, dtype=np.int32),
                "antenna1": np.asarray(applied.antenna1, dtype=np.int32),
                "antenna2": np.asarray(applied.antenna2, dtype=np.int32),
                "visibility": packed,
                "weight": script._hand_weight(
                    applied.flag, applied.weight, applied.time_s.size, packed.shape[1]
                ),
                "offset_p": source_relative_offsets(sky_lm, delta_p),
                "offset_q": source_relative_offsets(sky_lm, delta_q),
                "sky_lm": sky_lm,
                "uvw_m": np.asarray(applied.uvw_m, dtype=np.float64),
                "chi_p": script._chi(
                    applied.time_s, geom.phase_centre_rad, positions, applied.antenna1
                ),
                "chi_q": script._chi(
                    applied.time_s, geom.phase_centre_rad, positions, applied.antenna2
                ),
            }
        )
        print(f"c147: field {geom.field_id} n={applied.time_s.size}", flush=True)
    if band_frequencies is None:
        raise RuntimeError("C147 load produced no frequencies")
    stacked = {
        key: np.concatenate([item[key] for item in field_rows], axis=0)
        for key in (
            "field_id",
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
        )
    }
    print(f"c147: stacked n={stacked['visibility'].shape[0]}", flush=True)
    by_channel: list[dict[str, object]] = []
    for channel in channels:
        print(f"c147: channel {channel} predict", flush=True)
        freq = np.asarray([band_frequencies[int(channel)]], dtype=np.float64)
        vis = np.asarray(stacked["visibility"][:, int(channel)])
        wgt = np.asarray(stacked["weight"][:, int(channel)])
        source = script._setjy_source(freq, 0.0, 0.0, vis.shape[0])
        pred_kwargs = {
            "catalog": catalog,
            "frequencies": freq,
            "offset_p": stacked["offset_p"],
            "offset_q": stacked["offset_q"],
            "chi_p": stacked["chi_p"],
            "chi_q": stacked["chi_q"],
            "residual": residual,
            "ant1": stacked["antenna1"],
            "ant2": stacked["antenna2"],
            "source": source,
            "uvw_m": stacked["uvw_m"],
            "sky_lm": stacked["sky_lm"],
        }
        pred_diag = script._predict(offdiag=False, **pred_kwargs)
        pred_full = script._predict(offdiag=True, **pred_kwargs)
        if pred_diag.ndim == 4:
            pred_diag = pred_diag[:, 0]
            pred_full = pred_full[:, 0]
        if vis.ndim == 4:
            vis = vis[:, 0]
            wgt = wgt[:, 0]
        source_i = float(np.median(np.abs(source[:, 0, 0, 0]).real))
        scores = score_c147_channel(
            visibility=vis,
            pred_diag=pred_diag,
            pred_full=pred_full,
            weight=wgt,
            field_id=stacked["field_id"],
            partition=partition,
            source_i_jy=source_i,
        )
        by_channel.append(
            {
                "channel": int(channel),
                "frequency_hz": float(freq[0]),
                "n_row": int(vis.shape[0]),
                "source_i_jy": source_i,
                "scores": scores,
            }
        )

    report = {
        "development_only": True,
        "historical_holdout": True,
        "holoraster_shortcut": False,
        "notes": [OFFSET_RING_NOTE],
        "source_from": source_from,
        "partition": partition_to_dict(partition),
        "fields": [geometry_to_dict(item) for item in geometries],
        "channels": by_channel,
    }
    write_json_atomic(output_dir / "c147_geometry.json", report)
    return report
