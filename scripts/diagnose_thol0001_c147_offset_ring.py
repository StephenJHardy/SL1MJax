"""Tiny C147-* prediction diagnostic. Does not write scientific products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.beam_operator import unique_visibility_times
from sl1mjax.calibration_terms import parallactic_angle_rad as parallactic_angle
from sl1mjax.cassbeam_highres import DEFAULT_HIGHRES_ROOT, HighresCassbeamCatalog
from sl1mjax.holography import (
    CASA_SETJY_3C147_C_IM_FLUXD_JY_SPW4,
    THOL0001_SPW4_CHANNEL_32_HZ,
    perley_butler_2017_3c147_stokes_i_jy,
)
from sl1mjax.holography_beam_prior import (
    beams_from_native_unique,
    sky_frame_residual_numpy,
    unique_native_jones,
)
from sl1mjax.holography_c147_offset_ring import (
    apply_geometric_fringe,
    diagonal_rr_ll_closure,
    locked_convention,
    predict_dual_antenna_numpy,
    reconstruct_field_offsets,
    select_3c147_sky_direction,
)
from sl1mjax.holography_calibration_golden import (
    apply_imported_solution,
    import_thol0001_casa_tables,
)
from sl1mjax.holography_full_jones import _antenna_jones_planes
from sl1mjax.holography_ms import _read_antennas, _tables
from sl1mjax.polarization import Receptor, circular_stokes_to_coherency, pack_coherency
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_thol0001_c147_offset_ring import (
    SCIENTIFIC_CAL,
    SCIENTIFIC_MS,
    UNBLOCK,
    _ddid_for_spw,
    _load_offset_block,
    _read_fields_and_sources,
    _residual_jones,
    _scientific_tables,
)


def _source(frequencies, intensity_scale: float, n_row: int):
    table = perley_butler_2017_3c147_stokes_i_jy(frequencies)
    ref = float(perley_butler_2017_3c147_stokes_i_jy(np.array([THOL0001_SPW4_CHANNEL_32_HZ]))[0])
    intensity = table * (float(intensity_scale) / ref)
    sky = circular_stokes_to_coherency(intensity, 0.0 * intensity, 0.0 * intensity, 0.0 * intensity)
    return np.broadcast_to(sky, (n_row, frequencies.size, 2, 2)).copy()


def _predict(catalog, frequencies, offset, chi_p, chi_q, residual, ant1, ant2, source, state, uvw_m):
    convention = locked_convention()
    native, valid, inverse = unique_native_jones(offset, chi_p, frequencies, catalog, convention)
    beam, ok = beams_from_native_unique(
        native,
        valid,
        inverse,
        convention,
        catalog=catalog,
        frequencies_hz=frequencies,
        chi=chi_p,
        off_diagonal=False,
        calibration_state=state,
    )
    r_p = sky_frame_residual_numpy(_antenna_jones_planes(residual, ant1), chi_p)
    r_q = sky_frame_residual_numpy(_antenna_jones_planes(residual, ant2), chi_q)
    pred = apply_geometric_fringe(
        predict_dual_antenna_numpy(r_p, beam, source, beam, r_q),
        uvw_m,
        frequencies,
        offset,
    )
    return np.where(ok[..., None, None], pred, np.nan)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--cal-root", type=Path, default=SCIENTIFIC_CAL)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--field-id", type=int, default=2)
    arguments = parser.parse_args()
    tables = _tables()
    fields, sources = _read_fields_and_sources(tables, arguments.measurement_set)
    source_radec, source_from = select_3c147_sky_direction(fields, sources)
    geometries = {item.field_id: item for item in reconstruct_field_offsets(fields, source_radec, source_from=source_from)}
    geom = geometries[int(arguments.field_id)]
    _ids, _names, positions = _read_antennas(tables, arguments.measurement_set)
    positions = np.asarray(positions, dtype=np.float64)
    solution = import_thol0001_casa_tables(
        _scientific_tables(arguments.cal_root),
        measurement_set=arguments.measurement_set,
        spectral_window_id=4,
        product="fullpol",
    )
    raw = _load_offset_block(
        tables,
        arguments.measurement_set,
        field_id=geom.field_id,
        phase_centre_rad=geom.phase_centre_rad,
        data_desc_id=_ddid_for_spw(tables, arguments.measurement_set, 4),
        spectral_window_id=4,
        channel=32,
        channel_stop=33,
    )
    keep = slice(0, int(arguments.max_rows))
    from dataclasses import replace

    raw = replace(
        raw,
        uvw_m=raw.uvw_m[keep],
        visibility=raw.visibility[keep],
        weight=raw.weight[keep],
        flag=raw.flag[keep],
        time_s=raw.time_s[keep],
        antenna1=raw.antenna1[keep],
        antenna2=raw.antenna2[keep],
        field_id=None if raw.field_id is None else raw.field_id[keep],
        scan_id=None if raw.scan_id is None else raw.scan_id[keep],
        state_id=None,
        observation_id=None,
        feed1=None,
        feed2=None,
        interval_s=None,
    )
    applied = apply_imported_solution(raw, solution)
    packed = pack_coherency(applied.visibility, applied.correlations, (Receptor.R, Receptor.L))
    unique_times, inverse = unique_visibility_times(applied.time_s)
    chi = parallactic_angle(unique_times, geom.phase_centre_rad, positions)
    chi_p = chi[inverse, applied.antenna1]
    chi_q = chi[inverse, applied.antenna2]
    offset = np.broadcast_to(np.array([[geom.l_rad, geom.m_rad]]), (applied.time_s.size, 2)).copy()
    catalog = HighresCassbeamCatalog(arguments.artifact_root)
    residual = _residual_jones(arguments.product_dir)
    identity = {
        int(ant): np.eye(2, dtype=np.complex128)
        for ant in np.unique(np.concatenate([applied.antenna1, applied.antenna2]))
    }
    w = np.zeros_like(packed, dtype=np.float64)
    flag = np.asarray(applied.flag, dtype=bool)
    wgt = np.asarray(applied.weight, dtype=np.float64)
    w[:, :, 0, 0] = np.where(~flag[:, :, 0], wgt[:, :, 0], 0.0)
    w[:, :, 0, 1] = np.where(~flag[:, :, 1], wgt[:, :, 1], 0.0)
    w[:, :, 1, 0] = np.where(~flag[:, :, 2], wgt[:, :, 2], 0.0)
    w[:, :, 1, 1] = np.where(~flag[:, :, 3], wgt[:, :, 3], 0.0)
    report = {
        "field_id": geom.field_id,
        "name": geom.name,
        "l_rad": geom.l_rad,
        "m_rad": geom.m_rad,
        "n_rows": int(applied.time_s.size),
        "median_abs_rr": float(np.nanmedian(np.abs(packed[:, 0, 0, 0]))),
        "median_abs_ll": float(np.nanmedian(np.abs(packed[:, 0, 1, 1]))),
        "median_abs_rl": float(np.nanmedian(np.abs(packed[:, 0, 0, 1]))),
        "cases": {},
    }
    for label, scale, jones, state in (
        ("setjy_identity_parang", CASA_SETJY_3C147_C_IM_FLUXD_JY_SPW4, identity, "casa_parang_true"),
        ("setjy_field9_parang", CASA_SETJY_3C147_C_IM_FLUXD_JY_SPW4, residual, "casa_parang_true"),
        ("unitI_identity_parang", 1.0, identity, "casa_parang_true"),
        ("setjy_identity_uncal", CASA_SETJY_3C147_C_IM_FLUXD_JY_SPW4, identity, "uncalibrated"),
    ):
        pred = _predict(
            catalog,
            applied.frequency_hz,
            offset,
            chi_p,
            chi_q,
            jones,
            applied.antenna1,
            applied.antenna2,
            _source(applied.frequency_hz, scale, applied.time_s.size),
            state,
            applied.uvw_m,
        )
        closure = diagonal_rr_ll_closure(packed, pred, w)
        tt = 0.0
        tr = 0.0 + 0.0j
        for row, col in ((0, 0), (1, 1)):
            m = packed[:, 0, row, col]
            p = pred[:, 0, row, col]
            ww = w[:, 0, row, col]
            finite = np.isfinite(m) & np.isfinite(p) & (ww > 0.0)
            tt += float(np.sum(ww[finite] * np.abs(p[finite]) ** 2))
            tr += complex(np.sum(ww[finite] * np.conjugate(p[finite]) * m[finite]))
        alpha = (tr / tt) if tt > 0.0 else 0.0j
        scaled = pred * alpha
        report["cases"][label] = {
            "median_pred_rr": float(np.nanmedian(np.abs(pred[:, 0, 0, 0]))),
            "median_pred_ll": float(np.nanmedian(np.abs(pred[:, 0, 1, 1]))),
            "median_phase_rr_deg": float(
                np.rad2deg(np.nanmedian(np.angle(packed[:, 0, 0, 0] * np.conjugate(pred[:, 0, 0, 0]))))
            ),
            "closure": closure,
            "ls_scale": [alpha.real, alpha.imag],
            "ls_abs": abs(alpha),
            "ls_phase_deg": float(np.rad2deg(np.angle(alpha))),
            "closure_after_ls_scale": diagonal_rr_ll_closure(packed, scaled, w),
        }
        print(label, json.dumps(report["cases"][label], default=str))
    print(json.dumps({k: report[k] for k in report if k != "cases"}, indent=2))
    Path("/tmp/c147_offset_ring_diagnose.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
