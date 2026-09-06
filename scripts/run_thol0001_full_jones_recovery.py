"""Recover unfrozen SPW-4 empirical full Jones from the scientific Df chain.

Uses the scientific work MS CORRECTED_DATA (applied once from DATA),
per-row field-10 MODEL_DATA, Q/U = 0, and measured AZELGEO pointing.
Does not freeze the product or touch the production full-Jones factory.
SPW 5 is not opened.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from sl1mjax.holography import HolographyObservation
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_full_jones import recover_holography_full_jones
from sl1mjax.holography_ms import audit_holography_measurement_set

DEFAULT_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
DEFAULT_SETJY = Path(
    "/media/stephen/astro/vla/extracted/commissioning/products/scientific/setjy_3c147.json"
)


def _diag():
    path = Path(__file__).with_name("run_thol0001_diagonal_recovery.py")
    spec = importlib.util.spec_from_file_location("thol0001_diagonal_recovery", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_record(sample) -> dict:
    jones = np.asarray(sample.jones)
    return {
        "moving_antenna_id": int(sample.moving_antenna_id),
        "unique_time_s": float(sample.unique_time_s),
        "offset_lm_rad": [float(sample.offset_lm_rad[0]), float(sample.offset_lm_rad[1])],
        "frequency_hz": float(sample.frequency_hz),
        "jones_real": jones.real.tolist(),
        "jones_imag": jones.imag.tolist(),
        "copolar_valid": bool(sample.copolar_valid),
        "off_diagonal_valid": bool(sample.off_diagonal_valid),
        "n_reference": int(sample.n_reference),
        "raster": str(sample.raster),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_set", nargs="?", type=Path, default=DEFAULT_MS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--setjy-record", type=Path, default=DEFAULT_SETJY)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    diag = _diag()
    audit = audit_holography_measurement_set(arguments.measurement_set)
    from sl1mjax.holography_ms import _tables

    tables = _tables()
    spectral_window_id = 4
    ddid = diag._ddid_for_spw(tables, arguments.measurement_set, spectral_window_id)
    block, source = diag._holoraster_channel_block(
        tables,
        arguments.measurement_set,
        data_desc_id=ddid,
        channel=arguments.channel,
        data_column="CORRECTED_DATA",
        spectral_window_id=spectral_window_id,
    )
    fluxd = diag._fluxd_provenance(
        arguments.setjy_record,
        spectral_window_id=spectral_window_id,
        stokes_i=None,
    )
    observation = HolographyObservation(
        block=block,
        pointing=diag._subset_pointing(audit.resolved, block.time_s),
        antenna_position_m=diag._antenna_positions(tables, arguments.measurement_set),
        calibration_state="casa_parang_true",
        phase_centre_rad=block.phase_centre_rad,
        source_name="3C147",
        source_coherency_visibility=source,
        selected_spw_id=spectral_window_id,
        provenance={
            "measurement_set": str(arguments.measurement_set),
            "data_column": "CORRECTED_DATA",
            "model_column": "MODEL_DATA",
            "source": "field_10_model_data_per_row",
            "q_over_i": 0.0,
            "u_over_i": 0.0,
            "d_product": "Df",
            "spectral_window_id": spectral_window_id,
            "channel": arguments.channel,
            "casa_setjy_fluxd_jy": fluxd,
            "frozen": False,
            "applied_from": "DATA_once",
        },
    )
    artifact = recover_holography_full_jones(observation)
    if artifact.frozen:
        raise RuntimeError("full Jones artifact must remain unfrozen")
    n_off = int(np.sum(artifact.off_diagonal_valid))
    n_co = int(np.sum(artifact.valid))
    payload = {
        "schema": "thol0001_empirical_full_jones_spw4_v1",
        "frozen": False,
        "full_jones_unfrozen": True,
        "production_factory_untouched": True,
        "n_samples": len(artifact.samples),
        "n_copolar_valid": n_co,
        "n_off_diagonal_valid": n_off,
        "calibration_state": artifact.calibration_state,
        "source_name": artifact.source_name,
        "reference_combination": artifact.reference_combination,
        "receptor_convention": artifact.receptor_convention,
        "offset_sign": artifact.offset_sign,
        "frequency_hz": float(block.frequency_hz[0]),
        "spectral_window_id": spectral_window_id,
        "channel": arguments.channel,
        "provenance": dict(observation.provenance),
        "notes": list(artifact.notes),
        "samples": [_sample_record(sample) for sample in artifact.samples],
    }
    write_json(payload, arguments.output_dir / "full_jones_spw4.json")
    summary = {key: payload[key] for key in payload if key != "samples"}
    write_json(summary, arguments.output_dir / "full_jones_spw4_summary.json")
    print(arguments.output_dir / "full_jones_spw4.json")
    print("n_samples", payload["n_samples"], "offdiag", n_off, "copolar", n_co)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
