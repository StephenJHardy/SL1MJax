"""Exploratory 3C391 I, Q, U, V diagnostics on the polarisation golden.

This is the first imaging-ladder step: frozen CASA-compatible apply,
calibrator floors, global q,u,v, and dirty Stokes images. Results are
not evidence-grade while exact Jones still produces false V. The 3C84
residual is in-sample. Same-data 3C286 is an apply-back residual, not
an independent floor. Use validate_3c391_global_polarization.py for
the complex M_I fit, 3C286 solve/eval split, and seven-pointing check.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.calibration import (
    apply_calibration,
    identity_solution,
    import_casa_polarization_solution,
    load_casa_calibration_golden,
)
from sl1mjax.data.averaging import average_frequency_bins, average_time_bins
from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.polarization_diagnostics import (
    calibrator_polarization_floor,
    dirty_stokes_images,
    fit_global_fractional_polarization,
    global_fractional_polarization_as_dict,
    polarization_floor_as_dict,
)
from sl1mjax.sky import RegularGrid

DEFAULT_MS = Path(
    "/home/stephen/checkouts/SL1MJax-frozen-20260824/data/3c391_ctm_mosaic_10s_spw0.ms"
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POL = ROOT / "tests" / "fixtures" / "3c391_polarization_golden.npz"
DEFAULT_KBG = ROOT / "tests" / "fixtures" / "3c391_calibration_golden.npz"
DEFAULT_OUTPUT = ROOT / "outputs" / "3c391_polarization_diagnostics"


def _corrected_from_casa(case):
    return replace(
        case.block,
        visibility=case.corrected_visibility,
        flag=case.block.flag | case.post_apply_flag,
    )


def _polarization_terms_only(solution):
    """Keep Kcross/D/X/P. Identity G/K/B for already-corrected target DATA."""

    identity = identity_solution(
        antenna_count=solution.antenna_count,
        correlations=solution.correlations,
        frequency_hz=solution.bandpass_frequency_hz,
        time_s=solution.gain_time_s,
        reference_antenna=solution.reference_antenna,
    )
    return replace(
        identity,
        receptors=solution.receptors,
        reference_frequency_hz=solution.reference_frequency_hz,
        antenna_position_m=solution.antenna_position_m,
        cross_hand_delay_s=solution.cross_hand_delay_s,
        cross_hand_delay_valid=solution.cross_hand_delay_valid,
        leakage=solution.leakage,
        leakage_frequency_hz=solution.leakage_frequency_hz,
        leakage_valid=solution.leakage_valid,
        leakage_application=solution.leakage_application,
        rl_phase=solution.rl_phase,
        rl_phase_frequency_hz=solution.rl_phase_frequency_hz,
        rl_phase_valid=solution.rl_phase_valid,
        apply_parallactic_angle=True,
        provenance={
            **solution.provenance,
            "gkb": "identity_on_corrected_data",
            "evidence_grade": False,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL)
    parser.add_argument("--calibration-golden", type=Path, default=DEFAULT_KBG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=17)
    parser.add_argument("--pixel-arcsec", type=float, default=2.0)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--measurement-set", type=Path, default=None)
    parser.add_argument("--target-field", type=int, default=2)
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument("--frequency-bins", type=int, default=8)
    parser.add_argument("--time-bin-seconds", type=float, default=60.0)
    parser.add_argument("--target-image-size", type=int, default=65)
    parser.add_argument("--target-pixel-arcsec", type=float, default=8.0)
    arguments = parser.parse_args(argv)

    flux = load_casa_calibration_golden(arguments.polarization_golden, label="flux_angle")
    leakage = load_casa_calibration_golden(
        arguments.polarization_golden, label="leakage_calibrator"
    )
    casa_flux = _corrected_from_casa(flux)
    casa_leakage = _corrected_from_casa(leakage)
    solution = import_casa_polarization_solution(
        arguments.polarization_golden,
        arguments.calibration_golden,
        label="flux_angle",
    )
    jax_flux = apply_calibration(flux.block, solution, extrapolate=True)

    report = {
        "evidence_grade": False,
        "apply": "casa_parallel_preserving",
        "floors": {
            "casa_3c286": polarization_floor_as_dict(
                calibrator_polarization_floor(
                    casa_flux, independence="apply_back", label="flux_angle"
                )
            ),
            "jax_3c286": polarization_floor_as_dict(
                calibrator_polarization_floor(
                    jax_flux, independence="apply_back", label="flux_angle_jax"
                )
            ),
            "casa_3c84_in_sample": polarization_floor_as_dict(
                calibrator_polarization_floor(
                    casa_leakage,
                    independence="in_sample",
                    label="leakage_calibrator",
                )
            ),
        },
        "global_quv": {},
    }
    for name, block in (("casa_3c286", casa_flux), ("jax_3c286", jax_flux)):
        fitted = fit_global_fractional_polarization(block)
        report["global_quv"][name] = global_fractional_polarization_as_dict(fitted)

    arguments.output.mkdir(parents=True, exist_ok=True)
    if not arguments.skip_images:
        grid = RegularGrid(
            arguments.image_size, np.deg2rad(arguments.pixel_arcsec / 3600)
        )
        images = dirty_stokes_images(jax_flux, grid)
        np.savez(
            arguments.output / "3c286_dirty_stokes.npz",
            stokes_i=images.stokes_i,
            stokes_q=images.stokes_q,
            stokes_u=images.stokes_u,
            stokes_v=images.stokes_v,
            psf=images.psf,
        )
        report["dirty_stokes_3c286"] = {
            "peak_i": images.peak_i,
            "peak_q": images.peak_q,
            "peak_u": images.peak_u,
            "peak_v": images.peak_v,
            "peak_q_over_i": images.peak_q / images.peak_i,
            "peak_u_over_i": images.peak_u / images.peak_i,
            "peak_v_over_i": images.peak_v / images.peak_i,
            "evidence_grade": False,
        }

    measurement_set = arguments.measurement_set
    if measurement_set is None and DEFAULT_MS.is_dir():
        measurement_set = DEFAULT_MS
    if measurement_set is not None:
        dataset = extract_measurement_set(
            measurement_set,
            data_column="CORRECTED_DATA",
            fields=(arguments.target_field,),
            row_stride=arguments.row_stride,
        )
        if not dataset.blocks:
            raise ValueError(f"no blocks extracted for field {arguments.target_field}")
        target = dataset.blocks[0]
        if arguments.frequency_bins < target.frequency_hz.size:
            target = average_frequency_bins(target, bin_count=arguments.frequency_bins)
        if arguments.time_bin_seconds > 0:
            target = average_time_bins(target, bin_seconds=arguments.time_bin_seconds)
        pol_only = _polarization_terms_only(solution)
        corrected_target = apply_calibration(target, pol_only, extrapolate=True)
        report["target"] = {
            "field_id": arguments.target_field,
            "measurement_set": str(measurement_set),
            "data_column": "CORRECTED_DATA",
            "pre_polarization_state": (
                "CASA G/K/B already present in CORRECTED_DATA; "
                "identity G/K/B plus JAX Kcross/D/X/P applied here"
            ),
            "rows": int(corrected_target.visibility.shape[0]),
            "channels": int(corrected_target.frequency_hz.size),
            "global_quv": None,
            "note": (
                "global q,u,v on an extended field requires frozen complex M_I; "
                "run validate_3c391_global_polarization.py"
            ),
        }
        if not arguments.skip_images:
            target_grid = RegularGrid(
                arguments.target_image_size,
                np.deg2rad(arguments.target_pixel_arcsec / 3600),
            )
            target_images = dirty_stokes_images(corrected_target, target_grid)
            np.savez(
                arguments.output / "c1_dirty_stokes.npz",
                stokes_i=target_images.stokes_i,
                stokes_q=target_images.stokes_q,
                stokes_u=target_images.stokes_u,
                stokes_v=target_images.stokes_v,
                psf=target_images.psf,
            )
            report["target"]["dirty_stokes"] = {
                "peak_i": target_images.peak_i,
                "peak_q": target_images.peak_q,
                "peak_u": target_images.peak_u,
                "peak_v": target_images.peak_v,
                "peak_q_over_i": target_images.peak_q / target_images.peak_i,
                "peak_u_over_i": target_images.peak_u / target_images.peak_i,
                "peak_v_over_i": target_images.peak_v / target_images.peak_i,
                "image_size": arguments.target_image_size,
                "pixel_arcsec": arguments.target_pixel_arcsec,
                "evidence_grade": False,
            }

    destination = arguments.output / "report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
