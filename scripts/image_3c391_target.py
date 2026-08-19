"""Compare CASA-corrected and SL1MJax-calibrated 3C391 target imaging."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from casacore import tables

from sl1mjax.calibration import (
    apply_calibration,
    identity_solution,
    import_casa_golden_solution,
    load_casa_calibration_golden,
)
from sl1mjax.calibration_inference import (
    CalibrationSolveConfig,
    flux_scale_solution,
    solve_staged_calibration,
    solve_time_gains,
    transfer_flux_scale,
)
from sl1mjax.data.averaging import average_frequency_bins, average_time_bins
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.diagnostics import dirty_image_and_psf
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.imaging import ImagingConfig, ImagingResult, reconstruct
from sl1mjax.inference import InferenceConfig
from sl1mjax.output import write_products
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.sky import RegularGrid
from sl1mjax.split import calibration_split


def _flags_for_rows(table: Any, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = table.selectrows(rows)
    try:
        flag = np.asarray(selected.getcol("FLAG"), dtype=bool)
        flag_row = (
            np.asarray(selected.getcol("FLAG_ROW"), dtype=bool)
            if "FLAG_ROW" in selected.colnames()
            else np.zeros(rows.size, dtype=bool)
        )
        return flag, flag_row
    finally:
        selected.close()


def _block(
    *,
    visibility: np.ndarray,
    flag: np.ndarray,
    flag_row: np.ndarray,
    weight: np.ndarray,
    uvw_m: np.ndarray,
    frequency_hz: np.ndarray,
    time_s: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    scan_id: np.ndarray,
    field_id: int,
    interval_s: np.ndarray,
    phase_centre_rad: tuple[float, float],
    column: str,
) -> VisibilityBlock:
    selected_correlations = np.asarray([0, 3])
    shape = visibility[:, :, selected_correlations].shape
    spectral_weight = np.broadcast_to(
        weight[:, None, selected_correlations], shape
    ).copy()
    effective_flag = (
        flag[:, :, selected_correlations] | flag_row[:, None, None]
    )
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        visibility=visibility[:, :, selected_correlations],
        weight=spectral_weight,
        flag=effective_flag,
        time_s=time_s,
        antenna1=antenna1,
        antenna2=antenna2,
        field_id=np.full(time_s.size, field_id, dtype=np.int32),
        scan_id=scan_id,
        interval_s=interval_s,
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=phase_centre_rad,
        provenance={"source_column": column, "field_id": field_id},
    )


def _concatenate(blocks: list[VisibilityBlock], *, label: str) -> VisibilityBlock:
    first = blocks[0]

    def rows(name: str) -> np.ndarray:
        return np.concatenate(
            [np.asarray(getattr(block, name)) for block in blocks], axis=0
        )

    return VisibilityBlock(
        uvw_m=rows("uvw_m"),
        frequency_hz=first.frequency_hz,
        visibility=rows("visibility"),
        weight=rows("weight"),
        flag=rows("flag"),
        time_s=rows("time_s"),
        antenna1=rows("antenna1"),
        antenna2=rows("antenna2"),
        field_id=rows("field_id"),
        scan_id=rows("scan_id"),
        state_id=rows("state_id"),
        observation_id=rows("observation_id"),
        feed1=rows("feed1"),
        feed2=rows("feed2"),
        interval_s=rows("interval_s"),
        correlations=first.correlations,
        receptor_basis=first.receptor_basis,
        phase_centre_rad=first.phase_centre_rad,
        provenance={"comparison_case": label, "chunk_count": len(blocks)},
    )


def _select_rows(block: VisibilityBlock, rows: np.ndarray) -> VisibilityBlock:
    field_id = block.field_id
    scan_id = block.scan_id
    state_id = block.state_id
    observation_id = block.observation_id
    feed1 = block.feed1
    feed2 = block.feed2
    interval_s = block.interval_s
    assert field_id is not None
    assert scan_id is not None
    assert state_id is not None
    assert observation_id is not None
    assert feed1 is not None
    assert feed2 is not None
    assert interval_s is not None
    return replace(
        block,
        uvw_m=block.uvw_m[rows],
        visibility=block.visibility[rows],
        weight=block.weight[rows],
        flag=block.flag[rows],
        time_s=block.time_s[rows],
        antenna1=block.antenna1[rows],
        antenna2=block.antenna2[rows],
        field_id=field_id[rows],
        scan_id=scan_id[rows],
        state_id=state_id[rows],
        observation_id=observation_id[rows],
        feed1=feed1[rows],
        feed2=feed2[rows],
        interval_s=interval_s[rows],
    )


def _write_fits(
    image: np.ndarray,
    path: Path,
    grid: RegularGrid,
    phase_centre_rad: tuple[float, float],
    *,
    unit: str,
) -> None:
    header = fits.Header()
    header["BUNIT"] = unit
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = (grid.size + 1) / 2
    header["CRPIX2"] = (grid.size + 1) / 2
    header["CRVAL1"] = np.rad2deg(phase_centre_rad[0])
    header["CRVAL2"] = np.rad2deg(phase_centre_rad[1])
    header["CDELT1"] = -np.rad2deg(grid.pixel_size_rad)
    header["CDELT2"] = np.rad2deg(grid.pixel_size_rad)
    fits.PrimaryHDU(image, header=header).writeto(path, overwrite=True)


def _solve_calibration(
    golden: Path, config: CalibrationSolveConfig
) -> tuple[Any, dict[str, float]]:
    primary = load_casa_calibration_golden(golden, label="flux_bandpass")
    secondary = load_casa_calibration_golden(golden, label="time_gain")
    known = import_casa_golden_solution(golden, field_id=primary.field_id)
    initial = identity_solution(
        antenna_count=known.antenna_count,
        correlations=primary.block.correlations,
        frequency_hz=primary.block.frequency_hz,
        time_s=np.unique(primary.block.time_s),
        reference_antenna=known.reference_antenna,
    )
    initial = replace(
        initial,
        reference_frequency_hz=known.reference_frequency_hz,
        antenna_position_offset_m=known.antenna_position_offset_m,
        provenance={
            "known_term": "VLA antenna-position correction",
            "solved_terms": ["G", "K", "B"],
        },
    )
    primary_result = solve_staged_calibration(
        primary.block,
        reference_antenna=known.reference_antenna,
        config=config,
        initial_solution=initial,
    )[-1]
    secondary_result = solve_time_gains(
        secondary.block,
        primary_result.solution,
        split=calibration_split(secondary.block, seed=config.seed),
        config=config,
    )
    metrics = {
        "primary_train_rms": primary_result.train_rms,
        "primary_holdout_rms": primary_result.holdout_rms,
        "secondary_train_rms": secondary_result.train_rms,
        "secondary_holdout_rms": secondary_result.holdout_rms,
        "secondary_flux_jy": transfer_flux_scale(
            primary_result.solution, secondary_result.solution
        ),
    }
    return flux_scale_solution(
        secondary_result.solution, metrics["secondary_flux_jy"]
    ), metrics


def _extract_target(
    measurement_set: Path,
    solution: Any,
    *,
    field_id: int,
    frequency_bins: int,
    time_bin_s: float,
    chunk_rows: int,
) -> tuple[VisibilityBlock, VisibilityBlock]:
    input_flag_path = (
        Path(str(measurement_set) + ".flagversions")
        / "flags.sl1mjax_calibration_input"
    )
    casa_chunks: list[VisibilityBlock] = []
    jax_chunks: list[VisibilityBlock] = []
    with (
        tables.table(str(measurement_set), readonly=True, ack=False) as main,
        tables.table(str(input_flag_path), readonly=True, ack=False) as input_flags,
        tables.table(
            str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False
        ) as spectral_window,
        tables.table(
            str(measurement_set / "FIELD"), readonly=True, ack=False
        ) as field,
    ):
        selected = main.query(f"FIELD_ID=={field_id}")
        try:
            source_rows = np.asarray(selected.rownumbers(), dtype=np.int64)
            frequency_hz = np.asarray(
                spectral_window.getcell("CHAN_FREQ", 0), dtype=np.float64
            )
            direction = np.asarray(
                field.getcell("PHASE_DIR", field_id), dtype=np.float64
            ).reshape(-1, 2)[0]
            phase_centre = (float(direction[0]), float(direction[1]))
            for start in range(0, selected.nrows(), chunk_rows):
                count = min(chunk_rows, selected.nrows() - start)
                rows = source_rows[start : start + count]
                input_flag, input_flag_row = _flags_for_rows(input_flags, rows)
                post_flag = np.asarray(
                    selected.getcol("FLAG", startrow=start, nrow=count), dtype=bool
                )
                post_flag_row = np.asarray(
                    selected.getcol("FLAG_ROW", startrow=start, nrow=count),
                    dtype=bool,
                )
                weight = np.asarray(
                    selected.getcol("WEIGHT", startrow=start, nrow=count),
                    dtype=np.float64,
                )
                uvw_m = np.asarray(
                    selected.getcol("UVW", startrow=start, nrow=count),
                    dtype=np.float64,
                )
                time_s = np.asarray(
                    selected.getcol("TIME", startrow=start, nrow=count),
                    dtype=np.float64,
                )
                antenna1 = np.asarray(
                    selected.getcol("ANTENNA1", startrow=start, nrow=count),
                    dtype=np.int32,
                )
                antenna2 = np.asarray(
                    selected.getcol("ANTENNA2", startrow=start, nrow=count),
                    dtype=np.int32,
                )
                scan_id = np.asarray(
                    selected.getcol("SCAN_NUMBER", startrow=start, nrow=count),
                    dtype=np.int32,
                )
                interval_s = np.asarray(
                    selected.getcol("INTERVAL", startrow=start, nrow=count),
                    dtype=np.float64,
                )
                casa = _block(
                    visibility=np.asarray(
                        selected.getcol(
                            "CORRECTED_DATA", startrow=start, nrow=count
                        )
                    ),
                    flag=post_flag,
                    flag_row=post_flag_row,
                    weight=weight,
                    uvw_m=uvw_m,
                    frequency_hz=frequency_hz,
                    time_s=time_s,
                    antenna1=antenna1,
                    antenna2=antenna2,
                    scan_id=scan_id,
                    field_id=field_id,
                    interval_s=interval_s,
                    phase_centre_rad=phase_centre,
                    column="CORRECTED_DATA",
                )
                raw = _block(
                    visibility=np.asarray(
                        selected.getcol("DATA", startrow=start, nrow=count)
                    ),
                    flag=input_flag,
                    flag_row=input_flag_row,
                    weight=weight,
                    uvw_m=uvw_m,
                    frequency_hz=frequency_hz,
                    time_s=time_s,
                    antenna1=antenna1,
                    antenna2=antenna2,
                    scan_id=scan_id,
                    field_id=field_id,
                    interval_s=interval_s,
                    phase_centre_rad=phase_centre,
                    column="DATA",
                )
                calibrated = apply_calibration(raw, solution, extrapolate=True)
                casa_chunks.append(
                    average_frequency_bins(casa, bin_count=frequency_bins)
                )
                jax_chunks.append(
                    average_frequency_bins(calibrated, bin_count=frequency_bins)
                )
        finally:
            selected.close()
    casa = average_time_bins(
        _concatenate(casa_chunks, label="casa_corrected"),
        bin_seconds=time_bin_s,
    )
    jax_calibrated = average_time_bins(
        _concatenate(jax_chunks, label="jax_calibrated"),
        bin_seconds=time_bin_s,
    )
    return casa, jax_calibrated


def _image_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    normalized_rms = np.sqrt(
        np.sum((actual - reference) ** 2)
        / np.maximum(np.sum(reference**2), np.finfo(float).tiny)
    )
    return {
        "normalized_rms": float(normalized_rms),
        "correlation": float(np.corrcoef(actual.ravel(), reference.ravel())[0, 1]),
        "casa_peak": float(np.max(reference)),
        "jax_peak": float(np.max(actual)),
        "casa_rms": float(np.sqrt(np.mean(reference**2))),
        "jax_rms": float(np.sqrt(np.mean(actual**2))),
    }


def _plot_comparison(
    casa: np.ndarray,
    jax_image: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    difference = jax_image - casa
    limit = float(np.percentile(np.abs(np.concatenate((casa.ravel(), jax_image.ravel()))), 99.5))
    difference_limit = float(np.percentile(np.abs(difference), 99.5))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, image, label in zip(
        axes[:2], (casa, jax_image), ("CASA corrected", "SL1MJax calibrated"), strict=True
    ):
        shown = axis.imshow(
            image,
            origin="lower",
            cmap="inferno",
            vmin=-0.05 * limit,
            vmax=limit,
        )
        axis.set_title(label)
        figure.colorbar(shown, ax=axis, fraction=0.046)
    shown = axes[2].imshow(
        difference,
        origin="lower",
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[2].set_title("SL1MJax − CASA")
    figure.colorbar(shown, ax=axes[2], fraction=0.046)
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_set", type=Path)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("tests/fixtures/3c391_calibration_golden.npz"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/3c391_target"))
    parser.add_argument("--field-id", type=int, default=2)
    parser.add_argument("--frequency-bins", type=int, default=4)
    parser.add_argument("--time-bin-s", type=float, default=60.0)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--dirty-size", type=int, default=96)
    parser.add_argument("--dirty-pixel-arcsec", type=float, default=4.0)
    parser.add_argument("--reconstruction-size", type=int, default=40)
    parser.add_argument("--reconstruction-pixel-arcsec", type=float, default=10.0)
    parser.add_argument("--reconstruction-rows", type=int, default=1000)
    parser.add_argument("--reconstruction-steps", type=int, default=150)
    parser.add_argument(
        "--operator-mode",
        choices=("autodiff", "explicit"),
        default="autodiff",
    )
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    solve_config = CalibrationSolveConfig(
        iterations=300, learning_rate=0.03, seed=11
    )
    solution, calibration_metrics = _solve_calibration(
        arguments.golden, solve_config
    )
    casa, jax_calibrated = _extract_target(
        arguments.measurement_set,
        solution,
        field_id=arguments.field_id,
        frequency_bins=arguments.frequency_bins,
        time_bin_s=arguments.time_bin_s,
        chunk_rows=arguments.chunk_rows,
    )
    dirty_grid = RegularGrid(
        arguments.dirty_size,
        np.deg2rad(arguments.dirty_pixel_arcsec / 3600),
    )
    casa_dirty, casa_psf = dirty_image_and_psf(casa, dirty_grid)
    jax_dirty, jax_psf = dirty_image_and_psf(jax_calibrated, dirty_grid)
    _write_fits(
        casa_dirty,
        arguments.output / "casa_corrected_dirty.fits",
        dirty_grid,
        casa.phase_centre_rad,
        unit="Jy/beam",
    )
    _write_fits(
        jax_dirty,
        arguments.output / "jax_calibrated_dirty.fits",
        dirty_grid,
        casa.phase_centre_rad,
        unit="Jy/beam",
    )
    _write_fits(
        casa_psf,
        arguments.output / "dirty_psf.fits",
        dirty_grid,
        casa.phase_centre_rad,
        unit="",
    )
    _plot_comparison(
        casa_dirty,
        jax_dirty,
        arguments.output / "dirty_comparison.png",
        title="3C391 C1 naturally weighted dirty image",
    )

    common_rows = np.flatnonzero(
        np.any(casa.active & jax_calibrated.active, axis=(1, 2))
    )
    reconstruction_count = min(arguments.reconstruction_rows, common_rows.size)
    reconstruction_rows = common_rows[
        np.linspace(
            0, common_rows.size - 1, reconstruction_count, dtype=np.int64
        )
    ]
    casa_reconstruction_block = _select_rows(casa, reconstruction_rows)
    jax_reconstruction_block = _select_rows(jax_calibrated, reconstruction_rows)
    image_config = ImagingConfig(
        size=arguments.reconstruction_size,
        pixel_size_rad=np.deg2rad(arguments.reconstruction_pixel_arcsec / 3600),
        inference=InferenceConfig(
            steps=arguments.reconstruction_steps,
            learning_rate=0.03,
            sparsity_weight=1e-4,
            chunk_size=256,
            initial_intensity=1e-3,
            patience=60,
            operator_mode=arguments.operator_mode,
            direct_dft=DirectDFTConfig(
                visibility_chunk_size=arguments.visibility_tile_size,
                pixel_chunk_size=arguments.pixel_tile_size,
            ),
        ),
        split_seed=17,
    )
    casa_reconstruction: ImagingResult = reconstruct(
        casa_reconstruction_block, image_config
    )
    jax_reconstruction: ImagingResult = reconstruct(
        jax_reconstruction_block, image_config
    )
    write_products(
        casa_reconstruction,
        arguments.output / "casa_corrected_reconstruction.fits",
    )
    write_products(
        jax_reconstruction,
        arguments.output / "jax_calibrated_reconstruction.fits",
    )
    _plot_comparison(
        casa_reconstruction.image,
        jax_reconstruction.image,
        arguments.output / "reconstruction_comparison.png",
        title="3C391 C1 positive-grid reconstruction",
    )

    common = casa.active & jax_calibrated.active
    visibility_weight = np.where(
        common, np.minimum(casa.weight, jax_calibrated.weight), 0.0
    )
    visibility_metrics = {
        "normalized_rms": float(
            np.sqrt(
                np.sum(
                    visibility_weight
                    * np.abs(jax_calibrated.visibility - casa.visibility) ** 2
                )
                / np.sum(visibility_weight * np.abs(casa.visibility) ** 2)
            )
        ),
        "common_sample_count": int(np.sum(common)),
    }
    summary = {
        "field_id": arguments.field_id,
        "measurement_set": arguments.measurement_set.name,
        "averaged_shape": list(casa.shape),
        "calibration": calibration_metrics,
        "visibility_comparison": visibility_metrics,
        "dirty_image_comparison": _image_metrics(jax_dirty, casa_dirty),
        "dirty_psf_comparison": _image_metrics(jax_psf, casa_psf),
        "reconstruction_comparison": _image_metrics(
            jax_reconstruction.image, casa_reconstruction.image
        ),
        "reconstruction": {
            "operator_mode": arguments.operator_mode,
            "visibility_tile_size": arguments.visibility_tile_size,
            "pixel_tile_size": arguments.pixel_tile_size,
            "row_count": reconstruction_count,
            "casa_train_loss": casa_reconstruction.train_loss,
            "casa_holdout_loss": casa_reconstruction.holdout_loss,
            "jax_train_loss": jax_reconstruction.train_loss,
            "jax_holdout_loss": jax_reconstruction.holdout_loss,
        },
        "known_terms": {
            "antenna_position": "imported VLA/CASA correction",
            "flags": "frozen CASA tutorial calibration-input flags",
            "sky_model": "CASA 3C286 model in compact golden fixture",
        },
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
