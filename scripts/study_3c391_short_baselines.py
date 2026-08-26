#!/usr/bin/env python3
"""Cross-validate missing-sky, calibration-time, and beam explanations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import InferenceConfig, infer_mosaic_quadtree
from sl1mjax.quadtree import quadtree_sky_from_regular_grid
from sl1mjax.residual_audit import audit_visibility_residuals
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import GaussianApproximation


def time_half_masks(
    blocks: tuple[VisibilityBlock, ...],
) -> tuple[
    tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[float | None, ...]
]:
    """Split every pointing at its median active integration time."""

    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    boundaries: list[float] = []
    for block in blocks:
        active_rows = np.any(block.active, axis=(1, 2))
        times = np.unique(block.time_s[active_rows])
        if times.size < 2:
            raise ValueError("each block needs at least two active integration times")
        boundary = float(np.median(times))
        first_rows = block.time_s <= boundary
        first_mask = np.broadcast_to(first_rows[:, None, None], block.shape) & block.active
        second_mask = np.broadcast_to((~first_rows)[:, None, None], block.shape) & block.active
        if not np.any(first_mask) or not np.any(second_mask):
            raise ValueError("time split produced an empty half")
        first.append(first_mask)
        second.append(second_mask)
        boundaries.append(boundary)
    return tuple(first), tuple(second), tuple(boundaries)


def interleaved_time_bin_masks(
    blocks: tuple[VisibilityBlock, ...],
    *,
    bin_seconds: float,
) -> tuple[
    tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[float | None, ...]
]:
    """Alternate complete averaging-time bins to retain similar UV coverage."""

    if not np.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError("bin_seconds must be finite and positive")
    even: list[np.ndarray] = []
    odd: list[np.ndarray] = []
    boundaries: list[float | None] = []
    for block in blocks:
        time_bin = np.floor(block.time_s / bin_seconds).astype(np.int64)
        active_bins = np.unique(time_bin[np.any(block.active, axis=(1, 2))])
        if active_bins.size < 2:
            raise ValueError("each block needs at least two active time bins")
        even_bins = active_bins[::2]
        even_rows = np.isin(time_bin, even_bins)
        even_mask = block.active & even_rows[:, None, None]
        odd_mask = block.active & (~even_rows)[:, None, None]
        if not np.any(even_mask) or not np.any(odd_mask):
            raise ValueError("interleaved split produced an empty partition")
        even.append(even_mask)
        odd.append(odd_mask)
        boundaries.append(None)
    return tuple(even), tuple(odd), tuple(boundaries)


def _metrics(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    masks: tuple[np.ndarray, ...],
    *,
    minimum_uv_klambda: float = 0.0,
    maximum_uv_klambda: float = np.inf,
) -> dict[str, float | int]:
    speed_of_light_m_s = 299_792_458.0
    residual_power = 0.0
    signal_power = 0.0
    weight_sum = 0.0
    count = 0
    for block, prediction, mask in zip(blocks, predictions, masks, strict=True):
        uv_klambda = (
            np.linalg.norm(block.uvw_m[:, :2], axis=1)
            * float(np.mean(block.frequency_hz))
            / speed_of_light_m_s
            / 1000.0
        )
        rows = (uv_klambda >= minimum_uv_klambda) & (
            uv_klambda < maximum_uv_klambda
        )
        selected = mask & rows[:, None, None] & block.active
        if not np.any(selected):
            continue
        residual_power += float(
            np.sum(
                block.weight[selected]
                * np.abs(block.visibility[selected] - prediction[selected]) ** 2
            )
        )
        signal_power += float(
            np.sum(block.weight[selected] * np.abs(block.visibility[selected]) ** 2)
        )
        weight_sum += float(np.sum(block.weight[selected]))
        count += int(np.count_nonzero(selected))
    if count == 0 or weight_sum <= 0 or signal_power <= 0:
        raise ValueError("metric selection contains no usable samples")
    return {
        "sample_count": count,
        "weighted_complex_mse": residual_power / weight_sum,
        "normalized_residual_power": residual_power / signal_power,
    }


def _comparison(
    blocks: tuple[VisibilityBlock, ...],
    base: tuple[np.ndarray, ...],
    candidate: tuple[np.ndarray, ...],
    masks: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    bins = ((0.0, 0.75), (0.75, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, np.inf))
    result: dict[str, Any] = {}
    for label, lower, upper in (
        ("all", 0.0, np.inf),
        *((f"{lo:g}_{hi:g}_klambda", lo, hi) for lo, hi in bins[:-1]),
        ("6_inf_klambda", 6.0, np.inf),
    ):
        original = _metrics(
            blocks,
            base,
            masks,
            minimum_uv_klambda=lower,
            maximum_uv_klambda=upper,
        )
        updated = _metrics(
            blocks,
            candidate,
            masks,
            minimum_uv_klambda=lower,
            maximum_uv_klambda=upper,
        )
        result[label] = {
            "base": original,
            "candidate": updated,
            "relative_weighted_complex_mse_change": (
                updated["weighted_complex_mse"] / original["weighted_complex_mse"]
                - 1.0
            ),
        }
    return result


def fit_positive_visibility_atom(
    blocks: tuple[VisibilityBlock, ...],
    base_predictions: tuple[np.ndarray, ...],
    atoms: tuple[np.ndarray, ...],
    masks: tuple[np.ndarray, ...],
) -> float:
    """Fit one non-negative real source coefficient by weighted least squares."""

    numerator = 0.0
    denominator = 0.0
    for block, base, atom, mask in zip(
        blocks, base_predictions, atoms, masks, strict=True
    ):
        if base.shape != block.shape or atom.shape != block.shape:
            raise ValueError("base prediction and atom must match each visibility block")
        selected = mask & block.active
        weight = block.weight[selected]
        residual = block.visibility[selected] - base[selected]
        selected_atom = atom[selected]
        numerator += float(
            np.sum(weight * np.real(np.conj(selected_atom) * residual))
        )
        denominator += float(np.sum(weight * np.abs(selected_atom) ** 2))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("catalog source atom has no positive weighted response")
    return max(numerator / denominator, 0.0)


def _catalog_source_atom(
    blocks: tuple[VisibilityBlock, ...],
    *,
    ra_deg: float,
    dec_deg: float,
    beam: VLAPrimaryBeam,
) -> tuple[np.ndarray, ...]:
    ra = np.asarray([np.deg2rad(ra_deg)])
    dec = np.asarray([np.deg2rad(dec_deg)])
    atoms = []
    for block in blocks:
        local_l, local_m, _ = radec_to_lmn(
            block.phase_centre_rad[0],
            block.phase_centre_rad[1],
            ra,
            dec,
        )
        beam_weight = beam.power_weights(local_l, local_m, block.frequency_hz)
        atoms.append(
            np.asarray(
                predict_stokes_i(
                    np.ones(1),
                    local_l,
                    local_m,
                    block.uvw_m,
                    block.frequency_hz,
                    block.antenna1,
                    block.antenna2,
                    block.correlations,
                    beam_weights=beam_weight,
                )
            )
        )
    return tuple(atoms)


def _catalog_source_cross_validation(
    blocks: tuple[VisibilityBlock, ...],
    base_predictions: tuple[np.ndarray, ...],
    first_masks: tuple[np.ndarray, ...],
    second_masks: tuple[np.ndarray, ...],
    *,
    name: str,
    ra_deg: float,
    dec_deg: float,
    catalog_flux_jy: float,
) -> dict[str, Any]:
    atoms = _catalog_source_atom(
        blocks,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        beam=_beam("airy_extended"),
    )
    directions = {}
    for label, training, evaluation in (
        ("first_to_second", first_masks, second_masks),
        ("second_to_first", second_masks, first_masks),
    ):
        fitted_flux = fit_positive_visibility_atom(
            blocks,
            base_predictions,
            atoms,
            training,
        )
        candidate = tuple(
            base + fitted_flux * atom
            for base, atom in zip(base_predictions, atoms, strict=True)
        )
        directions[label] = {
            "fitted_flux_jy": fitted_flux,
            "training": _comparison(blocks, base_predictions, candidate, training),
            "cross_time_evaluation": _comparison(
                blocks,
                base_predictions,
                candidate,
                evaluation,
            ),
        }
    return {
        "name": name,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "catalog_flux_jy": catalog_flux_jy,
        "beam": "airy_extended",
        "directions": directions,
    }


def _beam(name: str) -> VLAPrimaryBeam:
    if name == "airy":
        return VLAPrimaryBeam(kind="airy")
    if name == "gaussian":
        return VLAPrimaryBeam(kind="gaussian")
    if name == "airy_extended":
        return VLAPrimaryBeam(
            kind="airy",
            catalog=replace(
                VLABeamCatalog(),
                airy_max_radius_rad_at_1ghz=np.deg2rad(4.0),
            ),
        )
    if name in {"airy_d24", "airy_d26"}:
        diameter = 24.0 if name == "airy_d24" else 26.0
        return VLAPrimaryBeam(
            kind="airy",
            catalog=replace(VLABeamCatalog(), dish_diameter_m=diameter),
        )
    raise ValueError(f"unknown beam study variant {name!r}")


def _residual_blocks(
    blocks: tuple[VisibilityBlock, ...], predictions: tuple[np.ndarray, ...]
) -> tuple[VisibilityBlock, ...]:
    return tuple(
        replace(
            block,
            visibility=block.visibility - prediction,
            model_visibility=None,
            provenance={**dict(block.provenance), "data": "frozen_sky_residual"},
        )
        for block, prediction in zip(blocks, predictions, strict=True)
    )


def mosaic_beam_sensitivity_weights(
    blocks: tuple[VisibilityBlock, ...],
    topology: Any,
    masks: tuple[np.ndarray, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    beam: VLAPrimaryBeam,
) -> np.ndarray:
    """Return relative weighted column norms from mosaic power-beam response."""

    reference_l, reference_m = topology.centers()
    sky_ra, sky_dec = lmn_to_radec(
        mosaic_phase_centre_rad[0],
        mosaic_phase_centre_rad[1],
        reference_l,
        reference_m,
    )
    sensitivity_squared = np.zeros(len(topology.leaves), dtype=np.float64)
    for block, mask in zip(blocks, masks, strict=True):
        local_l, local_m, _ = radec_to_lmn(
            block.phase_centre_rad[0],
            block.phase_centre_rad[1],
            sky_ra,
            sky_dec,
        )
        beam_weight = beam.power_weights(local_l, local_m, block.frequency_hz)
        sample_weight = np.sum(
            np.where(mask & block.active, block.weight, 0.0),
            axis=(0, 2),
        )
        sensitivity_squared += np.sum(
            np.square(beam_weight) * sample_weight[None, :],
            axis=1,
        )
    sensitivity = np.sqrt(sensitivity_squared)
    maximum = float(np.max(sensitivity))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("mosaic beam sensitivity has no positive finite entries")
    return sensitivity / maximum


def _write_component(
    path: Path,
    fit: Any,
    *,
    mosaic_phase_centre_rad: tuple[float, float],
) -> None:
    l, m = fit.topology.centers()
    widths = fit.topology.widths_rad()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("level", "iy", "ix", "flux_jy", "l_rad", "m_rad", "width_rad")
        )
        for leaf, flux, leaf_l, leaf_m, width in zip(
            fit.topology.leaves, fit.flux, l, m, widths, strict=True
        ):
            writer.writerow(
                (leaf.level, leaf.iy, leaf.ix, flux, leaf_l, leaf_m, width)
            )


def _component_summary(fit: Any, *, original_half_width_rad: float) -> dict[str, Any]:
    l, m = fit.topology.centers()
    radius = np.sqrt(l * l + m * m)
    flux = np.asarray(fit.flux)
    outside = (np.abs(l) > original_half_width_rad) | (
        np.abs(m) > original_half_width_rad
    )
    order = np.argsort(flux)[::-1][:20]
    return {
        "steps": fit.steps,
        "best_step": fit.best_step,
        "converged": fit.converged,
        "kkt_residual": fit.kkt_residual,
        "total_flux_jy": float(np.sum(flux)),
        "outside_original_field_flux_jy": float(np.sum(flux[outside])),
        "outside_original_field_fraction": float(
            np.sum(flux[outside]) / max(np.sum(flux), np.finfo(float).tiny)
        ),
        "brightest_leaves": [
            {
                "flux_jy": float(flux[index]),
                "l_arcmin": float(np.rad2deg(l[index]) * 60.0),
                "m_arcmin": float(np.rad2deg(m[index]) * 60.0),
                "radius_arcmin": float(np.rad2deg(radius[index]) * 60.0),
            }
            for index in order
            if flux[index] > 0
        ],
    }


def _fit_direction(
    blocks: tuple[VisibilityBlock, ...],
    residual_blocks: tuple[VisibilityBlock, ...],
    base_predictions: tuple[np.ndarray, ...],
    train_masks: tuple[np.ndarray, ...],
    evaluation_masks: tuple[np.ndarray, ...],
    *,
    beam_name: str,
    coarse_size: int,
    coarse_pixel_arcsec: float,
    steps: int,
    lambda_l1: float,
    kkt_tolerance: float,
    direct_config: DirectDFTConfig,
    output: Path,
    label: str,
    original_half_width_rad: float,
    penalty: str,
) -> dict[str, Any]:
    sky = quadtree_sky_from_regular_grid(
        coarse_size,
        np.deg2rad(coarse_pixel_arcsec / 3600.0),
        np.zeros(coarse_size**2),
    )
    configuration = InferenceConfig(
        solver="fista",
        steps=steps,
        sparsity_weight=lambda_l1,
        kkt_tolerance=kkt_tolerance,
        validation_interval=25,
        operator_mode="explicit",
        direct_dft=direct_config,
    )
    selected_beam = _beam(beam_name)
    sparsity_weights = (
        None
        if penalty == "intrinsic_flux"
        else mosaic_beam_sensitivity_weights(
            blocks,
            sky.topology,
            train_masks,
            blocks[0].phase_centre_rad,
            selected_beam,
        )
    )
    print(
        f"{beam_name} {label}: fitting coarse residual sky with {penalty} penalty",
        flush=True,
    )
    fit = infer_mosaic_quadtree(
        residual_blocks,
        sky.topology,
        train_masks,
        blocks[0].phase_centre_rad,
        configuration,
        primary_beam=selected_beam,
        approximation=GaussianApproximation.WIDE_FIELD,
        initial_flux=sky.flux,
        sparsity_weights=sparsity_weights,
    )
    candidate = tuple(
        base + addition
        for base, addition in zip(base_predictions, fit.predictions, strict=True)
    )
    _write_component(
        output / f"{beam_name}_{label}_component.csv",
        fit,
        mosaic_phase_centre_rad=blocks[0].phase_centre_rad,
    )
    return {
        "penalty": penalty,
        "sensitivity_weight_summary": None
        if sparsity_weights is None
        else {
            "minimum": float(np.min(sparsity_weights)),
            "median": float(np.median(sparsity_weights)),
            "maximum": float(np.max(sparsity_weights)),
            "positive_count": int(np.count_nonzero(sparsity_weights > 0)),
        },
        "component": _component_summary(
            fit, original_half_width_rad=original_half_width_rad
        ),
        "training": _comparison(blocks, base_predictions, candidate, train_masks),
        "cross_time_evaluation": _comparison(
            blocks, base_predictions, candidate, evaluation_masks
        ),
    }


def _parse_beams(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("beams must be a unique comma-separated list")
    for item in result:
        try:
            _beam(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--frozen-directory",
        type=Path,
        default=Path("outputs/3c391_mosaic_hierarchical_frozen_104"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_short_baseline_study"),
    )
    parser.add_argument(
        "--beams",
        type=_parse_beams,
        default=("airy", "airy_extended", "gaussian", "airy_d24", "airy_d26"),
    )
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument(
        "--catalog-source",
        nargs=4,
        action="append",
        metavar=("NAME", "RA_DEG", "DEC_DEG", "CATALOG_FLUX_JY"),
        help="cross-validate a fixed-position delta source through extended Airy sidelobes",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("contiguous_halves", "interleaved_time_bins"),
        default="contiguous_halves",
    )
    parser.add_argument("--interleaved-bin-seconds", type=float, default=60.0)
    parser.add_argument("--coarse-size", type=int, default=64)
    parser.add_argument("--coarse-pixel-arcsec", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lambda-l1", type=float, default=3e-4)
    parser.add_argument(
        "--penalty",
        choices=("intrinsic_flux", "beam_sensitivity"),
        default="intrinsic_flux",
    )
    parser.add_argument("--kkt-tolerance", type=float, default=3e-5)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    arguments = parser.parse_args()

    blocks = read_dataset(arguments.fixture).blocks
    with np.load(arguments.frozen_directory / "predictions.npz") as stored:
        predictions = tuple(
            stored[f"consensus_C{index + 1}"] for index in range(len(blocks))
        )
    if arguments.split_strategy == "contiguous_halves":
        first, second, boundaries = time_half_masks(blocks)
    else:
        first, second, boundaries = interleaved_time_bin_masks(
            blocks,
            bin_seconds=arguments.interleaved_bin_seconds,
        )
    first_to_second = audit_visibility_residuals(
        blocks,
        predictions,
        first,
        second,
        group_kinds=("pointing", "baseline", "antenna", "channel", "correlation", "scan"),
        minimum_group_samples=128,
        minimum_group_outlier_fraction=0.2,
    )
    second_to_first = audit_visibility_residuals(
        blocks,
        predictions,
        second,
        first,
        group_kinds=("pointing", "baseline", "antenna", "channel", "correlation", "scan"),
        minimum_group_samples=128,
        minimum_group_outlier_fraction=0.2,
    )
    first_baselines = {
        group.key: group
        for group in first_to_second.groups
        if group.kind == "baseline"
        and group.discovery.sample_count > 0
        and group.evaluation.sample_count > 0
    }
    discovery_fraction = np.asarray(
        [group.discovery.outlier_fraction for group in first_baselines.values()]
    )
    evaluation_fraction = np.asarray(
        [group.evaluation.outlier_fraction for group in first_baselines.values()]
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "fixture": str(arguments.fixture),
        "frozen_directory": str(arguments.frozen_directory),
        "split_strategy": arguments.split_strategy,
        "interleaved_bin_seconds": arguments.interleaved_bin_seconds,
        "time_boundaries_s": list(boundaries),
        "time_half_audit": {
            "first_to_second": asdict(first_to_second),
            "second_to_first": asdict(second_to_first),
            "baseline_outlier_fraction_correlation": float(
                np.corrcoef(discovery_fraction, evaluation_fraction)[0, 1]
            ),
        },
        "coarse_configuration": {
            "size": arguments.coarse_size,
            "pixel_arcsec": arguments.coarse_pixel_arcsec,
            "field_of_view_arcmin": (
                arguments.coarse_size * arguments.coarse_pixel_arcsec / 60.0
            ),
            "steps": arguments.steps,
            "lambda_l1": arguments.lambda_l1,
            "kkt_tolerance": arguments.kkt_tolerance,
            "penalty": arguments.penalty,
        },
        "beam_variants": {},
        "catalog_sources": [],
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    if not arguments.diagnostics_only:
        residual_blocks = _residual_blocks(blocks, predictions)
        direct_config = DirectDFTConfig(
            visibility_chunk_size=arguments.visibility_tile_size,
            pixel_chunk_size=arguments.pixel_tile_size,
            precision=arguments.precision,
        )
        frozen_summary = json.loads(
            (arguments.frozen_directory / "summary.json").read_text(encoding="utf-8")
        )
        original_half_width_rad = (
            int(frozen_summary["root_size"])
            * np.deg2rad(float(frozen_summary["root_pixel_arcsec"]) / 3600.0)
            / 2.0
        )
        for beam_name in arguments.beams:
            variant_path = arguments.output / f"{beam_name}.json"
            if variant_path.exists():
                print(f"{beam_name}: loading cached result", flush=True)
                variant = json.loads(variant_path.read_text(encoding="utf-8"))
            else:
                variant = {
                    "first_to_second": _fit_direction(
                        blocks,
                        residual_blocks,
                        predictions,
                        first,
                        second,
                        beam_name=beam_name,
                        coarse_size=arguments.coarse_size,
                        coarse_pixel_arcsec=arguments.coarse_pixel_arcsec,
                        steps=arguments.steps,
                        lambda_l1=arguments.lambda_l1,
                        kkt_tolerance=arguments.kkt_tolerance,
                        direct_config=direct_config,
                        output=arguments.output,
                        label="first_to_second",
                        original_half_width_rad=original_half_width_rad,
                        penalty=arguments.penalty,
                    ),
                    "second_to_first": _fit_direction(
                        blocks,
                        residual_blocks,
                        predictions,
                        second,
                        first,
                        beam_name=beam_name,
                        coarse_size=arguments.coarse_size,
                        coarse_pixel_arcsec=arguments.coarse_pixel_arcsec,
                        steps=arguments.steps,
                        lambda_l1=arguments.lambda_l1,
                        kkt_tolerance=arguments.kkt_tolerance,
                        direct_config=direct_config,
                        output=arguments.output,
                        label="second_to_first",
                        original_half_width_rad=original_half_width_rad,
                        penalty=arguments.penalty,
                    ),
                }
                variant_path.write_text(
                    json.dumps(variant, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            summary["beam_variants"][beam_name] = variant
    for source in arguments.catalog_source or ():
        name, ra_text, dec_text, flux_text = source
        summary["catalog_sources"].append(
            _catalog_source_cross_validation(
                blocks,
                predictions,
                first,
                second,
                name=name,
                ra_deg=float(ra_text),
                dec_deg=float(dec_text),
                catalog_flux_jy=float(flux_text),
            )
        )
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "first_half": asdict(first_to_second.discovery),
                "second_half": asdict(first_to_second.evaluation),
                "baseline_outlier_fraction_correlation": summary["time_half_audit"][
                    "baseline_outlier_fraction_correlation"
                ],
                "beam_variants": summary["beam_variants"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
