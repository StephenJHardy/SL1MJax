"""Fit one pointing-aware hierarchical sky to the seven-field 3C391 mosaic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from sl1mjax.beam import primary_beam_from_name
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import (
    InferenceConfig,
    MosaicQuadtreeInferenceResult,
    infer_mosaic_quadtree,
)
from sl1mjax.objective import (
    normalized_weighted_complex_mse,
    weighted_complex_mse,
)
from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeLeaf,
    QuadtreeTopology,
)
from sl1mjax.refinement import render_quadtree_surface_brightness
from sl1mjax.sky import GaussianApproximation


def _load_embedded_topology(
    path: Path,
    *,
    source_root_size: int,
    destination_root_size: int,
    root_pixel_size_rad: float,
) -> QuadtreeTopology:
    if destination_root_size < source_root_size:
        raise ValueError("destination root must not be smaller than source root")
    difference = destination_root_size - source_root_size
    if difference % 2:
        raise ValueError("source and destination root sizes need the same parity")
    offset = difference // 2
    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    source_leaves = tuple(
        QuadtreeLeaf(int(row["level"]), int(row["iy"]), int(row["ix"]))
        for row in rows
    )
    shifted = tuple(
        QuadtreeLeaf(
            leaf.level,
            leaf.iy + offset * 2**leaf.level,
            leaf.ix + offset * 2**leaf.level,
        )
        for leaf in source_leaves
    )
    grid = QuadtreeGrid(destination_root_size, root_pixel_size_rad)
    outer = tuple(
        leaf
        for leaf in grid.root_leaves()
        if not (
            offset <= leaf.iy < offset + source_root_size
            and offset <= leaf.ix < offset + source_root_size
        )
    )
    return QuadtreeTopology(grid, outer + shifted)


def _initial_hierarchical_flux(
    base_topology: QuadtreeTopology,
    base_flux: np.ndarray,
    hierarchy: QuadtreeTopology,
) -> np.ndarray:
    base_by_leaf = dict(zip(base_topology.leaves, base_flux, strict=True))
    return np.asarray(
        [
            base_by_leaf[
                QuadtreeLeaf(0, leaf.iy // 2**leaf.level, leaf.ix // 2**leaf.level)
            ]
            / 4**leaf.level
            for leaf in hierarchy.leaves
        ],
        dtype=np.float64,
    )


def _global_metrics(
    blocks: tuple[VisibilityBlock, ...],
    result: MosaicQuadtreeInferenceResult,
) -> dict[str, object]:
    per_pointing = []
    residual_numerator = 0.0
    signal_numerator = 0.0
    weight_sum = 0.0
    for index, (block, prediction) in enumerate(
        zip(blocks, result.predictions, strict=True), start=1
    ):
        active = block.active
        effective_weight = np.where(active, block.weight, 0.0)
        residual = np.where(active, prediction - block.visibility, 0.0)
        residual_numerator += float(
            np.sum(effective_weight * np.abs(residual) ** 2)
        )
        signal_numerator += float(
            np.sum(effective_weight * np.abs(block.visibility) ** 2)
        )
        weight_sum += float(np.sum(effective_weight))
        per_pointing.append(
            {
                "label": f"C{index}",
                "field_id": int(block.field_id[0]),
                "phase_centre_deg": [
                    float(np.rad2deg(block.phase_centre_rad[0])),
                    float(np.rad2deg(block.phase_centre_rad[1])),
                ],
                "active_samples": int(np.count_nonzero(active)),
                "weighted_complex_mse": float(
                    weighted_complex_mse(
                        prediction,
                        block.visibility,
                        block.weight,
                        ~active,
                    )
                ),
                "normalized_residual_power": float(
                    normalized_weighted_complex_mse(
                        prediction,
                        block.visibility,
                        block.weight,
                        ~active,
                    )
                ),
            }
        )
    return {
        "leaf_count": len(result.topology.leaves),
        "total_flux_jy": float(np.sum(result.flux)),
        "steps": result.steps,
        "best_step": result.best_step,
        "converged": result.converged,
        "kkt_residual": result.kkt_residual,
        "weighted_complex_mse": residual_numerator / weight_sum,
        "normalized_residual_power": residual_numerator / signal_numerator,
        "per_pointing": per_pointing,
    }


def _write_reconstruction(
    path: Path,
    result: MosaicQuadtreeInferenceResult,
) -> None:
    deepest_level = max(leaf.level for leaf in result.topology.leaves)
    pixel_size_rad = result.topology.grid.root_pixel_size_rad / 2**deepest_level
    brightness = render_quadtree_surface_brightness(
        result.topology,
        result.flux,
        level=deepest_level,
    )
    image = brightness * pixel_size_rad**2
    header = fits.Header()
    header["BUNIT"] = "Jy/pixel"
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = (image.shape[1] + 1) / 2
    header["CRPIX2"] = (image.shape[0] + 1) / 2
    header["CRVAL1"] = np.rad2deg(result.mosaic_phase_centre_rad[0])
    header["CRVAL2"] = np.rad2deg(result.mosaic_phase_centre_rad[1])
    header["CDELT1"] = -np.rad2deg(pixel_size_rad)
    header["CDELT2"] = np.rad2deg(pixel_size_rad)
    fits.PrimaryHDU(image, header=header).writeto(path, overwrite=True)


def _write_topology(
    path: Path,
    result: MosaicQuadtreeInferenceResult,
) -> None:
    l, m = result.topology.centers()
    widths = result.topology.widths_rad()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level", "iy", "ix", "flux_jy", "l_rad", "m_rad", "width_rad"))
        for leaf, flux, leaf_l, leaf_m, width in zip(
            result.topology.leaves,
            result.flux,
            l,
            m,
            widths,
            strict=True,
        ):
            writer.writerow((leaf.level, leaf.iy, leaf.ix, flux, leaf_l, leaf_m, width))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("outputs/3c391_mosaic_imaging_fixture.zarr"),
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path(
            "outputs/3c391_hierarchical_airy_fista_16arcsec_lambda3e-4_"
            "consensus_all/consensus_topology.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_mosaic_joint_fista"),
    )
    parser.add_argument("--root-size", type=int, default=96)
    parser.add_argument("--source-root-size", type=int, default=64)
    parser.add_argument("--root-pixel-arcsec", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lambda-l1", type=float, default=3e-4)
    parser.add_argument("--kkt-tolerance", type=float, default=3e-5)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--maximum-pointings",
        type=int,
        help="Use the first N pointings for a smoke test.",
    )
    arguments = parser.parse_args()

    dataset = read_dataset(arguments.fixture)
    blocks = dataset.blocks
    if arguments.maximum_pointings is not None:
        if not 1 <= arguments.maximum_pointings <= len(blocks):
            raise ValueError("maximum-pointings must be between one and the block count")
        blocks = blocks[: arguments.maximum_pointings]
    root_pixel_size_rad = np.deg2rad(arguments.root_pixel_arcsec / 3600.0)
    base_grid = QuadtreeGrid(arguments.root_size, root_pixel_size_rad)
    base_topology = QuadtreeTopology(base_grid, base_grid.root_leaves())
    hierarchy = _load_embedded_topology(
        arguments.topology,
        source_root_size=arguments.source_root_size,
        destination_root_size=arguments.root_size,
        root_pixel_size_rad=root_pixel_size_rad,
    )
    mosaic_phase_centre = blocks[0].phase_centre_rad
    config = InferenceConfig(
        solver="fista",
        steps=arguments.steps,
        sparsity_weight=arguments.lambda_l1,
        kkt_tolerance=arguments.kkt_tolerance,
        validation_interval=25,
        operator_mode="explicit",
        direct_dft=DirectDFTConfig(
            visibility_chunk_size=arguments.visibility_tile_size,
            pixel_chunk_size=arguments.pixel_tile_size,
            precision=arguments.precision,  # type: ignore[arg-type]
        ),
    )
    primary_beam = primary_beam_from_name("airy")
    train_masks = tuple(block.active for block in blocks)
    arguments.output.mkdir(parents=True, exist_ok=True)

    print(
        f"joint baseline: {len(blocks)} pointings, {len(base_topology.leaves)} leaves",
        flush=True,
    )
    baseline = infer_mosaic_quadtree(
        blocks,
        base_topology,
        train_masks,
        mosaic_phase_centre,
        config,
        primary_beam=primary_beam,
        approximation=GaussianApproximation.WIDE_FIELD,
        initial_flux=np.zeros(len(base_topology.leaves)),
    )
    print(
        f"baseline complete: steps={baseline.steps}, KKT={baseline.kkt_residual:.3g}",
        flush=True,
    )
    hierarchical_initial = _initial_hierarchical_flux(
        baseline.topology,
        baseline.flux,
        hierarchy,
    )
    print(f"joint hierarchy: {len(hierarchy.leaves)} leaves", flush=True)
    hierarchical = infer_mosaic_quadtree(
        blocks,
        hierarchy,
        train_masks,
        mosaic_phase_centre,
        config,
        primary_beam=primary_beam,
        approximation=GaussianApproximation.WIDE_FIELD,
        initial_flux=hierarchical_initial,
    )
    print(
        f"hierarchy complete: steps={hierarchical.steps}, "
        f"KKT={hierarchical.kkt_residual:.3g}",
        flush=True,
    )

    baseline_metrics = _global_metrics(blocks, baseline)
    hierarchical_metrics = _global_metrics(blocks, hierarchical)
    baseline_power = float(baseline_metrics["normalized_residual_power"])
    hierarchical_power = float(hierarchical_metrics["normalized_residual_power"])
    summary = {
        "schema_version": 1,
        "fixture": str(arguments.fixture),
        "pointing_count": len(blocks),
        "mosaic_phase_centre_deg": [
            float(np.rad2deg(mosaic_phase_centre[0])),
            float(np.rad2deg(mosaic_phase_centre[1])),
        ],
        "configuration": {
            "root_size": arguments.root_size,
            "root_pixel_arcsec": arguments.root_pixel_arcsec,
            "field_of_view_arcmin": (
                arguments.root_size * arguments.root_pixel_arcsec / 60.0
            ),
            "lambda_l1": arguments.lambda_l1,
            "steps": arguments.steps,
            "kkt_tolerance": arguments.kkt_tolerance,
            "primary_beam": "airy",
            "approximation": "wide_field",
        },
        "baseline": baseline_metrics,
        "hierarchical": hierarchical_metrics,
        "hierarchical_improvement_over_baseline": {
            "absolute_normalized_residual_power": baseline_power - hierarchical_power,
            "relative": 1.0 - hierarchical_power / baseline_power,
        },
    }
    _write_reconstruction(arguments.output / "baseline_reconstruction.fits", baseline)
    _write_reconstruction(
        arguments.output / "hierarchical_reconstruction.fits",
        hierarchical,
    )
    _write_topology(arguments.output / "hierarchical_topology.csv", hierarchical)
    np.savez(
        arguments.output / "predictions.npz",
        **{
            f"baseline_C{index + 1}": prediction
            for index, prediction in enumerate(baseline.predictions)
        },
        **{
            f"hierarchical_C{index + 1}": prediction
            for index, prediction in enumerate(hierarchical.predictions)
        },
    )
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
