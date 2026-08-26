#!/usr/bin/env python3
"""Cross-validate a central hierarchy plus wide-field and catalogue sky atoms."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.catalog import (
    CatalogGuardAtom,
    RadioCatalogSource,
    read_radio_catalog,
    select_catalog_guard_atoms,
)
from sl1mjax.composite import (
    MosaicPointComponent,
    MosaicQuadtreeComponent,
    MosaicSkyComponent,
    infer_mosaic_composite,
    mosaic_beam_sensitivity_weights,
)
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.inference import InferenceConfig
from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeLeaf,
    QuadtreeTopology,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.split import interleaved_time_fold_masks

VARIANT_COMPONENTS = {
    "central": ("central",),
    "central_catalog": ("central", "catalogue"),
    "central_coarse": ("central", "coarse"),
    "full": ("central", "coarse", "catalogue"),
}


def _load_topology(
    path: Path,
    *,
    root_size: int,
    root_pixel_size_rad: float,
) -> QuadtreeTopology:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    leaves = tuple(QuadtreeLeaf(int(row["level"]), int(row["iy"]), int(row["ix"])) for row in rows)
    if not leaves:
        raise ValueError("topology CSV must contain at least one leaf")
    return QuadtreeTopology(QuadtreeGrid(root_size, root_pixel_size_rad), leaves)


def _parse_csv_names(value: str, *, allowed: set[str]) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("value must be a unique comma-separated list")
    unknown = set(names) - allowed
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown values: {sorted(unknown)}")
    return names


def _parse_lambdas(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("lambdas must be comma-separated numbers") from error
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("lambdas must be a unique comma-separated list")
    if any(not np.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError("lambdas must be finite and non-negative")
    return values


def _command_line_catalog_sources(
    values: list[list[str]],
    *,
    reference_frequency_hz: float,
) -> tuple[RadioCatalogSource, ...]:
    return tuple(
        RadioCatalogSource(
            name=name,
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            reference_frequency_hz=reference_frequency_hz,
            integrated_flux_jy=float(flux_jy),
            catalog="command_line",
            reference_url="user-supplied",
        )
        for name, ra_deg, dec_deg, flux_jy in values
    )


def _component_templates(
    central_topology: QuadtreeTopology,
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    coarse_size: int,
    coarse_pixel_arcsec: float,
    catalog_atoms: tuple[CatalogGuardAtom, ...],
) -> dict[str, MosaicSkyComponent]:
    coarse = quadtree_sky_from_regular_grid(
        coarse_size,
        np.deg2rad(coarse_pixel_arcsec / 3600.0),
        np.zeros(coarse_size**2),
    )
    return {
        "central": MosaicQuadtreeComponent(
            "central",
            central_topology,
            np.zeros(len(central_topology.leaves)),
        ),
        "coarse": MosaicQuadtreeComponent("coarse", coarse.topology, coarse.flux),
        "catalogue": MosaicPointComponent(
            "catalogue",
            np.asarray([atom.l_rad for atom in catalog_atoms]),
            np.asarray([atom.m_rad for atom in catalog_atoms]),
            np.asarray([atom.initial_flux_jy for atom in catalog_atoms]),
        ),
    }


def _metrics(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    masks: tuple[np.ndarray, ...],
    *,
    maximum_uv_klambda: float = np.inf,
) -> dict[str, float | int]:
    residual_power = 0.0
    signal_power = 0.0
    weight_sum = 0.0
    sample_count = 0
    for block, prediction, mask in zip(blocks, predictions, masks, strict=True):
        uv_klambda = (
            np.linalg.norm(block.uvw_m[:, :2], axis=1)
            * np.mean(block.frequency_hz)
            / 299_792_458.0
            / 1_000.0
        )
        selected = mask & block.active & (uv_klambda < maximum_uv_klambda)[:, None, None]
        residual = block.visibility[selected] - prediction[selected]
        weight = block.weight[selected]
        residual_power += float(np.sum(weight * np.abs(residual) ** 2))
        signal_power += float(np.sum(weight * np.abs(block.visibility[selected]) ** 2))
        weight_sum += float(np.sum(weight))
        sample_count += int(np.count_nonzero(selected))
    if sample_count == 0 or weight_sum <= 0 or signal_power <= 0:
        raise ValueError("metric mask contains no usable samples")
    return {
        "sample_count": sample_count,
        "weighted_complex_mse": residual_power / weight_sum,
        "normalized_residual_power": residual_power / signal_power,
    }


def _fit_metrics(
    blocks: tuple[VisibilityBlock, ...],
    predictions: tuple[np.ndarray, ...],
    train_masks: tuple[np.ndarray, ...],
    validation_masks: tuple[np.ndarray, ...],
    test_masks: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    return {
        label: {
            "all": _metrics(blocks, predictions, masks),
            "shorter_than_0.75_klambda": _metrics(
                blocks,
                predictions,
                masks,
                maximum_uv_klambda=0.75,
            ),
        }
        for label, masks in (
            ("train", train_masks),
            ("validation", validation_masks),
            ("test", test_masks),
        )
    }


def _candidate_stem(variant: str, lambda_l1: float) -> str:
    return f"{variant}_lambda_{lambda_l1:.8g}".replace("+", "")


def _catalogue_flux_payload(
    components: tuple[MosaicSkyComponent, ...],
    catalog_atoms: tuple[CatalogGuardAtom, ...],
) -> dict[str, float]:
    catalogue = next((component for component in components if component.name == "catalogue"), None)
    if catalogue is None:
        return {}
    if catalogue.flux.size != len(catalog_atoms):
        raise ValueError("catalogue component and atom metadata have different lengths")
    return {
        atom.source.name: float(flux)
        for atom, flux in zip(catalog_atoms, catalogue.flux, strict=True)
    }


def _save_candidate(
    output: Path,
    stem: str,
    result: Any,
    payload: dict[str, Any],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for component in result.components:
        arrays[f"flux_{component.name}"] = np.asarray(component.flux)
    for block_index, prediction in enumerate(result.predictions, start=1):
        arrays[f"prediction_C{block_index}"] = prediction
    np.savez(output / f"{stem}.npz", **arrays)
    (output / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_candidate(
    output: Path,
    stem: str,
    components: tuple[MosaicSkyComponent, ...],
    block_count: int,
) -> tuple[tuple[MosaicSkyComponent, ...], tuple[np.ndarray, ...], dict[str, Any]] | None:
    json_path = output / f"{stem}.json"
    array_path = output / f"{stem}.npz"
    if not json_path.exists() or not array_path.exists():
        return None
    with np.load(array_path) as stored:
        fitted = tuple(
            replace(component, flux=stored[f"flux_{component.name}"]) for component in components
        )
        predictions = tuple(stored[f"prediction_C{index}"] for index in range(1, block_count + 1))
    return fitted, predictions, json.loads(json_path.read_text(encoding="utf-8"))


def _load_initial_components(
    directory: Path,
    stem: str,
    components: tuple[MosaicSkyComponent, ...],
) -> tuple[MosaicSkyComponent, ...]:
    """Load fitted component fluxes as a fresh optimizer warm start."""

    path = directory / f"{stem}.npz"
    if not path.exists():
        raise ValueError(f"initial candidate does not exist: {path}")
    fitted: list[MosaicSkyComponent] = []
    with np.load(path) as stored:
        for component in components:
            key = f"flux_{component.name}"
            if key not in stored:
                raise ValueError(f"initial candidate lacks {key}")
            flux = np.asarray(stored[key], dtype=np.float64)
            if flux.shape != component.flux.shape or np.any(~np.isfinite(flux)):
                raise ValueError(f"initial candidate {key} has incompatible values")
            if np.any(flux < 0):
                raise ValueError(f"initial candidate {key} contains negative flux")
            fitted.append(replace(component, flux=flux))
    return tuple(fitted)


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
        default=Path("outputs/3c391_composite_frozen_protocol"),
    )
    parser.add_argument(
        "--variants",
        type=lambda value: _parse_csv_names(value, allowed=set(VARIANT_COMPONENTS)),
        default=tuple(VARIANT_COMPONENTS),
    )
    parser.add_argument("--lambdas", type=_parse_lambdas, default=(1e-4, 3e-4, 1e-3))
    parser.add_argument("--time-bin-seconds", type=float, default=60.0)
    parser.add_argument("--coarse-size", type=int, default=64)
    parser.add_argument("--coarse-pixel-arcsec", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--kkt-tolerance", type=float, default=3e-5)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--airy-max-radius-deg-at-1ghz", type=float, default=4.0)
    parser.add_argument(
        "--catalog-file",
        type=Path,
        default=Path("config/3c391_radio_guard_catalog.csv"),
    )
    parser.add_argument("--catalog-reference-frequency-hz", type=float, default=3.0e9)
    parser.add_argument("--minimum-catalog-apparent-flux-mjy", type=float, default=0.5)
    parser.add_argument("--maximum-catalog-major-axis-arcsec", type=float, default=45.0)
    parser.add_argument("--default-catalog-spectral-index", type=float, default=-0.7)
    parser.add_argument(
        "--catalog-source",
        nargs=4,
        action="append",
        metavar=("NAME", "RA_DEG", "DEC_DEG", "CATALOG_FLUX_JY"),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--initial-candidate-directory",
        type=Path,
        help="Warm-start requested variant/lambda fits from saved component fluxes.",
    )
    arguments = parser.parse_args()

    blocks = read_dataset(arguments.fixture).blocks
    frozen_summary = json.loads(
        (arguments.frozen_directory / "summary.json").read_text(encoding="utf-8")
    )
    root_size = int(frozen_summary["root_size"])
    root_pixel_arcsec = float(frozen_summary["root_pixel_arcsec"])
    central_topology = _load_topology(
        arguments.frozen_directory / "consensus_topology.csv",
        root_size=root_size,
        root_pixel_size_rad=np.deg2rad(root_pixel_arcsec / 3600.0),
    )
    mosaic_phase_centre = blocks[0].phase_centre_rad
    beam = VLAPrimaryBeam(
        kind="airy",
        catalog=replace(
            VLABeamCatalog(),
            airy_max_radius_rad_at_1ghz=np.deg2rad(arguments.airy_max_radius_deg_at_1ghz),
        ),
    )
    source_catalog = (
        read_radio_catalog(arguments.catalog_file)
        if arguments.catalog_source is None
        else _command_line_catalog_sources(
            arguments.catalog_source,
            reference_frequency_hz=arguments.catalog_reference_frequency_hz,
        )
    )
    central_half_width = root_size * np.deg2rad(root_pixel_arcsec / 3600.0) / 2.0
    catalog_atoms = select_catalog_guard_atoms(
        source_catalog,
        blocks,
        mosaic_phase_centre,
        primary_beam=beam,
        central_half_width_rad=(central_half_width, central_half_width),
        minimum_apparent_flux_jy=(arguments.minimum_catalog_apparent_flux_mjy / 1_000.0),
        default_spectral_index=arguments.default_catalog_spectral_index,
        maximum_major_axis_arcsec=arguments.maximum_catalog_major_axis_arcsec,
    )
    if not catalog_atoms and any(
        "catalogue" in VARIANT_COMPONENTS[variant] for variant in arguments.variants
    ):
        raise ValueError("catalog selection produced no usable guard atoms")
    templates = _component_templates(
        central_topology,
        mosaic_phase_centre,
        coarse_size=arguments.coarse_size,
        coarse_pixel_arcsec=arguments.coarse_pixel_arcsec,
        catalog_atoms=catalog_atoms,
    )
    train_masks, validation_masks, test_masks = interleaved_time_fold_masks(
        blocks,
        bin_seconds=arguments.time_bin_seconds,
    )
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.visibility_tile_size,
        pixel_chunk_size=arguments.pixel_tile_size,
        precision=arguments.precision,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "fixture": str(arguments.fixture),
        "frozen_directory": str(arguments.frozen_directory),
        "central_topology_leaf_count": len(central_topology.leaves),
        "central_initial_flux": "zero_to_avoid_test-leaking_warm_start",
        "initial_candidate_directory": (
            None
            if arguments.initial_candidate_directory is None
            else str(arguments.initial_candidate_directory)
        ),
        "variants": list(arguments.variants),
        "lambdas": list(arguments.lambdas),
        "time_split": {
            "kind": "five_fold_interleaved_time_bins",
            "bin_seconds": arguments.time_bin_seconds,
            "training_folds": [0, 1, 2],
            "validation_fold": 3,
            "sealed_test_fold": 4,
        },
        "coarse_size": arguments.coarse_size,
        "coarse_pixel_arcsec": arguments.coarse_pixel_arcsec,
        "catalog_file": (
            None if arguments.catalog_source is not None else str(arguments.catalog_file)
        ),
        "catalog_atoms": [
            {
                "name": atom.source.name,
                "ra_deg": atom.source.ra_deg,
                "dec_deg": atom.source.dec_deg,
                "reference_frequency_hz": atom.source.reference_frequency_hz,
                "catalog_flux_jy": atom.source.integrated_flux_jy,
                "initial_flux_jy": atom.initial_flux_jy,
                "maximum_apparent_flux_jy": atom.maximum_apparent_flux_jy,
                "maximum_beam_power": atom.maximum_beam_power,
                "offset_arcmin": atom.offset_arcmin,
                "catalog": atom.source.catalog,
                "reference_url": atom.source.reference_url,
            }
            for atom in catalog_atoms
        ],
        "catalog_selection": {
            "minimum_apparent_flux_mjy": arguments.minimum_catalog_apparent_flux_mjy,
            "maximum_major_axis_arcsec": arguments.maximum_catalog_major_axis_arcsec,
            "default_spectral_index": arguments.default_catalog_spectral_index,
        },
        "primary_beam": "extended_airy",
        "airy_max_radius_deg_at_1ghz": arguments.airy_max_radius_deg_at_1ghz,
        "steps": arguments.steps,
        "validation_interval": arguments.validation_interval,
        "kkt_tolerance": arguments.kkt_tolerance,
        "penalty": "globally_normalized_beam_sensitivity_weighted_positive_l1",
        "precision": arguments.precision,
    }
    protocol_path = arguments.output / "protocol.json"
    if protocol_path.exists() and not arguments.no_resume:
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("existing output protocol does not match this invocation")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    candidates: dict[str, list[dict[str, Any]]] = {variant: [] for variant in arguments.variants}
    warm_components: dict[tuple[str, float], tuple[MosaicSkyComponent, ...]] = {}
    for lambda_l1 in arguments.lambdas:
        for variant in arguments.variants:
            component_names = VARIANT_COMPONENTS[variant]
            components = tuple(templates[name] for name in component_names)
            if variant != "central" and ("central", lambda_l1) in warm_components:
                central = warm_components[("central", lambda_l1)][0]
                components = tuple(
                    central if component.name == "central" else component
                    for component in components
                )
            if arguments.initial_candidate_directory is not None:
                components = _load_initial_components(
                    arguments.initial_candidate_directory,
                    _candidate_stem(variant, lambda_l1),
                    components,
                )
            sensitivity_weights = mosaic_beam_sensitivity_weights(
                blocks,
                components,
                train_masks,
                mosaic_phase_centre,
                primary_beam=beam,
            )
            components = tuple(
                replace(component, sparsity_weights=weights)
                for component, weights in zip(components, sensitivity_weights, strict=True)
            )
            stem = _candidate_stem(variant, lambda_l1)
            cached = (
                None
                if arguments.no_resume
                else _load_candidate(
                    arguments.output,
                    stem,
                    components,
                    len(blocks),
                )
            )
            if cached is None:
                print(
                    f"{variant}, lambda={lambda_l1:g}: fitting "
                    f"{sum(component.flux.size for component in components)} atoms",
                    flush=True,
                )
                result = infer_mosaic_composite(
                    blocks,
                    components,
                    train_masks,
                    mosaic_phase_centre,
                    InferenceConfig(
                        solver="fista",
                        steps=arguments.steps,
                        sparsity_weight=lambda_l1,
                        validation_interval=arguments.validation_interval,
                        kkt_tolerance=arguments.kkt_tolerance,
                        operator_mode="explicit",
                        direct_dft=direct,
                    ),
                    holdout_masks=validation_masks,
                    primary_beam=beam,
                )
                fitted_components = result.components
                predictions = result.predictions
                payload = {
                    "variant": variant,
                    "lambda_l1": lambda_l1,
                    "steps": result.steps,
                    "best_step": result.best_step,
                    "converged": result.converged,
                    "kkt_residual": result.kkt_residual,
                    "component_flux_jy": {
                        component.name: float(np.sum(component.flux))
                        for component in result.components
                    },
                    "catalogue_atom_flux_jy": _catalogue_flux_payload(
                        result.components,
                        catalog_atoms,
                    ),
                    "metrics": _fit_metrics(
                        blocks,
                        predictions,
                        train_masks,
                        validation_masks,
                        test_masks,
                    ),
                }
                _save_candidate(arguments.output, stem, result, payload)
            else:
                fitted_components, predictions, payload = cached
                print(f"{variant}, lambda={lambda_l1:g}: resumed", flush=True)
            warm_components[(variant, lambda_l1)] = fitted_components
            candidates[variant].append(payload)
            print(
                f"{variant}, lambda={lambda_l1:g}: validation MSE="
                f"{payload['metrics']['validation']['all']['weighted_complex_mse']:.6g}",
                flush=True,
            )

    selected: dict[str, Any] = {}
    for variant, variant_candidates in candidates.items():
        winner = min(
            variant_candidates,
            key=lambda item: item["metrics"]["validation"]["all"]["weighted_complex_mse"],
        )
        selected[variant] = winner
    summary = {**protocol, "candidates": candidates, "selected_by_validation": selected}
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                variant: {
                    "lambda_l1": result["lambda_l1"],
                    "validation_mse": result["metrics"]["validation"]["all"][
                        "weighted_complex_mse"
                    ],
                    "sealed_test_mse": result["metrics"]["test"]["all"]["weighted_complex_mse"],
                }
                for variant, result in selected.items()
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
