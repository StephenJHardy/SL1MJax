#!/usr/bin/env python3
"""Search quadtree leaves for native-resolution 3C391 sky variation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam, predict_beam_weights
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.quadtree import (
    QuadtreeGrid,
    QuadtreeLeaf,
    QuadtreeTopology,
    predict_quadtree_stokes_i_explicit,
)
from sl1mjax.sky_recovery import (
    BlindSpatialVariationSearchResult,
    SkyVariationCandidate,
    blind_search_quadtree_sky_variation,
    inject_sky_component,
    native_variation_candidates,
    split_search_baselines,
    temporal_support_mask,
)


def _load_topology(
    path: Path,
    *,
    root_size: int,
    root_pixel_size_rad: float,
) -> QuadtreeTopology:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    if not rows:
        raise ValueError("topology CSV must contain at least one leaf")
    leaves = tuple(
        QuadtreeLeaf(int(row["level"]), int(row["iy"]), int(row["ix"]))
        for row in rows
    )
    if len(leaves) != len(set(leaves)):
        raise ValueError("topology CSV contains duplicate leaves")
    return QuadtreeTopology(
        QuadtreeGrid(root_size, root_pixel_size_rad),
        leaves,
    )


def _local_centres(
    topology: QuadtreeTopology,
    block: VisibilityBlock,
    mosaic_phase_centre_rad: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    reference_l, reference_m = topology.centers()
    ra, dec = lmn_to_radec(
        mosaic_phase_centre_rad[0],
        mosaic_phase_centre_rad[1],
        reference_l,
        reference_m,
    )
    return radec_to_lmn(
        block.phase_centre_rad[0],
        block.phase_centre_rad[1],
        ra,
        dec,
    )[:2]


def _select_spatial_candidates(
    topology: QuadtreeTopology,
    flux_jy: np.ndarray,
    block: VisibilityBlock,
    beam: VLAPrimaryBeam,
    *,
    mosaic_phase_centre_rad: tuple[float, float],
    maximum_candidates: int,
    minimum_static_flux_jy: float,
) -> tuple[tuple[QuadtreeLeaf, ...], dict[QuadtreeLeaf, dict[str, float]]]:
    flux = np.asarray(flux_jy, dtype=np.float64)
    if flux.shape != (len(topology.leaves),):
        raise ValueError("checkpoint flux does not match the central topology")
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    if not np.isfinite(minimum_static_flux_jy) or minimum_static_flux_jy < 0:
        raise ValueError("minimum_static_flux_jy must be finite and non-negative")
    local_l, local_m = _local_centres(
        topology,
        block,
        mosaic_phase_centre_rad,
    )
    beam_i, _, _ = predict_beam_weights(beam, local_l, local_m, block.frequency_hz)
    assert beam_i is not None
    rms_beam = np.sqrt(np.mean(np.square(beam_i), axis=1))
    apparent_flux = flux * rms_beam
    eligible = np.flatnonzero((flux > 0) & (flux >= minimum_static_flux_jy))
    order = eligible[
        np.argsort(-apparent_flux[eligible], kind="stable")[:maximum_candidates]
    ]
    leaves = tuple(topology.leaves[int(index)] for index in order)
    metadata = {
        topology.leaves[int(index)]: {
            "static_flux_jy": float(flux[index]),
            "rms_beam_power": float(rms_beam[index]),
            "apparent_static_flux_jy": float(apparent_flux[index]),
            "local_l_rad": float(local_l[index]),
            "local_m_rad": float(local_m[index]),
        }
        for index in order
    }
    return leaves, metadata


def _central_time_candidate(
    candidates: tuple[SkyVariationCandidate, ...],
    block: VisibilityBlock,
    *,
    width: int,
) -> SkyVariationCandidate:
    selected = [
        candidate
        for candidate in candidates
        if candidate.kind == "temporal_interval" and candidate.bin_count == width
    ]
    if not selected:
        raise ValueError(f"candidate bank contains no temporal width {width}")
    target = float(np.median(np.unique(block.time_s)))

    def distance(candidate: SkyVariationCandidate) -> float:
        assert candidate.coordinate_start is not None
        assert candidate.coordinate_stop is not None
        return abs(0.5 * (candidate.coordinate_start + candidate.coordinate_stop) - target)

    return min(selected, key=lambda candidate: (distance(candidate), candidate.name))


def _injection_leaf(
    candidates: tuple[QuadtreeLeaf, ...],
    metadata: dict[QuadtreeLeaf, dict[str, float]],
    *,
    target_offset_arcmin: float,
) -> QuadtreeLeaf:
    target_rad = np.deg2rad(target_offset_arcmin / 60.0)
    return min(
        candidates,
        key=lambda leaf: (
            abs(
                np.hypot(
                    metadata[leaf]["local_l_rad"],
                    metadata[leaf]["local_m_rad"],
                )
                - target_rad
            ),
            -metadata[leaf]["apparent_static_flux_jy"],
            leaf,
        ),
    )


def _unit_leaf_response(
    block: VisibilityBlock,
    topology: QuadtreeTopology,
    leaf: QuadtreeLeaf,
    beam: VLAPrimaryBeam,
    mosaic_phase_centre_rad: tuple[float, float],
    direct: DirectDFTConfig,
) -> np.ndarray:
    one = QuadtreeTopology(topology.grid, (leaf,))
    local_l, local_m = _local_centres(one, block, mosaic_phase_centre_rad)
    beam_i, beam_rr, beam_ll = predict_beam_weights(
        beam,
        local_l,
        local_m,
        block.frequency_hz,
    )
    return np.asarray(
        predict_quadtree_stokes_i_explicit(
            np.ones(1),
            one,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            config=direct,
            beam_weights=beam_i,
            beam_weights_rr=beam_rr,
            beam_weights_ll=beam_ll,
            centers_lm=(local_l, local_m),
        )
    )


def _leaf_payload(
    leaf: QuadtreeLeaf | None,
    topology: QuadtreeTopology,
    metadata: dict[QuadtreeLeaf, dict[str, float]],
) -> dict[str, Any] | None:
    if leaf is None:
        return None
    l_rad, m_rad = topology.grid.leaf_center_rad(leaf)
    return {
        "level": leaf.level,
        "iy": leaf.iy,
        "ix": leaf.ix,
        "l_rad": l_rad,
        "m_rad": m_rad,
        "width_rad": topology.grid.leaf_width_rad(leaf.level),
        **metadata[leaf],
    }


def _result_payload(
    result: BlindSpatialVariationSearchResult,
    topology: QuadtreeTopology,
    metadata: dict[QuadtreeLeaf, dict[str, float]],
) -> dict[str, Any]:
    def score_payload(score: Any) -> dict[str, Any]:
        payload = asdict(score)
        payload["leaf"] = _leaf_payload(score.leaf, topology, metadata)
        return payload

    return {
        "spatial_candidate_count": result.spatial_candidate_count,
        "variation_candidate_count": result.variation_candidate_count,
        "spatial_shortlist_size": result.spatial_shortlist_size,
        "discovery_shortlist_size": result.discovery_shortlist_size,
        "selected_leaf": _leaf_payload(result.selected_leaf, topology, metadata),
        "selected_variation": (
            None if result.selected_variation is None else asdict(result.selected_variation)
        ),
        "refit_static_coefficient_jy": result.refit_static_coefficient,
        "refit_variation_coefficient_jy": result.refit_variation_coefficient,
        "selection_incremental_weighted_mse": (
            result.selection_incremental_weighted_mse
        ),
        "evaluation_static_weighted_mse": result.evaluation_static_weighted_mse,
        "evaluation_candidate_weighted_mse": (
            result.evaluation_candidate_weighted_mse
        ),
        "evaluation_incremental_weighted_mse": (
            result.evaluation_incremental_weighted_mse
        ),
        "evaluation_relative_improvement": result.evaluation_relative_improvement,
        "accepted": result.accepted,
        "top_spatial_discovery": [
            score_payload(score) for score in result.spatial_discovery_ranking[:32]
        ],
        "discovery_shortlist": [
            score_payload(score) for score in result.discovery_shortlist
        ],
        "selection_shortlist": [
            score_payload(score) for score in result.selection_shortlist
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-fixture",
        type=Path,
        default=Path("outputs/3c391_native_averaging_ablation/native_C1.zarr"),
    )
    parser.add_argument(
        "--sky-protocol",
        type=Path,
        default=Path("outputs/3c391_composite_catalogue_stage3/protocol.json"),
    )
    parser.add_argument(
        "--sky-checkpoint",
        type=Path,
        default=Path("outputs/3c391_recovery_policy_fit_zero/sealed_active_only.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_native_spatial_variability_search"),
    )
    parser.add_argument("--maximum-spatial-candidates", type=int, default=768)
    parser.add_argument("--minimum-static-flux-mjy", type=float, default=0.0)
    parser.add_argument("--spatial-shortlist-size", type=int, default=16)
    parser.add_argument("--discovery-shortlist-size", type=int, default=64)
    parser.add_argument("--selection-baseline-fraction", type=float, default=0.2)
    parser.add_argument("--evaluation-baseline-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--row-batch-size", type=int, default=256)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--injection-amplitude-mjy", type=float, default=0.0)
    parser.add_argument("--injection-time-width", type=int, default=3)
    parser.add_argument("--injection-offset-arcmin", type=float, default=3.0)
    arguments = parser.parse_args()
    if arguments.maximum_spatial_candidates < 1:
        parser.error("--maximum-spatial-candidates must be positive")
    if arguments.injection_amplitude_mjy < 0:
        parser.error("--injection-amplitude-mjy must be non-negative")

    stored = read_dataset(arguments.native_fixture).blocks
    if len(stored) != 1 or stored[0].model_visibility is None:
        raise ValueError("native fixture must contain one block with a frozen prediction")
    block = stored[0]
    protocol = json.loads(arguments.sky_protocol.read_text(encoding="utf-8"))
    frozen_directory = Path(protocol["frozen_directory"])
    frozen_summary = json.loads(
        (frozen_directory / "summary.json").read_text(encoding="utf-8")
    )
    topology = _load_topology(
        frozen_directory / "consensus_topology.csv",
        root_size=int(frozen_summary["root_size"]),
        root_pixel_size_rad=np.deg2rad(
            float(frozen_summary["root_pixel_arcsec"]) / 3600.0
        ),
    )
    with np.load(arguments.sky_checkpoint) as checkpoint:
        central_flux = np.asarray(checkpoint["flux_central"], dtype=np.float64)
    beam = VLAPrimaryBeam(
        kind="airy",
        catalog=replace(
            VLABeamCatalog(),
            airy_max_radius_rad_at_1ghz=np.deg2rad(
                float(protocol["airy_max_radius_deg_at_1ghz"])
            ),
        ),
    )
    mosaic_phase_centre_rad = block.phase_centre_rad
    spatial_candidates, metadata = _select_spatial_candidates(
        topology,
        central_flux,
        block,
        beam,
        mosaic_phase_centre_rad=mosaic_phase_centre_rad,
        maximum_candidates=arguments.maximum_spatial_candidates,
        minimum_static_flux_jy=arguments.minimum_static_flux_mjy * 1e-3,
    )
    variations = native_variation_candidates(block)
    direct = DirectDFTConfig(
        visibility_chunk_size=arguments.row_batch_size,
        pixel_chunk_size=arguments.candidate_batch_size,
        precision=arguments.precision,
    )
    injection: dict[str, Any] | None = None
    if arguments.injection_amplitude_mjy > 0:
        true_leaf = _injection_leaf(
            spatial_candidates,
            metadata,
            target_offset_arcmin=arguments.injection_offset_arcmin,
        )
        true_variation = _central_time_candidate(
            variations,
            block,
            width=arguments.injection_time_width,
        )
        assert true_variation.coordinate_start is not None
        assert true_variation.coordinate_stop is not None
        support = temporal_support_mask(
            block,
            start_s=true_variation.coordinate_start,
            duration_s=(
                true_variation.coordinate_stop - true_variation.coordinate_start
            ),
        )
        response = _unit_leaf_response(
            block,
            topology,
            true_leaf,
            beam,
            mosaic_phase_centre_rad,
            direct,
        )
        block = inject_sky_component(
            block,
            response,
            support,
            arguments.injection_amplitude_mjy * 1e-3,
        )
        injection = {
            "leaf": _leaf_payload(true_leaf, topology, metadata),
            "variation": asdict(true_variation),
            "amplitude_jy": arguments.injection_amplitude_mjy * 1e-3,
        }

    split = split_search_baselines(
        block,
        selection_fraction=arguments.selection_baseline_fraction,
        evaluation_fraction=arguments.evaluation_baseline_fraction,
        seed=arguments.seed,
    )
    print(
        f"Searching {len(spatial_candidates)} leaves x {len(variations)} variations",
        flush=True,
    )
    result = blind_search_quadtree_sky_variation(
        block,
        topology,
        spatial_candidates,
        variations,
        split,
        mosaic_phase_centre_rad=mosaic_phase_centre_rad,
        primary_beam=beam,
        direct_config=direct,
        spatial_shortlist_size=arguments.spatial_shortlist_size,
        discovery_shortlist_size=arguments.discovery_shortlist_size,
        candidate_batch_size=arguments.candidate_batch_size,
        row_batch_size=arguments.row_batch_size,
        progress=lambda message: print(message, flush=True),
    )
    payload = {
        "protocol": {
            "native_fixture": str(arguments.native_fixture),
            "sky_protocol": str(arguments.sky_protocol),
            "sky_checkpoint": str(arguments.sky_checkpoint),
            "topology_leaf_count": len(topology.leaves),
            "spatial_prefilter": (
                "largest frozen apparent flux = static flux times RMS beam power"
            ),
            "maximum_spatial_candidates": arguments.maximum_spatial_candidates,
            "minimum_static_flux_mjy": arguments.minimum_static_flux_mjy,
            "spatial_shortlist_size": arguments.spatial_shortlist_size,
            "discovery_shortlist_size": arguments.discovery_shortlist_size,
            "variation_candidate_count": len(variations),
            "seed": arguments.seed,
            "selection_baseline_fraction": arguments.selection_baseline_fraction,
            "evaluation_baseline_fraction": arguments.evaluation_baseline_fraction,
            "candidate_batch_size": arguments.candidate_batch_size,
            "row_batch_size": arguments.row_batch_size,
            "precision": arguments.precision,
        },
        "injection": injection,
        "result": _result_payload(result, topology, metadata),
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    label = (
        "null"
        if injection is None
        else f"injection_{arguments.injection_amplitude_mjy:g}mJy"
    )
    destination = arguments.output / f"{label}_seed{arguments.seed}.json"
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
