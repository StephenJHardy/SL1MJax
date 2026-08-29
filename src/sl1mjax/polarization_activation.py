"""Coarse joint linear polarisation on a frozen Stokes-I topology.

Regions are declared from the accepted I model only. The fitted sky keeps
``v=0``. This is a spatial-activation test, not a per-pixel polarisation
image and not self-cal or RM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.composite import MosaicQuadtreeComponent, MosaicSkyComponent
from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import circular_stokes_from_correlations, electric_vector_position_angle_rad
from sl1mjax.polarization_diagnostics import (
    _row_channel_mask,
    deterministic_visibility_partitions,
    stokes_visibility_planes,
)

RegionScheme = Literal[
    "central_radial_widefield",
    "component_dictionaries",
    "central_ancestor_cells",
]
MODEL_KINDS = ("unpolarised", "global", "regional")


@dataclass(frozen=True)
class CoarseStokesIRegion:
    name: str
    components: tuple[MosaicSkyComponent, ...]
    stokes_i_jy: float
    provenance: dict[str, Any]


@dataclass(frozen=True)
class LinearPolarizationFit:
    kind: str
    region_names: tuple[str, ...]
    q: tuple[float, ...]
    u: tuple[float, ...]
    fractional_linear: tuple[float, ...]
    n_samples: int
    null_linear_loss: float
    polarized_linear_loss: float
    v: float
    provenance: dict[str, Any]


def _component_flux_and_lm(
    component: MosaicSkyComponent,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux = np.asarray(component.flux, dtype=np.float64).reshape(-1)
    if isinstance(component, MosaicQuadtreeComponent):
        l_rad, m_rad = component.topology.centers()
    else:
        l_rad = np.asarray(component.l_rad, dtype=np.float64).reshape(-1)
        m_rad = np.asarray(component.m_rad, dtype=np.float64).reshape(-1)
    if flux.size != l_rad.size:
        raise ValueError(f"{component.name} flux and coordinates have different lengths")
    return flux, l_rad, m_rad


def _component_stokes_i(component: MosaicSkyComponent) -> float:
    flux, _, _ = _component_flux_and_lm(component)
    return float(np.sum(np.maximum(flux, 0.0)))


def _zero_quadtree_leaves(
    component: MosaicQuadtreeComponent, keep: np.ndarray, name: str
) -> MosaicQuadtreeComponent:
    flux = np.asarray(component.flux, dtype=np.float64).copy()
    flux[~np.asarray(keep, dtype=bool)] = 0.0
    return replace(component, name=name, flux=flux)


def _split_central_by_i_radius(
    component: MosaicQuadtreeComponent,
) -> tuple[MosaicQuadtreeComponent, MosaicQuadtreeComponent, dict[str, Any]]:
    flux, l_rad, m_rad = _component_flux_and_lm(component)
    weight = np.maximum(flux, 0.0)
    total = float(np.sum(weight))
    if total <= 0.0:
        raise ValueError("central Stokes-I component has no positive flux")
    centroid_l = float(np.sum(weight * l_rad) / total)
    centroid_m = float(np.sum(weight * m_rad) / total)
    radius = np.hypot(l_rad - centroid_l, m_rad - centroid_m)
    order = np.argsort(radius)
    cumulative = np.cumsum(weight[order])
    split_at = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    split_at = min(max(split_at, 1), order.size - 1)
    inner = np.zeros(flux.size, dtype=bool)
    inner[order[:split_at]] = True
    inner &= weight > 0
    outer = (weight > 0) & ~inner
    if not np.any(inner) or not np.any(outer):
        raise ValueError("I-weighted radial split needs positive flux on both sides")
    threshold = float(radius[order[split_at]])
    provenance = {
        "selector": "stokes_i_weighted_median_radius",
        "qu_peaks": False,
        "centroid_l_rad": centroid_l,
        "centroid_m_rad": centroid_m,
        "radius_threshold_rad": threshold,
        "inner_stokes_i_jy": float(np.sum(weight[inner])),
        "outer_stokes_i_jy": float(np.sum(weight[outer])),
        "inner_leaf_count": int(np.count_nonzero(inner)),
        "outer_leaf_count": int(np.count_nonzero(outer)),
    }
    return (
        _zero_quadtree_leaves(component, inner, "central_inner"),
        _zero_quadtree_leaves(component, outer, "central_outer"),
        provenance,
    )


def _ancestor_shift(root_pixel_size_rad: float, ancestor_arcsec: float) -> int:
    root_arcsec = float(np.rad2deg(root_pixel_size_rad) * 3600.0)
    if ancestor_arcsec < root_arcsec - 1.0e-9:
        raise ValueError("ancestor cells must be at least as large as the I root pixel")
    ratio = ancestor_arcsec / root_arcsec
    shift = int(round(np.log2(ratio)))
    if abs(2**shift - ratio) > 1.0e-6:
        raise ValueError("ancestor_arcsec must be a power-of-two multiple of the I root pixel")
    return shift


def _split_central_by_ancestor_cells(
    component: MosaicQuadtreeComponent,
    *,
    ancestor_arcsec: float,
    minimum_region_i_jy: float,
) -> tuple[list[MosaicQuadtreeComponent], MosaicQuadtreeComponent | None, dict[str, Any]]:
    if minimum_region_i_jy < 0.0:
        raise ValueError("minimum_region_i_jy must be non-negative")
    flux, _, _ = _component_flux_and_lm(component)
    shift = _ancestor_shift(component.topology.grid.root_pixel_size_rad, ancestor_arcsec)
    buckets: dict[tuple[int, int], list[int]] = {}
    totals: dict[tuple[int, int], float] = {}
    max_level: dict[tuple[int, int], int] = {}
    for index, (leaf, leaf_flux) in enumerate(zip(component.topology.leaves, flux, strict=True)):
        key = (leaf.iy // 2 ** (leaf.level + shift), leaf.ix // 2 ** (leaf.level + shift))
        buckets.setdefault(key, []).append(index)
        totals[key] = totals.get(key, 0.0) + float(max(leaf_flux, 0.0))
        max_level[key] = max(max_level.get(key, 0), leaf.level)
    kept = tuple(
        key
        for key, total in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
        if total >= minimum_region_i_jy
    )
    if not kept:
        raise ValueError("no ancestor I cell meets the minimum Stokes-I floor")
    assigned = np.zeros(flux.size, dtype=bool)
    regions: list[MosaicQuadtreeComponent] = []
    for iy, ix in kept:
        keep = np.zeros(flux.size, dtype=bool)
        keep[np.asarray(buckets[(iy, ix)], dtype=np.int64)] = True
        assigned |= keep
        regions.append(_zero_quadtree_leaves(component, keep, f"i{int(ancestor_arcsec)}_{iy}_{ix}"))
    remainder = None
    leftover = (~assigned) & (np.maximum(flux, 0.0) > 0.0)
    if np.any(leftover):
        remainder = _zero_quadtree_leaves(component, leftover, "central_remainder")
    provenance = {
        "selector": "frozen_hierarchical_i_ancestor_cells",
        "qu_peaks": False,
        "ancestor_arcsec": float(ancestor_arcsec),
        "minimum_region_i_jy": float(minimum_region_i_jy),
        "root_pixel_arcsec": float(
            np.rad2deg(component.topology.grid.root_pixel_size_rad) * 3600.0
        ),
        "kept_cell_count": len(kept),
        "kept_cells": [
            {
                "name": f"i{int(ancestor_arcsec)}_{iy}_{ix}",
                "iy": iy,
                "ix": ix,
                "stokes_i_jy": totals[(iy, ix)],
                "leaf_count": len(buckets[(iy, ix)]),
                "max_level": max_level[(iy, ix)],
            }
            for iy, ix in kept
        ],
        "remainder_stokes_i_jy": (
            0.0 if remainder is None else _component_stokes_i(remainder)
        ),
    }
    return regions, remainder, provenance


def declare_coarse_stokes_i_regions(
    components: tuple[MosaicSkyComponent, ...],
    *,
    scheme: RegionScheme = "central_radial_widefield",
    ancestor_arcsec: float = 64.0,
    minimum_region_i_jy: float = 0.2,
    include_widefield: bool = True,
) -> tuple[CoarseStokesIRegion, ...]:
    """Declare a small I-only region set. Q/U images are not consulted."""

    if not components:
        raise ValueError("frozen Stokes-I components are required")
    names = [component.name for component in components]
    if len(set(names)) != len(names):
        raise ValueError("component names must be unique")
    common = {
        "scheme": scheme,
        "qu_peaks": False,
        "source": "frozen_stokes_i_topology",
    }
    if scheme == "component_dictionaries":
        return tuple(
            CoarseStokesIRegion(
                name=component.name,
                components=(component,),
                stokes_i_jy=_component_stokes_i(component),
                provenance={**common, "dictionary": component.name},
            )
            for component in components
        )
    by_name = {component.name: component for component in components}
    if "central" not in by_name or not isinstance(by_name["central"], MosaicQuadtreeComponent):
        raise ValueError(f"{scheme} needs a central quadtree component")
    widefield = tuple(component for component in components if component.name != "central")
    declared: list[CoarseStokesIRegion] = []
    if scheme == "central_radial_widefield":
        inner, outer, split = _split_central_by_i_radius(by_name["central"])
        declared.extend(
            (
                CoarseStokesIRegion(
                    "central_inner",
                    (inner,),
                    _component_stokes_i(inner),
                    {**common, **split, "role": "central_inner"},
                ),
                CoarseStokesIRegion(
                    "central_outer",
                    (outer,),
                    _component_stokes_i(outer),
                    {**common, **split, "role": "central_outer"},
                ),
            )
        )
    elif scheme == "central_ancestor_cells":
        cells, remainder, split = _split_central_by_ancestor_cells(
            by_name["central"],
            ancestor_arcsec=ancestor_arcsec,
            minimum_region_i_jy=minimum_region_i_jy,
        )
        declared.extend(
            CoarseStokesIRegion(
                cell.name,
                (cell,),
                _component_stokes_i(cell),
                {**common, **split, "role": cell.name},
            )
            for cell in cells
        )
        if remainder is not None:
            declared.append(
                CoarseStokesIRegion(
                    remainder.name,
                    (remainder,),
                    _component_stokes_i(remainder),
                    {**common, **split, "role": "central_remainder"},
                )
            )
    else:
        raise ValueError(f"unknown coarse region scheme {scheme!r}")
    if include_widefield:
        if not widefield:
            raise ValueError(f"{scheme} with widefield needs a coarse or catalogue component")
        declared.append(
            CoarseStokesIRegion(
                "widefield",
                widefield,
                float(sum(_component_stokes_i(component) for component in widefield)),
                {
                    **common,
                    "role": "widefield",
                    "dictionaries": [component.name for component in widefield],
                },
            )
        )
    if not declared:
        raise ValueError("no coarse Stokes-I regions were declared")
    return tuple(declared)


def stokes_i_from_visibility(
    visibility: np.ndarray, correlations: tuple
) -> NDArray[np.complex128]:
    model_i, *_ = circular_stokes_from_correlations(visibility, correlations)
    return np.asarray(model_i, dtype=np.complex128)


def _linear_design(
    blocks: tuple[VisibilityBlock, ...],
    region_stokes_i: dict[str, tuple[np.ndarray, ...]],
    sample_masks: tuple[np.ndarray | None, ...] | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, int, float]:
    names = tuple(region_stokes_i)
    if not names:
        raise ValueError("at least one Stokes-I region is required")
    masks = sample_masks if sample_masks is not None else (None,) * len(blocks)
    if len(masks) != len(blocks):
        raise ValueError("sample_masks must contain one array per block")
    for name, planes in region_stokes_i.items():
        if len(planes) != len(blocks):
            raise ValueError(f"region {name} must supply one M_I plane per block")
    gram = np.zeros((len(names), len(names)), dtype=np.complex128)
    rhs = np.zeros(len(names), dtype=np.complex128)
    null_loss = 0.0
    n_samples = 0
    for block_index, (block, mask) in enumerate(zip(blocks, masks, strict=True)):
        planes = stokes_visibility_planes(block)
        selected = _row_channel_mask(block, mask) & (planes.weight_linear > 0)
        if not np.any(selected):
            continue
        weight = planes.weight_linear[selected]
        observed = (planes.stokes_q + 1j * planes.stokes_u)[selected]
        design = np.column_stack(
            [
                np.asarray(region_stokes_i[name][block_index], dtype=np.complex128)[selected]
                for name in names
            ]
        )
        weighted = weight[:, None] * design
        gram += weighted.conj().T @ design
        rhs += weighted.conj().T @ observed
        null_loss += float(np.sum(weight * np.abs(observed) ** 2))
        n_samples += int(np.count_nonzero(selected))
    if n_samples == 0:
        raise ValueError("linear polarisation fit needs finite Q/U samples")
    return names, gram, rhs, n_samples, null_loss


def _solve_complex_gram(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (gram + gram.conj().T)
    try:
        fitted = np.linalg.solve(hermitian, rhs)
    except np.linalg.LinAlgError:
        fitted, *_ = np.linalg.lstsq(hermitian, rhs, rcond=None)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if float(np.min(np.real(eigenvalues))) <= 1.0e-12 * max(float(np.max(np.real(eigenvalues))), 1.0):
        fitted, *_ = np.linalg.lstsq(hermitian, rhs, rcond=None)
    return np.asarray(fitted, dtype=np.complex128)


def _clip_unit_disk(values: np.ndarray) -> tuple[np.ndarray, bool]:
    clipped = values.copy()
    changed = False
    for index, value in enumerate(clipped):
        amplitude = float(np.abs(value))
        if amplitude > 1.0:
            clipped[index] = value / amplitude
            changed = True
    return clipped, changed


def _fit_from_gram(
    kind: str,
    names: tuple[str, ...],
    gram: np.ndarray,
    rhs: np.ndarray,
    n_samples: int,
    null_loss: float,
    extra_provenance: dict[str, Any],
) -> LinearPolarizationFit:
    if kind == "unpolarised":
        coefficients = np.zeros(len(names), dtype=np.complex128)
        polarized_loss = null_loss
        clipped = False
    else:
        coefficients, clipped = _clip_unit_disk(_solve_complex_gram(gram, rhs))
        polarized_loss = float(
            null_loss
            - 2.0 * np.real(np.vdot(coefficients, rhs))
            + np.real(np.vdot(coefficients, gram @ coefficients))
        )
        polarized_loss = max(polarized_loss, 0.0)
    q = tuple(float(np.real(value)) for value in coefficients)
    u = tuple(float(np.imag(value)) for value in coefficients)
    return LinearPolarizationFit(
        kind=kind,
        region_names=names,
        q=q,
        u=u,
        fractional_linear=tuple(float(np.hypot(qi, ui)) for qi, ui in zip(q, u, strict=True)),
        n_samples=n_samples,
        null_linear_loss=null_loss,
        polarized_linear_loss=polarized_loss,
        v=0.0,
        provenance={
            "kind": kind,
            "regressor": "complex_stokes_i_regions",
            "v": 0.0,
            "spatial_image": kind == "regional",
            "frequency_model": "constant",
            "rm": False,
            "unit_disk_clipped": clipped,
            **extra_provenance,
        },
    )


def fit_unpolarised_linear_polarization(
    blocks: tuple[VisibilityBlock, ...],
    region_stokes_i: dict[str, tuple[np.ndarray, ...]],
    *,
    sample_masks: tuple[np.ndarray | None, ...] | None = None,
) -> LinearPolarizationFit:
    names, gram, rhs, n_samples, null_loss = _linear_design(
        blocks, region_stokes_i, sample_masks
    )
    return _fit_from_gram(
        "unpolarised", names, gram, rhs, n_samples, null_loss, {"model": "q=u=v=0"}
    )


def fit_global_linear_polarization(
    blocks: tuple[VisibilityBlock, ...],
    region_stokes_i: dict[str, tuple[np.ndarray, ...]],
    *,
    sample_masks: tuple[np.ndarray | None, ...] | None = None,
) -> LinearPolarizationFit:
    """One shared ``q+iu`` against the summed frozen ``M_I``. ``v=0``."""

    names, gram, rhs, n_samples, null_loss = _linear_design(
        blocks, region_stokes_i, sample_masks
    )
    ones = np.ones(len(names), dtype=np.complex128)
    collapsed_gram = np.array([[ones.conj() @ gram @ ones]], dtype=np.complex128)
    collapsed_rhs = np.array([ones.conj() @ rhs], dtype=np.complex128)
    collapsed = _fit_from_gram(
        "global",
        ("global",),
        collapsed_gram,
        collapsed_rhs,
        n_samples,
        null_loss,
        {"model": "one_complex_q_plus_iu", "collapsed_regions": list(names)},
    )
    return LinearPolarizationFit(
        kind=collapsed.kind,
        region_names=names,
        q=tuple(collapsed.q[0] for _ in names),
        u=tuple(collapsed.u[0] for _ in names),
        fractional_linear=tuple(collapsed.fractional_linear[0] for _ in names),
        n_samples=collapsed.n_samples,
        null_linear_loss=collapsed.null_linear_loss,
        polarized_linear_loss=collapsed.polarized_linear_loss,
        v=0.0,
        provenance=collapsed.provenance,
    )


def fit_regional_linear_polarization(
    blocks: tuple[VisibilityBlock, ...],
    region_stokes_i: dict[str, tuple[np.ndarray, ...]],
    *,
    sample_masks: tuple[np.ndarray | None, ...] | None = None,
) -> LinearPolarizationFit:
    """Joint ``q_r+iu_r`` against the declared I regions. ``v=0``."""

    names, gram, rhs, n_samples, null_loss = _linear_design(
        blocks, region_stokes_i, sample_masks
    )
    return _fit_from_gram(
        "regional",
        names,
        gram,
        rhs,
        n_samples,
        null_loss,
        {"model": "joint_regional_q_plus_iu"},
    )


def evaluate_linear_polarization(
    blocks: tuple[VisibilityBlock, ...],
    region_stokes_i: dict[str, tuple[np.ndarray, ...]],
    q: tuple[float, ...],
    u: tuple[float, ...],
    *,
    sample_masks: tuple[np.ndarray | None, ...] | None = None,
) -> dict[str, float]:
    names, gram, rhs, n_samples, null_loss = _linear_design(
        blocks, region_stokes_i, sample_masks
    )
    if len(q) != len(names) or len(u) != len(names):
        raise ValueError("q and u must contain one value per region")
    coefficients = np.asarray(q, dtype=np.float64) + 1j * np.asarray(u, dtype=np.float64)
    polarized_loss = float(
        null_loss
        - 2.0 * np.real(np.vdot(coefficients, rhs))
        + np.real(np.vdot(coefficients, gram @ coefficients))
    )
    return {
        "null_linear_loss": null_loss,
        "polarized_linear_loss": max(polarized_loss, 0.0),
        "n_samples": float(n_samples),
        "linear_loss_ratio": (
            max(polarized_loss, 0.0) / null_loss if null_loss > 0.0 else None
        ),
    }


def compare_linear_polarization_models(
    blocks: tuple[VisibilityBlock, ...],
    region_stokes_i: dict[str, tuple[np.ndarray, ...]],
    *,
    sample_masks: tuple[np.ndarray | None, ...] | None = None,
) -> dict[str, LinearPolarizationFit]:
    return {
        "unpolarised": fit_unpolarised_linear_polarization(
            blocks, region_stokes_i, sample_masks=sample_masks
        ),
        "global": fit_global_linear_polarization(
            blocks, region_stokes_i, sample_masks=sample_masks
        ),
        "regional": fit_regional_linear_polarization(
            blocks, region_stokes_i, sample_masks=sample_masks
        ),
    }


def leave_one_block_out_scores(
    blocks: tuple[VisibilityBlock, ...],
    region_stokes_i: dict[str, tuple[np.ndarray, ...]],
    *,
    labels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Fit on all-but-one pointing and score the three models on the held block."""

    if len(blocks) < 2:
        raise ValueError("leave-one-block-out needs at least two pointings")
    selected_labels = labels or tuple(f"C{index}" for index in range(1, len(blocks) + 1))
    if len(selected_labels) != len(blocks):
        raise ValueError("labels must contain one name per block")
    per_holdout: dict[str, Any] = {}
    for held_index, held_label in enumerate(selected_labels):
        train_index = tuple(index for index in range(len(blocks)) if index != held_index)
        train_blocks = tuple(blocks[index] for index in train_index)
        train_regions = {
            name: tuple(planes[index] for index in train_index)
            for name, planes in region_stokes_i.items()
        }
        held_blocks = (blocks[held_index],)
        held_regions = {name: (planes[held_index],) for name, planes in region_stokes_i.items()}
        fitted = compare_linear_polarization_models(train_blocks, train_regions)
        scores = {}
        for kind, model in fitted.items():
            scores[kind] = evaluate_linear_polarization(
                held_blocks, held_regions, model.q, model.u
            )
        per_holdout[held_label] = {
            "selection_labels": [selected_labels[index] for index in train_index],
            "fits": {kind: linear_polarization_as_dict(model) for kind, model in fitted.items()},
            "holdout_scores": scores,
        }
    return per_holdout


def partitioned_linear_polarization_scores(
    blocks: tuple[VisibilityBlock, ...],
    region_stokes_i: dict[str, tuple[np.ndarray, ...]],
) -> dict[str, Any]:
    """Fit on one deterministic half and score on the complementary half."""

    partitions = tuple(deterministic_visibility_partitions(block) for block in blocks)
    pairs = (
        ("baseline_even", "baseline_odd"),
        ("time_even", "time_odd"),
        ("channel_even", "channel_odd"),
    )
    report: dict[str, Any] = {}
    for train_name, test_name in pairs:
        train_masks = tuple(partition[train_name] for partition in partitions)
        test_masks = tuple(partition[test_name] for partition in partitions)
        fitted = compare_linear_polarization_models(
            blocks, region_stokes_i, sample_masks=train_masks
        )
        report[f"{train_name}_to_{test_name}"] = {
            "fits": {kind: linear_polarization_as_dict(model) for kind, model in fitted.items()},
            "holdout_scores": {
                kind: evaluate_linear_polarization(
                    blocks, region_stokes_i, model.q, model.u, sample_masks=test_masks
                )
                for kind, model in fitted.items()
            },
        }
    return report


def region_i_centroid_lm(region: CoarseStokesIRegion) -> tuple[float, float, float]:
    weight_sum = 0.0
    moment_l = 0.0
    moment_m = 0.0
    for component in region.components:
        flux, l_rad, m_rad = _component_flux_and_lm(component)
        weight = np.maximum(flux, 0.0)
        weight_sum += float(np.sum(weight))
        moment_l += float(np.sum(weight * l_rad))
        moment_m += float(np.sum(weight * m_rad))
    if weight_sum <= 0.0:
        raise ValueError(f"region {region.name} has no positive Stokes I")
    return moment_l / weight_sum, moment_m / weight_sum, weight_sum


def region_pointing_radius_rad(
    region: CoarseStokesIRegion,
    block: VisibilityBlock,
    mosaic_phase_centre_rad: tuple[float, float],
) -> float:
    centroid_l, centroid_m, _ = region_i_centroid_lm(region)
    from sl1mjax.coordinates import lmn_to_radec

    ra, dec = lmn_to_radec(
        mosaic_phase_centre_rad[0],
        mosaic_phase_centre_rad[1],
        np.asarray([centroid_l]),
        np.asarray([centroid_m]),
    )
    local_l, local_m, _ = radec_to_lmn(
        block.phase_centre_rad[0],
        block.phase_centre_rad[1],
        ra,
        dec,
    )
    return float(np.hypot(local_l[0], local_m[0]))


def block_mean_parallactic_angle_rad(
    block: VisibilityBlock, antenna_position_m: np.ndarray
) -> float:
    angles = parallactic_angle_rad(block.time_s, block.phase_centre_rad, antenna_position_m)
    row_active = np.any(block.active, axis=(1, 2))
    if not np.any(row_active):
        raise ValueError("block has no active rows for parallactic angle")
    return float(np.mean(angles[row_active]))


def observing_geometry_report(
    regions: tuple[CoarseStokesIRegion, ...],
    blocks: tuple[VisibilityBlock, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    antenna_position_m: np.ndarray,
    *,
    labels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    selected_labels = labels or tuple(f"C{index}" for index in range(1, len(blocks) + 1))
    geometry: dict[str, Any] = {}
    for label, block in zip(selected_labels, blocks, strict=True):
        geometry[label] = {
            "mean_parallactic_angle_deg": float(
                np.rad2deg(block_mean_parallactic_angle_rad(block, antenna_position_m))
            ),
            "regions": {
                region.name: {
                    "pointing_radius_deg": float(
                        np.rad2deg(
                            region_pointing_radius_rad(
                                region, block, mosaic_phase_centre_rad
                            )
                        )
                    ),
                    "stokes_i_jy": region.stokes_i_jy,
                }
                for region in regions
            },
        }
    return geometry


def parallactic_cohort_masks(
    blocks: tuple[VisibilityBlock, ...], antenna_position_m: np.ndarray
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]:
    """Split rows by whether the row-mean PA is below the mosaic median."""

    row_angles: list[np.ndarray] = []
    for block in blocks:
        angles = parallactic_angle_rad(block.time_s, block.phase_centre_rad, antenna_position_m)
        row_angles.append(np.mean(angles, axis=1))
    median = float(np.median(np.concatenate(row_angles)))
    low: list[np.ndarray] = []
    high: list[np.ndarray] = []
    for block, angles in zip(blocks, row_angles, strict=True):
        low_rows = angles <= median
        high_rows = ~low_rows
        low.append(np.broadcast_to(low_rows[:, None, None], block.shape) & block.active)
        high.append(np.broadcast_to(high_rows[:, None, None], block.shape) & block.active)
    return tuple(low), tuple(high), median


def beam_radius_cohort_masks(
    regions: tuple[CoarseStokesIRegion, ...],
    blocks: tuple[VisibilityBlock, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    region_name: str,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]:
    """Split whole pointings by one region's I-centroid beam radius."""

    region = next(item for item in regions if item.name == region_name)
    radii = np.asarray(
        [
            region_pointing_radius_rad(region, block, mosaic_phase_centre_rad)
            for block in blocks
        ],
        dtype=np.float64,
    )
    median = float(np.median(radii))
    inner: list[np.ndarray] = []
    outer: list[np.ndarray] = []
    for block, radius in zip(blocks, radii, strict=True):
        if radius <= median:
            inner.append(block.active)
            outer.append(np.zeros(block.shape, dtype=bool))
        else:
            inner.append(np.zeros(block.shape, dtype=bool))
            outer.append(block.active)
    return tuple(inner), tuple(outer), median


def linear_polarization_as_dict(fitted: LinearPolarizationFit) -> dict[str, Any]:
    payload = asdict(fitted)
    payload["evpa_deg"] = [
        float(np.rad2deg(electric_vector_position_angle_rad(q, u)))
        for q, u in zip(fitted.q, fitted.u, strict=True)
    ]
    payload["linear_loss_ratio"] = (
        fitted.polarized_linear_loss / fitted.null_linear_loss
        if fitted.null_linear_loss > 0.0
        else None
    )
    return payload
