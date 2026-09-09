"""Build the compact SPW-4 C-band beam-validation bundle.

Reads named Bacchus science outputs, rejects superseded full-raster squint as
a publication estimator, and writes checksummed laptop-sized products. The
notebook never opens the Measurement Set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam_validation_outputs import (
    PUBLICATION_VERSION,
    default_bundle_root,
    write_validation_bundle,
)
from sl1mjax.beam_validation_statistics import (
    PHASE_AMP_FLOOR_JY,
    SPATIAL_MAP_BINS,
    binned_complex_map,
    bright_source_examples,
    cell_average_vi,
    channel32_source_i_jy,
    complex_visibility_score,
    phase_valid_mask,
    radial_coherence,
    raster_family_from_offset,
    residual_group_summary,
    scientific_voltage_masks,
    visibility_hand_weight,
)
from sl1mjax.evla_c_publication import (
    crosshand_quadrants_from_export,
    frequency_series_from_reports,
    holoraster_channel32_from_report,
    offset_ring_from_c147_report,
    residual_geometry_from_export,
    squint_from_refresh_export,
    supersession_ledger,
)
from sl1mjax.evla_c_validation_refresh import PRIMARY_ARMS, PUBLICATION_BUNDLE_VERSION
from sl1mjax.holography import (
    THOL0001_EXECUTION_BLOCK,
    THOL0001_HOLORASTER_FIELD,
    THOL0001_LOWER_C_NATIVE_HZ,
    THOL0001_PROJECT,
    THOL0001_SCHEDULING_BLOCK,
    THOL0001_SOURCE,
)
from sl1mjax.holography_calibration import (
    C147_OFFSET_FIELD_IDS,
    CALWT,
    FULLPOL_PARANG,
    ON_AXIS_3C147_FIELD_IDS,
    REFERENCE_ANTENNA,
)
from sl1mjax.holography_diagonal import THOL0001_REFERENCE_ANTENNA_NAMES
from sl1mjax.holography_highres_cassbeam import DEFAULT_CONVENTION

DEFAULT_HOLORASTER = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_cassbeam_comparison"
)
DEFAULT_OFFSET_RING = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/c147_offset_ring"
)
DEFAULT_FULL_JONES = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
DEFAULT_COORDINATE_FEED = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_coordinate_feed_comparison_v2"
)
PACKAGE_HOLOGRAPHY = Path(__file__).resolve().parents[1] / "src" / "sl1mjax" / "data" / "holography_thol0001_lower_c"
SCATTER_POINTS = 8000
ONAXIS_ARCMIN = 0.15
# THOL0001 ANTENNA table is ea02–ea28. ea01 is absent from this MS.
THOL0001_ANTENNA_NAMES = tuple(f"ea{index:02d}" for index in range(2, 29))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _hand(values: np.ndarray, row: int, col: int) -> np.ndarray:
    plane = np.asarray(values)
    if plane.ndim == 4:
        return plane[:, 0, row, col]
    return plane[:, row, col]


def _subsample(n: int, max_points: int = SCATTER_POINTS) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_points, dtype=np.int64)


def extract_holoraster_tables(
    npz_path: Path,
    *,
    antenna_names: tuple[str, ...] = THOL0001_ANTENNA_NAMES,
    predicted_key: str = "predicted_diag",
    full_key: str = "predicted_full",
    offset_key: str = "offset",
    moving_key: str = "moving",
    reference_key: str = "reference",
    mask_key: str | None = "mask",
) -> dict[str, object]:
    """Reduce the 114 MiB comparison arrays to plot tables. No new predict."""

    with np.load(npz_path, allow_pickle=False) as handle:
        measured = handle["measured"]
        predicted = handle[predicted_key]
        full = handle[full_key]
        weight = handle["weight"]
        offset = handle[offset_key]
        moving = handle[moving_key]
        reference = handle[reference_key]
        if mask_key is None or mask_key not in handle.files:
            mask = np.ones(measured.shape[0], dtype=bool)
        else:
            mask = np.asarray(handle[mask_key], dtype=bool)
    intensity = channel32_source_i_jy()
    regions = scientific_voltage_masks(measured, weight, intensity_jy=intensity)
    choose = np.flatnonzero(mask)
    index = choose[_subsample(choose.size)]
    radius = np.hypot(offset[:, 0], offset[:, 1]) * (180.0 * 60.0 / np.pi)
    w_rr = visibility_hand_weight(weight, 0, 0)
    w_ll = visibility_hand_weight(weight, 1, 1)
    maps = {}
    for name, values, hand_weight in (
        ("rr_measured", _hand(measured, 0, 0), w_rr),
        ("rr_cassbeam", _hand(predicted, 0, 0), w_rr),
        ("rr_residual", _hand(measured, 0, 0) - _hand(predicted, 0, 0), w_rr),
        ("ll_measured", _hand(measured, 1, 1), w_ll),
        ("ll_cassbeam", _hand(predicted, 1, 1), w_ll),
        ("ll_residual", _hand(measured, 1, 1) - _hand(predicted, 1, 1), w_ll),
    ):
        mapped = binned_complex_map(
            offset[mask], values[mask], hand_weight[mask], n_bin=SPATIAL_MAP_BINS
        )
        maps[name] = mapped["mean"]
        maps["l_arcmin"] = mapped["l_arcmin"]
        maps["m_arcmin"] = mapped["m_arcmin"]
        maps["weight"] = mapped["weight"]
    occupancy = binned_complex_map(
        offset[mask],
        np.ones(int(np.sum(mask)), dtype=np.complex128),
        np.ones(int(np.sum(mask)), dtype=np.float64),
        n_bin=SPATIAL_MAP_BINS,
    )
    lobe = mask & regions["main_lobe"]
    cells = {}
    for hand, row, col in (("rr", 0, 0), ("ll", 1, 1)):
        averaged = cell_average_vi(
            offset,
            _hand(measured, row, col),
            _hand(predicted, row, col),
            lobe,
            intensity_jy=intensity,
        )
        for key, values in averaged.items():
            cells[f"{hand}_{key}"] = np.asarray(values, dtype=np.float32)
    family = raster_family_from_offset(offset)
    pass_names = {1: "dense / pass-1 occupancy", 2: "sparse / pass-2 occupancy"}

    def _named(rows: list[dict[str, float]], *, family_labels: bool = False) -> list[dict[str, object]]:
        labeled: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            index = int(item["id"])
            if family_labels:
                item["name"] = pass_names.get(index, str(index))
            elif 0 <= index < len(antenna_names):
                item["name"] = antenna_names[index]
            else:
                item["name"] = str(index)
            labeled.append(item)
        return labeled

    strata = {
        "source_i_jy": intensity,
        "mask": "V/I_model using CASA MODEL_DATA 3C147 at channel 32",
        "raster_family": (
            "Nearest Memo 195 dense versus sparse lattice. "
            "This is a pass-1/pass-2 occupancy proxy, not a scan-id join."
        ),
        "mover": _named(
            residual_group_summary(
                _hand(measured, 0, 0) - _hand(predicted, 0, 0), moving, lobe
            )
        ),
        "reference": _named(
            residual_group_summary(
                _hand(measured, 0, 0) - _hand(predicted, 0, 0), reference, lobe
            )
        ),
        "pass": _named(
            residual_group_summary(
                _hand(measured, 0, 0) - _hand(predicted, 0, 0), family, lobe
            ),
            family_labels=True,
        ),
        "mover_ll": _named(
            residual_group_summary(
                _hand(measured, 1, 1) - _hand(predicted, 1, 1), moving, lobe
            )
        ),
        "reference_ll": _named(
            residual_group_summary(
                _hand(measured, 1, 1) - _hand(predicted, 1, 1), reference, lobe
            )
        ),
        "pass_ll": _named(
            residual_group_summary(
                _hand(measured, 1, 1) - _hand(predicted, 1, 1), family, lobe
            ),
            family_labels=True,
        ),
    }
    scatter = {
        "source_i_jy": np.asarray(intensity, dtype=np.float64),
        "rr_obs_real": _hand(measured, 0, 0)[index].real.astype(np.float32),
        "rr_obs_imag": _hand(measured, 0, 0)[index].imag.astype(np.float32),
        "rr_pred_real": _hand(predicted, 0, 0)[index].real.astype(np.float32),
        "rr_pred_imag": _hand(predicted, 0, 0)[index].imag.astype(np.float32),
        "rr_obs_abs": np.abs(_hand(measured, 0, 0)[index]).astype(np.float32),
        "rr_pred_abs": np.abs(_hand(predicted, 0, 0)[index]).astype(np.float32),
        "ll_obs_real": _hand(measured, 1, 1)[index].real.astype(np.float32),
        "ll_obs_imag": _hand(measured, 1, 1)[index].imag.astype(np.float32),
        "ll_pred_real": _hand(predicted, 1, 1)[index].real.astype(np.float32),
        "ll_pred_imag": _hand(predicted, 1, 1)[index].imag.astype(np.float32),
        "ll_obs_abs": np.abs(_hand(measured, 1, 1)[index]).astype(np.float32),
        "ll_pred_abs": np.abs(_hand(predicted, 1, 1)[index]).astype(np.float32),
        "rl_obs_real": _hand(measured, 0, 1)[index].real.astype(np.float32),
        "rl_pred_real": _hand(full, 0, 1)[index].real.astype(np.float32),
        "rl_diag_real": _hand(predicted, 0, 1)[index].real.astype(np.float32),
        "main_lobe_mask": regions["main_lobe"][index],
        "mid_mask": regions["mid"][index],
        "outer_mask": regions["outer_diagnostic"][index],
        "rr_onaxis": radius[index] <= ONAXIS_ARCMIN,
        "ll_onaxis": radius[index] <= ONAXIS_ARCMIN,
    }
    rr_obs = _hand(measured, 0, 0)
    rr_pred = _hand(predicted, 0, 0)
    ll_obs = _hand(measured, 1, 1)
    ll_pred = _hand(predicted, 1, 1)
    l_ax = np.asarray(maps["l_arcmin"], dtype=np.float64)
    m_ax = np.asarray(maps["m_arcmin"], dtype=np.float64)
    ll_grid, mm_grid = np.meshgrid(l_ax, m_ax, indexing="xy")
    map_radius = np.hypot(ll_grid, mm_grid)
    map_outer = {
        hand: complex_visibility_score(
            np.asarray(maps[f"{hand}_measured"])[
                phase_valid_mask(
                    maps[f"{hand}_measured"],
                    maps[f"{hand}_cassbeam"],
                    maps["weight"],
                )
                & (map_radius >= 40.0)
            ],
            np.asarray(maps[f"{hand}_cassbeam"])[
                phase_valid_mask(
                    maps[f"{hand}_measured"],
                    maps[f"{hand}_cassbeam"],
                    maps["weight"],
                )
                & (map_radius >= 40.0)
            ],
        )
        for hand in ("rr", "ll")
    }
    coherence = {
        "source_i_jy": intensity,
        "amp_floor_jy": PHASE_AMP_FLOOR_JY,
        "domain": "moving-reference visibilities, not recovered E",
        "phase_status": "exploratory",
        "raster_extent_arcmin": {
            "l_abs_max": float(np.nanmax(np.abs(offset[mask, 0])) * (180.0 * 60.0 / np.pi)),
            "m_abs_max": float(np.nanmax(np.abs(offset[mask, 1])) * (180.0 * 60.0 / np.pi)),
            "corner_max": float(np.nanmax(radius[mask])),
        },
        "rr": radial_coherence(offset, rr_obs, rr_pred, w_rr, mask),
        "ll": radial_coherence(offset, ll_obs, ll_pred, w_ll, mask),
        "map_phase_beyond_40_arcmin": map_outer,
    }
    movers = []
    for antenna in np.unique(np.asarray(moving)[mask]):
        choose = mask & (np.asarray(moving) == int(antenna))
        index_id = int(antenna)
        movers.append(
            {
                "id": index_id,
                "name": (
                    antenna_names[index_id]
                    if 0 <= index_id < len(antenna_names)
                    else str(index_id)
                ),
                "rr": radial_coherence(offset, rr_obs, rr_pred, w_rr, choose),
                "ll": radial_coherence(offset, ll_obs, ll_pred, w_ll, choose),
            }
        )
    movers.sort(key=lambda item: item["name"])
    examples = {
        "note": (
            "Array-average binned visibilities at example radii. "
            "A common scalar beam cancels voltage phase in Stokes I: "
            "E_p(s) E_q(s)* = |E(s)|^2."
        ),
        "source_i_jy": intensity,
        "examples": bright_source_examples(maps, source_i_jy=intensity),
    }
    return {
        "scatter": scatter,
        "maps": maps,
        "cells": cells,
        "occupancy": {
            "l_arcmin": occupancy["l_arcmin"],
            "m_arcmin": occupancy["m_arcmin"],
            "count": occupancy["weight"],
        },
        "strata": strata,
        "radial_coherence": coherence,
        "antenna_coherence": {"movers": movers, "phase_status": "exploratory"},
        "bright_source_examples": examples,
    }


def extract_coordinate_feed_tables(
    npz_path: Path,
    report: dict[str, Any],
) -> dict[str, object]:
    """Reduce the corrected-coordinate three-beam comparison for publication."""

    with np.load(npz_path, allow_pickle=False) as handle:
        arrays = {key: np.asarray(handle[key]) for key in handle.files}
    measured = np.asarray(arrays["measured_rr_ll"], dtype=np.complex128)
    weight = np.asarray(arrays["weight_rr_ll"], dtype=np.float64)
    source_lm = np.asarray(arrays["source_lm_feed"], dtype=np.float64)
    commanded = np.asarray(arrays["commanded_offset_azelgeo"], dtype=np.float64)
    if measured.ndim != 2 or measured.shape[1] != 2:
        raise ValueError("coordinate/feed measured rows must be (row, RR/LL)")
    n_row = measured.shape[0]
    models = {
        "generic_commanded": np.asarray(
            arrays["generic_commanded_rr_ll"], dtype=np.complex128
        ),
        "generic_source_lm": np.asarray(
            arrays["generic_source_lm_rr_ll"], dtype=np.complex128
        ),
        "evla_c_source_lm": np.asarray(
            arrays["evla_c_source_lm_rr_ll"], dtype=np.complex128
        ),
    }
    if any(values.shape != measured.shape for values in models.values()):
        raise ValueError("coordinate/feed prediction shapes disagree")
    valid = np.column_stack(
        (
            np.asarray(arrays["rr_valid"], dtype=bool),
            np.asarray(arrays["ll_valid"], dtype=bool),
        )
    )
    splits = {
        "development": np.ones(n_row, dtype=bool),
        "train": np.asarray(arrays["train"], dtype=bool),
        "spatial_holdout": np.asarray(arrays["spatial_holdout"], dtype=bool),
        "mover_holdout": np.asarray(arrays["mover_holdout"], dtype=bool),
        "reference_holdout": np.asarray(arrays["reference_holdout"], dtype=bool),
    }
    regions = {
        "all": np.ones(n_row, dtype=bool),
        "main_lobe": np.asarray(arrays["main_lobe"], dtype=bool),
        "mid": np.asarray(arrays["mid"], dtype=bool),
        "outer_diagnostic": np.asarray(arrays["outer_diagnostic"], dtype=bool),
    }
    intensity = float(np.asarray(arrays["source_i_jy"]).reshape(-1)[0])
    metrics: dict[str, object] = {}
    for model_name, predicted in models.items():
        model_metrics: dict[str, object] = {}
        for split_name, split_mask in splits.items():
            split_metrics: dict[str, object] = {}
            for region_name, region_mask in regions.items():
                hands: dict[str, object] = {}
                for hand_index, hand_name in enumerate(("rr", "ll")):
                    choose = split_mask & region_mask & valid[:, hand_index]
                    score = complex_visibility_score(
                        measured[choose, hand_index],
                        predicted[choose, hand_index],
                        weight[choose, hand_index],
                    )
                    residual = np.abs(
                        measured[choose, hand_index] - predicted[choose, hand_index]
                    )
                    score["median_abs_over_i"] = (
                        float(np.median(residual) / intensity) if residual.size else float("nan")
                    )
                    hands[hand_name] = score
                split_metrics[region_name] = hands
            model_metrics[split_name] = split_metrics
        metrics[model_name] = model_metrics

    choose = _subsample(n_row)
    scatter: dict[str, np.ndarray] = {
        "source_i_jy": np.asarray(intensity, dtype=np.float64),
        "measured_rr": measured[choose, 0].astype(np.complex64),
        "measured_ll": measured[choose, 1].astype(np.complex64),
        "main_lobe": regions["main_lobe"][choose],
        "mid": regions["mid"][choose],
        "outer_diagnostic": regions["outer_diagnostic"][choose],
        "train": splits["train"][choose],
        "spatial_holdout": splits["spatial_holdout"][choose],
        "mover_holdout": splits["mover_holdout"][choose],
    }
    for model_name, predicted in models.items():
        scatter[f"{model_name}_rr"] = predicted[choose, 0].astype(np.complex64)
        scatter[f"{model_name}_ll"] = predicted[choose, 1].astype(np.complex64)

    maps: dict[str, np.ndarray] = {}
    for hand_index, hand_name in enumerate(("rr", "ll")):
        choose_hand = valid[:, hand_index]
        measured_map = binned_complex_map(
            source_lm[choose_hand],
            measured[choose_hand, hand_index],
            weight[choose_hand, hand_index],
            n_bin=SPATIAL_MAP_BINS,
        )
        maps[f"{hand_name}_measured"] = measured_map["mean"]
        maps[f"{hand_name}_weight"] = measured_map["weight"]
        maps["l_arcmin"] = measured_map["l_arcmin"]
        maps["m_arcmin"] = measured_map["m_arcmin"]
        for model_name, predicted in models.items():
            model_map = binned_complex_map(
                source_lm[choose_hand],
                predicted[choose_hand, hand_index],
                weight[choose_hand, hand_index],
                n_bin=SPATIAL_MAP_BINS,
            )
            maps[f"{model_name}_{hand_name}"] = model_map["mean"]

    summary = {
        "artifact": report.get("artifact"),
        "scope": "SPW-4 frozen development rows",
        "frequency_hz": float(np.asarray(arrays["frequency_hz"]).reshape(-1)[0]),
        "source_i_jy": intensity,
        "n_rows": n_row,
        "coordinate_contract": {
            "commanded_offset_azelgeo": "DIRECTION - TARGET = POINTING_OFFSET",
            "source_lm_feed": "negative commanded displacement; CASSBEAM query coordinate",
            "axis_map": (report.get("interpretation") or {}).get("axis_map"),
        },
        "models": {
            "generic_commanded": "historical generic VLA artifact at commanded offsets",
            "generic_source_lm": "generic VLA artifact at source_lm_feed",
            "evla_c_source_lm": "CASA-derived EVLA-C feed parameters at source_lm_feed",
        },
        "metrics": metrics,
        "paired_scores": report.get("paired_scores"),
        "interpretation": report.get("interpretation"),
        "map_squint": report.get("map_squint"),
        "generic_plane_centroids": report.get("generic_plane_centroids"),
        "evla_plane_centroids": report.get("evla_plane_centroids"),
        "width_grids": report.get("width_grids"),
        "row_accounting": report.get("row_accounting"),
        "development_only": True,
        "spw5_closed": True,
        "production_beam_frozen": False,
        "commanded_source_max_abs_error_rad": float(
            np.max(np.abs(source_lm + commanded))
        ),
    }
    return {"summary": summary, "scatter": scatter, "maps": maps}


def _frequency_without_publication_squint(frequency: dict[str, Any]) -> dict[str, Any]:
    """Keep copolar/cross-hand scores. Demote full-raster frequency squint."""

    channels = []
    for item in frequency.get("channels") or ():
        row = {
            "channel": item["channel"],
            "frequency_hz": item["frequency_hz"],
            "diagonal": item.get("diagonal"),
            "experimental_full_jones": item.get("experimental_full_jones"),
        }
        channels.append(row)
    return {
        "channels": channels,
        "superseded_full_raster_squint": {
            "publication_estimator": False,
            "estimator": "full_raster_power_centroid",
            "measured": frequency.get("squint_measured"),
            "cassbeam": frequency.get("squint_cassbeam"),
            "note": "Channel-dependent squint was a full-raster centroid and is not published",
        },
    }


def _slim_offset_ring(report: dict[str, Any], smoke: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    fields = list(geometry.get("fields") or report.get("geometry", {}).get("fields") or ())
    radii = [float(item["radius_arcmin"]) for item in fields]
    return {
        "decision": report.get("decision"),
        "model_selected": bool(report.get("selected_model")),
        "full_jones_frozen": bool(report.get("full_jones_frozen")),
        "production_factory_modified": bool(report.get("production_factory_modified")),
        "spw5_closed": bool(report.get("spw5_closed", True)),
        "applied_from": report.get("applied_from"),
        "c147_offset_used_for_d": bool(report.get("c147_offset_used_for_d")),
        "convention_ladder_searched": bool(report.get("convention_ladder_searched")),
        "row_accounting": report.get("row_accounting"),
        "scores": {
            key: report.get("scores", {}).get(key)
            for key in ("inner_unit", "inner_hat", "sealed_unit", "sealed_hat", "rrll_unit")
        },
        "upper_limit": report.get("upper_limit"),
        "closure": smoke.get("closure"),
        "smoke": smoke.get("smoke"),
        "median_abs": smoke.get("median_abs"),
        "calibration_hashes": report.get("calibration_hashes"),
        "radius_arcmin": float(np.median(radii)) if radii else float("nan"),
        "q_u_is_frequency_holdout": False,
        "q_u_note": "The already-written ring Q/U used all 64 channels and is not a frequency holdout",
        "nuisance": {
            "q_over_i": (report.get("nuisance") or {}).get("q_over_i"),
            "u_over_i": (report.get("nuisance") or {}).get("u_over_i"),
            "residual_jones": (report.get("nuisance") or {}).get("residual_jones"),
        },
    }


def _observation_summary(occupancy: dict[str, Any]) -> dict[str, Any]:
    combined = occupancy.get("combined") or {}
    passes = occupancy.get("passes") or []
    pass1 = next((item for item in passes if item.get("name") == "pass_1"), {})
    pass2 = next((item for item in passes if item.get("name") == "pass_2"), {})
    return {
        "project": THOL0001_PROJECT,
        "scheduling_block": THOL0001_SCHEDULING_BLOCK,
        "execution_block": THOL0001_EXECUTION_BLOCK,
        "source": THOL0001_SOURCE,
        "holoraster_field": THOL0001_HOLORASTER_FIELD,
        "on_axis_field_ids": list(ON_AXIS_3C147_FIELD_IDS),
        "offset_field_ids": list(C147_OFFSET_FIELD_IDS),
        "native_frequencies_hz": list(THOL0001_LOWER_C_NATIVE_HZ),
        "version1_frequency_hz": THOL0001_LOWER_C_NATIVE_HZ[0],
        "spw5_frequency_hz": THOL0001_LOWER_C_NATIVE_HZ[1],
        "spw5_status": "sealed",
        "pointing_offset_frame": "AZELGEO",
        "on_source_is_selection": False,
        "snap_pass2_to_memo_lattice": False,
        "antenna_names": list(THOL0001_ANTENNA_NAMES),
        "reference_antenna_names": list(THOL0001_REFERENCE_ANTENNA_NAMES),
        "reference_antenna": REFERENCE_ANTENNA,
        "occupancy": {
            "n_clustered_cells": combined.get("n_clustered_cells"),
            "n_dense_family": combined.get("n_dense_family"),
            "n_sparse_family": combined.get("n_sparse_family"),
            "pass1_cells": pass1.get("n_clustered_cells"),
            "pass2_cells": pass2.get("n_clustered_cells"),
        },
        "timeline": [
            {"label": "flux/bandpass", "scan_start": 2, "n_scans": 1},
            {"label": "pass 1 raster", "scan_start": 18, "n_scans": int(len(pass1.get("scan_numbers") or ()))},
            {"label": "on-axis 3C147", "scan_start": 51, "n_scans": 1},
            {"label": "pass 2 raster", "scan_start": 57, "n_scans": int(len(pass2.get("scan_numbers") or ()))},
        ],
    }


def _calibration_summary(
    holography_dir: Path,
    offset_hashes: dict[str, Any],
    applyback: dict[str, Any] | None,
) -> dict[str, Any]:
    model = _load(holography_dir / "three_c147_point_model.json") if (holography_dir / "three_c147_point_model.json").is_file() else {}
    return {
        "reference_antenna": REFERENCE_ANTENNA,
        "calwt": CALWT,
        "diagonal_parang": False,
        "fullpol_parang": FULLPOL_PARANG,
        "apply_holoraster_from": "CORRECTED_DATA",
        "apply_c147_offset_from": "DATA",
        "apply_fields_fullpol": "0,9,10,11",
        "c147_offset_applied_in_memory": True,
        "source_model": {
            "name": (model.get("model") or {}).get("name", "3C147"),
            "kind": (model.get("model") or {}).get("kind", "point"),
            "standard": (model.get("model") or {}).get("standard", "Perley-Butler 2017"),
        },
        "table_hashes": offset_hashes,
        "three_c286": {
            "casaguide_two_chi_deg": (applyback or {}).get("casaguide_two_chi_expected_deg"),
            "iau_evpa_deg": (applyback or {}).get("iau_evpa_expected_deg"),
            "residual_x_deg": (applyback or {}).get("residual_x_deg"),
            "v_over_i": ((applyback or {}).get("circular_polarisation_floor") or {}).get("v_over_i"),
        },
    }


def _convention_gates(
    holography_dir: Path,
    comparison: dict[str, Any],
    goldens: dict[str, Any] | None,
) -> dict[str, Any]:
    compare = _load(holography_dir / "compare_summary.json") if (holography_dir / "compare_summary.json").is_file() else {}
    convention = DEFAULT_CONVENTION
    return {
        "software_gates_passed": bool((comparison.get("software_gates") or {}).get("passed")),
        "injected_operator_oracle": (goldens or {}).get("injected_operator_oracle", {}).get("status"),
        "fullpol_rescored_passed": (goldens or {}).get("fullpol_rescored_passed"),
        "diagonal_golden_passed": (goldens or {}).get("diagonal_passed"),
        "cumulative_residuals": compare.get("by_stage") or {},
        "cassbeam_convention": {
            "l_sign": convention.l_sign,
            "m_sign": convention.m_sign,
            "swap_lm": convention.swap_lm,
            "jones": convention.jones,
            "swap_rl": convention.swap_rl,
            "rotate_spatial": convention.rotate_spatial,
            "note": "Locked mount-frame native packing. The 128-convention search is not reopened.",
        },
        "not_physical_beam_evidence": True,
    }


def build_from_named_products(
    *,
    holoraster_dir: Path,
    offset_ring_dir: Path,
    full_jones_dir: Path | None,
    coordinate_feed_dir: Path,
    holography_dir: Path,
    output_dir: Path,
) -> Path:
    comparison = _load(holoraster_dir / "holoraster_cassbeam_comparison.json")
    channel32 = _load(holoraster_dir / "channel32_summaries.json")
    frequency = _frequency_without_publication_squint(_load(holoraster_dir / "frequency_summaries.json"))
    squint = _load(holoraster_dir / "squint_mainlobe_20pct.json")
    if "squint_publication" in comparison:
        channel32 = {
            **channel32,
            "n_rows": channel32.get("n_rows"),
            "frequency_hz": channel32.get("frequency_hz"),
            "model_selected": False,
            "full_jones_frozen": False,
            "production_factory_modified": False,
            "diagonal_support": comparison.get("diagonal_support"),
        }
    report = _load(offset_ring_dir / "report.json")
    smoke = _load(offset_ring_dir / "channel32_smoke.json")
    geometry = _load(offset_ring_dir / "geometry.json")
    occupancy_summary = _load(holography_dir / "occupancy_summary.json")
    applyback = None
    goldens = None
    if full_jones_dir is not None and (full_jones_dir / "three_c286_applyback.json").is_file():
        applyback = _load(full_jones_dir / "three_c286_applyback.json")
    if full_jones_dir is not None and (full_jones_dir / "goldens.json").is_file():
        goldens = _load(full_jones_dir / "goldens.json")
    tables = extract_holoraster_tables(
        holoraster_dir / "channel32_comparison.npz",
        antenna_names=THOL0001_ANTENNA_NAMES,
    )
    coordinate_feed_report = _load(coordinate_feed_dir / "report.json")
    coordinate_feed = extract_coordinate_feed_tables(
        coordinate_feed_dir / "channel32_model_comparison.npz",
        coordinate_feed_report,
    )
    fields = {
        "fields": [
            {
                "field_id": item["field_id"],
                "name": item["name"],
                "l_arcmin": float(item["l_rad"] * 180.0 * 60.0 / np.pi),
                "m_arcmin": float(item["m_rad"] * 180.0 * 60.0 / np.pi),
                "radius_arcmin": item["radius_arcmin"],
            }
            for item in geometry["fields"]
        ],
        "partition": {
            "training": list(geometry["partition"]["training"]),
            "inner_holdout": list(geometry["partition"]["inner_holdout"]),
            "sealed_holdout": list(geometry["partition"]["sealed_holdout"]),
        },
    }
    quadrants = {
        name: channel32["quadrants"][name]
        for name in ("east", "west", "north", "south", "east_west", "north_south")
        if name in channel32.get("quadrants", {})
    }
    return write_validation_bundle(
        output_dir,
        observation=_observation_summary(occupancy_summary),
        calibration=_calibration_summary(
            holography_dir, report.get("calibration_hashes") or {}, applyback
        ),
        convention_gates=_convention_gates(holography_dir, comparison, goldens),
        holoraster_channel32=channel32,
        holoraster_frequency=frequency,
        squint=squint,
        offset_ring=_slim_offset_ring(report, smoke, geometry),
        residual_geometry=channel32["residual_geometry"],
        residual_strata=tables["strata"],
        radial_coherence=tables["radial_coherence"],
        antenna_coherence=tables["antenna_coherence"],
        bright_source_examples=tables["bright_source_examples"],
        frequency_series=frequency,
        offset_ring_fields=fields,
        crosshand_quadrants=quadrants,
        scatter=tables["scatter"],
        maps=tables["maps"],
        cells=tables["cells"],
        occupancy=tables["occupancy"],
        coordinate_feed_comparison=coordinate_feed["summary"],
        coordinate_feed_scatter=coordinate_feed["scatter"],
        coordinate_feed_maps=coordinate_feed["maps"],
        provenance={
            "measurement_set_identity": "THOL0001.lowerC.spw45.scientific.ms",
            "revision": "evla_c_source_lm_impact",
            "source_products": {
                "holoraster_cassbeam_comparison": "named Bacchus product",
                "c147_offset_ring": "named Bacchus product",
                "holography_thol0001_lower_c": "in-repo occupancy and calibration summaries",
                "holoraster_coordinate_feed_comparison_v2": "named Bacchus product",
            },
            "predictions_recomputed": {
                "historical_generic_commanded": False,
                "coordinate_feed_comparison": True,
            },
            "full_raster_squint_published": False,
            "spw5_opened": False,
            "plot_code": PUBLICATION_VERSION,
        },
    )


def build_from_existing_bundle(
    *,
    existing_bundle: Path,
    coordinate_feed_dir: Path,
    output_dir: Path,
) -> Path:
    """Add the corrected-coordinate comparison without reopening the MS products."""

    source = Path(existing_bundle)
    tables = source / "plot_tables"

    def plot_table(name: str) -> dict[str, np.ndarray]:
        with np.load(tables / name, allow_pickle=False) as handle:
            return {key: np.asarray(handle[key]) for key in handle.files}

    coordinate_report = _load(coordinate_feed_dir / "report.json")
    coordinate = extract_coordinate_feed_tables(
        coordinate_feed_dir / "channel32_model_comparison.npz",
        coordinate_report,
    )
    old_manifest = _load(source / "manifest.json")
    provenance = {
        key: value
        for key, value in old_manifest.items()
        if key not in {"schema_version", "publication_version", "files", "bundle_sha256"}
    }
    provenance.update(
        {
            "revision": "evla_c_source_lm_impact",
            "previous_bundle_sha256": old_manifest.get("bundle_sha256"),
            "predictions_recomputed": {
                "historical_generic_commanded": False,
                "coordinate_feed_comparison": True,
            },
            "spw5_opened": False,
        }
    )
    return write_validation_bundle(
        output_dir,
        observation=_load(source / "observation_summary.json"),
        calibration=_load(source / "calibration_summary.json"),
        convention_gates=_load(source / "convention_gates.json"),
        holoraster_channel32=_load(source / "holoraster_channel32.json"),
        holoraster_frequency=_load(source / "holoraster_frequency.json"),
        squint=_load(source / "squint_publication.json"),
        offset_ring=_load(source / "offset_ring.json"),
        residual_geometry=_load(tables / "residual_geometry.json"),
        residual_strata=_load(tables / "residual_strata.json"),
        radial_coherence=_load(tables / "radial_coherence.json"),
        antenna_coherence=_load(tables / "antenna_coherence.json"),
        bright_source_examples=_load(tables / "bright_source_examples.json"),
        frequency_series=_load(tables / "frequency_series.json"),
        offset_ring_fields=_load(tables / "offset_ring_fields.json"),
        crosshand_quadrants=_load(tables / "crosshand_quadrants.json"),
        scatter=plot_table("holoraster_scatter.npz"),
        maps=plot_table("holoraster_maps.npz"),
        cells=plot_table("holoraster_cells.npz"),
        occupancy=plot_table("raster_occupancy.npz"),
        coordinate_feed_comparison=coordinate["summary"],
        coordinate_feed_scatter=coordinate["scatter"],
        coordinate_feed_maps=coordinate["maps"],
        provenance=provenance,
    )


def build_from_refresh(
    *,
    refresh_dir: Path,
    existing_bundle: Path,
    coordinate_feed_dir: Path,
    output_dir: Path,
) -> Path:
    """Build v2 from the EVLA-C refresh product. Reuse F01–F04 provenance only."""

    source = Path(existing_bundle)
    tables = source / "plot_tables"
    channel32_report = _load(refresh_dir / "channels" / "channel32_report.json")
    classification_path = refresh_dir / "phase4_classification.json"
    if classification_path.is_file():
        channel32_report = {
            **channel32_report,
            "classification": _load(classification_path),
        }
    holoraster_channel32 = holoraster_channel32_from_report(channel32_report)
    reports = [channel32_report]
    for path in sorted((refresh_dir / "channels").glob("channel*_report.json")):
        if path.name == "channel32_report.json":
            continue
        reports.append(_load(path))
    reports.sort(key=lambda item: int(item["channel"]))
    frequency = frequency_series_from_reports(reports)
    squint = squint_from_refresh_export(refresh_dir / "channels" / "channel32_export.npz")
    extracted = extract_holoraster_tables(
        refresh_dir / "channels" / "channel32_export.npz",
        predicted_key=PRIMARY_ARMS[0],
        full_key=PRIMARY_ARMS[1],
        offset_key="source_lm_feed",
        moving_key="moving_id",
        reference_key="reference_id",
        mask_key=None,
    )
    coordinate_report = _load(coordinate_feed_dir / "report.json")
    coordinate = extract_coordinate_feed_tables(
        coordinate_feed_dir / "channel32_model_comparison.npz",
        coordinate_report,
    )
    ring_path = refresh_dir / "c147_ring" / "c147_report.json"
    if ring_path.is_file():
        offset_ring = offset_ring_from_c147_report(_load(ring_path))
        offset_fields = {
            "fields": offset_ring.get("fields") or [],
            "partition": offset_ring.get("partition") or {},
            "historical_holdout": True,
        }
    else:
        offset_ring = {
            **_load(source / "offset_ring.json"),
            "status": "blocked",
            "reason": "EVLA-C nine-frequency ring not yet written",
            "historical": True,
        }
        offset_fields = _load(tables / "offset_ring_fields.json")
    old_manifest = _load(source / "manifest.json")
    provenance = {
        "measurement_set_identity": "THOL0001.lowerC.spw45.scientific.ms",
        "revision": "evla_c_full_jones_validation_refresh_v1",
        "publication_version": PUBLICATION_BUNDLE_VERSION,
        "previous_bundle_sha256": old_manifest.get("bundle_sha256"),
        "predictions_recomputed": {
            "historical_generic_commanded": False,
            "evla_c_source_diagonal": True,
            "evla_c_source_full_jones": True,
        },
        "full_raster_squint_published": False,
        "spw5_opened": False,
        "plot_code": PUBLICATION_BUNDLE_VERSION,
        "supersession": supersession_ledger(),
        "f01_f04_reused": True,
        "development_only": True,
    }
    return write_validation_bundle(
        output_dir,
        observation=_load(source / "observation_summary.json"),
        calibration=_load(source / "calibration_summary.json"),
        convention_gates=_load(source / "convention_gates.json"),
        holoraster_channel32=holoraster_channel32,
        holoraster_frequency=frequency,
        squint=squint,
        offset_ring=offset_ring,
        residual_geometry=residual_geometry_from_export(
            refresh_dir / "channels" / "channel32_export.npz"
        ),
        residual_strata=extracted["strata"],
        radial_coherence=extracted["radial_coherence"],
        antenna_coherence=extracted["antenna_coherence"],
        bright_source_examples=extracted["bright_source_examples"],
        frequency_series=frequency,
        offset_ring_fields=offset_fields,
        crosshand_quadrants=crosshand_quadrants_from_export(
            refresh_dir / "channels" / "channel32_export.npz",
            source_i_jy=float(channel32_report.get("source_i_jy") or 8.0),
        ),
        scatter=extracted["scatter"],
        maps=extracted["maps"],
        cells=extracted["cells"],
        occupancy=_load_occupancy(tables),
        coordinate_feed_comparison=coordinate["summary"],
        coordinate_feed_scatter=coordinate["scatter"],
        coordinate_feed_maps=coordinate["maps"],
        provenance=provenance,
        publication_version=PUBLICATION_BUNDLE_VERSION,
    )


def _load_occupancy(tables: Path) -> dict[str, np.ndarray]:
    with np.load(tables / "raster_occupancy.npz", allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holoraster-dir", type=Path, default=DEFAULT_HOLORASTER)
    parser.add_argument("--offset-ring-dir", type=Path, default=DEFAULT_OFFSET_RING)
    parser.add_argument("--full-jones-dir", type=Path, default=DEFAULT_FULL_JONES)
    parser.add_argument("--coordinate-feed-dir", type=Path, default=DEFAULT_COORDINATE_FEED)
    parser.add_argument("--holography-dir", type=Path, default=PACKAGE_HOLOGRAPHY)
    parser.add_argument("--output-dir", type=Path, default=default_bundle_root())
    parser.add_argument(
        "--existing-bundle",
        type=Path,
        help="augment an existing schema-3 bundle without reopening its Bacchus sources",
    )
    parser.add_argument(
        "--refresh-dir",
        type=Path,
        help="EVLA-C refresh product; writes a v2 bundle and does not overwrite v1",
    )
    arguments = parser.parse_args()
    if arguments.refresh_dir is not None:
        repo = Path(__file__).resolve().parents[1]
        v1 = repo / "src" / "sl1mjax" / "data" / PUBLICATION_VERSION
        v2 = repo / "src" / "sl1mjax" / "data" / PUBLICATION_BUNDLE_VERSION
        destination = arguments.output_dir
        if destination.resolve() == v1.resolve():
            destination = v2
        if destination.resolve() == v1.resolve():
            raise RuntimeError("refusing to overwrite vla_c_band_beam_validation_v1")
        path = build_from_refresh(
            refresh_dir=arguments.refresh_dir,
            existing_bundle=arguments.existing_bundle or v1,
            coordinate_feed_dir=arguments.coordinate_feed_dir,
            output_dir=destination,
        )
        print(path)
        return
    if arguments.existing_bundle is not None:
        path = build_from_existing_bundle(
            existing_bundle=arguments.existing_bundle,
            coordinate_feed_dir=arguments.coordinate_feed_dir,
            output_dir=arguments.output_dir,
        )
        print(path)
        return
    path = build_from_named_products(
        holoraster_dir=arguments.holoraster_dir,
        offset_ring_dir=arguments.offset_ring_dir,
        full_jones_dir=arguments.full_jones_dir if arguments.full_jones_dir.is_dir() else None,
        coordinate_feed_dir=arguments.coordinate_feed_dir,
        holography_dir=arguments.holography_dir,
        output_dir=arguments.output_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
