"""Shared HOLORASTER extract/predict for the EVLA-C refresh.

Uses the existing comparison extraction path. Exports all four correlations.
Does not inherit width or squint corrections.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.cassbeam_highres import HighresCassbeamCatalog
from sl1mjax.evla_c_metrics import hand_scores
from sl1mjax.evla_c_validation_refresh import (
    COORDINATE_QUERY,
    HOLORASTER_FIELD_ID,
    PRIMARY_ARMS,
    refuse_frozen_write,
)
from sl1mjax.holography_alignment import holoraster_pair_masks
from sl1mjax.holography_beam_prior import evaluate_holoraster_cassbeam, squeeze_sample_jones
from sl1mjax.holography_cassbeam_correction import region_masks_from_voltage
from sl1mjax.holography_cassbeam_holoraster_report import locked_convention
from sl1mjax.holography_coordinate_feed_comparison import query_coordinates
from sl1mjax.holography_diagonal_correction import (
    refuse_c147_training,
    refuse_spw5,
    spw4_correction_holdouts,
)
from sl1mjax.holography_full_jones import moving_reference_row_geometry


def squeeze_vis(values: ArrayLike) -> NDArray[np.complex128]:
    return squeeze_sample_jones(values)


def predict_primary_arms(
    *,
    catalog: HighresCassbeamCatalog,
    frequencies_hz: ArrayLike,
    offset_lm_rad: ArrayLike,
    chi_moving: ArrayLike,
    chi_reference: ArrayLike,
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    moving_is_p: ArrayLike,
    residual_jones: Mapping[int, ArrayLike],
    source: ArrayLike,
) -> dict[str, NDArray[np.complex128]]:
    kwargs = {
        "catalog": catalog,
        "frequencies_hz": frequencies_hz,
        "convention": locked_convention(),
        "offset_lm_rad": offset_lm_rad,
        "chi_moving": chi_moving,
        "chi_reference": chi_reference,
        "moving_id": moving_id,
        "reference_id": reference_id,
        "moving_is_p": moving_is_p,
        "residual_jones": residual_jones,
        "source": source,
    }
    diagonal = squeeze_vis(evaluate_holoraster_cassbeam(off_diagonal=False, **kwargs).visibility)
    full = squeeze_vis(evaluate_holoraster_cassbeam(off_diagonal=True, **kwargs).visibility)
    return {
        PRIMARY_ARMS[0]: diagonal,
        PRIMARY_ARMS[1]: full,
    }


def build_holdout_masks(
    observation,
    names: tuple[str, ...],
    keep: np.ndarray,
    *,
    holdouts=None,
) -> dict[str, np.ndarray]:
    if holdouts is None:
        holdouts = spw4_correction_holdouts(observation, names)
    return {
        "train": np.asarray(holdouts.train, dtype=bool)[keep],
        "spatial_holdout": np.asarray(holdouts.spatial_holdout, dtype=bool)[keep],
        "mover_holdout": np.asarray(holdouts.mover_holdout, dtype=bool)[keep],
        "reference_holdout": np.asarray(holdouts.reference_holdout, dtype=bool)[keep],
    }


def extract_holoraster_channel(cmp, *, channel: int, measurement_set, product_dir, names):
    """Reuse the comparison runner's MS loaders. One extraction per channel."""

    refuse_spw5(spectral_window_id=4, opened=False)
    from sl1mjax.holography_ms import _read_antennas, _tables

    print(f"extract ch{channel}: tables/antennas", flush=True)
    tables = _tables()
    _ids, loaded_names, positions = _read_antennas(tables, measurement_set)
    loaded_names = tuple(loaded_names)
    if names is not None and tuple(names) != loaded_names:
        raise ValueError("antenna name table changed")
    diag = getattr(cmp, "_cached_diag", None)
    if diag is None:
        print(f"extract ch{channel}: load diagonal helper", flush=True)
        diag = cmp._diag()
        cmp._cached_diag = diag
    print(f"extract ch{channel}: load HOLORASTER visibilities", flush=True)
    observation, chi, packed, intensity, voltage, rr_ok, ll_ok = cmp._load_holoraster_channel(
        diag,
        tables,
        measurement_set,
        positions,
        channel=int(channel),
    )
    print(f"extract ch{channel}: geometry/holdouts", flush=True)
    refuse_c147_training(observation.block.field_id)
    if np.any(np.asarray(observation.block.field_id) != HOLORASTER_FIELD_ID):
        raise ValueError("HOLORASTER extract must stay on field 10")
    geometry = moving_reference_row_geometry(observation, require_all_channels=False)
    pair = holoraster_pair_masks(observation)
    usable = np.asarray(geometry["usable"], dtype=bool) & np.asarray(
        pair["moving_reference"], dtype=bool
    )
    holdouts = spw4_correction_holdouts(observation, loaded_names)
    development = (
        holdouts.train
        | holdouts.spatial_holdout
        | holdouts.mover_holdout
        | holdouts.reference_holdout
    )
    keep = usable & development
    full_raster = usable
    fields = cmp._row_fields(geometry, pair, chi, np.flatnonzero(keep))
    commanded = np.asarray(geometry["commanded_offset_azelgeo"], dtype=np.float64)[fields["rows"]]
    source_lm = np.asarray(geometry["source_lm_feed"], dtype=np.float64)[fields["rows"]]
    np.testing.assert_allclose(source_lm, query_coordinates(commanded, query=COORDINATE_QUERY))
    source = np.asarray(observation.source_coherency_visibility, dtype=np.complex128)
    if source.shape == (2, 2):
        source_rows = source
    elif source.ndim == 3:
        source_rows = source[fields["rows"], None, :, :]
    else:
        source_rows = source[fields["rows"]]
    measured = np.asarray(packed, dtype=np.complex128)
    measured = measured[fields["rows"], 0] if measured.ndim == 4 else measured[fields["rows"]]
    weight = cmp._hand_weight(observation, keep)
    if weight.ndim == 4:
        weight = weight[:, 0]
    regions = region_masks_from_voltage(np.asarray(voltage, dtype=np.float64)[fields["rows"]])
    residual = cmp._residual_jones(product_dir)
    return {
        "observation": observation,
        "chi": chi,
        "packed": packed,
        "intensity": intensity,
        "voltage": voltage,
        "rr_ok": np.asarray(rr_ok, dtype=bool)[fields["rows"]],
        "ll_ok": np.asarray(ll_ok, dtype=bool)[fields["rows"]],
        "names": loaded_names,
        "positions": positions,
        "geometry": geometry,
        "fields": fields,
        "commanded": commanded,
        "source_lm": source_lm,
        "source": source_rows,
        "measured": measured,
        "weight": weight,
        "regions": regions,
        "residual": residual,
        "keep": keep,
        "full_raster": full_raster,
        "n_development": int(np.sum(keep)),
        "n_full_raster_usable": int(np.sum(full_raster)),
        "frequency_hz": float(np.asarray(observation.block.frequency_hz).reshape(-1)[0]),
        "masks": build_holdout_masks(observation, loaded_names, keep),
    }


def export_channel_arrays(
    path: Path, extract: Mapping[str, object], predictions: Mapping[str, ArrayLike]
) -> Path:
    refuse_frozen_write(path)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "measured": np.asarray(extract["measured"], dtype=np.complex128),
        "weight": np.asarray(extract["weight"], dtype=np.float64),
        "commanded_offset_azelgeo": np.asarray(extract["commanded"], dtype=np.float64),
        "source_lm_feed": np.asarray(extract["source_lm"], dtype=np.float64),
        "moving_id": np.asarray(extract["fields"]["moving"], dtype=np.int32),
        "reference_id": np.asarray(extract["fields"]["reference"], dtype=np.int32),
        "moving_is_p": np.asarray(extract["fields"]["moving_is_p"], dtype=bool),
        "chi_moving": np.asarray(extract["fields"]["chi_m"], dtype=np.float64),
        "chi_reference": np.asarray(extract["fields"]["chi_r"], dtype=np.float64),
        "row_id": np.asarray(extract["fields"]["rows"], dtype=np.int64),
        "train": np.asarray(extract["masks"]["train"], dtype=bool),
        "spatial_holdout": np.asarray(extract["masks"]["spatial_holdout"], dtype=bool),
        "mover_holdout": np.asarray(extract["masks"]["mover_holdout"], dtype=bool),
        "reference_holdout": np.asarray(extract["masks"]["reference_holdout"], dtype=bool),
        "main_lobe": np.asarray(extract["regions"]["main_lobe"], dtype=bool),
        "mid": np.asarray(extract["regions"]["mid"], dtype=bool),
        "outer_diagnostic": np.asarray(extract["regions"]["outer_diagnostic"], dtype=bool),
        "rr_valid": np.asarray(extract["rr_ok"], dtype=bool),
        "ll_valid": np.asarray(extract["ll_ok"], dtype=bool),
        "frequency_hz": np.asarray(extract["frequency_hz"], dtype=np.float64),
        "source_i_jy": np.asarray(extract["intensity"], dtype=np.float64).reshape(-1)[:1],
        "coordinate_query": np.asarray(COORDINATE_QUERY),
    }
    for name, vis in predictions.items():
        payload[name] = squeeze_vis(vis)
    if PRIMARY_ARMS[0] in predictions and PRIMARY_ARMS[1] in predictions:
        payload["full_minus_diagonal"] = squeeze_vis(predictions[PRIMARY_ARMS[1]]) - squeeze_vis(
            predictions[PRIMARY_ARMS[0]]
        )
    np.savez_compressed(destination, **payload)
    return destination


def identity_vs_prior_evla_diagonal(
    new_export: Path,
    prior_npz: Path,
    *,
    atol: float = 1.0e-6,
) -> dict[str, object]:
    """Compare new EVLA-C RR/LL to the prior coordinate-feed EVLA diagonal.

    Matching uses source-in-beam coordinates quantized to 1 µas-scale bins
    plus moving antenna when both files have it. This is not a full-raster
    versus development-subset comparison.
    """

    if not Path(prior_npz).is_file():
        return {"status": "blocked", "reason": f"missing prior export {prior_npz}"}
    with np.load(new_export, allow_pickle=False) as handle:
        new_lm = np.asarray(handle["source_lm_feed"], dtype=np.float64)
        new_pred = squeeze_vis(handle[PRIMARY_ARMS[0]])
        new_moving = (
            np.asarray(handle["moving_id"], dtype=np.int32)
            if "moving_id" in handle.files
            else None
        )
        new_reference = (
            np.asarray(handle["reference_id"], dtype=np.int32)
            if "reference_id" in handle.files
            else None
        )
    with np.load(prior_npz, allow_pickle=False) as handle:
        old_lm = np.asarray(handle["source_lm_feed"], dtype=np.float64)
        if "evla_c_source_lm_rr_ll" not in handle.files:
            return {"status": "blocked", "reason": "prior export has no evla_c_source_lm_rr_ll"}
        old_pred = np.asarray(handle["evla_c_source_lm_rr_ll"], dtype=np.complex128)
        old_moving = (
            np.asarray(handle["moving_id"], dtype=np.int32)
            if "moving_id" in handle.files
            else None
        )
        old_reference = (
            np.asarray(handle["reference_id"], dtype=np.int32)
            if "reference_id" in handle.files
            else None
        )
    scale = 1.0e10
    new_key = np.column_stack(
        (np.round(new_lm[:, 0] * scale), np.round(new_lm[:, 1] * scale))
    ).astype(np.int64)
    old_key = np.column_stack(
        (np.round(old_lm[:, 0] * scale), np.round(old_lm[:, 1] * scale))
    ).astype(np.int64)
    if new_moving is not None and old_moving is not None:
        new_key = np.column_stack((new_key, new_moving.astype(np.int64)))
        old_key = np.column_stack((old_key, old_moving.astype(np.int64)))
    if new_reference is not None and old_reference is not None:
        new_key = np.column_stack((new_key, new_reference.astype(np.int64)))
        old_key = np.column_stack((old_key, old_reference.astype(np.int64)))
    old_index = {tuple(row.tolist()): i for i, row in enumerate(old_key)}
    pairs = []
    for i, row in enumerate(new_key):
        j = old_index.get(tuple(row.tolist()))
        if j is not None:
            pairs.append((i, j))
    if not pairs:
        return {
            "status": "blocked",
            "reason": "no matched rows between new development export and prior EVLA-C",
            "n_new": int(new_lm.shape[0]),
            "n_prior": int(old_lm.shape[0]),
        }
    ii = np.asarray(pairs)[:, 0]
    jj = np.asarray(pairs)[:, 1]
    new_rr = new_pred[ii, 0, 0]
    new_ll = new_pred[ii, 1, 1]
    old_rr = old_pred[jj, 0]
    old_ll = old_pred[jj, 1]
    d_rr = np.abs(new_rr - old_rr)
    d_ll = np.abs(new_ll - old_ll)
    max_abs = float(max(np.nanmax(d_rr), np.nanmax(d_ll)))
    median_abs = float(np.nanmedian(np.concatenate([d_rr, d_ll])))
    same_index = None
    if new_pred.shape[0] == old_pred.shape[0]:
        direct_rr = np.abs(new_pred[:, 0, 0] - old_pred[:, 0])
        direct_ll = np.abs(new_pred[:, 1, 1] - old_pred[:, 1])
        same_index = {
            "max_abs": float(max(np.nanmax(direct_rr), np.nanmax(direct_ll))),
            "median_abs": float(np.nanmedian(np.concatenate([direct_rr, direct_ll]))),
        }
    n_unique_new = int(len({tuple(row.tolist()) for row in new_key}))
    aligned = bool(
        same_index is not None
        and same_index["max_abs"] <= atol
        and new_lm.shape[0] == old_lm.shape[0]
        and float(np.max(np.abs(new_lm - old_lm))) == 0.0
    )
    ok = bool(aligned or max_abs <= atol)
    return {
        "status": "pass" if ok else "explained_difference",
        "aligned_same_index": aligned,
        "n_matched": int(len(pairs)),
        "n_unique_new_keys": n_unique_new,
        "n_new": int(new_lm.shape[0]),
        "n_prior": int(old_lm.shape[0]),
        "max_abs": max_abs,
        "median_abs": median_abs,
        "same_index": same_index,
        "atol": float(atol),
        "note": (
            "Same EVLA-C physical model should match on shared rows. "
            "A difference must be versioned before new scores replace old ones."
        ),
    }


def score_extract(
    extract: Mapping[str, object],
    predictions: Mapping[str, ArrayLike],
    *,
    source_i_jy: float,
) -> dict[str, object]:
    measured = extract["measured"]
    weight = extract["weight"]
    masks = dict(extract["masks"])
    regions = {
        "all": np.ones(measured.shape[0], dtype=bool),
        "main_lobe": np.asarray(extract["regions"]["main_lobe"], dtype=bool),
        "mid": np.asarray(extract["regions"]["mid"], dtype=bool),
        "outer": np.asarray(extract["regions"]["outer_diagnostic"], dtype=bool),
    }
    out: dict[str, object] = {
        "n_development": int(extract["n_development"]),
        "n_full_raster_usable": int(extract["n_full_raster_usable"]),
        "frequency_hz": float(extract["frequency_hz"]),
        "source_i_jy": float(source_i_jy),
        "arms": {},
    }
    for arm, pred in predictions.items():
        arm_scores: dict[str, object] = {}
        for split_name, split in masks.items():
            arm_scores[split_name] = {}
            for region_name, region in regions.items():
                choose = np.asarray(split, dtype=bool) & np.asarray(region, dtype=bool)
                if not bool(np.any(choose)):
                    arm_scores[split_name][region_name] = {"n": 0}
                    continue
                arm_scores[split_name][region_name] = hand_scores(
                    measured[choose], pred[choose], weight[choose], source_i_jy=source_i_jy
                )
        out["arms"][arm] = arm_scores
    return out
