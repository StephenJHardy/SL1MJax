"""Diagonal HOLORASTER scores for the band-wide survey.

Opens a declared SPW/channel on the scientific MS. Does not apply residual
Jones and does not use the sealed SPW-5 comparison product.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_validation_statistics import classify_diagonal_region_support
from sl1mjax.cassbeam_highres import HighresCassbeamCatalog
from sl1mjax.evla_c_diagonal_survey import (
    AXIS_MAP,
    COORDINATE_QUERY,
    MAIN_LOBE_ACCEPTED_MAX,
    PRIMARY_ARM,
    RESIDUAL_JONES_POLICY,
    execution_from_measurement_set,
    refuse_frozen_write,
    refuse_residual_jones_apply,
    slot_product_stem,
    taper_in_documented_range,
    write_json_atomic,
)
from sl1mjax.evla_c_holoraster_compare import build_holdout_masks, squeeze_vis
from sl1mjax.evla_c_metrics import HANDS, hand_scores
from sl1mjax.evla_c_validation_refresh import HOLORASTER_FIELD_ID
from sl1mjax.holography import HolographyObservation
from sl1mjax.holography_alignment import apparent_voltage_response, holoraster_pair_masks
from sl1mjax.holography_beam_prior import evaluate_holoraster_cassbeam, vis_planes
from sl1mjax.holography_cassbeam_correction import (
    identity_residual_jones,
    region_masks_from_voltage,
)
from sl1mjax.holography_cassbeam_holoraster_report import locked_convention
from sl1mjax.holography_coordinate_feed_comparison import query_coordinates
from sl1mjax.holography_diagonal import copolar_hand_active_rows, source_model_stokes_i
from sl1mjax.holography_diagonal_correction import (
    holoraster_correction_holdouts,
    refuse_c147_training,
)
from sl1mjax.holography_full_jones import moving_reference_row_geometry
from sl1mjax.holography_ms import _read_antennas, _tables, audit_holography_measurement_set
from sl1mjax.polarization import Receptor, pack_coherency

SURVEY_MODEL_ID = "cassbeam_evla_c_diagonal_survey_v1"
FIXED_MAIN_LOBE_ARCMIN = 8.0
FIXED_MID_ARCMIN = 20.0
NUMERICAL_QUALIFICATION = "incomplete_no_band_wide_convergence"
PROVENANCE_KEYS = (
    "time_s",
    "scan_id",
    "antenna1",
    "antenna2",
    "field_id",
    "row_id",
)


def open_survey_catalog(root: Path) -> HighresCassbeamCatalog:
    """Opt-in survey catalog. Exact native frequency only. Not the production factory."""

    refuse_frozen_write(root)
    if RESIDUAL_JONES_POLICY != "not_applied":
        refuse_residual_jones_apply()
    return HighresCassbeamCatalog(Path(root), expected_model_id=SURVEY_MODEL_ID)


def require_survey_frequency(catalog: HighresCassbeamCatalog, frequency_hz: float) -> int:
    """Refuse nearest-plane substitution."""

    return catalog.require_exact_mhz(frequency_hz)


def load_comparison_module(scripts_dir: Path):
    path = Path(scripts_dir) / "run_thol0001_holoraster_cassbeam_comparison.py"
    spec = importlib.util.spec_from_file_location("thol0001_holoraster_cassbeam_comparison", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def empty_residual_jones(antenna_ids: ArrayLike) -> dict[int, NDArray[np.complex128]]:
    if RESIDUAL_JONES_POLICY != "not_applied":
        refuse_residual_jones_apply()
    return identity_residual_jones(antenna_ids)


def holoraster_audit_for_slot(
    cmp,
    measurement_set: Path,
    *,
    frequency_hz: float,
    spectral_window_id: int,
):
    """Pointing audit once per work MS and SPW. Visibilities are not cached."""

    cache = getattr(cmp, "_survey_audit_cache", None)
    if cache is None:
        cache = {}
        cmp._survey_audit_cache = cache
    key = (str(Path(measurement_set).resolve()), int(spectral_window_id))
    if key not in cache:
        cache[key] = audit_holography_measurement_set(
            measurement_set,
            target_frequency_hz=(float(frequency_hz),),
        )
    return cache[key]


def load_holoraster_slot(
    cmp,
    *,
    measurement_set: Path,
    spectral_window_id: int,
    channel: int,
    data_column: str = "CORRECTED_DATA",
) -> dict[str, object]:
    """Extract one native channel. Residual Jones is identity."""

    from sl1mjax.beam_operator import unique_visibility_times
    from sl1mjax.calibration_terms import parallactic_angle_rad as parallactic_angle

    tables = _tables()
    _ids, names, positions = _read_antennas(tables, measurement_set)
    names = tuple(names)
    diag = getattr(cmp, "_cached_diag", None)
    if diag is None:
        diag = cmp._diag()
        cmp._cached_diag = diag
    ddid = diag._ddid_for_spw(tables, measurement_set, int(spectral_window_id))
    with tables.table(str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False) as window:
        frequencies = np.asarray(
            window.getcell("CHAN_FREQ", int(spectral_window_id)), dtype=np.float64
        ).reshape(-1)
    frequency_hz = float(frequencies[int(channel)])
    audit = holoraster_audit_for_slot(
        cmp,
        measurement_set,
        frequency_hz=frequency_hz,
        spectral_window_id=int(spectral_window_id),
    )
    block, source = diag._holoraster_channel_block(
        tables,
        measurement_set,
        data_desc_id=ddid,
        channel=int(channel),
        channel_stop=int(channel) + 1,
        data_column=data_column,
        spectral_window_id=int(spectral_window_id),
    )
    observation = HolographyObservation(
        block=block,
        pointing=diag._subset_pointing(audit.resolved, block.time_s),
        antenna_position_m=positions,
        calibration_state="casa_parang_true",
        phase_centre_rad=block.phase_centre_rad,
        source_name="3C147",
        source_coherency_visibility=source,
        selected_spw_id=int(spectral_window_id),
    )
    unique_times, _ = unique_visibility_times(block.time_s)
    chi = parallactic_angle(unique_times, observation.phase_centre_rad, positions)
    packed = pack_coherency(block.visibility, block.correlations, (Receptor.R, Receptor.L))
    intensity = source_model_stokes_i(observation)
    rr_ok, ll_ok = copolar_hand_active_rows(observation)
    voltage = apparent_voltage_response(vis_planes(packed)[:, 0], intensity, rr_ok, ll_ok)
    refuse_c147_training(observation.block.field_id)
    if np.any(np.asarray(observation.block.field_id) != HOLORASTER_FIELD_ID):
        raise ValueError("HOLORASTER extract must stay on field 10")
    geometry = moving_reference_row_geometry(observation, require_all_channels=False)
    pair = holoraster_pair_masks(observation)
    usable = np.asarray(geometry["usable"], dtype=bool) & np.asarray(
        pair["moving_reference"], dtype=bool
    )
    holdouts = holoraster_correction_holdouts(observation, names)
    development = (
        holdouts.train
        | holdouts.spatial_holdout
        | holdouts.mover_holdout
        | holdouts.reference_holdout
    )
    keep = usable & development
    fields = cmp._row_fields(geometry, pair, chi, np.flatnonzero(keep))
    commanded = np.asarray(geometry["commanded_offset_azelgeo"], dtype=np.float64)[fields["rows"]]
    source_lm = np.asarray(geometry["source_lm_feed"], dtype=np.float64)[fields["rows"]]
    np.testing.assert_allclose(source_lm, query_coordinates(commanded, query=COORDINATE_QUERY))
    source_arr = np.asarray(observation.source_coherency_visibility, dtype=np.complex128)
    if source_arr.shape == (2, 2):
        source_rows = source_arr
    elif source_arr.ndim == 3:
        source_rows = source_arr[fields["rows"], None, :, :]
    else:
        source_rows = source_arr[fields["rows"]]
    measured = np.asarray(packed, dtype=np.complex128)
    measured = measured[fields["rows"], 0] if measured.ndim == 4 else measured[fields["rows"]]
    weight = cmp._hand_weight(observation, keep)
    if weight.ndim == 4:
        weight = weight[:, 0]
    return {
        "observation": observation,
        "names": names,
        "positions": positions,
        "fields": fields,
        "commanded": commanded,
        "source_lm": source_lm,
        "source": source_rows,
        "measured": measured,
        "weight": weight,
        "regions": region_masks_from_voltage(np.asarray(voltage, dtype=np.float64)[fields["rows"]]),
        "residual": empty_residual_jones(np.arange(len(names))),
        "keep": keep,
        "masks": build_holdout_masks(observation, names, keep, holdouts=holdouts),
        "frequency_hz": frequency_hz,
        "spectral_window_id": int(spectral_window_id),
        "channel": int(channel),
        "execution": execution_from_measurement_set(measurement_set),
        "data_column": data_column,
        "intensity": intensity,
        "n_development": int(np.sum(keep)),
        "time_s": np.asarray(observation.block.time_s, dtype=np.float64)[fields["rows"]],
        "scan_id": np.asarray(observation.block.scan_id, dtype=np.int32)[fields["rows"]],
        "antenna1": np.asarray(observation.block.antenna1, dtype=np.int32)[fields["rows"]],
        "antenna2": np.asarray(observation.block.antenna2, dtype=np.int32)[fields["rows"]],
        "field_id": np.asarray(observation.block.field_id, dtype=np.int32)[fields["rows"]],
        "row_id": np.asarray(fields["rows"], dtype=np.int64),
    }


def predict_diagonal(
    *,
    catalog: HighresCassbeamCatalog,
    extract: Mapping[str, object],
) -> NDArray[np.complex128]:
    fields = extract["fields"]
    return squeeze_vis(
        evaluate_holoraster_cassbeam(
            catalog=catalog,
            frequencies_hz=np.asarray([extract["frequency_hz"]], dtype=np.float64),
            convention=locked_convention(),
            offset_lm_rad=extract["source_lm"],
            chi_moving=fields["chi_m"],
            chi_reference=fields["chi_r"],
            moving_id=fields["moving"],
            reference_id=fields["reference"],
            moving_is_p=fields["moving_is_p"],
            residual_jones=extract["residual"],
            source=extract["source"],
            off_diagonal=False,
        ).visibility
    )


def fixed_radius_region_masks(offset_lm: ArrayLike) -> dict[str, NDArray[np.bool_]]:
    """Geometry-only regions. Independent of measured amplitude and flags."""

    offset = np.asarray(offset_lm, dtype=np.float64)
    radius = np.hypot(offset[:, 0], offset[:, 1]) * (180.0 / np.pi) * 60.0
    finite = np.isfinite(radius)
    return {
        "main_lobe": finite & (radius < FIXED_MAIN_LOBE_ARCMIN),
        "mid": finite & (radius >= FIXED_MAIN_LOBE_ARCMIN) & (radius < FIXED_MID_ARCMIN),
        "outer_diagnostic": finite & (radius >= FIXED_MID_ARCMIN),
    }


def score_fixed_geometry(
    *,
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    offset_lm: ArrayLike,
    source_i_jy: float,
) -> dict[str, object]:
    """Rescore an existing export on the shared radius mask. Does not drop outliers."""

    regions = fixed_radius_region_masks(offset_lm)
    region_scores = {
        name: _masked_scores(measured, predicted, weight, mask, source_i_jy)
        for name, mask in regions.items()
    }
    rr = float(region_scores["main_lobe"].get("RR", {}).get("residual_power", float("nan")))
    ll = float(region_scores["main_lobe"].get("LL", {}).get("residual_power", float("nan")))
    both = bool(
        np.isfinite(rr) and np.isfinite(ll) and rr <= MAIN_LOBE_ACCEPTED_MAX and ll <= MAIN_LOBE_ACCEPTED_MAX
    )
    return {
        "mask": "fixed_radius_arcmin",
        "main_lobe_arcmin": FIXED_MAIN_LOBE_ARCMIN,
        "mid_arcmin": FIXED_MID_ARCMIN,
        "regions": region_scores,
        "main_lobe_both_hands_accepted": both,
        "empirical_main_lobe_accepted": both,
        "numerically_qualified": False,
        "n_main_lobe": int(np.sum(regions["main_lobe"])),
        "n_mid": int(np.sum(regions["mid"])),
        "n_outer": int(np.sum(regions["outer_diagnostic"])),
    }


def outlier_influence_by_group(
    *,
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    mask: ArrayLike,
    group: ArrayLike,
    hand: str = "RR",
) -> list[dict[str, object]]:
    """Leave-one-group-out residual-power change. Does not drop groups."""

    from sl1mjax.evla_c_metrics import residual_power

    keep = np.asarray(mask, dtype=bool).reshape(-1)
    labels = np.asarray(group).reshape(-1)
    if labels.size != keep.size:
        raise ValueError("group length must match samples")
    vis = np.asarray(measured)
    pred = np.asarray(predicted)
    wt = np.asarray(weight)
    baseline = residual_power(vis[keep], pred[keep], wt[keep], hand)
    rows: list[dict[str, object]] = []
    for label in sorted({item for item in labels[keep].tolist()}, key=str):
        leave = keep & (labels != label)
        if not bool(np.any(leave)):
            continue
        score = residual_power(vis[leave], pred[leave], wt[leave], hand)
        rows.append(
            {
                "group": label if isinstance(label, (int, float, str)) else str(label),
                "n": int(np.sum(keep & (labels == label))),
                "residual_power_without_group": float(score),
                "delta_from_baseline": float(score - baseline),
            }
        )
    rows.sort(key=lambda row: abs(float(row["delta_from_baseline"])), reverse=True)
    return rows


def _masked_scores(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    mask: ArrayLike,
    source_i_jy: float,
) -> dict[str, dict[str, float]]:
    keep = np.asarray(mask, dtype=bool).reshape(-1)
    if keep.size != np.asarray(measured).shape[0]:
        raise ValueError("mask length must match samples")
    if not bool(np.any(keep)):
        return {hand: {"n": 0, "residual_power": float("nan")} for hand in HANDS}
    return hand_scores(measured[keep], predicted[keep], weight[keep], source_i_jy=source_i_jy)


def score_extract(
    extract: Mapping[str, object],
    predicted: ArrayLike,
) -> dict[str, object]:
    measured = np.asarray(extract["measured"])
    predicted = squeeze_vis(predicted)
    weight = np.asarray(extract["weight"])
    intensity = np.asarray(extract["intensity"], dtype=np.float64).reshape(-1)
    finite = intensity[np.isfinite(intensity)]
    source_i = float(np.nanmedian(finite)) if finite.size else float("nan")
    regions = extract["regions"]
    region_scores = {
        name: _masked_scores(measured, predicted, weight, regions[name], source_i)
        for name in ("main_lobe", "mid", "outer_diagnostic")
        if name in regions
    }
    classifier_payload = {
        name: {
            "rr": {
                "residual_power": float(
                    region_scores[name].get("RR", {}).get("residual_power", float("nan"))
                )
            },
            "ll": {
                "residual_power": float(
                    region_scores[name].get("LL", {}).get("residual_power", float("nan"))
                )
            },
        }
        for name in region_scores
    }
    support = classify_diagonal_region_support(classifier_payload)
    rr = float(region_scores.get("main_lobe", {}).get("RR", {}).get("residual_power", float("nan")))
    ll = float(region_scores.get("main_lobe", {}).get("LL", {}).get("residual_power", float("nan")))
    both = bool(
        np.isfinite(rr)
        and np.isfinite(ll)
        and rr <= MAIN_LOBE_ACCEPTED_MAX
        and ll <= MAIN_LOBE_ACCEPTED_MAX
    )
    status = "scientifically-qualified" if both else "complete"
    if not (np.isfinite(rr) and np.isfinite(ll)):
        status = "unsupported"
    fixed = score_fixed_geometry(
        measured=measured,
        predicted=predicted,
        weight=weight,
        offset_lm=extract["source_lm"],
        source_i_jy=source_i,
    )
    return {
        "spectral_window_id": extract["spectral_window_id"],
        "channel": extract["channel"],
        "execution": extract.get("execution") or "lower_c",
        "frequency_hz": extract["frequency_hz"],
        "coordinate_query": COORDINATE_QUERY,
        "axis_map": AXIS_MAP,
        "primary_arm": PRIMARY_ARM,
        "residual_jones_policy": RESIDUAL_JONES_POLICY,
        "data_column": extract["data_column"],
        "n_development": extract["n_development"],
        "source_i_jy": source_i,
        "taper_in_documented_range": taper_in_documented_range(float(extract["frequency_hz"])),
        "all_data": hand_scores(measured, predicted, weight, source_i_jy=source_i),
        "regions": region_scores,
        "support": support,
        "main_lobe_both_hands_accepted": both,
        "empirical_main_lobe_accepted": both,
        "numerically_qualified": False,
        "numerical_qualification": NUMERICAL_QUALIFICATION,
        "fixed_geometry": fixed,
        "status": status,
    }


def write_slot_products(
    output_dir: Path,
    *,
    extract: Mapping[str, object],
    predicted: ArrayLike,
    report: Mapping[str, object],
) -> dict[str, str]:
    refuse_frozen_write(output_dir)
    stem = slot_product_stem(
        output_dir,
        execution=str(report.get("execution") or extract.get("execution") or "lower_c"),
        spectral_window_id=int(extract["spectral_window_id"]),
        channel=int(extract["channel"]),
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    arrays = stem.with_name(stem.name + "_export.npz")
    payload = {
        "measured": np.asarray(extract["measured"]),
        "predicted": squeeze_vis(predicted),
        "weight": np.asarray(extract["weight"]),
        "source_lm_feed": np.asarray(extract["source_lm"]),
        "commanded_offset_azelgeo": np.asarray(extract["commanded"]),
        "frequency_hz": np.asarray(extract["frequency_hz"]),
        "main_lobe": np.asarray(extract["regions"]["main_lobe"]),
        "mid": np.asarray(extract["regions"]["mid"]),
        "outer_diagnostic": np.asarray(extract["regions"]["outer_diagnostic"]),
        "fixed_main_lobe": fixed_radius_region_masks(extract["source_lm"])["main_lobe"],
        "fixed_mid": fixed_radius_region_masks(extract["source_lm"])["mid"],
        "fixed_outer": fixed_radius_region_masks(extract["source_lm"])["outer_diagnostic"],
    }
    for key in PROVENANCE_KEYS:
        if key in extract and extract[key] is not None:
            payload[key] = np.asarray(extract[key])
    masks = extract.get("masks") or {}
    for key in ("train", "spatial_holdout", "mover_holdout", "reference_holdout"):
        if key in masks:
            payload[f"mask_{key}"] = np.asarray(masks[key], dtype=bool)
    np.savez_compressed(arrays, **payload)
    report_path = write_json_atomic(stem.with_name(stem.name + "_report.json"), report)
    return {"export": str(arrays), "report": str(report_path)}
