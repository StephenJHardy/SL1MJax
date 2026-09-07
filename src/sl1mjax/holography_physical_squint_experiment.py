"""Direct squint measurement and predeclared SPW-4 candidate models.

Development evidence only. Native CASSBEAM squint stays the physics prior
until a gate says otherwise. SPW 5 stays sealed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import evla195_total_squint_rad
from sl1mjax.holography_beam_prior import sky_frame_residual_numpy
from sl1mjax.holography_cassbeam_correction import ARCMIN_TO_RAD, FeedFrameLookup
from sl1mjax.holography_cassbeam_correction_audit import leave_one_mover_sensitivity
from sl1mjax.holography_cassbeam_holoraster_report import (
    mainlobe_power_mask,
    squint_from_voltage_maps,
)
from sl1mjax.holography_diagonal_correction import (
    OFFSET_CELL_QUANT_PER_RAD,
    complex_visibility_loss,
    mover_cluster_ids,
    packed_offset_keys,
    quantized_offset_keys,
    refuse_magnitude_loss,
    refuse_spw5,
    score_paired_holdout,
    spatial_cluster_ids,
)
from sl1mjax.holography_full_jones import _antenna_jones_planes
from sl1mjax.holography_physical_squint import (
    IDENTITY_PHYSICAL,
    PhysicalBeamState,
    native_separation_rad,
    physical_path_stages,
)
from sl1mjax.polarization import circular_parallactic_jones, invert_jones

WIDTH_GRID = np.linspace(0.90, 1.12, 23)
PUBLISHED_HOLOGRAPHY_SQUINT_ARCMIN = 0.518
CANDIDATE_NAMES = (
    "no_squint",
    "native",
    "native_plus_width",
    "empirical",
    "empirical_plus_width",
)


def invert_moving_sky_jones(
    visibilities: ArrayLike,
    source: ArrayLike,
    residual_moving: ArrayLike,
    residual_reference: ArrayLike,
    moving_is_p: ArrayLike,
) -> NDArray[np.complex128]:
    """Undo :math:`V=R_m E_m S R_r^H` for the moving beam. Vectorized."""

    vis = np.asarray(visibilities, dtype=np.complex128)
    if vis.ndim == 4:
        vis = vis[:, 0]
    sky = np.asarray(source, dtype=np.complex128)
    if sky.ndim == 2:
        sky = np.broadcast_to(sky, vis.shape)
    elif sky.ndim == 4:
        sky = sky[:, 0]
    r_m = np.asarray(residual_moving, dtype=np.complex128)
    r_r = np.asarray(residual_reference, dtype=np.complex128)
    mover_p = np.asarray(moving_is_p, dtype=bool).reshape(-1)
    inv_m = invert_jones(r_m)
    inv_r = invert_jones(r_r)
    inv_s = invert_jones(sky)
    inv_r_h = np.conjugate(np.swapaxes(inv_r, -1, -2))
    inv_m_h = np.conjugate(np.swapaxes(inv_m, -1, -2))
    e_p = inv_m @ vis @ inv_r_h @ inv_s
    e_h = inv_s @ inv_r @ vis @ inv_m_h
    e_q = np.conjugate(np.swapaxes(e_h, -1, -2))
    return np.where(mover_p[:, None, None], e_p, e_q)


def sky_jones_to_feed_frame(
    sky_jones: ArrayLike,
    chi_moving: ArrayLike,
) -> NDArray[np.complex128]:
    """Undo CASA ``parang=True`` parallactic rotation."""

    sky = np.asarray(sky_jones, dtype=np.complex128)
    para = circular_parallactic_jones(np.asarray(chi_moving, dtype=np.float64).reshape(-1))
    conjugate = np.conjugate(np.swapaxes(para, -1, -2))
    return para @ sky @ conjugate


def recovered_feed_jones(
    visibilities: ArrayLike,
    source: ArrayLike,
    residual_jones: Mapping[int, ArrayLike],
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    moving_is_p: ArrayLike,
    chi_moving: ArrayLike,
    chi_reference: ArrayLike,
) -> NDArray[np.complex128]:
    r_m = sky_frame_residual_numpy(_antenna_jones_planes(residual_jones, moving_id), chi_moving)
    r_r = sky_frame_residual_numpy(
        _antenna_jones_planes(residual_jones, reference_id), chi_reference
    )
    sky = invert_moving_sky_jones(visibilities, source, r_m, r_r, moving_is_p)
    return sky_jones_to_feed_frame(sky, chi_moving)


def _hand_ok(weight: ArrayLike, flag: ArrayLike | None, hand: str) -> NDArray[np.bool_]:
    wgt = np.asarray(weight, dtype=np.float64)
    if wgt.ndim == 4:
        wgt = wgt[:, 0]
    row, col = {"rr": (0, 0), "ll": (1, 1)}[hand]
    ok = np.isfinite(wgt[:, row, col]) & (wgt[:, row, col] > 0.0)
    if flag is not None:
        ok = ok & np.asarray(flag, dtype=bool).reshape(-1)
    return ok


def cell_hand_maps(
    offset_lm_rad: ArrayLike,
    power: ArrayLike,
    weight: ArrayLike,
    mask: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Equal-cell map: visibility weights only combine repeats inside a cell."""

    keys = quantized_offset_keys(offset_lm_rad)
    choose = np.asarray(mask, dtype=bool).reshape(-1)
    if not bool(np.any(choose)):
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, np.zeros(0), np.zeros(0)
    selected = keys[choose]
    unique, inverse = np.unique(selected, axis=0, return_inverse=True)
    ww = np.asarray(weight, dtype=np.float64).reshape(-1)[choose]
    pp = np.asarray(power, dtype=np.float64).reshape(-1)[choose]
    numer = np.bincount(inverse, weights=ww * pp)
    denom = np.bincount(inverse, weights=ww)
    mean = np.divide(numer, denom, out=np.full(unique.shape[0], np.nan), where=denom > 0.0)
    centres = unique.astype(np.float64) / OFFSET_CELL_QUANT_PER_RAD
    return centres, mean, denom


def common_support_maps(
    offset_lm_rad: ArrayLike,
    rr_power: ArrayLike,
    ll_power: ArrayLike,
    rr_weight: ArrayLike,
    ll_weight: ArrayLike,
    rr_ok: ArrayLike,
    ll_ok: ArrayLike,
) -> dict[str, NDArray]:
    rr_c, rr_p, _rr_w = cell_hand_maps(offset_lm_rad, rr_power, rr_weight, rr_ok)
    ll_c, ll_p, _ll_w = cell_hand_maps(offset_lm_rad, ll_power, ll_weight, ll_ok)
    rr_pack = packed_offset_keys(rr_c)
    ll_pack = packed_offset_keys(ll_c)
    _common, rr_idx, ll_idx = np.intersect1d(rr_pack, ll_pack, return_indices=True)
    centres = rr_c[rr_idx]
    return {
        "offset": centres,
        "rr_power": rr_p[rr_idx],
        "ll_power": ll_p[ll_idx],
        "weight": np.ones(centres.shape[0], dtype=np.float64),
        "n_cells": int(centres.shape[0]),
    }


def _normalize_origin_power(maps: Mapping[str, NDArray]) -> dict[str, NDArray]:
    offset = np.asarray(maps["offset"], dtype=np.float64)
    radius = np.hypot(offset[:, 0], offset[:, 1])
    if radius.size == 0:
        return dict(maps)
    origin = int(np.argmin(radius))
    rr0 = max(float(maps["rr_power"][origin]), 1.0e-12)
    ll0 = max(float(maps["ll_power"][origin]), 1.0e-12)
    rr = np.asarray(maps["rr_power"], dtype=np.float64) / rr0
    ll = np.asarray(maps["ll_power"], dtype=np.float64) / ll0
    out = dict(maps)
    out["rr_power"] = rr
    out["ll_power"] = ll
    return out


def squint_from_common_map(
    maps: Mapping[str, NDArray],
    *,
    series: str,
    frequency_hz: float | None = None,
    common_mask: bool = False,
) -> dict[str, object]:
    offset = maps["offset"]
    rr = maps["rr_power"]
    ll = maps["ll_power"]
    weight = maps["weight"]
    if common_mask:
        mask = mainlobe_power_mask(0.5 * (rr + ll), weight)
        rr_w = np.where(mask, weight, 0.0)
        ll_w = rr_w
    else:
        rr_w = weight
        ll_w = weight
    return squint_from_voltage_maps(
        offset,
        rr,
        ll,
        rr_w,
        ll_w,
        frequency_hz=frequency_hz,
        series=series,
    )


def _unit_bootstrap(
    labels: ArrayLike,
    offset: ArrayLike,
    rr_power: ArrayLike,
    ll_power: ArrayLike,
    rr_ok: ArrayLike,
    ll_ok: ArrayLike,
    rr_weight: ArrayLike,
    ll_weight: ArrayLike,
    *,
    n_boot: int,
    seed: int,
    frequency_hz: float,
) -> dict[str, object]:
    ids = np.asarray(labels)
    unique = np.unique(ids)
    rng = np.random.default_rng(int(seed))
    draws = np.zeros((int(n_boot), 2), dtype=np.float64)
    for boot in range(int(n_boot)):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        keep = np.isin(ids, chosen)
        maps = common_support_maps(
            np.asarray(offset)[keep],
            np.asarray(rr_power)[keep],
            np.asarray(ll_power)[keep],
            np.asarray(rr_weight)[keep],
            np.asarray(ll_weight)[keep],
            np.asarray(rr_ok)[keep],
            np.asarray(ll_ok)[keep],
        )
        maps = _normalize_origin_power(maps)
        record = squint_from_common_map(maps, series="bootstrap", frequency_hz=frequency_hz)
        draws[boot, 0] = float(record["rr_l_arcmin"]) - float(record["ll_l_arcmin"])
        draws[boot, 1] = float(record["rr_m_arcmin"]) - float(record["ll_m_arcmin"])
    finite = np.isfinite(draws).all(axis=1)
    sample = draws[finite]
    mean = np.mean(sample, axis=0) if sample.size else np.full(2, np.nan)
    mag = np.hypot(sample[:, 0], sample[:, 1]) if sample.size else np.array([])
    return {
        "mean_dl_arcmin": float(mean[0]),
        "mean_dm_arcmin": float(mean[1]),
        "mean_magnitude_arcmin": float(np.hypot(*mean)),
        "magnitude_lo": float(np.quantile(mag, 0.025)) if mag.size else float("nan"),
        "magnitude_hi": float(np.quantile(mag, 0.975)) if mag.size else float("nan"),
        "n_boot": int(sample.shape[0]),
        "n_units": int(unique.size),
    }


def group_squint_vectors(
    labels: ArrayLike,
    names: Sequence[str],
    offset: ArrayLike,
    rr_power: ArrayLike,
    ll_power: ArrayLike,
    rr_ok: ArrayLike,
    ll_ok: ArrayLike,
    rr_weight: ArrayLike,
    ll_weight: ArrayLike,
    *,
    frequency_hz: float,
) -> list[dict[str, object]]:
    ids = np.asarray(labels)
    lookup = {index: str(name) for index, name in enumerate(names)}
    rows: list[dict[str, object]] = []
    for antenna in np.unique(ids):
        keep = ids == int(antenna)
        maps = common_support_maps(
            np.asarray(offset)[keep],
            np.asarray(rr_power)[keep],
            np.asarray(ll_power)[keep],
            np.asarray(rr_weight)[keep],
            np.asarray(ll_weight)[keep],
            np.asarray(rr_ok)[keep],
            np.asarray(ll_ok)[keep],
        )
        if int(maps["n_cells"]) < 8:
            continue
        maps = _normalize_origin_power(maps)
        record = squint_from_common_map(
            maps, series=lookup.get(int(antenna), str(antenna)), frequency_hz=frequency_hz
        )
        rows.append(
            {
                "name": lookup.get(int(antenna), str(antenna)),
                "id": int(antenna),
                "n_cells": int(maps["n_cells"]),
                "dl_arcmin": float(record["rr_l_arcmin"]) - float(record["ll_l_arcmin"]),
                "dm_arcmin": float(record["rr_m_arcmin"]) - float(record["ll_m_arcmin"]),
                "separation_arcmin": float(record["separation_arcmin"]),
                "position_angle_rad": float(
                    np.arctan2(
                        float(record["rr_m_arcmin"]) - float(record["ll_m_arcmin"]),
                        float(record["rr_l_arcmin"]) - float(record["ll_l_arcmin"]),
                    )
                ),
            }
        )
    return rows


def measure_direct_squint(
    *,
    offset_lm_rad: ArrayLike,
    measured: ArrayLike,
    source: ArrayLike,
    residual_jones: Mapping[int, ArrayLike],
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    moving_is_p: ArrayLike,
    chi_moving: ArrayLike,
    chi_reference: ArrayLike,
    weight: ArrayLike,
    rr_ok: ArrayLike,
    ll_ok: ArrayLike,
    train: ArrayLike,
    antenna_names: Sequence[str],
    frequency_hz: float,
    visit_id: ArrayLike | None = None,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, object]:
    """Map-domain 20%-of-peak squint on training cells with common R/L support."""

    refuse_spw5(opened=False)
    choose = np.asarray(train, dtype=bool).reshape(-1)
    feed = recovered_feed_jones(
        np.asarray(measured)[choose],
        source if np.asarray(source).shape == (2, 2) else np.asarray(source)[choose],
        residual_jones,
        np.asarray(moving_id)[choose],
        np.asarray(reference_id)[choose],
        np.asarray(moving_is_p)[choose],
        np.asarray(chi_moving)[choose],
        np.asarray(chi_reference)[choose],
    )
    offset = np.asarray(offset_lm_rad, dtype=np.float64)[choose]
    wgt = np.asarray(weight, dtype=np.float64)
    if wgt.ndim == 4:
        wgt = wgt[:, 0]
    wgt = wgt[choose]
    rr_flag = _hand_ok(wgt, np.asarray(rr_ok)[choose], "rr")
    ll_flag = _hand_ok(wgt, np.asarray(ll_ok)[choose], "ll")
    rr_power = np.abs(feed[:, 0, 0]) ** 2
    ll_power = np.abs(feed[:, 1, 1]) ** 2
    maps = _normalize_origin_power(
        common_support_maps(
            offset, rr_power, ll_power, wgt[:, 0, 0], wgt[:, 1, 1], rr_flag, ll_flag
        )
    )
    independent = squint_from_common_map(
        maps, series="measured_train_independent_masks", frequency_hz=frequency_hz
    )
    common = squint_from_common_map(
        maps,
        series="measured_train_common_mask",
        frequency_hz=frequency_hz,
        common_mask=True,
    )
    delta = np.array(
        [
            float(independent["rr_l_arcmin"]) - float(independent["ll_l_arcmin"]),
            float(independent["rr_m_arcmin"]) - float(independent["ll_m_arcmin"]),
        ],
        dtype=np.float64,
    )
    native = native_separation_rad() / ARCMIN_TO_RAD
    memo = float(np.asarray(evla195_total_squint_rad(frequency_hz)).reshape(-1)[0] / ARCMIN_TO_RAD)
    movers = group_squint_vectors(
        np.asarray(moving_id)[choose],
        antenna_names,
        offset,
        rr_power,
        ll_power,
        rr_flag,
        ll_flag,
        wgt[:, 0, 0],
        wgt[:, 1, 1],
        frequency_hz=frequency_hz,
    )
    refs = group_squint_vectors(
        np.asarray(reference_id)[choose],
        antenna_names,
        offset,
        rr_power,
        ll_power,
        rr_flag,
        ll_flag,
        wgt[:, 0, 0],
        wgt[:, 1, 1],
        frequency_hz=frequency_hz,
    )
    visits: list[dict[str, object]] = []
    if visit_id is not None:
        visit = np.asarray(visit_id)[choose]
        if visit.size:
            names = [f"visit_{int(item)}" for item in range(int(np.max(visit)) + 1)]
            visits = group_squint_vectors(
                visit,
                names,
                offset,
                rr_power,
                ll_power,
                rr_flag,
                ll_flag,
                wgt[:, 0, 0],
                wgt[:, 1, 1],
                frequency_hz=frequency_hz,
            )
    mover_boot = _unit_bootstrap(
        np.asarray(moving_id)[choose],
        offset,
        rr_power,
        ll_power,
        rr_flag,
        ll_flag,
        wgt[:, 0, 0],
        wgt[:, 1, 1],
        n_boot=n_boot,
        seed=seed,
        frequency_hz=frequency_hz,
    )
    ref_boot = _unit_bootstrap(
        np.asarray(reference_id)[choose],
        offset,
        rr_power,
        ll_power,
        rr_flag,
        ll_flag,
        wgt[:, 0, 0],
        wgt[:, 1, 1],
        n_boot=n_boot,
        seed=seed + 1,
        frequency_hz=frequency_hz,
    )
    return {
        "domain": "map",
        "estimator": "mainlobe_20pct_peak",
        "weighting": "equal_spatial_cell; visibility weights only inside cells",
        "n_train_rows": int(np.sum(choose)),
        "n_common_cells": int(maps["n_cells"]),
        "independent_masks": independent,
        "common_mask": common,
        "delta_lm_arcmin": [float(delta[0]), float(delta[1])],
        "delta_lm_rad": [float(delta[0] * ARCMIN_TO_RAD), float(delta[1] * ARCMIN_TO_RAD)],
        "magnitude_arcmin": float(np.hypot(*delta)),
        "position_angle_rad": float(np.arctan2(delta[1], delta[0])),
        "comparisons": {
            "zero": 0.0,
            "native_cassbeam_arcmin": float(np.hypot(*native)),
            "memo195_arcmin": memo,
            "published_holography_arcmin": PUBLISHED_HOLOGRAPHY_SQUINT_ARCMIN,
        },
        "by_mover": movers,
        "by_reference": refs,
        "by_visit": visits,
        "mover_bootstrap": mover_boot,
        "reference_bootstrap": ref_boot,
        "maps": maps,
    }


def _predict_rows(samples, lookup: FeedFrameLookup, state: PhysicalBeamState, mask: ArrayLike):
    choose = np.asarray(mask, dtype=bool).reshape(-1)
    source = samples.source
    if np.asarray(source).ndim != 2:
        source = np.asarray(source)[choose]
    return physical_path_stages(
        samples.offset_lm_rad[choose],
        lookup,
        state,
        residual_jones=samples.residual_jones,
        moving_id=samples.moving_id[choose],
        reference_id=samples.reference_id[choose],
        moving_is_p=samples.moving_is_p[choose],
        chi_moving=samples.parallactic_angle_rad[choose],
        chi_reference=samples.reference_parallactic_angle_rad[choose],
        source=source,
    )["visibility"]


def native_direction_unit() -> NDArray[np.float64]:
    native = native_separation_rad()
    scale = float(np.hypot(*native))
    if scale <= 0.0:
        raise ValueError("native CASSBEAM separation is zero")
    return native / scale


def delta_from_parallel_perp(*, parallel_scale: float, perp_arcmin: float) -> NDArray[np.float64]:
    """``parallel_scale=1``, ``perp=0`` is native CASSBEAM separation."""

    parallel = native_direction_unit()
    perp = np.array([-parallel[1], parallel[0]], dtype=np.float64)
    return (float(parallel_scale) * native_separation_rad()) + (
        float(perp_arcmin) * ARCMIN_TO_RAD * perp
    )


def mover_direction_stable(
    records: Sequence[Mapping[str, object]],
    *,
    max_circ_std_rad: float = 1.05,
) -> bool:
    if len(records) < 3:
        return False
    angles = np.array([float(item["position_angle_rad"]) for item in records], dtype=np.float64)
    mean = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    wrap = np.angle(np.exp(1j * (angles - mean)))
    return bool(np.all(np.isfinite(wrap)) and float(np.std(wrap)) <= float(max_circ_std_rad))


def candidate_state(
    name: str, *, empirical_delta_rad: ArrayLike, width: float
) -> PhysicalBeamState:
    if name == "no_squint":
        return PhysicalBeamState(width=1.0, delta_lm_rad=(0.0, 0.0))
    if name == "native":
        return IDENTITY_PHYSICAL
    if name == "native_plus_width":
        return PhysicalBeamState(width=float(width))
    if name == "empirical":
        delta = np.asarray(empirical_delta_rad, dtype=np.float64).reshape(2)
        return PhysicalBeamState(delta_lm_rad=(float(delta[0]), float(delta[1])))
    if name == "empirical_plus_width":
        delta = np.asarray(empirical_delta_rad, dtype=np.float64).reshape(2)
        return PhysicalBeamState(
            width=float(width),
            delta_lm_rad=(float(delta[0]), float(delta[1])),
        )
    raise ValueError(f"unknown candidate {name!r}")


def fit_width_on_train(
    samples,
    lookup: FeedFrameLookup,
    *,
    delta_lm_rad: ArrayLike,
    grid: ArrayLike = WIDTH_GRID,
) -> dict[str, object]:
    """Training-row width only. Pointing stays zero."""

    refuse_spw5(opened=False)
    refuse_magnitude_loss("complex")
    widths = np.asarray(grid, dtype=np.float64).reshape(-1)
    losses = np.empty(widths.size, dtype=np.float64)
    delta = np.asarray(delta_lm_rad, dtype=np.float64).reshape(2)
    train = samples.train
    for index, width in enumerate(widths):
        state = PhysicalBeamState(
            width=float(width),
            delta_lm_rad=(float(delta[0]), float(delta[1])),
        )
        vis = _predict_rows(samples, lookup, state, train)
        losses[index] = complex_visibility_loss(samples.measured[train], vis, samples.weight[train])
    best = int(np.nanargmin(losses))
    return {
        "grid": [float(item) for item in widths],
        "loss": [float(item) for item in losses],
        "width": float(widths[best]),
        "interior": bool(best not in {0, widths.size - 1}),
        "min_loss": float(losses[best]),
    }


def rr_ll_difference_loss(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    rr_ok: ArrayLike,
    ll_ok: ArrayLike,
) -> float:
    meas = np.asarray(measured)
    pred = np.asarray(predicted)
    wgt = np.asarray(weight, dtype=np.float64)
    if meas.ndim == 4:
        meas = meas[:, 0]
        pred = pred[:, 0]
        wgt = wgt[:, 0]
    ok = (
        np.asarray(rr_ok, dtype=bool).reshape(-1)
        & np.asarray(ll_ok, dtype=bool).reshape(-1)
        & np.isfinite(wgt[:, 0, 0])
        & np.isfinite(wgt[:, 1, 1])
    )
    ww = 0.5 * (wgt[:, 0, 0] + wgt[:, 1, 1])
    obs = meas[:, 0, 0] - meas[:, 1, 1]
    hat = pred[:, 0, 0] - pred[:, 1, 1]
    denom = float(np.sum(ww[ok] * np.abs(obs[ok]) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(ww[ok] * np.abs(obs[ok] - hat[ok]) ** 2) / denom)


def hand_loss(measured, predicted, weight, hand: str) -> float:
    meas = np.asarray(measured)
    pred = np.asarray(predicted)
    wgt = np.asarray(weight, dtype=np.float64)
    if meas.ndim == 4:
        meas = meas[:, 0]
        pred = pred[:, 0]
        wgt = wgt[:, 0]
    row, col = {"rr": (0, 0), "ll": (1, 1)}[hand]
    ok = np.isfinite(meas[:, row, col]) & np.isfinite(pred[:, row, col]) & (wgt[:, row, col] > 0.0)
    denom = float(np.sum(wgt[ok, row, col] * np.abs(meas[ok, row, col]) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(
        np.sum(wgt[ok, row, col] * np.abs(meas[ok, row, col] - pred[ok, row, col]) ** 2) / denom
    )


def score_model(
    samples,
    predicted: ArrayLike,
    baseline: ArrayLike,
    *,
    rr_ok: ArrayLike,
    ll_ok: ArrayLike,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, object]:
    pred = np.asarray(predicted)
    base = np.asarray(baseline)
    splits = {
        "train": samples.train,
        "spatial_holdout": samples.spatial_holdout,
        "mover_holdout": samples.mover_holdout,
        "reference_holdout": getattr(
            samples, "reference_holdout", np.zeros(samples.train.size, dtype=bool)
        ),
    }
    regions = {
        "all": np.ones(samples.train.size, dtype=bool),
        "main_lobe": samples.main_lobe,
        "mid": samples.mid,
        "outer": samples.outer,
    }
    losses: dict[str, object] = {}
    for split_name, split in splits.items():
        losses[split_name] = {}
        for region_name, region in regions.items():
            choose = np.asarray(split, dtype=bool) & np.asarray(region, dtype=bool)
            if not bool(np.any(choose)):
                losses[split_name][region_name] = {"n": 0, "combined": float("nan")}
                continue
            losses[split_name][region_name] = {
                "n": int(np.sum(choose)),
                "combined": complex_visibility_loss(
                    samples.measured[choose], pred[choose], samples.weight[choose]
                ),
                "rr": hand_loss(
                    samples.measured[choose], pred[choose], samples.weight[choose], "rr"
                ),
                "ll": hand_loss(
                    samples.measured[choose], pred[choose], samples.weight[choose], "ll"
                ),
                "rr_minus_ll": rr_ll_difference_loss(
                    samples.measured[choose],
                    pred[choose],
                    samples.weight[choose],
                    np.asarray(rr_ok)[choose],
                    np.asarray(ll_ok)[choose],
                ),
            }
    spatial = score_paired_holdout(
        samples.measured[samples.spatial_holdout],
        pred[samples.spatial_holdout],
        base[samples.spatial_holdout],
        samples.weight[samples.spatial_holdout],
        spatial_cluster_ids(samples.offset_lm_rad, samples.spatial_holdout),
        axis="spatial",
        cluster_kind="spatial_cell",
        n_boot=n_boot,
        seed=seed,
    )
    moving = score_paired_holdout(
        samples.measured[samples.mover_holdout],
        pred[samples.mover_holdout],
        base[samples.mover_holdout],
        samples.weight[samples.mover_holdout],
        mover_cluster_ids(samples.moving_id, samples.mover_holdout),
        axis="moving",
        cluster_kind="moving_antenna",
        n_boot=n_boot,
        seed=seed,
    )
    movers = leave_one_mover_sensitivity(samples, pred, base, n_boot=n_boot, seed=seed)
    mover = samples.mover_holdout
    return {
        "losses": losses,
        "baseline_rr_minus_ll": rr_ll_difference_loss(
            samples.measured[mover],
            base[mover],
            samples.weight[mover],
            np.asarray(rr_ok)[mover],
            np.asarray(ll_ok)[mover],
        ),
        "candidate_rr_minus_ll": rr_ll_difference_loss(
            samples.measured[mover],
            pred[mover],
            samples.weight[mover],
            np.asarray(rr_ok)[mover],
            np.asarray(ll_ok)[mover],
        ),
        "spatial": {
            "delta": float(spatial.delta),
            "delta_lo": float(spatial.delta_lo),
            "delta_hi": float(spatial.delta_hi),
            "improves": bool(spatial.improves()),
        },
        "moving": {
            "delta": float(moving.delta),
            "delta_lo": float(moving.delta_lo),
            "delta_hi": float(moving.delta_hi),
            "improves": bool(moving.improves()),
        },
        "movers": movers,
    }


def joint_width_squint_surface(
    samples,
    lookup: FeedFrameLookup,
    *,
    width_grid: ArrayLike,
    magnitude_scale_grid: ArrayLike,
) -> dict[str, object]:
    """Train-row complex loss versus common width and physical squint magnitude.

    Squint direction stays on the native CASSBEAM axis. This is an identifiability
    diagnostic, not a model-selection surface.
    """

    refuse_spw5(opened=False)
    refuse_magnitude_loss("complex")
    widths = np.asarray(width_grid, dtype=np.float64).reshape(-1)
    scales = np.asarray(magnitude_scale_grid, dtype=np.float64).reshape(-1)
    loss = np.empty((scales.size, widths.size), dtype=np.float64)
    train = samples.train
    native = native_separation_rad()
    for i, scale in enumerate(scales):
        delta = scale * native
        for j, width in enumerate(widths):
            state = PhysicalBeamState(
                width=float(width),
                delta_lm_rad=(float(delta[0]), float(delta[1])),
            )
            vis = _predict_rows(samples, lookup, state, train)
            loss[i, j] = complex_visibility_loss(
                samples.measured[train], vis, samples.weight[train]
            )
    flat = int(np.nanargmin(loss))
    i_min, j_min = np.unravel_index(flat, loss.shape)
    return {
        "domain": "visibility",
        "estimator": "complex_visibility_loss",
        "selection_surface": False,
        "width_grid": [float(item) for item in widths],
        "magnitude_scale_grid": [float(item) for item in scales],
        "loss": loss,
        "min_width": float(widths[j_min]),
        "min_magnitude_scale": float(scales[i_min]),
        "min_loss": float(loss[i_min, j_min]),
        "interior": bool(i_min not in {0, scales.size - 1} and j_min not in {0, widths.size - 1}),
    }


def perpendicular_squint_profile(
    samples,
    lookup: FeedFrameLookup,
    *,
    width: float,
    perp_arcmin_grid: ArrayLike,
) -> dict[str, object]:
    """Train-row loss versus the component of squint perpendicular to CASSBEAM."""

    refuse_spw5(opened=False)
    refuse_magnitude_loss("complex")
    perps = np.asarray(perp_arcmin_grid, dtype=np.float64).reshape(-1)
    losses = np.empty(perps.size, dtype=np.float64)
    train = samples.train
    for index, perp in enumerate(perps):
        delta = delta_from_parallel_perp(parallel_scale=1.0, perp_arcmin=float(perp))
        state = PhysicalBeamState(
            width=float(width),
            delta_lm_rad=(float(delta[0]), float(delta[1])),
        )
        vis = _predict_rows(samples, lookup, state, train)
        losses[index] = complex_visibility_loss(samples.measured[train], vis, samples.weight[train])
    best = int(np.nanargmin(losses))
    return {
        "domain": "visibility",
        "component": "perpendicular_to_native_cassbeam",
        "grid_arcmin": [float(item) for item in perps],
        "loss": [float(item) for item in losses],
        "min_perp_arcmin": float(perps[best]),
        "interior": bool(best not in {0, perps.size - 1}),
        "min_loss": float(losses[best]),
    }


def interpret_experiment(
    *,
    native_vs_none: Mapping[str, object],
    empirical_vs_native: Mapping[str, object],
    width_native_vs_native: Mapping[str, object] | None,
    width_empirical_vs_empirical: Mapping[str, object] | None,
    direct: Mapping[str, object],
    width_native: Mapping[str, object] | None,
    width_empirical: Mapping[str, object] | None,
    identity_ok: bool,
    centroid_direction_ok: bool,
) -> dict[str, object]:
    refuse_spw5(opened=False)
    if not identity_ok or not centroid_direction_ok:
        return {
            "outcome": "parameterisation_or_convention_failure",
            "keep_native_squint": True,
            "store_empirical_correction": False,
            "width_survived": False,
            "spw5_ready": False,
            "development_only": True,
        }
    native_rrll = float(native_vs_none["losses"]["mover_holdout"]["all"]["rr_minus_ll"])
    none_rrll = float(native_vs_none.get("baseline_rr_minus_ll") or np.nan)
    native_beats_zero = bool(
        native_vs_none["spatial"]["improves"]
        or native_vs_none["moving"]["improves"]
        or (np.isfinite(native_rrll) and np.isfinite(none_rrll) and native_rrll < none_rrll)
    )
    emp = empirical_vs_native
    direction_ok = mover_direction_stable(direct.get("by_mover") or ())
    empirical_supported = bool(
        float(direct["magnitude_arcmin"]) > 0.0
        and np.isfinite(direct["mover_bootstrap"]["magnitude_lo"])
        and float(direct["mover_bootstrap"]["magnitude_lo"]) > 0.0
        and direction_ok
        and emp["spatial"]["improves"]
        and emp["moving"]["improves"]
        and emp["movers"]["unit_gate_passes"]
        and centroid_direction_ok
    )
    width_ok = False
    if (
        width_native is not None
        and width_empirical is not None
        and width_native_vs_native is not None
        and width_empirical_vs_empirical is not None
    ):
        width_ok = bool(
            width_native.get("interior")
            and width_empirical.get("interior")
            and abs(float(width_native["width"]) - float(width_empirical["width"])) <= 0.03
            and width_native_vs_native["spatial"]["improves"]
            and width_native_vs_native["moving"]["improves"]
            and width_native_vs_native["movers"]["unit_gate_passes"]
            and width_empirical_vs_empirical["spatial"]["improves"]
            and width_empirical_vs_empirical["moving"]["improves"]
        )
    if empirical_supported:
        outcome = "empirical_squint_correction_supported"
        store = True
    elif native_beats_zero and not (emp["spatial"]["improves"] and emp["moving"]["improves"]):
        outcome = "native_squint_detected_and_adequate"
        store = False
    else:
        outcome = "native_squint_retained_but_unresolved"
        store = False
    return {
        "outcome": outcome,
        "keep_native_squint": True,
        "store_empirical_correction": store,
        "width_survived": width_ok,
        "spw5_ready": bool(
            (store or outcome == "native_squint_detected_and_adequate") and width_ok
        ),
        "development_only": True,
        "native_beats_zero_squint_observable": native_beats_zero,
        "direction_stable": direction_ok,
    }
