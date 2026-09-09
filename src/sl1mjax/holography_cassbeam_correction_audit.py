"""Consistency audit of the validation-selected SPW-4 diagonal correction.

The accepted prefix is identity + R/L squint scale + beam width. Pointing,
sidelobe, and azimuthal terms stay rejected. This module does not open SPW 5
and does not refit the frozen family.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.holography_cassbeam_correction import (
    ARCMIN_TO_RAD,
    CorrectionSamples,
    CorrectionState,
    FeedFrameLookup,
    apply_feed_frame_correction,
    cassbeam_hand_peaks_rad,
    mover_units_to_dict,
    predict_from_state,
)
from sl1mjax.holography_cassbeam_holoraster_report import (
    squint_from_voltage_maps,
    weighted_hand_stats,
)
from sl1mjax.holography_diagonal_correction import (
    COPOLAR_HANDS,
    SPW4_HOLDOUT_MOVING_ANTENNA_NAMES,
    complex_visibility_loss,
    mover_cluster_ids,
    mover_paired_deltas,
    score_paired_holdout,
)

VALIDATION_SELECTED_SPW4 = "validation_selected_spw4_diagonal_correction"
ACCEPTED_SQUINT_SCALE = 0.85
ACCEPTED_WIDTH_SCALE = 1.04
ACCEPTED_TERMS = ("identity", "rl_squint_scale", "beam_width")
REJECTED_LADDER_TERMS = (
    "pointing_offset",
    "first_sidelobe_radius_amplitude",
    "low_order_azimuthal",
)
PUBLICATION_MEASURED_SQUINT_ARCMIN = 0.515
PUBLICATION_CASSBEAM_SQUINT_ARCMIN = 0.411
SURFACE_SQUINT_GRID = np.linspace(0.70, 1.50, 17)
SURFACE_WIDTH_GRID = np.linspace(0.90, 1.12, 12)
MODEL_GRID_HALF_ARCMIN = 8.0
MODEL_GRID_STEP_ARCMIN = 0.10
EDGE_TOLERANCE = 1.0e-12
TRADEOFF_LOSS_FRACTION = 0.05


def accepted_validation_state() -> CorrectionState:
    """Frozen validation-selected prefix. Pointing stays rejected."""

    return CorrectionState(
        squint_scale=ACCEPTED_SQUINT_SCALE,
        width_scale=ACCEPTED_WIDTH_SCALE,
        accepted_terms=ACCEPTED_TERMS,
    )


def refuse_reopening_rejected_ladder_terms(state: CorrectionState) -> None:
    extra = tuple(term for term in state.accepted_terms if term in REJECTED_LADDER_TERMS)
    if extra:
        raise RuntimeError(
            "pointing, sidelobe, and azimuthal stay rejected in this ladder; "
            f"refusing {extra}"
        )
    if state.accepted_terms != ACCEPTED_TERMS and not state.is_identity():
        if any(term in REJECTED_LADDER_TERMS for term in state.accepted_terms):
            raise RuntimeError("rejected first-ladder terms cannot be reopened")


def warped_hand_peaks_rad(
    state: CorrectionState,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Commanded-frame peaks if the CASSBEAM 20%-of-peak centroids just warp."""

    refuse_reopening_rejected_ladder_terms(state)
    peak_r, peak_l = cassbeam_hand_peaks_rad()
    factor = float(state.width_scale) + float(state.squint_scale) - 1.0
    return factor * peak_r, factor * peak_l


def warped_squint_separation_arcmin(state: CorrectionState) -> float:
    right, left = warped_hand_peaks_rad(state)
    return float(np.hypot(*(right - left)) / ARCMIN_TO_RAD)


def model_offset_grid(
    *,
    half_arcmin: float = MODEL_GRID_HALF_ARCMIN,
    step_arcmin: float = MODEL_GRID_STEP_ARCMIN,
) -> tuple[NDArray[np.float64], int]:
    axis = np.arange(-float(half_arcmin), float(half_arcmin) + 0.5 * float(step_arcmin), float(step_arcmin))
    ll, mm = np.meshgrid(axis, axis, indexing="xy")
    offset = np.stack([ll.reshape(-1), mm.reshape(-1)], axis=1) * ARCMIN_TO_RAD
    return offset, int(axis.size)


def render_feed_frame_beams(
    lookup: FeedFrameLookup,
    state: CorrectionState,
    offset_lm_rad: ArrayLike,
) -> NDArray[np.complex128]:
    refuse_reopening_rejected_ladder_terms(state)
    return apply_feed_frame_correction(offset_lm_rad, lookup, state)


def squint_from_feed_jones(
    offset_lm_rad: ArrayLike,
    jones: ArrayLike,
    weight: ArrayLike | None = None,
    *,
    series: str,
    frequency_hz: float | None = None,
) -> dict[str, object]:
    values = np.asarray(jones, dtype=np.complex128)
    if values.ndim != 3 or values.shape[-2:] != (2, 2):
        raise ValueError("feed Jones must have shape (sample, 2, 2)")
    if weight is None:
        rr_w = np.ones(values.shape[0], dtype=np.float64)
        ll_w = rr_w
    else:
        wgt = np.asarray(weight, dtype=np.float64)
        if wgt.ndim == 3:
            rr_w = wgt[:, 0, 0]
            ll_w = wgt[:, 1, 1]
        else:
            rr_w = wgt.reshape(-1)
            ll_w = rr_w
    return squint_from_voltage_maps(
        offset_lm_rad,
        np.abs(values[:, 0, 0]) ** 2,
        np.abs(values[:, 1, 1]) ** 2,
        rr_w,
        ll_w,
        frequency_hz=frequency_hz,
        series=series,
    )


def squint_from_visibilities(
    offset_lm_rad: ArrayLike,
    vis: ArrayLike,
    weight: ArrayLike,
    *,
    series: str,
    frequency_hz: float | None = None,
) -> dict[str, object]:
    values = np.asarray(vis, dtype=np.complex128)
    if values.ndim == 4:
        values = values[:, 0]
    wgt = np.asarray(weight, dtype=np.float64)
    if wgt.ndim == 4:
        wgt = wgt[:, 0]
    return squint_from_voltage_maps(
        offset_lm_rad,
        np.abs(values[:, 0, 0]) ** 2,
        np.abs(values[:, 1, 1]) ** 2,
        wgt[:, 0, 0],
        wgt[:, 1, 1],
        frequency_hz=frequency_hz,
        series=series,
    )


def hand_residual_power(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    *,
    hand: str,
) -> float:
    meas = np.asarray(measured)
    pred = np.asarray(predicted)
    wgt = np.asarray(weight, dtype=np.float64)
    if meas.ndim == 4:
        meas = meas[:, 0]
        pred = pred[:, 0]
        wgt = wgt[:, 0]
    lookup = {name: (row, col) for name, row, col in COPOLAR_HANDS}
    if hand not in lookup:
        raise ValueError(f"hand must be rr or ll, not {hand!r}")
    row, col = lookup[hand]
    obs = meas[:, row, col]
    hat = pred[:, row, col]
    ww = wgt[:, row, col]
    finite = np.isfinite(obs) & np.isfinite(hat) & np.isfinite(ww) & (ww > 0.0)
    denom = float(np.sum(ww[finite] * np.abs(obs[finite]) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(ww[finite] * np.abs(obs[finite] - hat[finite]) ** 2) / denom)


def split_region_hand_losses(
    samples: CorrectionSamples,
    predicted: ArrayLike,
) -> dict[str, dict[str, dict[str, float]]]:
    """Train / spatial / mover losses by copolar hand and radial region."""

    pred = np.asarray(predicted)
    splits = {
        "train": samples.train,
        "spatial_holdout": samples.spatial_holdout,
        "mover_holdout": samples.mover_holdout,
    }
    regions = {
        "all": np.ones(samples.train.size, dtype=bool),
        "main_lobe": samples.main_lobe,
        "mid": samples.mid,
        "outer": samples.outer,
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, split in splits.items():
        out[split_name] = {}
        for region_name, region in regions.items():
            choose = np.asarray(split, dtype=bool) & np.asarray(region, dtype=bool)
            if not bool(np.any(choose)):
                out[split_name][region_name] = {
                    "n": 0,
                    "loss": float("nan"),
                    "rr": float("nan"),
                    "ll": float("nan"),
                }
                continue
            meas = samples.measured[choose]
            hat = pred[choose]
            wgt = samples.weight[choose]
            stats_rr = weighted_hand_stats(meas[:, 0, 0], hat[:, 0, 0], wgt[:, 0, 0])
            stats_ll = weighted_hand_stats(meas[:, 1, 1], hat[:, 1, 1], wgt[:, 1, 1])
            out[split_name][region_name] = {
                "n": int(np.sum(choose)),
                "loss": complex_visibility_loss(meas, hat, wgt),
                "rr": float(stats_rr["residual_power"]),
                "ll": float(stats_ll["residual_power"]),
            }
    return out


def joint_squint_width_surface(
    samples: CorrectionSamples,
    lookup: FeedFrameLookup,
    *,
    squint_grid: ArrayLike = SURFACE_SQUINT_GRID,
    width_grid: ArrayLike = SURFACE_WIDTH_GRID,
    row_mask: ArrayLike | None = None,
) -> dict[str, object]:
    """Train-row loss on a joint (squint, width) grid. Does not refit."""

    mask = samples.train if row_mask is None else np.asarray(row_mask, dtype=bool)
    squint = np.asarray(squint_grid, dtype=np.float64).reshape(-1)
    width = np.asarray(width_grid, dtype=np.float64).reshape(-1)
    loss = np.full((squint.size, width.size), np.nan, dtype=np.float64)
    for i, scale in enumerate(squint):
        for j, wide in enumerate(width):
            state = CorrectionState(
                squint_scale=float(scale),
                width_scale=float(wide),
                accepted_terms=ACCEPTED_TERMS,
            )
            refuse_reopening_rejected_ladder_terms(state)
            vis = predict_from_state(samples, lookup, state)
            loss[i, j] = complex_visibility_loss(
                samples.measured[mask], vis[mask], samples.weight[mask]
            )
    finite = np.isfinite(loss)
    if not bool(np.any(finite)):
        raise ValueError("joint squint–width surface produced no finite losses")
    flat = int(np.nanargmin(loss))
    i_min, j_min = np.unravel_index(flat, loss.shape)
    accepted = accepted_validation_state()
    i_acc = int(np.argmin(np.abs(squint - accepted.squint_scale)))
    j_acc = int(np.argmin(np.abs(width - accepted.width_scale)))
    min_loss = float(loss[i_min, j_min])
    span = float(np.nanmax(loss) - min_loss)
    near = finite & ((loss - min_loss) <= max(TRADEOFF_LOSS_FRACTION * span, 0.0))
    return {
        "squint_grid": [float(item) for item in squint],
        "width_grid": [float(item) for item in width],
        "loss": loss,
        "min_squint": float(squint[i_min]),
        "min_width": float(width[j_min]),
        "min_loss": min_loss,
        "accepted_squint": float(accepted.squint_scale),
        "accepted_width": float(accepted.width_scale),
        "accepted_loss": float(loss[i_acc, j_acc]),
        "accepted_on_grid_edge": bool(
            abs(float(squint[i_acc]) - float(squint[0])) <= EDGE_TOLERANCE
            or abs(float(squint[i_acc]) - float(squint[-1])) <= EDGE_TOLERANCE
        ),
        "minimum_on_grid_edge": bool(
            i_min in {0, squint.size - 1} or j_min in {0, width.size - 1}
        ),
        "n_near_minimum": int(np.sum(near)),
        "identifiable_interior": bool(
            (not (i_min in {0, squint.size - 1} or j_min in {0, width.size - 1}))
            and int(np.sum(near)) <= 3
        ),
    }


def leave_one_mover_sensitivity(
    samples: CorrectionSamples,
    candidate: ArrayLike,
    baseline: ArrayLike,
    *,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, object]:
    """Per-mover ΔL and mover-holdout ΔL after dropping each holdout antenna."""

    units = mover_paired_deltas(
        samples.measured[samples.mover_holdout],
        np.asarray(candidate)[samples.mover_holdout],
        np.asarray(baseline)[samples.mover_holdout],
        samples.weight[samples.mover_holdout],
        samples.moving_id[samples.mover_holdout],
        samples.antenna_names,
        mainlobe_mask=samples.main_lobe[samples.mover_holdout],
    )
    names = {str(name): index for index, name in enumerate(samples.antenna_names)}
    loo: list[dict[str, object]] = []
    for name in SPW4_HOLDOUT_MOVING_ANTENNA_NAMES:
        antenna = names[name]
        keep = samples.mover_holdout & (samples.moving_id != int(antenna))
        if not bool(np.any(keep)):
            raise ValueError(f"leave-one-mover dropped every row for {name}")
        scored = score_paired_holdout(
            samples.measured[keep],
            np.asarray(candidate)[keep],
            np.asarray(baseline)[keep],
            samples.weight[keep],
            mover_cluster_ids(samples.moving_id, keep),
            axis="moving",
            cluster_kind="moving_antenna",
            n_boot=n_boot,
            seed=seed,
        )
        loo.append(
            {
                "left_out": name,
                "delta": float(scored.delta),
                "delta_lo": float(scored.delta_lo),
                "delta_hi": float(scored.delta_hi),
                "improves": bool(scored.improves()),
                "n_clusters": int(scored.n_clusters),
            }
        )
    return {
        "movers": mover_units_to_dict(units),
        "leave_one_out": loo,
        "all_five_improve": bool(units.n_improving == len(SPW4_HOLDOUT_MOVING_ANTENNA_NAMES)),
        "loo_all_improve": bool(all(bool(item["improves"]) for item in loo)),
        "unit_gate_passes": bool(units.passes()),
    }


def squint_moves_toward_holography(
    measured_arcmin: float,
    cassbeam_arcmin: float,
    corrected_arcmin: float,
) -> bool:
    """True when the correction closes the measured-minus-CASSBEAM gap."""

    before = abs(float(measured_arcmin) - float(cassbeam_arcmin))
    after = abs(float(measured_arcmin) - float(corrected_arcmin))
    return after < before


def interpret_physical_coherence(
    *,
    measured_squint_arcmin: float,
    cassbeam_squint_arcmin: float,
    corrected_squint_arcmin: float,
    surface: Mapping[str, object],
    movers: Mapping[str, object],
) -> dict[str, object]:
    """Decide whether the family is a physical beam or a validation-selected warp."""

    toward = squint_moves_toward_holography(
        measured_squint_arcmin, cassbeam_squint_arcmin, corrected_squint_arcmin
    )
    identifiable = bool(surface.get("identifiable_interior"))
    coherent = bool(toward and identifiable and movers.get("unit_gate_passes"))
    reasons: list[str] = []
    if not toward:
        reasons.append(
            "corrected 20%-of-peak squint moves away from the holography measurement"
        )
    if bool(surface.get("accepted_on_grid_edge")):
        reasons.append("accepted squint scale sits on the original ladder grid edge")
    if not identifiable:
        reasons.append("joint train surface is not an isolated interior minimum")
    if not bool(movers.get("unit_gate_passes")):
        reasons.append("holdout-mover unit gate failed")
    if coherent:
        status = "physically_coherent_candidate"
        freeze = True
    else:
        status = "validation_selected_not_general"
        freeze = False
    return {
        "status": status,
        "physically_coherent": coherent,
        "freeze_spw4_family": freeze,
        "open_spw5_transfer": False if not freeze else True,
        "squint_moves_toward_holography": toward,
        "identifiable_interior": identifiable,
        "reasons": reasons,
        "description": (
            "validation-selected SPW-4 diagonal correction"
            if not freeze
            else "frozen SPW-4 diagonal family ready for a no-refit SPW 5 transfer"
        ),
    }


def audit_payload(
    *,
    model_identity_squint: Mapping[str, object],
    model_corrected_squint: Mapping[str, object],
    visibility_measured_squint: Mapping[str, object],
    visibility_identity_squint: Mapping[str, object],
    visibility_corrected_squint: Mapping[str, object],
    warped_separation_arcmin: float,
    losses_identity: Mapping[str, object],
    losses_corrected: Mapping[str, object],
    surface: Mapping[str, object],
    movers: Mapping[str, object],
) -> dict[str, object]:
    interpretation = interpret_physical_coherence(
        measured_squint_arcmin=float(visibility_measured_squint["separation_arcmin"]),
        cassbeam_squint_arcmin=float(visibility_identity_squint["separation_arcmin"]),
        corrected_squint_arcmin=float(model_corrected_squint["separation_arcmin"]),
        surface=surface,
        movers=movers,
    )
    return {
        "artifact": VALIDATION_SELECTED_SPW4,
        "accepted": {
            "accepted_terms": list(ACCEPTED_TERMS),
            "squint_scale": ACCEPTED_SQUINT_SCALE,
            "width_scale": ACCEPTED_WIDTH_SCALE,
            "pointing_offset": "rejected",
            "sidelobe": "not_opened",
            "azimuthal": "not_opened",
        },
        "spw5_closed": True,
        "spw5_opened": False,
        "model_selected": False,
        "description": interpretation["description"],
        "squint": {
            "estimator": "mainlobe_20pct_peak",
            "publication_estimator": True,
            "warped_peak_separation_arcmin": float(warped_separation_arcmin),
            "model_identity": dict(model_identity_squint),
            "model_corrected": dict(model_corrected_squint),
            "visibility_measured": dict(visibility_measured_squint),
            "visibility_identity": dict(visibility_identity_squint),
            "visibility_corrected": dict(visibility_corrected_squint),
        },
        "losses": {"identity": losses_identity, "corrected": losses_corrected},
        "joint_surface": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in surface.items()
        },
        "movers": movers,
        "interpretation": interpretation,
    }
