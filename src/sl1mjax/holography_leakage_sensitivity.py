"""SPW-4 full-Jones leakage sensitivity and aggregation diagnosis.

The empirical LORO full-versus-diagonal test failed because the training
SNR mask reduced the model to the diagonal beam. That is not evidence that
the physical off-diagonal beam is zero. This module injects known
feed-frame leakage, inspects unmasked estimates, aggregates where
redundancy exists, and compares CASSBEAM full Jones to its diagonal on
the same held-out rows. Full Jones stays unfrozen. SPW 5 stays closed.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import BeamCalibrationState, require_beam_calibration_state
from sl1mjax.beam_operator import unique_visibility_times
from sl1mjax.cassbeam_beam import (
    _bilinear_jones,
    _diagonal_only,
    _nearest_table,
    _normalize_on_axis,
    load_cassbeam_cband_artifact,
)
from sl1mjax.holography import HolographyObservation, Memo195LowerCRaster, moving_reference_row_mask
from sl1mjax.holography_alignment import voltage_response_region_masks
from sl1mjax.holography_diagonal import _raster_label, _source_coherency
from sl1mjax.holography_full_jones import (
    HolographyFullJonesArtifact,
    HolographyFullJonesSample,
    _antenna_jones_planes,
    _predict_vis_jax,
    _sky_frame_residual_jax,
)
from sl1mjax.polarization import (
    Receptor,
    circular_parallactic_jones,
    invert_jones,
    pack_coherency,
    unpack_coherency,
)

INJECTED_LEAKAGE_AMPLITUDES = (0.001, 0.003, 0.01, 0.03)
LORO_LEAKAGE_SENSITIVITY = "loro_leakage_sensitivity"
LEAKAGE_SENSITIVITY_NOTE = (
    "The failed LORO full-versus-diagonal test established only that no "
    "transferable off-diagonal signal was detected at the present "
    "resolution. It is not evidence that the physical leak is zero or "
    "that CASSBEAM is wrong. The next artifact is the injection-derived "
    "detection curve."
)
FEED_FRAME_INJECTION_NOTE = (
    "Injection left-multiplies the mover feed-frame Jones by "
    "[[1, ε], [ε, 1]] and keeps the real flags, weights, coordinates, "
    "and residual Jones. Recovered leakage is E_RL/E_RR and E_LR/E_LL."
)
AGGREGATION_BEFORE_SNR_NOTE = (
    "Phase-align leakage ratios in the feed frame, then combine "
    "references and repeated dwells before the SNR decision. "
    "Thresholding per-time estimates can discard a coherent signal."
)
CASSBEAM_DIRECT_NOTE = (
    "Direct CASSBEAM full Jones versus CASSBEAM diagonal on held-out "
    "rows bypasses empirical recovery. Production full Jones stays "
    "unfrozen; this diagnostic uses allow_unfrozen=True."
)
_RECEPTORS = (Receptor.R, Receptor.L)


def feed_frame_leakage_jones(epsilon: complex) -> NDArray[np.complex128]:
    """Return :math:`L=[[1,\\varepsilon],[\\varepsilon,1]]`."""

    plane = np.eye(2, dtype=np.complex128)
    value = complex(epsilon)
    plane[0, 1] = value
    plane[1, 0] = value
    return plane


def inject_feed_frame_leakage(
    observation: HolographyObservation,
    *,
    epsilon: complex,
    residual_jones: ArrayLike | Mapping[int, ArrayLike] | None = None,
    parallactic_angle_rad: ArrayLike | None = None,
    row_mask: ArrayLike | None = None,
    visibility: ArrayLike | None = None,
    on_axis_atol_rad: float = 1.0e-6,
) -> NDArray[np.complex128]:
    """Inject feed-frame leakage into real visibilities.

    For a mover on antenna 1, :math:`V'=R_m L R_m^{-1} V`. For a mover on
    antenna 2, :math:`V'=V R_m^{-H} L^H R_m^H`. Origin rows are left
    unchanged so an on-axis gauge cannot absorb :math:`L`. Flags and
    weights are unchanged. Rows outside the moving–reference mask stay
    untouched.
    """

    measured = (
        np.asarray(observation.block.visibility, dtype=np.complex128)
        if visibility is None
        else np.asarray(visibility, dtype=np.complex128)
    )
    if measured.shape != observation.block.visibility.shape:
        raise ValueError("visibility must match the observation visibility shape")
    packed = pack_coherency(measured, observation.block.correlations, _RECEPTORS)
    injected = np.array(packed, copy=True)
    moving_ref = moving_reference_row_mask(observation.block, observation.pointing)
    active = observation.active_row_mask(mask_unsettled=True, moving_reference_only=True)
    if row_mask is not None:
        active = active & np.asarray(row_mask, dtype=bool).reshape(-1)
    rows = np.flatnonzero(moving_ref & active)
    if rows.size == 0:
        return measured
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = np.full(offsets.shape[:2], "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    unique_times, _ = unique_visibility_times(observation.block.time_s)
    time_index = np.asarray(inverse, dtype=np.int32)[rows]
    antenna_p = np.asarray(observation.block.antenna1, dtype=np.int32)[rows]
    antenna_q = np.asarray(observation.block.antenna2, dtype=np.int32)[rows]
    keep = pointing_valid[time_index, antenna_p] & pointing_valid[time_index, antenna_q]
    role_p = roles[time_index, antenna_p]
    moving_is_p = role_p == "moving"
    moving_is_q = roles[time_index, antenna_q] == "moving"
    keep &= moving_is_p ^ moving_is_q
    if not bool(np.any(keep)):
        return measured
    rows = rows[keep]
    time_index = time_index[keep]
    antenna_p = antenna_p[keep]
    antenna_q = antenna_q[keep]
    moving_is_p = moving_is_p[keep]
    moving = np.where(moving_is_p, antenna_p, antenna_q).astype(np.int32)
    radius = np.hypot(
        offsets[time_index, moving, 0],
        offsets[time_index, moving, 1],
    )
    off_axis = radius > float(on_axis_atol_rad)
    if not bool(np.any(off_axis)):
        return measured
    rows = rows[off_axis]
    time_index = time_index[off_axis]
    antenna_p = antenna_p[off_axis]
    antenna_q = antenna_q[off_axis]
    moving_is_p = moving_is_p[off_axis]
    moving = moving[off_axis]
    r_m = _antenna_jones_planes(residual_jones, moving)
    if residual_jones is not None:
        if parallactic_angle_rad is not None:
            chi = np.asarray(parallactic_angle_rad, dtype=np.float64)
        elif observation.antenna_position_m and observation.phase_centre_rad is not None:
            from sl1mjax.calibration_terms import parallactic_angle_rad as _chi

            chi = _chi(
                unique_times,
                observation.phase_centre_rad,
                np.asarray(observation.antenna_position_m, dtype=np.float64),
            )
        else:
            chi = None
        if chi is not None:
            import jax.numpy as jnp

            r_m = np.asarray(
                _sky_frame_residual_jax(jnp.asarray(r_m), jnp.asarray(chi[time_index, moving]))
            )
    leakage = feed_frame_leakage_jones(epsilon)
    inv_r = invert_jones(r_m)
    sandwich = r_m @ leakage @ inv_r
    selected = injected[rows]
    vis_p = sandwich[:, None, :, :] @ selected
    vis_q = selected @ np.conjugate(np.swapaxes(sandwich, -1, -2))[:, None, :, :]
    injected[rows] = np.where(moving_is_p[:, None, None, None], vis_p, vis_q)
    return unpack_coherency(injected, observation.block.correlations, _RECEPTORS)


def feed_frame_leakage_ratios(
    jones: ArrayLike,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return :math:`E_{RL}/E_{RR}` and :math:`E_{LR}/E_{LL}`."""

    planes = np.asarray(jones, dtype=np.complex128)
    rr = planes[..., 0, 0]
    ll = planes[..., 1, 1]
    rl = planes[..., 0, 1]
    lr = planes[..., 1, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        d_rl = np.where(np.abs(rr) > 1.0e-15, rl / rr, np.nan + 1j * np.nan)
        d_lr = np.where(np.abs(ll) > 1.0e-15, lr / ll, np.nan + 1j * np.nan)
    return d_rl, d_lr


def coherent_mean(values: ArrayLike, weights: ArrayLike | None = None) -> complex:
    """Weighted complex mean. This is not a median of magnitudes."""

    samples = np.asarray(values, dtype=np.complex128).reshape(-1)
    if weights is None:
        mass = np.ones(samples.shape, dtype=np.float64)
    else:
        mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(samples) & np.isfinite(mass) & (mass > 0.0)
    if not bool(np.any(finite)):
        return complex(np.nan, np.nan)
    total = float(np.sum(mass[finite]))
    if total <= 0.0:
        return complex(np.nan, np.nan)
    return complex(np.sum(mass[finite] * samples[finite]) / total)


def phase_coherence(values: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Return :math:`|\\langle z\\rangle| / \\langle|z|\\rangle`."""

    samples = np.asarray(values, dtype=np.complex128).reshape(-1)
    if weights is None:
        mass = np.ones(samples.shape, dtype=np.float64)
    else:
        mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(samples) & (mass > 0.0)
    if not bool(np.any(finite)):
        return float("nan")
    mean = coherent_mean(samples[finite], mass[finite])
    amplitude = float(np.sum(mass[finite] * np.abs(samples[finite])) / np.sum(mass[finite]))
    if amplitude <= 0.0:
        return float("nan")
    return float(np.abs(mean) / amplitude)


def _weighted_scatter(values: ArrayLike, weights: ArrayLike, mean: complex) -> float:
    samples = np.asarray(values, dtype=np.complex128).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(samples) & (mass > 0.0) & np.isfinite(mean)
    if int(np.sum(finite)) < 2:
        return float("nan")
    return float(
        np.sqrt(np.sum(mass[finite] * np.abs(samples[finite] - mean) ** 2) / np.sum(mass[finite]))
    )


def _group_scatter(
    values: NDArray[np.complex128],
    weights: NDArray[np.float64],
    labels: NDArray,
) -> float:
    """Scatter of per-label coherent means about the global coherent mean."""

    finite = np.isfinite(values) & (weights > 0.0)
    if not bool(np.any(finite)):
        return float("nan")
    keys, inverse = np.unique(np.asarray(labels)[finite], return_inverse=True)
    if keys.size < 2:
        return float("nan")
    total = np.zeros(keys.size, dtype=np.float64)
    stacked = np.zeros(keys.size, dtype=np.complex128)
    np.add.at(total, inverse, weights[finite])
    np.add.at(stacked, inverse, weights[finite] * values[finite])
    ok = total > 0.0
    if int(np.sum(ok)) < 2:
        return float("nan")
    means = stacked[ok] / total[ok]
    global_mean = coherent_mean(values[finite], weights[finite])
    return float(np.sqrt(np.mean(np.abs(means - global_mean) ** 2)))


def _safe_median(values: ArrayLike) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def _snr_bins(
    snr: NDArray[np.float64], edges: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 10.0)
) -> dict[str, int]:
    finite = snr[np.isfinite(snr)]
    counts = {}
    last = None
    for edge in edges:
        if last is None:
            counts[f"lt_{edge:g}"] = int(np.sum(finite < edge))
        else:
            counts[f"{last:g}_to_{edge:g}"] = int(np.sum((finite >= last) & (finite < edge)))
        last = edge
    counts[f"ge_{edges[-1]:g}"] = int(np.sum(finite >= edges[-1]))
    counts["n_finite"] = int(finite.size)
    return counts


def unmasked_leakage_diagnostics(
    rows: Mapping[str, ArrayLike],
    *,
    voltage: ArrayLike | None = None,
    min_snr: float = 3.0,
) -> dict[str, object]:
    """Inspect leakage ratios before the SNR mask.

    Reports the complex coherent mean, phase coherence, thermal / reference
    / repeat-visit scatter, SNR versus radius and voltage, and the count
    that entered every estimate.
    """

    jones = np.asarray(rows["jones"], dtype=np.complex128)
    weight = np.asarray(rows["weight"], dtype=np.float64)
    d_rl, d_lr = feed_frame_leakage_ratios(jones)
    w_rl = weight[..., 0, 1]
    w_lr = weight[..., 1, 0]
    radius = np.hypot(
        np.asarray(rows["offset_lm_rad"], dtype=np.float64)[:, 0],
        np.asarray(rows["offset_lm_rad"], dtype=np.float64)[:, 1],
    )
    thermal_rl = np.where(w_rl > 0.0, 1.0 / np.sqrt(w_rl), np.nan)
    thermal_lr = np.where(w_lr > 0.0, 1.0 / np.sqrt(w_lr), np.nan)
    mean_rl = coherent_mean(d_rl, w_rl)
    mean_lr = coherent_mean(d_lr, w_lr)
    snr_rl = np.abs(d_rl) / np.maximum(thermal_rl, 1.0e-30)
    snr_lr = np.abs(d_lr) / np.maximum(thermal_lr, 1.0e-30)
    report: dict[str, object] = {
        "n": int(d_rl.size),
        "n_finite_rl": int(np.sum(np.isfinite(d_rl) & (w_rl > 0.0))),
        "n_finite_lr": int(np.sum(np.isfinite(d_lr) & (w_lr > 0.0))),
        "coherent_mean_rl": [mean_rl.real, mean_rl.imag],
        "coherent_mean_lr": [mean_lr.real, mean_lr.imag],
        "coherent_abs_rl": float(np.abs(mean_rl)),
        "coherent_abs_lr": float(np.abs(mean_lr)),
        "median_abs_rl": float(np.nanmedian(np.abs(d_rl))),
        "median_abs_lr": float(np.nanmedian(np.abs(d_lr))),
        "phase_coherence_rl": phase_coherence(d_rl, w_rl),
        "phase_coherence_lr": phase_coherence(d_lr, w_lr),
        "thermal_scatter_rl": float(np.nanmedian(thermal_rl[np.isfinite(thermal_rl)]))
        if np.any(np.isfinite(thermal_rl))
        else float("nan"),
        "thermal_scatter_lr": float(np.nanmedian(thermal_lr[np.isfinite(thermal_lr)]))
        if np.any(np.isfinite(thermal_lr))
        else float("nan"),
        "reference_scatter_rl": _group_scatter(d_rl, w_rl, np.asarray(rows["reference_id"])),
        "reference_scatter_lr": _group_scatter(d_lr, w_lr, np.asarray(rows["reference_id"])),
        "repeat_visit_scatter_rl": _group_scatter(d_rl, w_rl, np.asarray(rows["visit_id"])),
        "repeat_visit_scatter_lr": _group_scatter(d_lr, w_lr, np.asarray(rows["visit_id"])),
        "snr_rl": _snr_bins(snr_rl),
        "snr_lr": _snr_bins(snr_lr),
        "fraction_snr_ge_min_rl": float(np.mean(snr_rl[np.isfinite(snr_rl)] >= float(min_snr)))
        if np.any(np.isfinite(snr_rl))
        else float("nan"),
        "fraction_snr_ge_min_lr": float(np.mean(snr_lr[np.isfinite(snr_lr)] >= float(min_snr)))
        if np.any(np.isfinite(snr_lr))
        else float("nan"),
        "n_meaning": "unmasked per-row leakage ratios after RIME inversion",
    }
    voltage_row = None if voltage is None else np.asarray(voltage, dtype=np.float64)
    if voltage_row is not None and voltage_row.size == np.asarray(rows["row"]).size:
        row_voltage = voltage_row
    elif voltage_row is not None:
        row_voltage = voltage_row[np.asarray(rows["row"], dtype=np.int64)]
    else:
        row_voltage = None
    if row_voltage is not None:
        regions = voltage_response_region_masks(row_voltage)
        by_voltage = {}
        for name, mask in regions.items():
            by_voltage[name] = {
                "n": int(np.sum(mask)),
                "coherent_abs_rl": float(np.abs(coherent_mean(d_rl[mask], w_rl[mask]))),
                "coherent_abs_lr": float(np.abs(coherent_mean(d_lr[mask], w_lr[mask]))),
                "median_snr_rl": _safe_median(snr_rl[mask]),
                "median_snr_lr": _safe_median(snr_lr[mask]),
            }
        report["by_voltage"] = by_voltage
    radius_edges = (0.0, 2.0e-3, 5.0e-3, 1.0e-2, 2.0e-2)
    by_radius = {}
    last = radius_edges[0]
    for edge in radius_edges[1:]:
        mask = (radius >= last) & (radius < edge)
        by_radius[f"{last:g}_to_{edge:g}_rad"] = {
            "n": int(np.sum(mask)),
            "coherent_abs_rl": float(np.abs(coherent_mean(d_rl[mask], w_rl[mask]))),
            "median_snr_rl": _safe_median(snr_rl[mask]),
        }
        last = edge
    by_radius[f"ge_{radius_edges[-1]:g}_rad"] = {
        "n": int(np.sum(radius >= last)),
        "coherent_abs_rl": float(np.abs(coherent_mean(d_rl[radius >= last], w_rl[radius >= last]))),
        "median_snr_rl": _safe_median(snr_rl[radius >= last]),
    }
    report["by_radius"] = by_radius
    return report


def _cell_keys(offset_lm_rad: ArrayLike, *, offset_atol_rad: float) -> NDArray[np.int64]:
    scale = 1.0 / float(offset_atol_rad)
    offset = np.asarray(offset_lm_rad, dtype=np.float64)
    packed = np.rint(offset[:, 0] * scale).astype(np.int64) * 1_000_003 + np.rint(
        offset[:, 1] * scale
    ).astype(np.int64)
    return packed


def aggregate_feed_frame_leakage(
    rows: Mapping[str, ArrayLike],
    *,
    offset_atol_rad: float = 2.908882086657216e-05,
    min_snr: float = 3.0,
    observation: HolographyObservation | None = None,
) -> HolographyFullJonesArtifact:
    """Phase-align feed-frame leakage, then combine refs and repeated dwells.

    Leakage is averaged as :math:`E_{RL}/E_{RR}` and :math:`E_{LR}/E_{LL}`
    so a dwell-dependent copolar phase cannot cancel a stable leak. The
    SNR mask is applied only after that aggregation.
    """

    jones = np.asarray(rows["jones"], dtype=np.complex128)
    weight = np.asarray(rows["weight"], dtype=np.float64)
    d_rl, d_lr = feed_frame_leakage_ratios(jones)
    moving = np.asarray(rows["moving_id"], dtype=np.int32)
    frequency = np.asarray(rows["frequency_hz"], dtype=np.float64)
    offset = np.asarray(rows["offset_lm_rad"], dtype=np.float64)
    cells = _cell_keys(offset, offset_atol_rad=offset_atol_rad)
    freq_key = np.rint(frequency).astype(np.int64)
    keys = np.stack([moving.astype(np.int64), cells, freq_key], axis=1)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_group = int(uniq.shape[0])
    w_rl = weight[..., 0, 1]
    w_lr = weight[..., 1, 0]
    w_rr = weight[..., 0, 0]
    w_ll = weight[..., 1, 1]
    rl_w = np.zeros(n_group, dtype=np.float64)
    lr_w = np.zeros(n_group, dtype=np.float64)
    rr_w = np.zeros(n_group, dtype=np.float64)
    ll_w = np.zeros(n_group, dtype=np.float64)
    rl_sum = np.zeros(n_group, dtype=np.complex128)
    lr_sum = np.zeros(n_group, dtype=np.complex128)
    rr_sum = np.zeros(n_group, dtype=np.complex128)
    ll_sum = np.zeros(n_group, dtype=np.complex128)
    counts = np.zeros(n_group, dtype=np.int32)
    np.add.at(rl_w, inverse, np.where(np.isfinite(d_rl), w_rl, 0.0))
    np.add.at(lr_w, inverse, np.where(np.isfinite(d_lr), w_lr, 0.0))
    np.add.at(rr_w, inverse, np.where(np.isfinite(jones[:, 0, 0]), w_rr, 0.0))
    np.add.at(ll_w, inverse, np.where(np.isfinite(jones[:, 1, 1]), w_ll, 0.0))
    np.add.at(rl_sum, inverse, np.where(np.isfinite(d_rl), w_rl * d_rl, 0.0))
    np.add.at(lr_sum, inverse, np.where(np.isfinite(d_lr), w_lr * d_lr, 0.0))
    np.add.at(rr_sum, inverse, np.where(np.isfinite(jones[:, 0, 0]), w_rr * jones[:, 0, 0], 0.0))
    np.add.at(ll_sum, inverse, np.where(np.isfinite(jones[:, 1, 1]), w_ll * jones[:, 1, 1], 0.0))
    np.add.at(counts, inverse, 1)
    d_rl_mean = np.where(rl_w > 0.0, rl_sum / np.maximum(rl_w, 1.0e-30), np.nan)
    d_lr_mean = np.where(lr_w > 0.0, lr_sum / np.maximum(lr_w, 1.0e-30), np.nan)
    e_rr = np.where(rr_w > 0.0, rr_sum / np.maximum(rr_w, 1.0e-30), 0.0)
    e_ll = np.where(ll_w > 0.0, ll_sum / np.maximum(ll_w, 1.0e-30), 0.0)
    planes = np.zeros((n_group, 2, 2), dtype=np.complex128)
    planes[:, 0, 0] = e_rr
    planes[:, 1, 1] = e_ll
    planes[:, 0, 1] = d_rl_mean * e_rr
    planes[:, 1, 0] = d_lr_mean * e_ll
    rl_scatter = np.zeros(n_group, dtype=np.float64)
    lr_scatter = np.zeros(n_group, dtype=np.float64)
    np.add.at(
        rl_scatter,
        inverse,
        np.where(np.isfinite(d_rl), w_rl * np.abs(d_rl - d_rl_mean[inverse]) ** 2, 0.0),
    )
    np.add.at(
        lr_scatter,
        inverse,
        np.where(np.isfinite(d_lr), w_lr * np.abs(d_lr - d_lr_mean[inverse]) ** 2, 0.0),
    )
    sigma_rms_rl = np.sqrt(rl_scatter / np.maximum(rl_w, 1.0e-30))
    sigma_rms_lr = np.sqrt(lr_scatter / np.maximum(lr_w, 1.0e-30))
    n_eff_rl = np.maximum(counts.astype(np.float64), 1.0)
    n_eff_lr = np.maximum(counts.astype(np.float64), 1.0)
    sigma_d_rl = np.where(
        sigma_rms_rl > 0.0,
        sigma_rms_rl / np.sqrt(n_eff_rl),
        1.0 / np.sqrt(np.maximum(rl_w, 1.0e-30)),
    )
    sigma_d_lr = np.where(
        sigma_rms_lr > 0.0,
        sigma_rms_lr / np.sqrt(n_eff_lr),
        1.0 / np.sqrt(np.maximum(lr_w, 1.0e-30)),
    )
    sigma = np.full((n_group, 2, 2), np.nan, dtype=np.float64)
    sigma[:, 0, 0] = 1.0 / np.sqrt(np.maximum(rr_w, 1.0e-30))
    sigma[:, 1, 1] = 1.0 / np.sqrt(np.maximum(ll_w, 1.0e-30))
    sigma[:, 0, 1] = sigma_d_rl * np.abs(e_rr)
    sigma[:, 1, 0] = sigma_d_lr * np.abs(e_ll)
    memo = Memo195LowerCRaster()
    samples = []
    leakage = []
    copolar = []
    times = np.asarray(rows["unique_time_s"], dtype=np.float64)
    time_acc = np.zeros(n_group, dtype=np.float64)
    time_w = np.zeros(n_group, dtype=np.float64)
    off_acc = np.zeros((n_group, 2), dtype=np.float64)
    np.add.at(time_acc, inverse, times)
    np.add.at(time_w, inverse, 1.0)
    np.add.at(off_acc, inverse, offset)
    for index in range(n_group):
        cell_offset = off_acc[index] / max(float(time_w[index]), 1.0)
        plane = planes[index]
        snr_rl = np.abs(d_rl_mean[index]) / np.maximum(sigma_d_rl[index], 1.0e-30)
        snr_lr = np.abs(d_lr_mean[index]) / np.maximum(sigma_d_lr[index], 1.0e-30)
        leak_ok = bool(
            rl_w[index] > 0.0
            and lr_w[index] > 0.0
            and np.isfinite(snr_rl)
            and np.isfinite(snr_lr)
            and (snr_rl >= float(min_snr) or snr_lr >= float(min_snr))
        )
        if not leak_ok:
            plane = np.array(plane, copy=True)
            plane[0, 1] = 0.0
            plane[1, 0] = 0.0
        sample = HolographyFullJonesSample(
            moving_antenna_id=int(uniq[index, 0]),
            unique_time_s=float(time_acc[index] / max(float(time_w[index]), 1.0)),
            offset_lm_rad=np.asarray(cell_offset, dtype=np.float64),
            frequency_hz=float(uniq[index, 2]),
            jones=plane,
            sigma=sigma[index],
            n_reference=int(counts[index]),
            weight=float(0.25 * (rr_w[index] + ll_w[index] + rl_w[index] + lr_w[index])),
            copolar_valid=bool(rr_w[index] > 0.0 and ll_w[index] > 0.0),
            off_diagonal_valid=leak_ok,
            raster=_raster_label(cell_offset, memo),
        )
        samples.append(sample)
        copolar.append(sample.copolar_valid)
        leakage.append(leak_ok)
        planes[index] = plane
    source_name = "3C147" if observation is None else observation.source_name
    calibration_state = "casa_parang_true" if observation is None else observation.calibration_state
    source_version = (
        "stokes_i_point"
        if observation is None or observation.source_model is None
        else observation.source_model.standard
    )
    offset_sign = "commanded_pointing" if observation is None else observation.pointing.offset_sign
    return HolographyFullJonesArtifact(
        samples=tuple(samples),
        jones=np.asarray(planes, dtype=np.complex128),
        valid=np.asarray(copolar, dtype=bool),
        off_diagonal_valid=np.asarray(leakage, dtype=bool),
        calibration_state=calibration_state,
        source_name=source_name,
        source_model_version=source_version,
        reference_combination="feed_frame_phase_aligned_cell_average",
        receptor_convention="circular_R_L",
        offset_sign=offset_sign,
        frozen=False,
        notes=(
            "empirical full Jones; not frozen",
            AGGREGATION_BEFORE_SNR_NOTE,
            f"off-diagonals with aggregated SNR < {float(min_snr):.1f} are zeroed",
        ),
    )


def combine_channel_leakage_after_delay(
    rows: Mapping[str, ArrayLike],
    *,
    offset_atol_rad: float = 2.908882086657216e-05,
) -> dict[str, NDArray]:
    """Remove a linear delay/phase per cell, then coherently average channels.

    Training channels stay in the returned estimates. This is an SPW-4
    sensitivity tool. It does not open SPW 5.
    """

    jones = np.asarray(rows["jones"], dtype=np.complex128)
    weight = np.asarray(rows["weight"], dtype=np.float64)
    d_rl, d_lr = feed_frame_leakage_ratios(jones)
    frequency = np.asarray(rows["frequency_hz"], dtype=np.float64)
    moving = np.asarray(rows["moving_id"], dtype=np.int64)
    cells = _cell_keys(rows["offset_lm_rad"], offset_atol_rad=offset_atol_rad)
    aligned_rl = _remove_linear_delay_grouped(d_rl, frequency, weight[..., 0, 1], moving, cells)
    aligned_lr = _remove_linear_delay_grouped(d_lr, frequency, weight[..., 1, 0], moving, cells)
    cell_keys = np.stack([moving, cells], axis=1)
    cell_uniq, cell_inverse = np.unique(cell_keys, axis=0, return_inverse=True)
    n_cell = int(cell_uniq.shape[0])
    w_rl = weight[..., 0, 1]
    w_lr = weight[..., 1, 0]
    w_rr = weight[..., 0, 0]
    w_ll = weight[..., 1, 1]
    rl_w = np.zeros(n_cell, dtype=np.float64)
    lr_w = np.zeros(n_cell, dtype=np.float64)
    rr_w = np.zeros(n_cell, dtype=np.float64)
    ll_w = np.zeros(n_cell, dtype=np.float64)
    rl_sum = np.zeros(n_cell, dtype=np.complex128)
    lr_sum = np.zeros(n_cell, dtype=np.complex128)
    rr_sum = np.zeros(n_cell, dtype=np.complex128)
    ll_sum = np.zeros(n_cell, dtype=np.complex128)
    np.add.at(rl_w, cell_inverse, np.where(np.isfinite(aligned_rl), w_rl, 0.0))
    np.add.at(lr_w, cell_inverse, np.where(np.isfinite(aligned_lr), w_lr, 0.0))
    np.add.at(rr_w, cell_inverse, np.where(np.isfinite(jones[:, 0, 0]), w_rr, 0.0))
    np.add.at(ll_w, cell_inverse, np.where(np.isfinite(jones[:, 1, 1]), w_ll, 0.0))
    np.add.at(rl_sum, cell_inverse, np.where(np.isfinite(aligned_rl), w_rl * aligned_rl, 0.0))
    np.add.at(lr_sum, cell_inverse, np.where(np.isfinite(aligned_lr), w_lr * aligned_lr, 0.0))
    np.add.at(
        rr_sum, cell_inverse, np.where(np.isfinite(jones[:, 0, 0]), w_rr * jones[:, 0, 0], 0.0)
    )
    np.add.at(
        ll_sum, cell_inverse, np.where(np.isfinite(jones[:, 1, 1]), w_ll * jones[:, 1, 1], 0.0)
    )
    e_rr = np.where(rr_w > 0.0, rr_sum / np.maximum(rr_w, 1.0e-30), 0.0)
    e_ll = np.where(ll_w > 0.0, ll_sum / np.maximum(ll_w, 1.0e-30), 0.0)
    d_rl_m = np.where(rl_w > 0.0, rl_sum / np.maximum(rl_w, 1.0e-30), np.nan)
    d_lr_m = np.where(lr_w > 0.0, lr_sum / np.maximum(lr_w, 1.0e-30), np.nan)
    out_jones = np.zeros((n_cell, 2, 2), dtype=np.complex128)
    out_jones[:, 0, 0] = e_rr
    out_jones[:, 1, 1] = e_ll
    out_jones[:, 0, 1] = d_rl_m * e_rr
    out_jones[:, 1, 0] = d_lr_m * e_ll
    offset = np.asarray(rows["offset_lm_rad"], dtype=np.float64)
    off_acc = np.zeros((n_cell, 2), dtype=np.float64)
    off_w = np.zeros(n_cell, dtype=np.float64)
    np.add.at(off_acc, cell_inverse, offset)
    np.add.at(off_w, cell_inverse, 1.0)
    return {
        "moving_id": cell_uniq[:, 0].astype(np.int32),
        "cell_id": cell_uniq[:, 1],
        "offset_lm_rad": off_acc / np.maximum(off_w[:, None], 1.0),
        "jones": out_jones,
        "n_channel_combined": np.bincount(cell_inverse, minlength=n_cell).astype(np.int32),
        "coherent_abs_rl": np.abs(d_rl_m),
        "coherent_abs_lr": np.abs(d_lr_m),
    }


def _remove_linear_delay_grouped(
    values: NDArray[np.complex128],
    frequency_hz: NDArray[np.float64],
    weights: NDArray[np.float64],
    moving_id: NDArray[np.int64],
    cell_id: NDArray[np.int64],
) -> NDArray[np.complex128]:
    """Derotate a per-cell linear phase versus frequency. No Python groups."""

    samples = np.asarray(values, dtype=np.complex128)
    freq = np.asarray(frequency_hz, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    group_keys = np.stack(
        [np.asarray(moving_id, dtype=np.int64), np.asarray(cell_id, dtype=np.int64)], axis=1
    )
    group_uniq, group_inverse = np.unique(group_keys, axis=0, return_inverse=True)
    freq_values, freq_inverse = np.unique(freq, return_inverse=True)
    n_group = int(group_uniq.shape[0])
    n_freq = int(freq_values.size)
    slot = group_inverse * n_freq + freq_inverse
    finite = np.isfinite(samples) & (mass > 0.0)
    grid_w = np.zeros(n_group * n_freq, dtype=np.float64)
    grid_sum = np.zeros(n_group * n_freq, dtype=np.complex128)
    np.add.at(grid_w, slot, np.where(finite, mass, 0.0))
    np.add.at(grid_sum, slot, np.where(finite, mass * samples, 0.0))
    grid = np.where(grid_w > 0.0, grid_sum / np.maximum(grid_w, 1.0e-30), 1.0 + 0.0j)
    grid = grid.reshape(n_group, n_freq)
    ok = grid_w.reshape(n_group, n_freq) > 0.0
    phases = np.angle(grid)
    if n_freq >= 2:
        delta = np.diff(phases, axis=1)
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        unwrapped = np.concatenate(
            [phases[:, :1], phases[:, :1] + np.cumsum(delta, axis=1)],
            axis=1,
        )
    else:
        unwrapped = phases
    mask = ok.astype(np.float64)
    sum_w = np.sum(mask, axis=1)
    sum_wx = mask @ freq_values
    sum_wy = np.sum(mask * unwrapped, axis=1)
    sum_wxx = mask @ (freq_values * freq_values)
    sum_wxy = np.sum(mask * unwrapped * freq_values, axis=1)
    denom = sum_w * sum_wxx - sum_wx * sum_wx
    slope = np.where(np.abs(denom) > 0.0, (sum_w * sum_wxy - sum_wx * sum_wy) / denom, 0.0)
    intercept = np.where(sum_w > 0.0, (sum_wy - slope * sum_wx) / np.maximum(sum_w, 1.0e-30), 0.0)
    model = intercept[group_inverse] + slope[group_inverse] * freq
    return samples * np.exp(-1j * model)


def recovered_leakage_from_artifact(
    artifact: HolographyFullJonesArtifact,
    *,
    off_axis_only: bool = False,
    on_axis_atol_rad: float = 1.0e-6,
) -> dict[str, float]:
    """Coherent feed-frame leakage of an unmasked or aggregated artifact."""

    d_rl, d_lr = feed_frame_leakage_ratios(artifact.jones)
    weight = np.asarray([sample.weight for sample in artifact.samples], dtype=np.float64)
    valid = np.asarray(artifact.off_diagonal_valid, dtype=bool)
    if off_axis_only:
        radius = np.asarray(
            [
                np.hypot(float(sample.offset_lm_rad[0]), float(sample.offset_lm_rad[1]))
                for sample in artifact.samples
            ],
            dtype=np.float64,
        )
        keep = radius > float(on_axis_atol_rad)
        d_rl = d_rl[keep]
        d_lr = d_lr[keep]
        weight = weight[keep]
        valid = valid[keep]
    mean_rl = coherent_mean(d_rl, weight)
    mean_lr = coherent_mean(d_lr, weight)
    return {
        "coherent_abs_rl": float(np.abs(mean_rl)),
        "coherent_abs_lr": float(np.abs(mean_lr)),
        "median_abs_rl": float(np.nanmedian(np.abs(d_rl))),
        "median_abs_lr": float(np.nanmedian(np.abs(d_lr))),
        "n_samples": int(d_rl.size),
        "n_off_diagonal_valid": int(np.sum(valid)),
        "off_diagonal_fraction": float(np.mean(valid)) if valid.size else float("nan"),
    }


def injection_recovery_point(
    *,
    injected_amplitude: float,
    recovered: Mapping[str, float],
    heldout_scores: Mapping[str, object] | None = None,
    baseline_abs: float = 0.0,
    match_fraction: float = 0.5,
) -> dict[str, object]:
    """One amplitude on the detection curve.

    Recovery is the increment above the uninjected baseline, not the
    absolute coherent mean. A constant leak already in the data must
    not count as detecting a smaller injection.
    """

    injected = float(injected_amplitude)
    recovered_abs = float(recovered.get("coherent_abs_rl") or np.nan)
    increment = recovered_abs - float(baseline_abs)
    recovered_ok = bool(
        np.isfinite(recovered_abs)
        and injected > 0.0
        and increment >= float(match_fraction) * injected
    )
    main = {} if heldout_scores is None else dict(heldout_scores)
    return {
        "injected_amplitude": injected,
        "baseline_abs_rl": float(baseline_abs),
        "recovered_increment_abs_rl": float(increment),
        "recovered_coherent_abs_rl": recovered_abs,
        "recovered_coherent_abs_lr": float(recovered.get("coherent_abs_lr") or np.nan),
        "recovered_median_abs_rl": float(recovered.get("median_abs_rl") or np.nan),
        "n_off_diagonal_valid": recovered.get("n_off_diagonal_valid"),
        "off_diagonal_fraction": recovered.get("off_diagonal_fraction"),
        "recovered": recovered_ok,
        "rl_improved": main.get("rl_improved"),
        "lr_improved": main.get("lr_improved"),
        "rr_ll_regression": main.get("rr_ll_regression"),
        "full_median_abs_rl_over_i": main.get("full_median_abs_rl_over_i"),
        "diagonal_median_abs_rl_over_i": main.get("diagonal_median_abs_rl_over_i"),
        "full_median_abs_lr_over_i": main.get("full_median_abs_lr_over_i"),
        "diagonal_median_abs_lr_over_i": main.get("diagonal_median_abs_lr_over_i"),
    }


def evaluate_cassbeam_jones_at_offsets(
    offset_lm_rad: ArrayLike,
    frequency_hz: ArrayLike,
    parallactic_angle_rad: ArrayLike,
    *,
    off_diagonal: bool,
    calibration_state: BeamCalibrationState | str = "casa_parang_true",
) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
    """Vectorized CASSBEAM lookup at holography commanded offsets.

    The source is at the phase centre. ``offset_lm_rad`` is the mover
    commanded pointing. Production full Jones stays unfrozen.
    """

    state = require_beam_calibration_state(calibration_state)
    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    chi = np.asarray(parallactic_angle_rad, dtype=np.float64).reshape(-1)
    if frequency.size == 1:
        frequency = np.full(offset.shape[0], float(frequency[0]), dtype=np.float64)
    if chi.size == 1:
        chi = np.full(offset.shape[0], float(chi[0]), dtype=np.float64)
    if frequency.size != offset.shape[0] or chi.size != offset.shape[0]:
        raise ValueError("offset, frequency_hz, and parallactic_angle_rad must match")
    artifact = load_cassbeam_cband_artifact()
    l_off = -offset[:, 0]
    m_off = -offset[:, 1]
    cosine = np.cos(chi)
    sine = np.sin(chi)
    l_ant = l_off * cosine + m_off * sine
    m_ant = -l_off * sine + m_off * cosine
    jones = np.zeros((offset.shape[0], 2, 2), dtype=np.complex128)
    valid = np.zeros(offset.shape[0], dtype=bool)
    freq_keys = np.rint(frequency).astype(np.int64)
    for key in np.unique(freq_keys):
        members = np.flatnonzero(freq_keys == key)
        table = _nearest_table(artifact.tables, float(frequency[members[0]]))
        plane, ok = _bilinear_jones(table, l_ant[members], m_ant[members])
        if not off_diagonal:
            plane = _diagonal_only(plane)
        plane = _normalize_on_axis(plane, table, off_diagonal=off_diagonal)
        para = circular_parallactic_jones(chi[members])
        if state is BeamCalibrationState.CASA_PARANG_TRUE:
            conjugate = np.conjugate(np.swapaxes(para, -1, -2))
            plane = conjugate @ plane @ para
        elif state is BeamCalibrationState.UNCALIBRATED:
            plane = plane @ para
        else:
            raise ValueError(f"unsupported beam calibration state {state!r}")
        jones[members] = plane
        valid[members] = ok
    return jones, valid


def cassbeam_predicted_leakage_amplitude(
    offset_lm_rad: ArrayLike,
    frequency_hz: ArrayLike,
    parallactic_angle_rad: ArrayLike,
) -> dict[str, float]:
    """CASSBEAM |E_RL/E_RR| at the measured holography coordinates."""

    jones, valid = evaluate_cassbeam_jones_at_offsets(
        offset_lm_rad,
        frequency_hz,
        parallactic_angle_rad,
        off_diagonal=True,
    )
    d_rl, d_lr = feed_frame_leakage_ratios(jones)
    usable = valid & np.isfinite(d_rl) & np.isfinite(d_lr)
    return {
        "n": int(np.sum(usable)),
        "median_abs_rl_over_rr": float(np.median(np.abs(d_rl[usable])))
        if np.any(usable)
        else float("nan"),
        "median_abs_lr_over_ll": float(np.median(np.abs(d_lr[usable])))
        if np.any(usable)
        else float("nan"),
        "p90_abs_rl_over_rr": float(np.percentile(np.abs(d_rl[usable]), 90))
        if np.any(usable)
        else float("nan"),
    }


def predict_moving_reference_from_cassbeam(
    observation: HolographyObservation,
    *,
    residual_jones: Mapping[int, ArrayLike],
    row_mask: ArrayLike,
    parallactic_angle_rad: ArrayLike | None = None,
    off_diagonal: bool,
) -> NDArray[np.complex128]:
    """Predict mover–reference visibilities from CASSBEAM, not empirical maps."""

    import jax.numpy as jnp

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    source = np.asarray(_source_coherency(observation), dtype=np.complex128)
    packed = pack_coherency(
        observation.block.visibility, observation.block.correlations, _RECEPTORS
    )
    predicted = np.full(packed.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = np.full(offsets.shape[:2], "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return predicted
    time_index = np.asarray(inverse, dtype=np.int32)[rows]
    antenna_p = np.asarray(observation.block.antenna1, dtype=np.int32)[rows]
    antenna_q = np.asarray(observation.block.antenna2, dtype=np.int32)[rows]
    keep = pointing_valid[time_index, antenna_p] & pointing_valid[time_index, antenna_q]
    if not bool(np.any(keep)):
        return predicted
    rows = rows[keep]
    time_index = time_index[keep]
    antenna_p = antenna_p[keep]
    antenna_q = antenna_q[keep]
    role_p = roles[time_index, antenna_p]
    moving = np.where(role_p == "moving", antenna_p, antenna_q).astype(np.int32)
    reference = np.where(role_p == "moving", antenna_q, antenna_p).astype(np.int32)
    offset = offsets[time_index, moving]
    n_chan = int(observation.block.frequency_hz.size)
    unique_times, _ = unique_visibility_times(observation.block.time_s)
    if parallactic_angle_rad is None:
        from sl1mjax.calibration_terms import parallactic_angle_rad as _chi

        chi_table = _chi(
            unique_times,
            observation.phase_centre_rad,
            np.asarray(observation.antenna_position_m, dtype=np.float64),
        )
    else:
        chi_table = np.asarray(parallactic_angle_rad, dtype=np.float64)
    chi = chi_table[time_index, moving]
    query = np.repeat(offset, n_chan, axis=0)
    freqs = np.broadcast_to(observation.block.frequency_hz, (moving.size, n_chan)).reshape(-1)
    chi_f = np.repeat(chi, n_chan)
    beams, ok = evaluate_cassbeam_jones_at_offsets(
        query,
        freqs,
        chi_f,
        off_diagonal=off_diagonal,
        calibration_state=observation.calibration_state,
    )
    beams = beams.reshape(moving.size, n_chan, 2, 2)
    ok = ok.reshape(moving.size, n_chan)
    if not off_diagonal:
        beams[..., 0, 1] = 0.0
        beams[..., 1, 0] = 0.0
    r_m = _antenna_jones_planes(residual_jones, moving)
    r_r = _antenna_jones_planes(residual_jones, reference)
    r_m = np.asarray(_sky_frame_residual_jax(jnp.asarray(r_m), jnp.asarray(chi)))
    r_r = np.asarray(
        _sky_frame_residual_jax(jnp.asarray(r_r), jnp.asarray(chi_table[time_index, reference]))
    )
    source_rows = source[rows]
    if source_rows.ndim == 3:
        source_rows = source_rows[:, None, :, :]
    vis = np.asarray(
        _predict_vis_jax(
            jnp.asarray(r_m),
            jnp.asarray(beams),
            jnp.asarray(source_rows),
            jnp.asarray(r_r),
            jnp.asarray(moving == antenna_p),
        )
    )
    predicted[rows] = np.where(ok[..., None, None], vis, predicted[rows])
    return predicted


def nearest_injection_amplitude(cassbeam_amplitude: float) -> float:
    """Injection grid point nearest the CASSBEAM-predicted leak."""

    if not np.isfinite(cassbeam_amplitude) or cassbeam_amplitude <= 0.0:
        return float(INJECTED_LEAKAGE_AMPLITUDES[-2])
    return float(
        min(INJECTED_LEAKAGE_AMPLITUDES, key=lambda amp: abs(amp - float(cassbeam_amplitude)))
    )


def classify_leakage_sensitivity(
    *,
    cassbeam_amplitude: float,
    injection_curve: Mapping[str, Mapping[str, object]],
    cassbeam_rl_improved: bool | None,
    cassbeam_lr_improved: bool | None,
    cassbeam_rr_ll_regression: bool | None,
    real_coherent_abs: float,
) -> dict[str, object]:
    """Turn the four diagnosis tests into a primary outcome.

    Full Jones stays unfrozen. SPW 5 stays closed.
    """

    nearest = nearest_injection_amplitude(cassbeam_amplitude)
    key = f"{nearest:.6g}"
    if key not in injection_curve:
        matches = [name for name in injection_curve if abs(float(name) - nearest) < 1.0e-12]
        key = matches[0] if matches else next(iter(injection_curve), key)
    point = dict(injection_curve.get(key) or {})
    injections_recovered = bool(point.get("recovered"))
    cassbeam_direct_improves = bool(
        cassbeam_rl_improved and cassbeam_lr_improved and not cassbeam_rr_ll_regression
    )
    real_absent = bool(
        np.isfinite(real_coherent_abs)
        and np.isfinite(cassbeam_amplitude)
        and cassbeam_amplitude > 0.0
        and real_coherent_abs < 0.5 * float(cassbeam_amplitude)
    )
    tests = {
        "injections_not_recovered_at_cassbeam_amplitude": not injections_recovered,
        "injections_recovered_real_leakage_absent": injections_recovered and real_absent,
        "cassbeam_direct_improves_crosshands": cassbeam_direct_improves,
        "cassbeam_direct_also_null": (not cassbeam_direct_improves)
        and (not injections_recovered or real_absent),
    }
    if tests["injections_not_recovered_at_cassbeam_amplitude"]:
        outcome = "injections_not_recovered_at_cassbeam_amplitude"
        interpretation = "This dataset/estimator lacks sensitivity at the CASSBEAM amplitude."
    elif tests["injections_recovered_real_leakage_absent"]:
        outcome = "injections_recovered_real_leakage_absent"
        interpretation = (
            "Injections recover but real leakage does not. The empirical "
            "data argue against that leakage amplitude or coherence."
        )
    elif tests["cassbeam_direct_improves_crosshands"]:
        outcome = "cassbeam_direct_improves_crosshands"
        interpretation = (
            "Direct CASSBEAM improves held-out RL/LR. The empirical estimator is the problem."
        )
    else:
        outcome = "cassbeam_direct_also_null"
        interpretation = (
            "Direct CASSBEAM also makes no measurable difference. SPW-4 "
            "THOL0001 cannot validate the off-diagonal model at the "
            "current residual floor."
        )
    return {
        "gate": LORO_LEAKAGE_SENSITIVITY,
        "status": "fail",
        "blocking": True,
        "outcome": outcome,
        "interpretation": interpretation,
        "tests": tests,
        "cassbeam_amplitude": float(cassbeam_amplitude)
        if np.isfinite(cassbeam_amplitude)
        else None,
        "nearest_injection_amplitude": nearest,
        "nearest_injection": point,
        "real_coherent_abs": float(real_coherent_abs) if np.isfinite(real_coherent_abs) else None,
        "full_jones_blocked": True,
        "full_jones_frozen": False,
        "spw5_closed": True,
        "most_important_next_artifact": "cassbeam_diagonal_low_order_correction",
        "notes": (
            LEAKAGE_SENSITIVITY_NOTE,
            FEED_FRAME_INJECTION_NOTE,
            AGGREGATION_BEFORE_SNR_NOTE,
            CASSBEAM_DIRECT_NOTE,
        ),
    }
