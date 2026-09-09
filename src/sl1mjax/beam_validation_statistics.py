"""Publication statistics derived from the compact validation bundle.

Notebook cells must call these functions. They must not hard-code scientific
numbers or re-open the Measurement Set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_validation_claims import refuse_full_raster_squint
from sl1mjax.beam_validation_outputs import ValidationBundle
from sl1mjax.holography import (
    CASA_SETJY_3C147_C_IM_MODEL_DATA_JY_SPW4_CH32,
    Memo195LowerCRaster,
)
from sl1mjax.holography_alignment import (
    apparent_voltage_response,
    voltage_response_region_masks,
)

CLAIM_VERSION = "vla_c_band_beam_validation_v1"
MAIN_LOBE_ACCEPTED_MAX = 0.01
MID_BEAM_QUALIFIED_MAX = 0.12
REGION_SUPPORT_CLASS = {
    "main_lobe": "accepted",
    "mid": "qualified",
    "outer_diagnostic": "diagnostic",
}


def _threshold_support_class(name: str, matches: bool, intended: str) -> str:
    if matches:
        return intended
    if name == "main_lobe":
        return "rejected"
    if name == "mid":
        return "unqualified"
    return intended


def classify_diagonal_region_support(
    hand_residual_power: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, object]:
    """Region-qualified diagonal support. Not a single whole-raster label."""

    regions: dict[str, object] = {}
    for name, intended in REGION_SUPPORT_CLASS.items():
        hands = hand_residual_power.get(name) or {}
        rr = float((hands.get("rr") or {}).get("residual_power", float("nan")))
        ll = float((hands.get("ll") or {}).get("residual_power", float("nan")))
        if name == "main_lobe":
            matches = bool(
                np.isfinite(rr)
                and np.isfinite(ll)
                and rr <= MAIN_LOBE_ACCEPTED_MAX
                and ll <= MAIN_LOBE_ACCEPTED_MAX
            )
        elif name == "mid":
            matches = bool(
                np.isfinite(rr)
                and np.isfinite(ll)
                and rr <= MID_BEAM_QUALIFIED_MAX
                and ll <= MID_BEAM_QUALIFIED_MAX
            )
        else:
            matches = True
        actual = _threshold_support_class(name, matches, intended)
        regions[name] = {
            "class": actual,
            "intended_class": intended,
            "residual_power_rr": rr,
            "residual_power_ll": ll,
            "matches_class": matches,
        }
    return {
        "main_lobe": str(regions["main_lobe"]["class"]),
        "mid_beam": str(regions["mid"]["class"]),
        "outer_raster": str(regions["outer_diagnostic"]["class"]),
        "regions": regions,
    }


def visibility_hand_weight(weight: ArrayLike, row: int, col: int) -> NDArray[np.float64]:
    """Per-sample weight for one visibility hand. Does not reuse RR for LL."""

    wgt = np.asarray(weight, dtype=np.float64)
    if wgt.ndim == 4:
        return wgt[:, 0, row, col]
    if wgt.ndim == 3:
        return wgt[:, row, col]
    if wgt.ndim == 1:
        return wgt
    raise ValueError("weight must be (sample,), (sample, 2, 2), or (sample, 1, 2, 2)")


def hand_metric(hands: Mapping[str, Mapping[str, Any]], hand: str, name: str) -> float:
    return float((hands.get(hand) or {}).get(name))


def channel32_source_i_jy() -> float:
    """HOLORASTER channel-32 source I from the locked CASA MODEL_DATA value."""

    return float(CASA_SETJY_3C147_C_IM_MODEL_DATA_JY_SPW4_CH32)


def rms_visibility_error(residual_power: float) -> float:
    """Amplitude-equivalent RMS of a residual-power score: sqrt(L)."""

    value = float(residual_power)
    return math.sqrt(value) if value >= 0.0 and math.isfinite(value) else float("nan")


def residual_power_table(channel32: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Region-qualified RR/LL residual power and amplitude-equivalent errors."""

    scored = channel32.get("regions") or {}
    regions = scored.get("hand_residual_power") or {}
    out: dict[str, dict[str, float]] = {}
    for name in ("main_lobe", "mid", "outer_diagnostic", "all"):
        hands = regions.get(name) or {}
        region = scored.get(name) or {}
        rr = hand_metric(hands, "rr", "residual_power")
        ll = hand_metric(hands, "ll", "residual_power")
        out[name] = {
            "rr": rr,
            "ll": ll,
            "rr_rms": rms_visibility_error(rr),
            "ll_rms": rms_visibility_error(ll),
            "rr_median_abs_over_i": float(region.get("median_abs_rr_over_i", float("nan"))),
            "ll_median_abs_over_i": float(region.get("median_abs_ll_over_i", float("nan"))),
            "rr_median_abs_jy": float(region.get("median_abs_rr_jy", float("nan"))),
            "ll_median_abs_jy": float(region.get("median_abs_ll_jy", float("nan"))),
            "n_rr": float((hands.get("rr") or {}).get("n") or 0),
            "n_ll": float((hands.get("ll") or {}).get("n") or 0),
            "rr_correlation": hand_metric(hands, "rr", "correlation_abs"),
            "ll_correlation": hand_metric(hands, "ll", "correlation_abs"),
            "rr_median_abs_ratio": hand_metric(hands, "rr", "median_abs_ratio"),
            "ll_median_abs_ratio": hand_metric(hands, "ll", "median_abs_ratio"),
        }
    return out


def copolar_slope(channel32: Mapping[str, Any], hand: str) -> complex:
    row = (channel32.get("diagonal") or {}).get(hand) or {}
    return complex(float(row["slope_real"]), float(row["slope_imag"]))


def copolar_correlation(channel32: Mapping[str, Any], hand: str) -> float:
    return hand_metric(channel32.get("diagonal") or {}, hand, "correlation_abs")


def crosshand_non_detection(channel32: Mapping[str, Any]) -> dict[str, float]:
    experimental = channel32.get("experimental_full_jones") or {}
    return {
        "rl_correlation": hand_metric(experimental, "rl", "correlation_abs"),
        "lr_correlation": hand_metric(experimental, "lr", "correlation_abs"),
        "rl_residual_power": hand_metric(experimental, "rl", "residual_power"),
        "lr_residual_power": hand_metric(experimental, "lr", "residual_power"),
        "rl_median_abs_obs": hand_metric(experimental, "rl", "median_abs_obs"),
        "lr_median_abs_obs": hand_metric(experimental, "lr", "median_abs_obs"),
        "rl_median_abs_pred": hand_metric(experimental, "rl", "median_abs_pred"),
        "lr_median_abs_pred": hand_metric(experimental, "lr", "median_abs_pred"),
    }


def publication_squint_pair(squint: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    measured = dict(squint["measured"])
    cassbeam = dict(squint["cassbeam"])
    refuse_full_raster_squint(measured)
    refuse_full_raster_squint(cassbeam)
    return {"measured": measured, "cassbeam": cassbeam}


def offset_ring_diagonal_closure(offset_ring: Mapping[str, Any]) -> dict[str, Any]:
    closure = offset_ring.get("closure") or {}
    return {
        "rr": float((closure.get("rr") or {}).get("relative_power")),
        "ll": float((closure.get("ll") or {}).get("relative_power")),
        "passed": bool(closure.get("passed")),
        "n_rr": int((closure.get("rr") or {}).get("n") or 0),
        "n_ll": int((closure.get("ll") or {}).get("n") or 0),
        "radius_arcmin": float(offset_ring.get("radius_arcmin", float("nan"))),
        "q_u_is_frequency_holdout": bool(offset_ring.get("q_u_is_frequency_holdout", False)),
    }


def frequency_copolar_series(frequency: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-channel copolar scores. Frequency squint is not a publication series."""

    rows = []
    for item in frequency.get("channels") or ():
        diagonal = item.get("diagonal") or {}
        rows.append(
            {
                "channel": int(item["channel"]),
                "frequency_hz": float(item["frequency_hz"]),
                "rr_correlation": hand_metric(diagonal, "rr", "correlation_abs"),
                "ll_correlation": hand_metric(diagonal, "ll", "correlation_abs"),
                "rr_residual_power": hand_metric(diagonal, "rr", "residual_power"),
                "ll_residual_power": hand_metric(diagonal, "ll", "residual_power"),
                "rl_correlation": hand_metric(
                    item.get("experimental_full_jones") or {}, "rl", "correlation_abs"
                ),
                "lr_correlation": hand_metric(
                    item.get("experimental_full_jones") or {}, "lr", "correlation_abs"
                ),
                "rl_residual_power": hand_metric(
                    item.get("experimental_full_jones") or {}, "rl", "residual_power"
                ),
                "lr_residual_power": hand_metric(
                    item.get("experimental_full_jones") or {}, "lr", "residual_power"
                ),
                "region": str(item.get("region") or "main_lobe"),
                "n_main_lobe": int(item.get("n_main_lobe") or 0),
                "n_development": int(item.get("n_development") or 0),
            }
        )
    return rows


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _complex_text(value: complex) -> str:
    sign = "+" if value.imag >= 0.0 else "-"
    return f"{value.real:.3f}{sign}{abs(value.imag):.3f}i"


def build_claims(bundle: ValidationBundle) -> dict[str, Any]:
    """Machine-readable claim registry bound to this bundle's hashes."""

    channel32 = bundle.holoraster_channel32
    residuals = residual_power_table(channel32)
    support = classify_diagonal_region_support(
        {
            name: {
                "rr": {"residual_power": residuals[name]["rr"]},
                "ll": {"residual_power": residuals[name]["ll"]},
            }
            for name in ("main_lobe", "mid", "outer_diagnostic")
        }
    )
    squint = publication_squint_pair(bundle.squint)
    ring = offset_ring_diagonal_closure(bundle.offset_ring)
    cross = crosshand_non_detection(channel32)
    coordinate_feed = bundle.coordinate_feed_comparison
    coordinate_contract = coordinate_feed.get("coordinate_contract") or {}
    paired = coordinate_feed.get("paired_scores") or {}
    feed_effect = paired.get("evla_c_source_lm_vs_generic_source_lm") or {}
    feed_spatial = feed_effect.get("spatial") or {}
    feed_moving = feed_effect.get("moving") or {}
    feed_effect_passes = bool(feed_spatial.get("improves") and feed_moving.get("improves"))
    classification = dict(channel32.get("classification") or {})
    outcome = str(classification.get("outcome") or "")
    if outcome == "supported_on_spw4_development":
        c05_status = "warn"
        c05_title = "EVLA-C full Jones is supported only on SPW-4 development partitions"
        c05_class = "supported_on_spw4_development"
        c05_evidence = (
            "Paired RL/LR intervals improved on both spatial and mover partitions "
            "with no main-lobe copolar regression. This is not a production freeze."
        )
    elif outcome == "disfavoured_on_tested_support":
        c05_status = "warn"
        c05_title = "EVLA-C full Jones is disfavoured on the tested SPW-4 support"
        c05_class = "disfavoured_on_tested_support"
        c05_evidence = (
            "Unit-model injections were recoverable, but the real-data paired "
            "RL/LR intervals lie above zero on both development partitions."
        )
    elif outcome == "inconclusive_sensitivity":
        c05_status = "warn"
        c05_title = "EVLA-C full Jones is inconclusive on these SPW-4 observations"
        c05_class = "inconclusive_sensitivity"
        c05_evidence = (
            "The experiment cannot distinguish the unit full-Jones prediction "
            "from the diagonal at the predeclared sensitivity."
        )
    elif outcome == "blocked_implementation_or_contract":
        c05_status = "blocked"
        c05_title = "EVLA-C full Jones is blocked by a numerical or contract failure"
        c05_class = "blocked_implementation_or_contract"
        c05_evidence = str(classification.get("reason") or "numerical or provenance failure")
    else:
        c05_status = "blocked"
        c05_title = "Historical generic CASSBEAM full Jones is an experimental non-detection"
        c05_class = "experimental_non_detection"
        c05_evidence = (
            f"RL/LR correlation {cross['rl_correlation']:.3f} / "
            f"{cross['lr_correlation']:.3f}; residual power "
            f"{cross['rl_residual_power']:.3f} / {cross['lr_residual_power']:.3f}; "
            "this v1 panel has not been promoted as EVLA-C evidence"
        )
    files = dict(bundle.manifest.get("files") or {})
    hashes = {
        "holoraster_channel32": files.get("holoraster_channel32.json"),
        "squint_publication": files.get("squint_publication.json"),
        "offset_ring": files.get("offset_ring.json"),
        "manifest": bundle.manifest.get("bundle_sha256"),
    }
    claims = (
        {
            "id": "C01",
            "title": "CASSBEAM is the SPW-4 diagonal C-band reference in the main lobe",
            "status": "pass" if support["main_lobe"] == "accepted" else "fail",
            "support_class": support["main_lobe"],
            "metric": {
                "rr_residual_power": residuals["main_lobe"]["rr"],
                "ll_residual_power": residuals["main_lobe"]["ll"],
                "rr_rms": residuals["main_lobe"]["rr_rms"],
                "ll_rms": residuals["main_lobe"]["ll_rms"],
                "rr_median_abs_over_i": residuals["main_lobe"]["rr_median_abs_over_i"],
                "ll_median_abs_over_i": residuals["main_lobe"]["ll_median_abs_over_i"],
                "threshold": 0.01,
            },
            "evidence": (
                "HOLORASTER channel-32 residual power RR "
                f"{_percent(residuals['main_lobe']['rr'])}, "
                f"LL {_percent(residuals['main_lobe']['ll'])} "
                f"(RMS visibility {_percent(residuals['main_lobe']['rr_rms'])} / "
                f"{_percent(residuals['main_lobe']['ll_rms'])}; "
                "median |ΔV|/I "
                f"{_percent(residuals['main_lobe']['rr_median_abs_over_i'])} / "
                f"{_percent(residuals['main_lobe']['ll_median_abs_over_i'])})"
            ),
            "figures": ["F05", "F08", "F10", "F12", "F15"],
            "input_hashes": hashes,
            "validation_split": "HOLORASTER field 10, native channel 32",
        },
        {
            "id": "C02",
            "title": "Middle-beam CASSBEAM is useful but imperfect",
            "status": "warn" if support["mid_beam"] == "qualified" else "fail",
            "support_class": support["mid_beam"],
            "metric": {
                "rr_residual_power": residuals["mid"]["rr"],
                "ll_residual_power": residuals["mid"]["ll"],
                "threshold": 0.12,
            },
            "evidence": (
                f"HOLORASTER mid-beam residual power RR {_percent(residuals['mid']['rr'])}, "
                f"LL {_percent(residuals['mid']['ll'])}"
            ),
            "figures": ["F10", "F12"],
            "input_hashes": hashes,
            "validation_split": "HOLORASTER field 10, mid-beam mask",
        },
        {
            "id": "C03",
            "title": "Outer-raster CASSBEAM is diagnostic only",
            "status": "warn",
            "support_class": "diagnostic",
            "metric": {
                "rr_residual_power": residuals["outer_diagnostic"]["rr"],
                "ll_residual_power": residuals["outer_diagnostic"]["ll"],
                "rr_correlation": residuals["outer_diagnostic"]["rr_correlation"],
                "ll_correlation": residuals["outer_diagnostic"]["ll_correlation"],
                "rr_median_abs_ratio": residuals["outer_diagnostic"]["rr_median_abs_ratio"],
                "ll_median_abs_ratio": residuals["outer_diagnostic"]["ll_median_abs_ratio"],
            },
            "evidence": (
                "HOLORASTER outer residual power RR "
                f"{_percent(residuals['outer_diagnostic']['rr'])}, "
                f"LL {_percent(residuals['outer_diagnostic']['ll'])}; "
                "complex correlation "
                f"{residuals['outer_diagnostic']['rr_correlation']:.3f} / "
                f"{residuals['outer_diagnostic']['ll_correlation']:.3f}; "
                "median |V| ratio "
                f"{residuals['outer_diagnostic']['rr_median_abs_ratio']:.2f} / "
                f"{residuals['outer_diagnostic']['ll_median_abs_ratio']:.2f}. "
                "Outer phase is exploratory and is not a claim metric."
            ),
            "figures": ["F10", "F12", "F20", "F21", "F22", "F23"],
            "input_hashes": hashes,
            "validation_split": "HOLORASTER field 10, outer-raster mask",
        },
        {
            "id": "C04",
            "title": "The C147-* offset ring independently supports the diagonal beam",
            "status": "pass" if ring["passed"] else "fail",
            "support_class": "accepted",
            "metric": {
                "rr_residual_power": ring["rr"],
                "ll_residual_power": ring["ll"],
                "radius_arcmin": ring["radius_arcmin"],
            },
            "evidence": (
                f"{ring['radius_arcmin']:.2f}′ ring RR/LL residual power "
                f"{_percent(ring['rr'])} / {_percent(ring['ll'])} after the geometric fringe"
            ),
            "figures": ["F17"],
            "input_hashes": hashes,
            "validation_split": "C147-* fields 1–8; unused in calibration and beam fitting",
            "limitations": (
                "The already-written ring Q/U used all 64 channels and is not a frequency holdout",
            ),
        },
        {
            "id": "C05",
            "title": c05_title,
            "status": c05_status,
            "support_class": c05_class,
            "metric": {
                "rl_correlation": cross["rl_correlation"],
                "lr_correlation": cross["lr_correlation"],
                "rl_residual_power": cross["rl_residual_power"],
                "lr_residual_power": cross["lr_residual_power"],
                "outcome": outcome or None,
            },
            "evidence": c05_evidence,
            "figures": ["F18"],
            "input_hashes": hashes,
            "validation_split": "HOLORASTER RL/LR versus experimental full Jones",
            "limitations": (
                "SPW-4 development partitions only; residual Jones is the "
                "channel-32 field-9 plane applied to all publication channels",
            ),
        },
        {
            "id": "C06",
            "title": "SPW 5 remains sealed",
            "status": "not_run",
            "support_class": "frequency_transfer",
            "metric": {},
            "evidence": (
                "Reserved for one diagonal frequency-transfer test after a "
                "frozen SPW-4 correction"
            ),
            "figures": ["F19"],
            "input_hashes": hashes,
            "validation_split": "SPW 5 sealed",
        },
        {
            "id": "C07",
            "title": "Publication squint uses the 20%-of-peak main-lobe estimator",
            "status": "pass",
            "support_class": "squint",
            "metric": {
                "measured_arcmin": float(squint["measured"]["separation_arcmin"]),
                "cassbeam_arcmin": float(squint["cassbeam"]["separation_arcmin"]),
                "memo195_arcmin": float(squint["measured"]["memo195_separation_arcmin"]),
                "full_raster_measured_arcmin": float(
                    squint["measured"]["full_raster_separation_arcmin"]
                ),
            },
            "evidence": (
                f"Channel 32 measured {float(squint['measured']['separation_arcmin']):.3f}′, "
                f"Memo 195 {float(squint['measured']['memo195_separation_arcmin']):.3f}′, "
                f"CASSBEAM {float(squint['cassbeam']['separation_arcmin']):.3f}′. "
                "The superseded full-raster measured centroid is retained only as a bias diagnostic"
            ),
            "figures": ["F13"],
            "input_hashes": hashes,
            "validation_split": "HOLORASTER channel 32, 20%-of-peak main-lobe mask",
        },
        {
            "id": "C08",
            "title": "HOLORASTER beam queries use source-in-feed coordinates",
            "status": "pass"
            if float(coordinate_feed.get("commanded_source_max_abs_error_rad", float("inf")))
            <= 1.0e-12
            else "fail",
            "support_class": "coordinate_contract",
            "metric": {
                "max_abs_source_plus_commanded_rad": coordinate_feed.get(
                    "commanded_source_max_abs_error_rad"
                ),
                "axis_map": coordinate_contract.get("axis_map"),
            },
            "evidence": (
                "POINTING_OFFSET is the commanded AZELGEO displacement; a source at "
                "the phase centre is queried at its negative in the feed frame"
            ),
            "figures": ["F26", "F28", "F29"],
            "input_hashes": hashes,
            "validation_split": "Geometry contract; no visibility model selected",
        },
        {
            "id": "C09",
            "title": "EVLA-C feed parameters improve SPW-4 development holdouts",
            "status": "pass" if feed_effect_passes else "fail",
            "support_class": "development_only",
            "metric": {
                "spatial_delta": feed_spatial.get("delta"),
                "spatial_delta_lo": feed_spatial.get("delta_lo"),
                "spatial_delta_hi": feed_spatial.get("delta_hi"),
                "mover_delta": feed_moving.get("delta"),
                "mover_delta_lo": feed_moving.get("delta_lo"),
                "mover_delta_hi": feed_moving.get("delta_hi"),
                "evla_c_squint_arcmin": (
                    coordinate_feed.get("evla_plane_centroids") or {}
                ).get("separation_arcmin"),
            },
            "evidence": (
                "Against generic VLA at the same source-in-beam coordinates, "
                f"spatial ΔL {float(feed_spatial.get('delta', float('nan'))):+.4f} "
                f"and mover ΔL {float(feed_moving.get('delta', float('nan'))):+.4f}; "
                "both paired 95% intervals are below zero"
            ),
            "figures": ["F26", "F27", "F28", "F29"],
            "input_hashes": hashes,
            "validation_split": "Frozen SPW-4 spatial and mover development holdouts",
            "limitations": (
                "One frequency only; production beam is not frozen and SPW 5 remains sealed",
            ),
        },
    )
    return {
        "version": str(bundle.manifest.get("publication_version") or CLAIM_VERSION),
        "diagonal_reference": "evla_c_source_lm_spw4_development",
        "full_jones": outcome or "not_retested_after_coordinate_feed_update",
        "spw5": "sealed_diagonal_frequency_transfer",
        "convention_search": False,
        "rr_slope": _complex_text(copolar_slope(channel32, "rr")),
        "ll_slope": _complex_text(copolar_slope(channel32, "ll")),
        "rr_correlation": copolar_correlation(channel32, "rr"),
        "ll_correlation": copolar_correlation(channel32, "ll"),
        "diagonal_support": support,
        "claims": list(claims),
    }


def claim_table(bundle: ValidationBundle) -> list[dict[str, Any]]:
    registry = bundle.claims if bundle.claims.get("claims") else build_claims(bundle)
    return list(registry["claims"])


def claim_by_id(bundle: ValidationBundle, claim_id: str) -> dict[str, Any]:
    for item in claim_table(bundle):
        if item["id"] == claim_id:
            return item
    raise KeyError(claim_id)


ARC_MIN_PER_RAD = 180.0 * 60.0 / np.pi
SPATIAL_MAP_BINS = 51


def binned_complex_map(
    offset_lm_rad: ArrayLike,
    values: ArrayLike,
    weight: ArrayLike,
    *,
    n_bin: int = SPATIAL_MAP_BINS,
) -> dict[str, NDArray]:
    """Weighted-mean complex map on a regular (l, m) grid. Vectorized."""

    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    val = np.asarray(values, dtype=np.complex128).reshape(-1)
    wgt = np.asarray(weight, dtype=np.float64).reshape(-1)
    if offset.shape[0] != val.size:
        raise ValueError("offset and values must share the sample axis")
    l_arcmin = offset[:, 0] * ARC_MIN_PER_RAD
    m_arcmin = offset[:, 1] * ARC_MIN_PER_RAD
    finite = (
        np.isfinite(l_arcmin)
        & np.isfinite(m_arcmin)
        & np.isfinite(val)
        & np.isfinite(wgt)
        & (wgt > 0.0)
    )
    span = float(np.nanmax(np.abs(np.concatenate([l_arcmin[finite], m_arcmin[finite]]))))
    span = max(span, 1.0)
    edges = np.linspace(-span, span, int(n_bin) + 1)
    ix = np.clip(np.digitize(l_arcmin, edges) - 1, 0, n_bin - 1)
    iy = np.clip(np.digitize(m_arcmin, edges) - 1, 0, n_bin - 1)
    flat = iy * n_bin + ix
    count = np.zeros(n_bin * n_bin, dtype=np.float64)
    real = np.zeros(n_bin * n_bin, dtype=np.float64)
    imag = np.zeros(n_bin * n_bin, dtype=np.float64)
    np.add.at(count, flat[finite], wgt[finite])
    np.add.at(real, flat[finite], wgt[finite] * val[finite].real)
    np.add.at(imag, flat[finite], wgt[finite] * val[finite].imag)
    safe = count > 0.0
    mean = np.full(n_bin * n_bin, np.nan, dtype=np.complex128)
    mean[safe] = real[safe] / count[safe] + 1j * imag[safe] / count[safe]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "l_arcmin": centers,
        "m_arcmin": centers,
        "mean": mean.reshape(n_bin, n_bin),
        "weight": count.reshape(n_bin, n_bin),
    }


def _hand_plane(values: ArrayLike, row: int, col: int) -> NDArray:
    plane = np.asarray(values)
    if plane.ndim == 4:
        return plane[:, 0, row, col]
    if plane.ndim == 3:
        return plane[:, row, col]
    return plane


def scientific_voltage_masks(
    measured: ArrayLike,
    weight: ArrayLike,
    *,
    intensity_jy: float | None = None,
) -> dict[str, NDArray[np.bool_]]:
    """Main-lobe / mid / outer masks from V/I_model, not a peak-amplitude percentile."""

    intensity = float(channel32_source_i_jy() if intensity_jy is None else intensity_jy)
    wgt = np.asarray(weight)
    if wgt.ndim == 4:
        rr_ok = np.isfinite(wgt[:, 0, 0, 0]) & (wgt[:, 0, 0, 0] > 0.0)
        ll_ok = np.isfinite(wgt[:, 0, 1, 1]) & (wgt[:, 0, 1, 1] > 0.0)
    else:
        rr_ok = np.ones(np.asarray(measured).shape[0], dtype=bool)
        ll_ok = rr_ok
    voltage = apparent_voltage_response(measured, intensity, rr_ok, ll_ok)
    return voltage_response_region_masks(voltage)


def raster_family_from_offset(offset_lm_rad: ArrayLike) -> NDArray[np.int32]:
    """Dense (1) versus sparse (2) Memo 195 family. Pass-1/pass-2 occupancy proxy."""

    raster = Memo195LowerCRaster()
    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    l_arcmin = offset[:, 0] * ARC_MIN_PER_RAD
    m_arcmin = offset[:, 1] * ARC_MIN_PER_RAD
    half_d = (raster.dense_n - 1) // 2
    half_s = (raster.sparse_n - 1) // 2
    ix_d = np.clip(np.round(l_arcmin / raster.dense_spacing_arcmin), -half_d, half_d)
    iy_d = np.clip(np.round(m_arcmin / raster.dense_spacing_arcmin), -half_d, half_d)
    ix_s = np.clip(np.round(l_arcmin / raster.sparse_spacing_arcmin), -half_s, half_s)
    iy_s = np.clip(np.round(m_arcmin / raster.sparse_spacing_arcmin), -half_s, half_s)
    dist_d = np.hypot(
        l_arcmin - ix_d * raster.dense_spacing_arcmin,
        m_arcmin - iy_d * raster.dense_spacing_arcmin,
    )
    dist_s = np.hypot(
        l_arcmin - ix_s * raster.sparse_spacing_arcmin,
        m_arcmin - iy_s * raster.sparse_spacing_arcmin,
    )
    return np.where(dist_d <= dist_s, 1, 2).astype(np.int32)


def cell_average_vi(
    offset_lm_rad: ArrayLike,
    observed: ArrayLike,
    predicted: ArrayLike,
    choose: ArrayLike,
    *,
    intensity_jy: float,
    n_bin: int = 21,
) -> dict[str, NDArray]:
    """Spatial-cell means of V/I with within-cell standard deviations."""

    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    obs = np.asarray(observed, dtype=np.complex128).reshape(-1) / float(intensity_jy)
    pred = np.asarray(predicted, dtype=np.complex128).reshape(-1) / float(intensity_jy)
    mask = np.asarray(choose, dtype=bool).reshape(-1)
    l_arcmin = offset[:, 0] * ARC_MIN_PER_RAD
    m_arcmin = offset[:, 1] * ARC_MIN_PER_RAD
    finite = (
        mask
        & np.isfinite(obs)
        & np.isfinite(pred)
        & np.isfinite(l_arcmin)
        & np.isfinite(m_arcmin)
    )
    if not bool(np.any(finite)):
        empty = np.zeros(0, dtype=np.float64)
        return {
            "pred_abs": empty,
            "obs_abs": empty,
            "err_abs": empty,
            "pred_real": empty,
            "obs_real": empty,
            "err_real": empty,
            "n": empty,
        }
    span = float(np.nanmax(np.abs(np.concatenate([l_arcmin[finite], m_arcmin[finite]]))))
    span = max(span, 1.0)
    edges = np.linspace(-span, span, int(n_bin) + 1)
    ix = np.clip(np.digitize(l_arcmin, edges) - 1, 0, n_bin - 1)
    iy = np.clip(np.digitize(m_arcmin, edges) - 1, 0, n_bin - 1)
    flat = iy * n_bin + ix
    n_cell = n_bin * n_bin
    count = np.zeros(n_cell, dtype=np.float64)
    obs_abs = np.zeros(n_cell, dtype=np.float64)
    pred_abs = np.zeros(n_cell, dtype=np.float64)
    obs_abs2 = np.zeros(n_cell, dtype=np.float64)
    obs_re = np.zeros(n_cell, dtype=np.float64)
    pred_re = np.zeros(n_cell, dtype=np.float64)
    obs_re2 = np.zeros(n_cell, dtype=np.float64)
    np.add.at(count, flat[finite], 1.0)
    np.add.at(obs_abs, flat[finite], np.abs(obs[finite]))
    np.add.at(pred_abs, flat[finite], np.abs(pred[finite]))
    np.add.at(obs_abs2, flat[finite], np.abs(obs[finite]) ** 2)
    np.add.at(obs_re, flat[finite], obs[finite].real)
    np.add.at(pred_re, flat[finite], pred[finite].real)
    np.add.at(obs_re2, flat[finite], obs[finite].real ** 2)
    keep = count >= 4.0
    mean_obs_abs = obs_abs[keep] / count[keep]
    mean_pred_abs = pred_abs[keep] / count[keep]
    mean_obs_re = obs_re[keep] / count[keep]
    mean_pred_re = pred_re[keep] / count[keep]
    var_abs = np.maximum(obs_abs2[keep] / count[keep] - mean_obs_abs**2, 0.0)
    var_re = np.maximum(obs_re2[keep] / count[keep] - mean_obs_re**2, 0.0)
    return {
        "pred_abs": mean_pred_abs,
        "obs_abs": mean_obs_abs,
        "err_abs": np.sqrt(var_abs),
        "pred_real": mean_pred_re,
        "obs_real": mean_obs_re,
        "err_real": np.sqrt(var_re),
        "n": count[keep],
    }


def residual_group_summary(
    residual: ArrayLike,
    groups: ArrayLike,
    choose: ArrayLike,
) -> list[dict[str, float]]:
    """Median |residual| and count for each group label. Vectorized."""

    res = np.asarray(residual, dtype=np.complex128).reshape(-1)
    labels = np.asarray(groups).reshape(-1)
    mask = np.asarray(choose, dtype=bool).reshape(-1) & np.isfinite(res)
    if not bool(np.any(mask)):
        return []
    used = labels[mask]
    amp = np.abs(res[mask])
    unique, inverse = np.unique(used, return_inverse=True)
    rows: list[dict[str, float]] = []
    for index, label in enumerate(unique):
        values = amp[inverse == index]
        rows.append(
            {
                "id": int(label) if np.issubdtype(type(label), np.integer) else float(label),
                "n": int(values.size),
                "median_abs": float(np.median(values)),
            }
        )
    rows.sort(key=lambda item: item["median_abs"], reverse=True)
    return rows


PHASE_AMP_FLOOR_JY = 0.05
PHASE_WEIGHT_FRACTION = 0.1
RADIAL_EDGES_ARCMIN = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0)
BRIGHT_SOURCE_RADII_ARCMIN = (0.0, 10.0, 20.0, 40.0, 55.0, 65.0)


def amplitude_db(amp: ArrayLike, peak: float) -> NDArray[np.float64]:
    """Voltage amplitude in dB relative to ``peak``. Clipped at −40 dB."""

    value = np.abs(np.asarray(amp, dtype=np.complex128))
    scale = max(float(peak), 1.0e-12)
    return 20.0 * np.log10(np.maximum(value / scale, 1.0e-2))


def wrapped_phase_residual(measured: ArrayLike, predicted: ArrayLike) -> NDArray[np.float64]:
    """Wrapped phase residual arg(E_meas E_pred*)."""

    obs = np.asarray(measured, dtype=np.complex128)
    pred = np.asarray(predicted, dtype=np.complex128)
    return np.angle(obs * np.conjugate(pred))


def phase_valid_mask(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    *,
    amp_floor_jy: float = PHASE_AMP_FLOOR_JY,
    weight_fraction: float = PHASE_WEIGHT_FRACTION,
) -> NDArray[np.bool_]:
    """Amplitude-floor and occupancy mask. Null-adjacent cells are excluded."""

    obs = np.asarray(measured, dtype=np.complex128)
    pred = np.asarray(predicted, dtype=np.complex128)
    wgt = np.asarray(weight, dtype=np.float64)
    support = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(wgt) & (wgt > 0.0)
    occupied = wgt[support]
    floor = float(np.median(occupied)) * float(weight_fraction) if occupied.size else 0.0
    return (
        support
        & (np.abs(obs) >= float(amp_floor_jy))
        & (np.abs(pred) >= float(amp_floor_jy))
        & (wgt >= floor)
    )


def complex_visibility_score(
    observed: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike | None = None,
) -> dict[str, float]:
    """Weighted complex correlation, slope and residual power. Vectorized."""

    obs = np.asarray(observed, dtype=np.complex128).reshape(-1)
    pred = np.asarray(predicted, dtype=np.complex128).reshape(-1)
    if weight is None:
        wgt = np.ones(obs.shape, dtype=np.float64)
    else:
        wgt = np.asarray(weight, dtype=np.float64).reshape(-1)
    finite = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(wgt) & (wgt > 0.0)
    obs = obs[finite]
    pred = pred[finite]
    wgt = wgt[finite]
    empty = {
        "n": 0.0,
        "slope_real": float("nan"),
        "slope_imag": float("nan"),
        "correlation_abs": float("nan"),
        "residual_power": float("nan"),
        "median_abs_obs": float("nan"),
        "median_abs_pred": float("nan"),
        "median_abs_ratio": float("nan"),
        "median_phase_deg": float("nan"),
        "circular_phase_deg": float("nan"),
        "circular_phase_std_deg": float("nan"),
    }
    if obs.size == 0:
        return empty
    tt = float(np.sum(wgt * np.abs(pred) ** 2))
    rr = float(np.sum(wgt * np.abs(obs) ** 2))
    tr = complex(np.sum(wgt * np.conjugate(pred) * obs))
    residual = float(np.sum(wgt * np.abs(obs - pred) ** 2))
    slope = tr / tt if tt > 0.0 else complex(np.nan, np.nan)
    denom = math.sqrt(tt * rr) if tt > 0.0 and rr > 0.0 else float("nan")
    phase = wrapped_phase_residual(obs, pred)
    resultant = complex(np.sum(wgt * np.exp(1j * phase))) / float(np.sum(wgt))
    rho = float(np.abs(resultant))
    med_obs = float(np.median(np.abs(obs)))
    med_pred = float(np.median(np.abs(pred)))
    return {
        "n": float(obs.size),
        "slope_real": float(slope.real),
        "slope_imag": float(slope.imag),
        "correlation_abs": float(np.abs(tr) / denom) if math.isfinite(denom) else float("nan"),
        "residual_power": residual / rr if rr > 0.0 else float("nan"),
        "median_abs_obs": med_obs,
        "median_abs_pred": med_pred,
        "median_abs_ratio": med_obs / med_pred if med_pred > 0.0 else float("nan"),
        "median_phase_deg": float(np.degrees(np.median(phase))),
        "circular_phase_deg": float(np.degrees(np.angle(resultant))),
        "circular_phase_std_deg": (
            float(np.degrees(math.sqrt(-2.0 * math.log(rho)))) if 0.0 < rho < 1.0 else 0.0
        ),
    }


def radial_coherence(
    offset_lm_rad: ArrayLike,
    observed: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    choose: ArrayLike,
    *,
    edges_arcmin: Sequence[float] = RADIAL_EDGES_ARCMIN,
    amp_floor_jy: float = PHASE_AMP_FLOOR_JY,
) -> list[dict[str, float]]:
    """Annular complex scores. Loops only over the reduced radial bins."""

    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    obs = np.asarray(observed, dtype=np.complex128).reshape(-1)
    pred = np.asarray(predicted, dtype=np.complex128).reshape(-1)
    wgt = np.asarray(weight, dtype=np.float64).reshape(-1)
    mask = np.asarray(choose, dtype=bool).reshape(-1)
    radius = np.hypot(offset[:, 0], offset[:, 1]) * ARC_MIN_PER_RAD
    edges = np.asarray(tuple(edges_arcmin), dtype=np.float64)
    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        band = mask & (radius >= lo) & (radius < hi)
        phase_ok = phase_valid_mask(obs, pred, wgt, amp_floor_jy=amp_floor_jy)
        score = complex_visibility_score(obs[band], pred[band], wgt[band])
        phase = complex_visibility_score(
            obs[band & phase_ok], pred[band & phase_ok], wgt[band & phase_ok]
        )
        rows.append(
            {
                "r_lo_arcmin": float(lo),
                "r_hi_arcmin": float(hi),
                "r_mid_arcmin": float(0.5 * (lo + hi)),
                **score,
                "n_phase": phase["n"],
                "median_phase_deg": phase["median_phase_deg"],
                "circular_phase_deg": phase["circular_phase_deg"],
                "circular_phase_std_deg": phase["circular_phase_std_deg"],
            }
        )
    return rows


def map_axis_cut(
    grid: ArrayLike,
    l_arcmin: ArrayLike,
    m_arcmin: ArrayLike,
    *,
    kind: str,
) -> dict[str, NDArray]:
    """Nearest-axis or diagonal cut through a regular (l, m) map."""

    values = np.asarray(grid)
    l_ax = np.asarray(l_arcmin, dtype=np.float64).reshape(-1)
    m_ax = np.asarray(m_arcmin, dtype=np.float64).reshape(-1)
    i0 = int(np.argmin(np.abs(m_ax)))
    j0 = int(np.argmin(np.abs(l_ax)))
    if kind == "m0":
        return {"x_arcmin": l_ax, "value": values[i0, :]}
    if kind == "l0":
        return {"x_arcmin": m_ax, "value": values[:, j0]}
    if kind == "diag":
        index = np.arange(min(values.shape[0], values.shape[1]))
        return {"x_arcmin": l_ax[index], "value": values[index, index]}
    if kind == "antidiag":
        index = np.arange(min(values.shape[0], values.shape[1]))
        return {"x_arcmin": l_ax[index], "value": values[index, values.shape[1] - 1 - index]}
    raise ValueError(f"unknown cut {kind!r}")


def bright_source_examples(
    maps: Mapping[str, ArrayLike],
    *,
    radii_arcmin: Sequence[float] = BRIGHT_SOURCE_RADII_ARCMIN,
    source_i_jy: float,
) -> list[dict[str, object]]:
    """Array-average complex V at example radii. Not per-antenna E."""

    l_ax = np.asarray(maps["l_arcmin"], dtype=np.float64).reshape(-1)
    m_ax = np.asarray(maps["m_arcmin"], dtype=np.float64).reshape(-1)
    ll, mm = np.meshgrid(l_ax, m_ax, indexing="xy")
    radius = np.hypot(ll, mm)
    weight = np.asarray(maps["weight"], dtype=np.float64)
    rr_obs = np.asarray(maps["rr_measured"], dtype=np.complex128)
    rr_pred = np.asarray(maps["rr_cassbeam"], dtype=np.complex128)
    ll_obs = np.asarray(maps["ll_measured"], dtype=np.complex128)
    ll_pred = np.asarray(maps["ll_cassbeam"], dtype=np.complex128)
    examples: list[dict[str, object]] = []
    for target in radii_arcmin:
        supported = (weight > 0.0) & np.isfinite(rr_obs) & np.isfinite(rr_pred)
        if not bool(np.any(supported)):
            continue
        if float(target) <= 1.0e-6:
            choose = supported & (radius <= 1.5)
        else:
            choose = supported & (np.abs(radius - float(target)) <= 3.0) & (np.abs(mm) <= 4.0)
            if not bool(np.any(choose)):
                choose = supported & (np.abs(radius - float(target)) <= 4.0)
        if not bool(np.any(choose)):
            continue
        peak = np.argmax(np.where(choose, np.abs(rr_pred), -1.0))
        iy, ix = np.unravel_index(int(peak), rr_pred.shape)
        examples.append(
            {
                "label": "boresight" if float(target) <= 1.0e-6 else f"{float(target):.0f}′",
                "l_arcmin": float(ll[iy, ix]),
                "m_arcmin": float(mm[iy, ix]),
                "radius_arcmin": float(radius[iy, ix]),
                "source_i_jy": float(source_i_jy),
                "rr_meas_re": float(rr_obs[iy, ix].real),
                "rr_meas_im": float(rr_obs[iy, ix].imag),
                "rr_pred_re": float(rr_pred[iy, ix].real),
                "rr_pred_im": float(rr_pred[iy, ix].imag),
                "ll_meas_re": float(ll_obs[iy, ix].real),
                "ll_meas_im": float(ll_obs[iy, ix].imag),
                "ll_pred_re": float(ll_pred[iy, ix].real),
                "ll_pred_im": float(ll_pred[iy, ix].imag),
                "rr_stokes_i_factor_meas": float(np.abs(rr_obs[iy, ix] / source_i_jy) ** 2),
                "rr_stokes_i_factor_pred": float(np.abs(rr_pred[iy, ix] / source_i_jy) ** 2),
            }
        )
    return examples
