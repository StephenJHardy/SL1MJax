"""Publication statistics derived from the compact validation bundle.

Notebook cells must call these functions. They must not hard-code scientific
numbers or re-open the Measurement Set.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sl1mjax.beam_validation_claims import refuse_full_raster_squint
from sl1mjax.beam_validation_outputs import ValidationBundle

CLAIM_VERSION = "vla_c_band_beam_validation_v1"
MAIN_LOBE_ACCEPTED_MAX = 0.01
MID_BEAM_QUALIFIED_MAX = 0.12
REGION_SUPPORT_CLASS = {
    "main_lobe": "accepted",
    "mid": "qualified",
    "outer_diagnostic": "diagnostic",
}


def classify_diagonal_region_support(
    hand_residual_power: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, object]:
    """Region-qualified diagonal support. Not a single whole-raster label."""

    regions: dict[str, object] = {}
    for name, support in REGION_SUPPORT_CLASS.items():
        hands = hand_residual_power.get(name) or {}
        rr = float((hands.get("rr") or {}).get("residual_power", float("nan")))
        ll = float((hands.get("ll") or {}).get("residual_power", float("nan")))
        if name == "main_lobe":
            matches = bool(rr <= MAIN_LOBE_ACCEPTED_MAX and ll <= MAIN_LOBE_ACCEPTED_MAX)
        elif name == "mid":
            matches = bool(rr <= MID_BEAM_QUALIFIED_MAX and ll <= MID_BEAM_QUALIFIED_MAX)
        else:
            matches = True
        regions[name] = {
            "class": support,
            "residual_power_rr": rr,
            "residual_power_ll": ll,
            "matches_class": matches,
        }
    return {
        "main_lobe": "accepted",
        "mid_beam": "qualified",
        "outer_raster": "diagnostic",
        "regions": regions,
    }


def hand_metric(hands: Mapping[str, Mapping[str, Any]], hand: str, name: str) -> float:
    return float((hands.get(hand) or {}).get(name))


def residual_power_table(channel32: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Region-qualified RR/LL residual power from the HOLORASTER comparison."""

    regions = (channel32.get("regions") or {}).get("hand_residual_power") or {}
    out: dict[str, dict[str, float]] = {}
    for name in ("main_lobe", "mid", "outer_diagnostic", "all"):
        hands = regions.get(name) or {}
        out[name] = {
            "rr": hand_metric(hands, "rr", "residual_power"),
            "ll": hand_metric(hands, "ll", "residual_power"),
            "n_rr": float((hands.get("rr") or {}).get("n") or 0),
            "n_ll": float((hands.get("ll") or {}).get("n") or 0),
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
            "support_class": "accepted",
            "metric": {
                "rr_residual_power": residuals["main_lobe"]["rr"],
                "ll_residual_power": residuals["main_lobe"]["ll"],
                "threshold": 0.01,
            },
            "evidence": (
                "HOLORASTER channel-32 residual power RR "
                f"{_percent(residuals['main_lobe']['rr'])}, "
                f"LL {_percent(residuals['main_lobe']['ll'])}"
            ),
            "figures": ["F08", "F10", "F12"],
            "input_hashes": hashes,
            "validation_split": "HOLORASTER field 10, native channel 32",
        },
        {
            "id": "C02",
            "title": "Middle-beam CASSBEAM is useful but imperfect",
            "status": "warn" if support["mid_beam"] == "qualified" else "fail",
            "support_class": "qualified",
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
            },
            "evidence": (
                "HOLORASTER outer residual power RR "
                f"{_percent(residuals['outer_diagnostic']['rr'])}, "
                f"LL {_percent(residuals['outer_diagnostic']['ll'])}"
            ),
            "figures": ["F10", "F12"],
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
            "title": "CASSBEAM full Jones is an experimental non-detection",
            "status": "blocked",
            "support_class": "experimental_non_detection",
            "metric": {
                "rl_correlation": cross["rl_correlation"],
                "lr_correlation": cross["lr_correlation"],
                "rl_residual_power": cross["rl_residual_power"],
                "lr_residual_power": cross["lr_residual_power"],
            },
            "evidence": (
                f"RL/LR correlation {cross['rl_correlation']:.3f} / "
                f"{cross['lr_correlation']:.3f}; residual power "
                f"{cross['rl_residual_power']:.3f} / {cross['lr_residual_power']:.3f}; "
                "predicted signal below the observed cloud"
            ),
            "figures": ["F18"],
            "input_hashes": hashes,
            "validation_split": "HOLORASTER RL/LR versus experimental full Jones",
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
    )
    return {
        "version": CLAIM_VERSION,
        "diagonal_reference": "cassbeam_diagonal_cband_reference",
        "full_jones": "experimental_non_detection",
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
