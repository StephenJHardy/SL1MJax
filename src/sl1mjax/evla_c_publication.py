"""Convert EVLA-C refresh exports into the v2 publication JSON shape.

Does not reopen the Measurement Set. Historical generic/commanded scores
stay labelled historical.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from sl1mjax.evla_c_metrics import HANDS, hand_scores
from sl1mjax.evla_c_validation_refresh import (
    PRIMARY_ARMS,
    PUBLICATION_BUNDLE_VERSION,
    write_json_atomic,
)

_HAND_LOWER = {name: name.lower() for name in HANDS}


def _hand_record(scores: Mapping[str, Mapping[str, float]], hand: str) -> dict[str, float]:
    row = dict(scores.get(hand) or {})
    return {
        "n": int(row.get("n") or 0),
        "slope_real": float(row.get("slope_real", float("nan"))),
        "slope_imag": float(row.get("slope_imag", float("nan"))),
        "correlation_abs": float(row.get("correlation_abs", float("nan"))),
        "residual_power": float(row.get("residual_power", float("nan"))),
        "median_abs_obs": float("nan"),
        "median_abs_pred": float("nan"),
        "median_abs_ratio": float("nan"),
        "median_abs_jy": float(row.get("median_abs_jy", float("nan"))),
        "median_abs_over_i": float(row.get("median_abs_over_i", float("nan"))),
        "residual_rms": float(row.get("residual_rms", float("nan"))),
    }


def _region_block(scores: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    rr = _hand_record(scores, "RR")
    ll = _hand_record(scores, "LL")
    return {
        "n": int(max(rr["n"], ll["n"])),
        "n_meaning": "scored development rows in this region after the row mask",
        "n_rr": rr["n"],
        "n_ll": ll["n"],
        "n_finite": int(min(rr["n"], ll["n"])),
        "median_abs_rr_over_i": rr["median_abs_over_i"],
        "median_abs_ll_over_i": ll["median_abs_over_i"],
        "median_abs_rr_jy": rr["median_abs_jy"],
        "median_abs_ll_jy": ll["median_abs_jy"],
    }


def holoraster_channel32_from_report(report: Mapping[str, object]) -> dict[str, object]:
    """Map a refresh channel report onto the publication channel-32 document."""

    scores = dict(report.get("scores") or {})
    arms = dict(scores.get("arms") or {})
    # Publication region metrics are descriptive; use the train split.
    all_split = dict((arms.get(PRIMARY_ARMS[0]) or {}).get("spatial_holdout") or {})
    train_all = dict((arms.get(PRIMARY_ARMS[0]) or {}).get("train") or {})
    regions_src = {
        "all": train_all.get("all") or {},
        "main_lobe": train_all.get("main_lobe") or {},
        "mid": train_all.get("mid") or {},
        "outer_diagnostic": train_all.get("outer") or {},
    }
    hand_power = {
        name: {
            _HAND_LOWER[hand]: _hand_record(block, hand)
            for hand in HANDS
            if hand in block
        }
        for name, block in regions_src.items()
    }
    full_all = dict((arms.get(PRIMARY_ARMS[1]) or {}).get("train") or {}).get("all") or {}
    diag_all = train_all.get("all") or {}
    return {
        "model": "evla_c_source_lm",
        "coordinate_query": "source_lm_feed",
        "development_only": True,
        "n_development": int(report.get("n_development") or 0),
        "n_rows": int(report.get("n_development") or 0),
        "n_full_raster_usable": int(report.get("n_full_raster_usable") or 0),
        "frequency_hz": float(report.get("frequency_hz") or 0.0),
        "source_i_jy": float(report.get("source_i_jy") or 0.0),
        "residual_jones_native_channel": 32,
        "diagonal": {
            _HAND_LOWER[hand]: _hand_record(diag_all, hand)
            for hand in HANDS
            if hand in diag_all
        },
        "experimental_full_jones": {
            _HAND_LOWER[hand]: _hand_record(full_all, hand) for hand in HANDS if hand in full_all
        },
        "regions": {
            **{name: _region_block(block) for name, block in regions_src.items()},
            "voltage_response": {
                "definition": "mean(|V_RR|, |V_LL|) / I_model on individually valid hands",
                "regions": {
                    "main_lobe": ">= 0.5",
                    "mid": "0.2 to 0.5",
                    "outer_diagnostic": "< 0.2",
                },
            },
            "hand_residual_power": hand_power,
        },
        "sensitivity": report.get("sensitivity") or {},
        "classification": report.get("classification") or {},
        "spatial_holdout_present": bool(all_split),
        "full_jones_frozen": False,
        "production_factory_modified": False,
        "model_selected": False,
        "spw5_opened": False,
    }


def frequency_series_from_reports(reports: list[Mapping[str, object]]) -> dict[str, object]:
    channels = []
    for report in reports:
        scores = dict(report.get("scores") or {})
        arms = dict(scores.get("arms") or {})
        train_diag = dict((arms.get(PRIMARY_ARMS[0]) or {}).get("train") or {})
        train_full = dict((arms.get(PRIMARY_ARMS[1]) or {}).get("train") or {})
        main = dict(train_diag.get("main_lobe") or {})
        all_rows = dict(train_diag.get("all") or {})
        full_main = dict(train_full.get("main_lobe") or {})
        full_all = dict(train_full.get("all") or {})
        channels.append(
            {
                "channel": int(report["channel"]),
                "frequency_hz": float(report["frequency_hz"]),
                "n_development": int(report.get("n_development") or 0),
                "n_main_lobe": int(
                    report.get("main_lobe_n") or (main.get("RR") or {}).get("n") or 0
                ),
                "region": "main_lobe",
                "diagonal": {
                    _HAND_LOWER[hand]: _hand_record(main, hand) for hand in HANDS if hand in main
                },
                "diagonal_all": {
                    _HAND_LOWER[hand]: _hand_record(all_rows, hand)
                    for hand in HANDS
                    if hand in all_rows
                },
                "experimental_full_jones": {
                    _HAND_LOWER[hand]: _hand_record(full_main, hand)
                    for hand in HANDS
                    if hand in full_main
                },
                "experimental_full_jones_all": {
                    _HAND_LOWER[hand]: _hand_record(full_all, hand)
                    for hand in HANDS
                    if hand in full_all
                },
            }
        )
    return {
        "model": "evla_c_source_lm",
        "coordinate_query": "source_lm_feed",
        "development_only": True,
        "nearest_plane_substitution": False,
        "channels": channels,
    }


def supersession_ledger() -> dict[str, object]:
    return {
        "publication_version": PUBLICATION_BUNDLE_VERSION,
        "production_accepted": False,
        "entries": [
            {
                "item": "generic commanded HOLORASTER comparison",
                "status": "historical",
                "reason": "Used generic VLA feed at commanded AZELGEO offsets",
            },
            {
                "item": "channel-32 EVLA-C diagonal vs generic at source_lm_feed",
                "status": "development_evidence",
                "reason": "One-frequency feed comparison; not a production freeze",
            },
            {
                "item": "historical generic full-Jones non-detection",
                "status": "historical",
                "reason": "Not a test of EVLA-C at source-in-beam coordinates",
            },
            {
                "item": "SPW 5",
                "status": "sealed",
                "reason": "Reserved until an SPW-4 model is frozen",
            },
            {
                "item": "width 1.04 / empirical squint orientation",
                "status": "not_inherited",
                "reason": "Unit physical EVLA-C model only in this refresh",
            },
        ],
    }


def squint_from_refresh_export(npz_path: Path) -> dict[str, object]:
    """20%-of-peak squint on named source-in-beam coordinates."""

    from sl1mjax.holography_cassbeam_holoraster_report import squint_from_voltage_maps

    with np.load(npz_path, allow_pickle=False) as handle:
        offset = np.asarray(handle["source_lm_feed"], dtype=np.float64)
        measured = np.asarray(handle["measured"])
        predicted = np.asarray(handle[PRIMARY_ARMS[0]])
        weight = np.asarray(handle["weight"], dtype=np.float64)
        frequency = float(np.asarray(handle["frequency_hz"]).reshape(-1)[0])
    meas = squint_from_voltage_maps(
        offset,
        np.abs(measured[:, 0, 0]) ** 2,
        np.abs(measured[:, 1, 1]) ** 2,
        weight[:, 0, 0],
        weight[:, 1, 1],
        frequency_hz=frequency,
        series="measured_source_lm",
    )
    model = squint_from_voltage_maps(
        offset,
        np.abs(predicted[:, 0, 0]) ** 2,
        np.abs(predicted[:, 1, 1]) ** 2,
        weight[:, 0, 0],
        weight[:, 1, 1],
        frequency_hz=frequency,
        series="evla_c_source_lm",
    )
    return {
        "measured": meas,
        "cassbeam": model,
        "frequency_hz": frequency,
        "predictions_recomputed": True,
        "coordinate_query": "source_lm_feed",
        "model": "evla_c_source_lm",
    }


def residual_geometry_from_export(npz_path: Path) -> dict[str, object]:
    from sl1mjax.holography_beam_prior import vis_planes
    from sl1mjax.holography_cassbeam_holoraster_report import residual_geometry_summaries

    with np.load(npz_path, allow_pickle=False) as handle:
        measured = np.asarray(handle["measured"])
        predicted = np.asarray(handle[PRIMARY_ARMS[0]])
        weight = np.asarray(handle["weight"])
        offset = np.asarray(handle["source_lm_feed"], dtype=np.float64)
        frequency = float(np.asarray(handle["frequency_hz"]).reshape(-1)[0])
    residual = vis_planes(measured) - vis_planes(predicted)
    mask = np.ones(measured.shape[0], dtype=bool)
    geometry = residual_geometry_summaries(residual, weight, offset, 32, mask)
    geometry["coordinate_query"] = "source_lm_feed"
    geometry["frequency_hz"] = frequency
    return geometry


def crosshand_quadrants_from_export(npz_path: Path, *, source_i_jy: float) -> dict[str, object]:
    from sl1mjax.holography_cassbeam_holoraster_report import spatial_quadrant_masks

    with np.load(npz_path, allow_pickle=False) as handle:
        measured = np.asarray(handle["measured"])
        predicted = np.asarray(handle[PRIMARY_ARMS[1]])
        weight = np.asarray(handle["weight"])
        offset = np.asarray(handle["source_lm_feed"], dtype=np.float64)
    masks = spatial_quadrant_masks(offset)
    out: dict[str, object] = {}
    for name, mask in masks.items():
        if not bool(np.any(mask)):
            continue
        scored = hand_scores(
            measured[mask], predicted[mask], weight[mask], source_i_jy=source_i_jy
        )
        out[name] = {
            "rl": _hand_record(scored, "RL"),
            "lr": _hand_record(scored, "LR"),
        }
    return out


def offset_ring_from_c147_report(report: Mapping[str, object]) -> dict[str, object]:
    """Map a nine-frequency EVLA-C C147 report onto the publication ring shape."""

    out = dict(report)
    if out.get("status") == "blocked" or out.get("historical"):
        out.setdefault("q_u_is_frequency_holdout", False)
        return out
    fields = list(out.get("fields") or [])
    radii = [float(item.get("radius_arcmin") or float("nan")) for item in fields]
    finite = [value for value in radii if np.isfinite(value)]
    channels = list(out.get("channels") or [])
    chosen = next((item for item in channels if int(item.get("channel") or -1) == 32), None)
    if chosen is None and channels:
        chosen = channels[0]
    scores = dict((chosen or {}).get("scores") or {})
    block = {}
    for name in ("inner_holdout", "training", "sealed_holdout"):
        if scores.get(name):
            block = dict(scores[name])
            if name == "inner_holdout":
                break
    diagonal = dict(block.get("diagonal") or {})
    rr = dict(diagonal.get("RR") or {})
    ll = dict(diagonal.get("LL") or {})
    rr_l = float(rr.get("residual_power", float("nan")))
    ll_l = float(ll.get("residual_power", float("nan")))
    out["radius_arcmin"] = float(np.nanmedian(finite)) if finite else float("nan")
    out["q_u_is_frequency_holdout"] = False
    out["historical_holdout"] = True
    out["development_only"] = True
    out["closure"] = {
        "rr": {"relative_power": rr_l, "n": int(rr.get("n") or 0), "ok": int(rr_l < 0.05)},
        "ll": {"relative_power": ll_l, "n": int(ll.get("n") or 0), "ok": int(ll_l < 0.05)},
        "passed": bool(np.isfinite(rr_l) and np.isfinite(ll_l) and rr_l < 0.05 and ll_l < 0.05),
        "partition": next(
            (name for name in ("inner_holdout", "training") if scores.get(name)),
            None,
        ),
        "channel": int((chosen or {}).get("channel") or -1),
    }
    empty = []
    for item in channels:
        inner = dict((item.get("scores") or {}).get("inner_holdout") or {})
        rr_power = dict(inner.get("diagonal") or {}).get("RR") or {}
        power = float(rr_power.get("residual_power", float("nan")))
        if not np.isfinite(power):
            empty.append(int(item.get("channel") or -1))
    out["empty_inner_holdout_channels"] = empty
    return out


def write_publication_documents(output_dir: Path, *, channel32: Mapping[str, object]) -> Path:
    root = Path(output_dir)
    write_json_atomic(
        root / "holoraster_channel32.json",
        holoraster_channel32_from_report(channel32),
    )
    write_json_atomic(root / "supersession_ledger.json", supersession_ledger())
    return root
