"""Separate HOLORASTER coordinate sign from EVLA-C versus generic VLA feed.

Compare, in order:

1. generic CASSBEAM queried at commanded AZELGEO offsets;
2. generic CASSBEAM queried at ``source_lm_feed``;
3. EVLA-C CASSBEAM queried at ``source_lm_feed``, when that artifact exists.

The 128-member Jones ladder stays closed. SPW 5 stays sealed. Frozen
generic high-resolution products are not overwritten.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.cassbeam_highres import HighresCassbeamPlane
from sl1mjax.holography_cassbeam_holoraster_report import squint_from_voltage_maps
from sl1mjax.holography_diagonal_correction import refuse_spw5
from sl1mjax.holography_holoraster_coordinates import (
    LOCKED_AXIS_MAP,
    source_lm_feed_from_commanded_azelgeo,
    squint_vector_kinds,
)
from sl1mjax.holography_physical_squint_experiment import score_model

COORDINATE_FEED_EXPERIMENT = "spw4_coordinate_feed_comparison_v1"
GENERIC_COMMANDED = "generic_commanded"
GENERIC_SOURCE = "generic_source_lm"
EVLA_SOURCE = "evla_c_source_lm"


def refuse_frozen_generic_artifact(path: Path) -> None:
    if "cassbeam_cband_full_jones_g1024_p32_20260906" in Path(path).resolve().as_posix():
        if Path(path).name.endswith(".in") or "evla" in Path(path).name:
            return
        raise RuntimeError("refusing to write into the frozen generic CASSBEAM artifact")


def query_coordinates(
    commanded_azelgeo_rad: ArrayLike,
    *,
    query: str,
    source_lm_feed: ArrayLike | None = None,
) -> NDArray[np.float64]:
    if query == "commanded":
        return np.asarray(commanded_azelgeo_rad, dtype=np.float64)
    if query != "source_lm_feed":
        raise ValueError(f"unknown query kind {query!r}")
    if source_lm_feed is not None:
        return np.asarray(source_lm_feed, dtype=np.float64)
    return source_lm_feed_from_commanded_azelgeo(commanded_azelgeo_rad)


def plane_hand_centroids_arcmin(
    plane: HighresCassbeamPlane,
    *,
    frequency_hz: float,
    series: str,
) -> dict[str, object]:
    """20%-of-peak RR/LL centroids on the CASSBEAM plane itself."""

    ll, mm = np.meshgrid(plane.l_rad, plane.m_rad, indexing="xy")
    offset = np.stack([ll.reshape(-1), mm.reshape(-1)], axis=1)
    rr = np.abs(plane.jones_norm[..., 0, 0]) ** 2
    left = np.abs(plane.jones_norm[..., 1, 1]) ** 2
    weight = np.ones(rr.size, dtype=np.float64)
    record = squint_from_voltage_maps(
        offset,
        rr.reshape(-1),
        left.reshape(-1),
        weight,
        weight,
        frequency_hz=frequency_hz,
        series=series,
    )
    record["r_minus_l"] = squint_vector_kinds(
        r_minus_l_lm=(
            float(record["rr_l_arcmin"]) - float(record["ll_l_arcmin"]),
            float(record["rr_m_arcmin"]) - float(record["ll_m_arcmin"]),
        )
    )
    return record


def interpret_coordinate_feed_scores(
    paired: Mapping[str, Mapping[str, object]],
    *,
    evla_present: bool,
) -> dict[str, object]:
    refuse_spw5(opened=False)
    source_vs_commanded = paired["generic_source_lm_vs_generic_commanded"]
    source_beats = bool(
        source_vs_commanded["spatial"]["improves"] and source_vs_commanded["moving"]["improves"]
    )
    out = {
        "keep_physical_squint": True,
        "native_orientation_retained_as_production_prior": False,
        "source_lm_feed_beats_commanded": source_beats,
        "axis_map": LOCKED_AXIS_MAP.name,
        "evla_c_compared": bool(evla_present),
        "spw5_ready": False,
        "development_only": True,
        "convention_ladder_reopened": False,
    }
    if evla_present and "evla_c_source_lm_vs_generic_source_lm" in paired:
        evla_vs_generic = paired["evla_c_source_lm_vs_generic_source_lm"]
        out["evla_c_beats_generic_source"] = bool(
            evla_vs_generic["spatial"]["improves"] and evla_vs_generic["moving"]["improves"]
        )
    return out


def score_candidate_against_baseline(
    samples,
    predicted: ArrayLike,
    baseline: ArrayLike,
    *,
    rr_ok: ArrayLike,
    ll_ok: ArrayLike,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    return score_model(
        samples,
        predicted,
        baseline,
        rr_ok=rr_ok,
        ll_ok=ll_ok,
        n_boot=n_boot,
        seed=seed,
    )
