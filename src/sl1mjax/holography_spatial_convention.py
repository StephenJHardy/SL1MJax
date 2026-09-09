"""Discrete SPW-4 diagonal spatial-convention test.

CASSBEAM squint stays physically present. This enumerates only ``l``/``m``
signs, axis swap, and R/L swap of the complete diagonal beam. The locked
mount-frame CASSBEAM convention is not reopened. Width is frozen at 1.04.
SPW 5 stays sealed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.holography_beam_prior import predict_from_moving_beams
from sl1mjax.holography_cassbeam_correction import (
    FeedFrameLookup,
    apply_parallactic_after_correction,
    renormalize_to_cassbeam_origin,
)
from sl1mjax.holography_diagonal_correction import refuse_spw5, require_boresight_unity
from sl1mjax.holography_physical_squint import (
    IDENTITY_PHYSICAL,
    native_hand_centers_rad,
    physical_query_coordinates,
)

SPATIAL_CONVENTION_EXPERIMENT = "spw4_spatial_convention_v1"
FROZEN_WIDTH = 1.04
IDENTITY_ATOL = 1.0e-12
MEASURED_INDEPENDENT_DELTA_ARCMIN = (0.25244491429213367, -0.6747099228808454)


def refuse_convention_ladder() -> None:
    raise RuntimeError(
        "this test enumerates only l/m signs, axis swap, and R/L swap; "
        "the 128-member CASSBEAM convention ladder stays closed"
    )


def refuse_magnitude_fit() -> None:
    raise RuntimeError("magnitude stays frozen until a spatial convention is locked")


@dataclass(frozen=True)
class SpatialTransform:
    """One discrete spatial reorientation of the complete diagonal beam."""

    l_sign: int = 1
    m_sign: int = 1
    swap_lm: bool = False
    swap_rl: bool = False

    def __post_init__(self) -> None:
        if int(self.l_sign) not in {-1, 1} or int(self.m_sign) not in {-1, 1}:
            raise ValueError("axis signs must be ±1")
        object.__setattr__(self, "l_sign", int(self.l_sign))
        object.__setattr__(self, "m_sign", int(self.m_sign))

    @property
    def name(self) -> str:
        axis = f"l{self.l_sign:+d}_m{self.m_sign:+d}"
        if self.swap_lm:
            axis += "_swaplm"
        if self.swap_rl:
            axis += "_swaprl"
        return axis

    def is_identity(self) -> bool:
        return self.l_sign == 1 and self.m_sign == 1 and not self.swap_lm and not self.swap_rl

    def inverse(self) -> SpatialTransform:
        if self.swap_lm:
            return SpatialTransform(self.m_sign, self.l_sign, True, self.swap_rl)
        return SpatialTransform(self.l_sign, self.m_sign, False, self.swap_rl)


IDENTITY_TRANSFORM = SpatialTransform()


def spatial_transform_catalog() -> tuple[SpatialTransform, ...]:
    """The 16 sign / swap / R-L candidates. No Jones or sky-frame search."""

    return tuple(
        SpatialTransform(l_sign, m_sign, swap_lm, swap_rl)
        for swap_lm in (False, True)
        for l_sign in (1, -1)
        for m_sign in (1, -1)
        for swap_rl in (False, True)
    )


def apply_spatial_transform(
    offset_lm_rad: ArrayLike,
    transform: SpatialTransform,
) -> NDArray[np.float64]:
    offset = np.asarray(offset_lm_rad, dtype=np.float64)
    if offset.ndim == 1:
        offset = offset.reshape(1, 2)
    left = float(transform.l_sign) * offset[:, 0]
    right = float(transform.m_sign) * offset[:, 1]
    if transform.swap_lm:
        return np.stack([right, left], axis=1)
    return np.stack([left, right], axis=1)


def transformed_hand_centers_rad(
    transform: SpatialTransform,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    center_r, center_l = native_hand_centers_rad()
    mapped_r = apply_spatial_transform(center_r, transform)[0]
    mapped_l = apply_spatial_transform(center_l, transform)[0]
    if transform.swap_rl:
        return mapped_l, mapped_r
    return mapped_r, mapped_l


def transformed_separation_rad(transform: SpatialTransform) -> NDArray[np.float64]:
    right, left = transformed_hand_centers_rad(transform)
    return right - left


def vector_alignment(
    candidate_lm: ArrayLike,
    reference_lm: ArrayLike,
) -> dict[str, float]:
    cand = np.asarray(candidate_lm, dtype=np.float64).reshape(2)
    ref = np.asarray(reference_lm, dtype=np.float64).reshape(2)
    n_c = float(np.hypot(*cand))
    n_r = float(np.hypot(*ref))
    if n_c <= 0.0 or n_r <= 0.0:
        return {
            "cosine": float("nan"),
            "angle_deg": float("nan"),
            "aligns": False,
        }
    cosine = float(np.dot(cand, ref) / (n_c * n_r))
    cosine = min(1.0, max(-1.0, cosine))
    return {
        "cosine": cosine,
        "angle_deg": float(np.degrees(np.arccos(cosine))),
        "aligns": bool(cosine > 0.0),
    }


def transformed_query_coordinates(
    offset_lm_rad: ArrayLike,
    transform: SpatialTransform,
    *,
    width: float = FROZEN_WIDTH,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Width about native hand centres after the inverse spatial map."""

    if float(width) <= 0.0:
        raise ValueError("width must be positive")
    offset = apply_spatial_transform(offset_lm_rad, transform.inverse())
    center_r, center_l = native_hand_centers_rad()
    query_r = center_r + (offset - center_r) / float(width)
    query_l = center_l + (offset - center_l) / float(width)
    return query_r, query_l


def apply_transformed_feed_frame(
    offset_lm_rad: ArrayLike,
    lookup: FeedFrameLookup,
    transform: SpatialTransform,
    *,
    width: float = FROZEN_WIDTH,
) -> NDArray[np.complex128]:
    """Complete diagonal CASSBEAM after a discrete spatial transform."""

    refuse_spw5(opened=False)
    offset = np.asarray(offset_lm_rad, dtype=np.float64)
    if offset.ndim == 1:
        offset = offset.reshape(1, 2)
    query_r, query_l = transformed_query_coordinates(offset, transform, width=width)
    looked_r = np.asarray(lookup(query_r), dtype=np.complex128)
    looked_l = np.asarray(lookup(query_l), dtype=np.complex128)
    jones = np.zeros((offset.shape[0], 2, 2), dtype=np.complex128)
    if transform.swap_rl:
        jones[:, 0, 0] = looked_l[:, 1, 1]
        jones[:, 1, 1] = looked_r[:, 0, 0]
    else:
        jones[:, 0, 0] = looked_r[:, 0, 0]
        jones[:, 1, 1] = looked_l[:, 1, 1]
    origin = np.zeros((1, 2), dtype=np.float64)
    q0_r, q0_l = transformed_query_coordinates(origin, transform, width=width)
    raw0 = np.zeros((2, 2), dtype=np.complex128)
    if transform.swap_rl:
        raw0[0, 0] = np.asarray(lookup(q0_l), dtype=np.complex128)[0, 1, 1]
        raw0[1, 1] = np.asarray(lookup(q0_r), dtype=np.complex128)[0, 0, 0]
    else:
        raw0[0, 0] = np.asarray(lookup(q0_r), dtype=np.complex128)[0, 0, 0]
        raw0[1, 1] = np.asarray(lookup(q0_l), dtype=np.complex128)[0, 1, 1]
    cass0 = np.asarray(lookup(origin), dtype=np.complex128)[0]
    corrected = renormalize_to_cassbeam_origin(jones, raw0, cass0)
    origin_corr = renormalize_to_cassbeam_origin(raw0[None, ...], raw0, cass0)[0]
    require_boresight_unity(
        origin_corr[0, 0] / cass0[0, 0],
        origin_corr[1, 1] / cass0[1, 1],
    )
    return corrected


def transformed_path_stages(
    offset_lm_rad: ArrayLike,
    lookup: FeedFrameLookup,
    transform: SpatialTransform,
    *,
    width: float,
    residual_jones: Mapping[int, ArrayLike],
    moving_id: ArrayLike,
    reference_id: ArrayLike,
    moving_is_p: ArrayLike,
    chi_moving: ArrayLike,
    chi_reference: ArrayLike,
    source: ArrayLike,
) -> dict[str, NDArray[np.complex128]]:
    feed = apply_transformed_feed_frame(offset_lm_rad, lookup, transform, width=width)
    sky = apply_parallactic_after_correction(feed, chi_moving)
    vis = predict_from_moving_beams(
        residual_jones,
        moving_id,
        reference_id,
        moving_is_p,
        chi_moving,
        chi_reference,
        sky,
        source,
    )
    if vis.ndim == 4:
        if vis.shape[1] != 1:
            raise ValueError("spatial-convention predictions must stay on one native channel")
        vis = vis[:, 0]
    return {"feed": feed, "sky": sky, "visibility": vis}


def identity_queries_match_physical(offset_lm_rad: ArrayLike) -> None:
    """Identity transform plus ``w=1`` must be the physical identity query."""

    query_r, query_l = transformed_query_coordinates(offset_lm_rad, IDENTITY_TRANSFORM, width=1.0)
    physical_r, physical_l = physical_query_coordinates(offset_lm_rad, IDENTITY_PHYSICAL)
    if not np.allclose(query_r, physical_r, rtol=0.0, atol=IDENTITY_ATOL):
        raise ValueError("identity spatial query must match the physical identity")
    if not np.allclose(query_l, physical_l, rtol=0.0, atol=IDENTITY_ATOL):
        raise ValueError("identity spatial query must match the physical identity")


def select_spatial_transform(
    records: Sequence[Mapping[str, object]],
    *,
    measured_delta_arcmin: ArrayLike = MEASURED_INDEPENDENT_DELTA_ARCMIN,
) -> dict[str, object]:
    """Rank on training RR−LL. Reject RR/LL regressions and anti-aligned maps."""

    if not records:
        raise ValueError("no spatial-transform records")
    ranked = sorted(
        records,
        key=lambda item: (
            float(item["train_rr_minus_ll"]),
            float(item["train_combined"]),
        ),
    )
    identity = next(item for item in records if bool(item["is_identity"]))
    chosen = None
    rejected: list[dict[str, object]] = []
    for item in ranked:
        align = vector_alignment(item["delta_arcmin"], measured_delta_arcmin)
        rr_ok = float(item["train_rr"]) <= float(identity["train_rr"]) * 1.10 + 0.002
        ll_ok = float(item["train_ll"]) <= float(identity["train_ll"]) * 1.10 + 0.002
        if align["aligns"] and rr_ok and ll_ok:
            chosen = dict(item)
            chosen["alignment"] = align
            break
        rejected.append(
            {
                "name": item["name"],
                "aligns": align["aligns"],
                "angle_deg": align["angle_deg"],
                "rr_ok": rr_ok,
                "ll_ok": ll_ok,
            }
        )
    return {
        "selected": chosen,
        "ranked_names": [str(item["name"]) for item in ranked],
        "rejected": rejected,
        "locked": chosen is not None,
    }
