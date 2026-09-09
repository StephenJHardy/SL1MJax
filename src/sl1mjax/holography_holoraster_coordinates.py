"""HOLORASTER commanded versus source-in-beam coordinates.

``POINTING_OFFSET`` is the commanded AZELGEO displacement of the moving
antenna. 3C147 remains at ``TARGET``, so its coordinate inside that
moved beam is ``TARGET-DIRECTION = -POINTING_OFFSET`` after the locked
AZELGEO-to-CASSBEAM axis map. Bare ``offset_lm_rad`` is refused on this
path. The 128-member Jones convention ladder stays closed. SPW 5 stays
sealed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.holography import (
    AntennaPointingTable,
    commanded_offset_from_direction_target,
    source_relative_lm_rad,
)

ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)
MEMO195_PLUS_L = "right when looking outward"
MEMO195_PLUS_M = "toward local zenith"
LOCKED_AZELGEO_TO_CASSBEAM = "az_to_plus_l_el_to_plus_m_no_swap"
SQUINT_CODE_KIND = "r_minus_l"
SQUINT_MEMO195_KIND = "r_to_l"


@dataclass(frozen=True)
class AzelgeoCassbeamAxisMap:
    """Locked tangent-axis map from AZELGEO ``(az, el)`` into CASSBEAM ``(l, m)``.

    Memo 195 and CASSBEAM take ``+l`` as right looking outward and ``+m``
    toward local zenith. The locked hypothesis is therefore no axis swap:
    the AZELGEO azimuth tangent is ``+l`` and the elevation tangent is
    ``+m``. Signs are applied to the AZELGEO components before that
    assignment.
    """

    az_sign: int = 1
    el_sign: int = 1
    swap_azel: bool = False
    name: str = LOCKED_AZELGEO_TO_CASSBEAM

    def __post_init__(self) -> None:
        if int(self.az_sign) not in {-1, 1} or int(self.el_sign) not in {-1, 1}:
            raise ValueError("AZELGEO axis signs must be ±1")
        object.__setattr__(self, "az_sign", int(self.az_sign))
        object.__setattr__(self, "el_sign", int(self.el_sign))


LOCKED_AXIS_MAP = AzelgeoCassbeamAxisMap()


def refuse_bare_offset_lm_rad() -> None:
    raise RuntimeError(
        "HOLORASTER beam queries require commanded_offset_azelgeo or "
        "source_lm_feed; bare offset_lm_rad is ambiguous"
    )


def refuse_convention_ladder() -> None:
    raise RuntimeError(
        "the 128-member CASSBEAM convention ladder stays closed; "
        "only the deterministic AZELGEO-to-CASSBEAM map is locked here"
    )


def azelgeo_to_cassbeam_lm(
    azelgeo_rad: ArrayLike,
    axis_map: AzelgeoCassbeamAxisMap = LOCKED_AXIS_MAP,
) -> NDArray[np.float64]:
    offset = np.asarray(azelgeo_rad, dtype=np.float64)
    if offset.ndim == 1:
        offset = offset.reshape(1, 2)
    if offset.ndim != 2 or offset.shape[1] != 2:
        raise ValueError("AZELGEO offsets must have shape (sample, 2)")
    azimuth = float(axis_map.az_sign) * offset[:, 0]
    elevation = float(axis_map.el_sign) * offset[:, 1]
    if axis_map.swap_azel:
        return np.stack([elevation, azimuth], axis=1)
    return np.stack([azimuth, elevation], axis=1)


def commanded_offset_azelgeo(azelgeo_rad: ArrayLike) -> NDArray[np.float64]:
    """Name the commanded mount-frame displacement. Do not query CASSBEAM with it."""

    return azelgeo_to_cassbeam_lm(azelgeo_rad, LOCKED_AXIS_MAP)


def source_lm_feed_from_commanded_azelgeo(
    commanded_azelgeo_rad: ArrayLike,
    *,
    axis_map: AzelgeoCassbeamAxisMap = LOCKED_AXIS_MAP,
) -> NDArray[np.float64]:
    """Source coordinate in the moved beam: ``TARGET-DIRECTION = -Δ``.

    3C147 stays at ``TARGET``. After the locked AZELGEO-to-CASSBEAM map,
    that is the CASSBEAM query coordinate.
    """

    commanded = azelgeo_to_cassbeam_lm(commanded_azelgeo_rad, axis_map)
    sky = np.zeros((1, 2), dtype=np.float64)
    relative = source_relative_lm_rad(sky, commanded, offset_sign="commanded_pointing")
    return np.asarray(relative[:, 0, :], dtype=np.float64)


def squint_vector_kinds(*, r_minus_l_lm: ArrayLike) -> dict[str, object]:
    """Publish both code ``c_R-c_L`` and Memo 195 ``c_L-c_R`` labels."""

    code = np.asarray(r_minus_l_lm, dtype=np.float64).reshape(2)
    memo = -code
    return {
        "r_minus_l": [float(code[0]), float(code[1])],
        "r_to_l": [float(memo[0]), float(memo[1])],
        "code_kind": SQUINT_CODE_KIND,
        "memo195_kind": SQUINT_MEMO195_KIND,
        "note": (
            "code products report r_minus_l = c_R-c_L; "
            "Memo 195 position angle is r_to_l = c_L-c_R = -r_minus_l"
        ),
    }


def holoraster_ms_geometry_oracle(
    table: AntennaPointingTable,
    phase_centre_rad: tuple[float, float],
    *,
    residual_limit_arcmin: float = 0.05,
) -> dict[str, object]:
    """Prove ``DIRECTION-TARGET = POINTING_OFFSET`` on preserved MS columns."""

    names = [column.name for column in table.columns]
    if any(name not in names for name in ("DIRECTION", "TARGET", "POINTING_OFFSET")):
        raise ValueError("geometry oracle needs DIRECTION, TARGET, and POINTING_OFFSET")
    predicted = commanded_offset_from_direction_target(table, phase_centre_rad)
    stored = np.asarray(table.column("POINTING_OFFSET").values_rad, dtype=np.float64)
    residual = predicted - stored
    radius = np.hypot(stored[:, 0], stored[:, 1])
    nonzero = radius > 0.05 * ARCMIN_TO_RAD
    residual_arcmin = residual / ARCMIN_TO_RAD
    if bool(np.any(nonzero)):
        radial = np.hypot(residual_arcmin[nonzero, 0], residual_arcmin[nonzero, 1])
        max_residual = float(np.max(radial))
    else:
        max_residual = float("nan")
    source_from_target = -predicted
    source_from_offset = source_lm_feed_from_commanded_azelgeo(stored)
    source_agree = bool(
        np.allclose(
            source_from_target[nonzero],
            source_from_offset[nonzero],
            rtol=0.0,
            atol=1.0e-15,
        )
    )
    within_limit = bool(np.isfinite(max_residual) and max_residual <= residual_limit_arcmin)
    passed = bool(within_limit and source_agree)
    return {
        "n_row": int(stored.shape[0]),
        "n_nonzero": int(np.sum(nonzero)),
        "max_direction_minus_target_minus_offset_arcmin": max_residual,
        "source_is_negative_commanded": source_agree,
        "passes": passed,
        "direction_measure_ref": table.column("DIRECTION").measure_ref,
        "target_measure_ref": table.column("TARGET").measure_ref,
        "pointing_offset_measure_ref": table.column("POINTING_OFFSET").measure_ref,
        "axis_map": LOCKED_AXIS_MAP.name,
        "plus_l": MEMO195_PLUS_L,
        "plus_m": MEMO195_PLUS_M,
    }


def manufactured_asymmetric_copolar(offset_lm_rad: ArrayLike) -> NDArray[np.float64]:
    """Analytic copolar voltage brighter at ``+l`` and ``+m``.

    One manufactured beam is enough to distinguish sign reversal from an
    axis swap. It is not a physical VLA model.
    """

    offset = np.asarray(offset_lm_rad, dtype=np.float64)
    if offset.ndim == 1:
        offset = offset.reshape(1, 2)
    scale = 2.0 * ARCMIN_TO_RAD
    voltage = 1.0 + 0.6 * np.tanh(offset[:, 0] / scale) + 0.4 * np.tanh(offset[:, 1] / scale)
    return np.asarray(voltage, dtype=np.float64)


def lock_azelgeo_cassbeam_axis_map(
    commanded_azelgeo_rad: ArrayLike,
) -> dict[str, object]:
    """Lock the Memo 195 no-swap map with one asymmetric beam and one track."""

    commanded = np.asarray(commanded_azelgeo_rad, dtype=np.float64)
    if commanded.ndim == 1:
        commanded = commanded.reshape(1, 2)
    physical = manufactured_asymmetric_copolar(
        source_lm_feed_from_commanded_azelgeo(commanded, axis_map=LOCKED_AXIS_MAP)
    )
    candidates = {
        LOCKED_AXIS_MAP.name: LOCKED_AXIS_MAP,
        "negate_az": AzelgeoCassbeamAxisMap(-1, 1, False, "negate_az"),
        "negate_el": AzelgeoCassbeamAxisMap(1, -1, False, "negate_el"),
        "negate_both": AzelgeoCassbeamAxisMap(-1, -1, False, "negate_both"),
        "swap": AzelgeoCassbeamAxisMap(1, 1, True, "swap"),
        "swap_negate_both": AzelgeoCassbeamAxisMap(-1, -1, True, "swap_negate_both"),
        "commanded_as_source": None,
    }
    residuals: dict[str, float] = {}
    for name, axis_map in candidates.items():
        if axis_map is None:
            query = azelgeo_to_cassbeam_lm(commanded, LOCKED_AXIS_MAP)
        else:
            query = source_lm_feed_from_commanded_azelgeo(commanded, axis_map=axis_map)
        model = manufactured_asymmetric_copolar(query)
        residuals[name] = float(np.max(np.abs(model - physical)))
    winner = min(residuals, key=residuals.get)
    unique = residuals[winner] < 0.5 * min(
        value for name, value in residuals.items() if name != winner
    )
    locked = winner == LOCKED_AXIS_MAP.name and unique and residuals[winner] <= 1.0e-12
    if not locked:
        raise ValueError(
            "AZELGEO-to-CASSBEAM map failed to lock uniquely on the manufactured track; "
            f"winner={winner} residuals={residuals}"
        )
    return {
        "locked": True,
        "axis_map": LOCKED_AXIS_MAP.name,
        "plus_l": MEMO195_PLUS_L,
        "plus_m": MEMO195_PLUS_M,
        "residuals": residuals,
        "n_track": int(commanded.shape[0]),
        "commanded_as_source_residual": residuals["commanded_as_source"],
    }


def holoraster_row_coordinates(
    commanded_azelgeo_rad: ArrayLike,
) -> dict[str, NDArray[np.float64] | str]:
    """Return separately named commanded and source-in-beam coordinates."""

    commanded = commanded_offset_azelgeo(commanded_azelgeo_rad)
    source = source_lm_feed_from_commanded_azelgeo(commanded_azelgeo_rad)
    return {
        "commanded_offset_azelgeo": commanded,
        "source_lm_feed": source,
        "commanded_meaning": "DIRECTION-TARGET = POINTING_OFFSET",
        "source_meaning": "TARGET-DIRECTION after locked AZELGEO-to-CASSBEAM map",
        "axis_map": LOCKED_AXIS_MAP.name,
    }


def require_geometry_oracle(report: Mapping[str, object]) -> None:
    if not bool(report.get("passes")):
        raise ValueError(
            "HOLORASTER geometry oracle failed; "
            "DIRECTION-TARGET must equal POINTING_OFFSET and "
            "source_lm_feed must be the negative commanded displacement"
        )
