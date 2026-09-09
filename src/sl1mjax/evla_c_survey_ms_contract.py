"""Visibility-reader contracts for survey Measurement Set reuse.

The HOLORASTER channel reader originally assumed circular order, ignored
FLAG_ROW, and broadcast WEIGHT. Those choices were justified for one MS.
New executions must satisfy these tests before reuse.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.polarization import Correlation

REQUIRED_CIRCULAR_ORDER = (
    Correlation.RR,
    Correlation.RL,
    Correlation.LR,
    Correlation.LL,
)
CASA_CIRCULAR_CORR_TYPE = (5, 6, 7, 8)


def combine_channel_and_row_flags(
    flag: ArrayLike,
    flag_row: ArrayLike | None,
) -> NDArray[np.bool_]:
    """OR FLAG with FLAG_ROW. A true row flag marks every channel and hand."""

    plane = np.asarray(flag, dtype=bool)
    if flag_row is None:
        return plane
    row = np.asarray(flag_row, dtype=bool).reshape(-1)
    if row.size != plane.shape[0]:
        raise ValueError("FLAG_ROW length must match FLAG rows")
    shaped = row.reshape((row.size,) + (1,) * (plane.ndim - 1))
    return plane | shaped


def select_row_or_spectrum_weight(
    weight: ArrayLike,
    *,
    weight_spectrum: ArrayLike | None,
    spectrum_defined: bool,
    n_chan: int,
) -> NDArray[np.float64]:
    """Use WEIGHT_SPECTRUM only when its tiles are defined; else broadcast WEIGHT."""

    row = np.asarray(weight, dtype=np.float64)
    if spectrum_defined and weight_spectrum is not None:
        spectrum = np.asarray(weight_spectrum, dtype=np.float64)
        if spectrum.shape[0] != row.shape[0]:
            raise ValueError("WEIGHT_SPECTRUM row count must match WEIGHT")
        return spectrum
    if row.ndim == 2:
        return np.repeat(row[:, None, :], int(n_chan), axis=1)
    raise ValueError("WEIGHT must have shape (row, corr)")


def require_circular_correlation_order(
    correlations: Sequence[object],
    *,
    corr_type: Sequence[int] | None = None,
) -> tuple[Correlation, ...]:
    """Refuse a silent reorder. Wrong CORR_TYPE is a hard failure."""

    order = tuple(correlations)
    if order != REQUIRED_CIRCULAR_ORDER:
        raise ValueError(f"survey reader requires {REQUIRED_CIRCULAR_ORDER}, got {order}")
    if corr_type is not None and tuple(int(code) for code in corr_type) != CASA_CIRCULAR_CORR_TYPE:
        raise ValueError(
            f"POLARIZATION.CORR_TYPE {tuple(corr_type)} is not circular RR,RL,LR,LL"
        )
    return REQUIRED_CIRCULAR_ORDER
