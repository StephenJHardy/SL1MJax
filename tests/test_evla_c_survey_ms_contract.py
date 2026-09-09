from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.evla_c_survey_ms_contract import (
    CASA_CIRCULAR_CORR_TYPE,
    combine_channel_and_row_flags,
    require_circular_correlation_order,
    select_row_or_spectrum_weight,
)
from sl1mjax.polarization import Correlation


def test_flag_row_marks_every_channel_and_hand() -> None:
    flag = np.zeros((3, 2, 4), dtype=bool)
    flag[1, 0, 0] = True
    flag_row = np.array([False, False, True])
    combined = combine_channel_and_row_flags(flag, flag_row)
    assert combined[1, 0, 0]
    assert np.all(combined[2])
    assert not combined[0].any()


def test_weight_spectrum_is_used_only_when_defined() -> None:
    weight = np.full((2, 4), 4.0)
    spectrum = np.arange(16, dtype=np.float64).reshape(2, 2, 4)
    broadcast = select_row_or_spectrum_weight(
        weight, weight_spectrum=spectrum, spectrum_defined=False, n_chan=2
    )
    assert broadcast.shape == (2, 2, 4)
    np.testing.assert_allclose(broadcast[:, 0, :], 4.0)
    used = select_row_or_spectrum_weight(
        weight, weight_spectrum=spectrum, spectrum_defined=True, n_chan=2
    )
    np.testing.assert_array_equal(used, spectrum)


def test_wrong_correlation_order_and_corr_type_fail() -> None:
    require_circular_correlation_order(
        (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        corr_type=CASA_CIRCULAR_CORR_TYPE,
    )
    with pytest.raises(ValueError, match="requires"):
        require_circular_correlation_order(
            (Correlation.LL, Correlation.LR, Correlation.RL, Correlation.RR)
        )
    with pytest.raises(ValueError, match="CORR_TYPE"):
        require_circular_correlation_order(
            (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
            corr_type=(8, 7, 6, 5),
        )
