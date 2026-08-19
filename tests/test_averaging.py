import numpy as np

from sl1mjax.data.averaging import average_frequency_bins, average_time_bins
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis


def _block() -> VisibilityBlock:
    values = np.arange(16, dtype=np.float64).reshape(4, 4, 1)
    flag = np.zeros(values.shape, dtype=bool)
    flag[0, 1, 0] = True
    return VisibilityBlock(
        uvw_m=np.array(
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [8.0, 9.0, 10.0]]
        ),
        frequency_hz=np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9]),
        visibility=values.astype(np.complex128),
        model_visibility=2 * values.astype(np.complex128),
        weight=np.ones(values.shape),
        flag=flag,
        time_s=np.array([1.0, 9.0, 21.0, 29.0]),
        antenna1=np.array([0, 0, 0, 0]),
        antenna2=np.array([1, 1, 1, 1]),
        scan_id=np.array([3, 3, 3, 3]),
        correlations=(Correlation.I,),
        receptor_basis=ReceptorBasis.STOKES,
    )


def test_frequency_averaging_honours_flags_and_weights() -> None:
    averaged = average_frequency_bins(_block(), bin_count=2)

    assert averaged.shape == (4, 2, 1)
    np.testing.assert_array_equal(averaged.frequency_hz, [1.05e9, 1.25e9])
    assert averaged.visibility[0, 0, 0] == 0
    assert averaged.weight[0, 0, 0] == 1
    assert averaged.visibility[1, 0, 0] == 4.5
    assert averaged.model_visibility is not None
    assert averaged.model_visibility[1, 0, 0] == 9


def test_time_averaging_groups_by_baseline_scan_and_time_bin() -> None:
    averaged = average_time_bins(_block(), bin_seconds=10.0)

    assert averaged.shape == (2, 4, 1)
    np.testing.assert_allclose(averaged.uvw_m[0], np.array([15.0, 22.0, 29.0]) / 7)
    assert averaged.visibility[0, 0, 0] == 2
    assert averaged.weight[0, 0, 0] == 2
    assert averaged.time_s[0] == 39 / 7
    assert averaged.interval_s is not None
    np.testing.assert_array_equal(averaged.interval_s, 10.0)
