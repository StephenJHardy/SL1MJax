import jax.numpy as jnp
import numpy as np
import pytest

from sl1mjax.polarization import Correlation
from sl1mjax.rime import SPEED_OF_LIGHT_M_S, predict_stokes_i


def _predict(
    intensity,
    l,
    m,
    uvw_m,
    correlations=(Correlation.I,),
    *,
    frequency_hz=(SPEED_OF_LIGHT_M_S,),
    antenna1=None,
    antenna2=None,
    **kwargs,
):
    rows = np.asarray(uvw_m).shape[0]
    return predict_stokes_i(
        intensity,
        l,
        m,
        uvw_m,
        frequency_hz,
        np.zeros(rows, dtype=np.int32) if antenna1 is None else antenna1,
        np.ones(rows, dtype=np.int32) if antenna2 is None else antenna2,
        correlations,
        **kwargs,
    )


def test_zero_baseline_flux_includes_projection() -> None:
    intensity = np.array([1.25, 2.5])
    l = np.array([0.0, 0.3])
    m = np.array([0.0, 0.4])

    actual = _predict(intensity, l, m, np.zeros((2, 3)))
    expected_flux = np.sum(intensity / np.sqrt(1.0 - l**2 - m**2))

    np.testing.assert_allclose(actual[..., 0], expected_flux, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("correlations", "factors"),
    [
        (
            (Correlation.XX, Correlation.XY, Correlation.YX, Correlation.YY),
            (1.0, 0.0, 0.0, 1.0),
        ),
        (
            (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
            (1.0, 0.0, 0.0, 1.0),
        ),
    ],
)
def test_all_linear_and_circular_products_map_stokes_i(
    correlations: tuple[Correlation, ...], factors: tuple[float, ...]
) -> None:
    actual = _predict(
        [3.5],
        [0.0],
        [0.0],
        np.zeros((1, 3)),
        correlations,
        include_projection=False,
    )

    np.testing.assert_array_equal(actual[0, 0], 3.5 * np.asarray(factors))


def test_geometric_phase_matches_measurement_equation() -> None:
    intensity = 2.25
    l = 0.2
    m = -0.1
    uvw = np.array([[1.5, -0.75, 2.0]])
    n = np.sqrt(1.0 - l**2 - m**2)
    expected = intensity * np.exp(
        -2j * np.pi * (uvw[0, 0] * l + uvw[0, 1] * m + uvw[0, 2] * (n - 1.0))
    )

    actual = _predict(
        [intensity], [l], [m], uvw, include_projection=False
    )

    np.testing.assert_allclose(actual[0, 0, 0], expected, rtol=1e-14, atol=1e-14)


def test_reversing_baseline_conjugates_visibility() -> None:
    uvw = np.array(
        [
            [7.0, -3.0, 1.5],
            [-2.0, 4.0, -0.5],
        ]
    )
    forward = _predict([0.5, 1.75], [0.1, -0.2], [-0.15, 0.05], uvw)
    reverse = _predict([0.5, 1.75], [0.1, -0.2], [-0.15, 0.05], -uvw)

    np.testing.assert_allclose(reverse, np.conj(forward), rtol=1e-14, atol=1e-14)


def test_fixed_scalar_gains_apply_ordered_baseline_response() -> None:
    gains = np.array([1.0 + 2.0j, -0.5 + 0.25j, 2.0 - 1.0j])
    antenna1 = np.array([0, 2], dtype=np.int32)
    antenna2 = np.array([1, 0], dtype=np.int32)
    actual = _predict(
        [4.0],
        [0.0],
        [0.0],
        np.zeros((2, 3)),
        antenna1=antenna1,
        antenna2=antenna2,
        fixed_gains=gains,
        include_projection=False,
    )
    expected = 4.0 * gains[antenna1] * np.conj(gains[antenna2])

    np.testing.assert_allclose(actual[..., 0], expected[:, None], rtol=1e-14, atol=1e-14)


def test_chunked_prediction_matches_single_chunk() -> None:
    uvw = np.array(
        [
            [12.0, -4.0, 1.0],
            [-5.0, 8.0, 2.0],
            [3.0, 6.0, -1.0],
        ]
    )
    kwargs = dict(
        intensity=[0.5, 1.0, 2.0],
        l=[-0.1, 0.0, 0.2],
        m=[0.05, -0.15, 0.1],
        uvw_m=uvw,
        frequency_hz=[0.75 * SPEED_OF_LIGHT_M_S, 1.25 * SPEED_OF_LIGHT_M_S],
        correlations=(Correlation.XX, Correlation.XY, Correlation.YX, Correlation.YY),
    )

    one_sample_chunks = _predict(chunk_size=1, **kwargs)
    one_chunk = _predict(chunk_size=uvw.shape[0] * 2, **kwargs)

    np.testing.assert_allclose(one_sample_chunks, one_chunk, rtol=1e-14, atol=1e-14)


def test_prediction_uses_x64_complex_precision() -> None:
    result = _predict(
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.zeros((1, 3), dtype=jnp.float32),
    )

    assert result.dtype == jnp.complex128
