import jax
import jax.numpy as jnp
import numpy as np

from sl1mjax.objective import sky_prior, weighted_complex_mse
from sl1mjax.polarization import Correlation
from sl1mjax.rime import SPEED_OF_LIGHT_M_S, predict_stokes_i
from sl1mjax.sky import RegularGrid, physical_intensity


def test_weighted_flagged_multi_correlation_loss() -> None:
    prediction = np.array(
        [
            [[2.0 + 1.0j, 9.0 + 9.0j], [0.0 + 1.0j, 2.0 - 1.0j]],
            [[4.0 + 0.0j, 3.0 + 3.0j], [1.0 - 2.0j, -8.0 + 2.0j]],
        ]
    )
    observation = np.array(
        [
            [[1.0 + 1.0j, np.nan + 0.0j], [0.0 + 0.0j, 0.0 - 1.0j]],
            [[2.0 + 0.0j, 1.0 + 1.0j], [1.0 + 0.0j, 100.0 + 100.0j]],
        ]
    )
    weight = np.array(
        [
            [[2.0, 100.0], [1.0, 0.0]],
            [[0.5, np.inf], [3.0, 7.0]],
        ]
    )
    flag = np.array(
        [
            [[False, False], [False, False]],
            [[False, False], [False, True]],
        ]
    )
    # Active squared residuals are 1, 1, 4, and 4 with weights 2, 1, 0.5, and 3.
    expected = (2.0 * 1.0 + 1.0 * 1.0 + 0.5 * 4.0 + 3.0 * 4.0) / 6.5

    actual = weighted_complex_mse(prediction, observation, weight, flag)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_sky_prior_is_normalized_per_pixel_and_per_edge() -> None:
    small = np.array([[1.0, 3.0], [5.0, 7.0]])
    tiled = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)

    small_sparsity = sky_prior(
        small, size=2, sparsity_weight=2.0, smoothness_weight=0.0
    )
    tiled_sparsity = sky_prior(
        tiled, size=4, sparsity_weight=2.0, smoothness_weight=0.0
    )
    smoothness = sky_prior(
        small, size=2, sparsity_weight=0.0, smoothness_weight=3.0
    )

    np.testing.assert_allclose(small_sparsity, 8.0)
    np.testing.assert_allclose(tiled_sparsity, small_sparsity)
    # Horizontal mean square is 4 and vertical mean square is 16.
    np.testing.assert_allclose(smoothness, 3.0 * (4.0 + 16.0) / 2.0)


def test_raw_sky_autodiff_matches_central_finite_differences() -> None:
    grid = RegularGrid(size=2, pixel_size_rad=0.08)
    l, m = grid.coordinates
    uvw_m = np.array(
        [
            [2.0, -1.0, 0.5],
            [-1.5, 0.25, -0.75],
        ]
    )
    frequencies = np.array([0.8, 1.1]) * SPEED_OF_LIGHT_M_S
    antenna1 = np.array([0, 1], dtype=np.int32)
    antenna2 = np.array([1, 0], dtype=np.int32)
    correlations = (Correlation.XX, Correlation.XY, Correlation.YX, Correlation.YY)
    raw = jnp.array([-1.2, -0.4, 0.3, 1.1], dtype=jnp.float64)
    reference_raw = jnp.array([-1.0, -0.2, 0.1, 0.8], dtype=jnp.float64)
    observation = predict_stokes_i(
        physical_intensity(reference_raw),
        l,
        m,
        uvw_m,
        frequencies,
        antenna1,
        antenna2,
        correlations,
        chunk_size=3,
    )
    observation = observation.at[0, 0, 0].add(0.01 - 0.02j)
    weight = jnp.arange(1, observation.size + 1, dtype=jnp.float64).reshape(
        observation.shape
    )
    flag = jnp.zeros(observation.shape, dtype=bool).at[1, 1, 2].set(True)

    def objective(raw_parameters: jax.Array) -> jax.Array:
        intensity = physical_intensity(raw_parameters)
        prediction = predict_stokes_i(
            intensity,
            l,
            m,
            uvw_m,
            frequencies,
            antenna1,
            antenna2,
            correlations,
            chunk_size=3,
        )
        return weighted_complex_mse(prediction, observation, weight, flag) + sky_prior(
            intensity,
            size=grid.size,
            sparsity_weight=0.07,
            smoothness_weight=0.11,
        )

    autodiff = np.asarray(jax.grad(objective)(raw))
    epsilon = 1e-5
    finite_difference = np.empty(raw.shape, dtype=np.float64)
    raw_numpy = np.asarray(raw)
    for index in range(raw.size):
        step = np.zeros(raw.shape, dtype=np.float64)
        step[index] = epsilon
        finite_difference[index] = (
            float(objective(jnp.asarray(raw_numpy + step)))
            - float(objective(jnp.asarray(raw_numpy - step)))
        ) / (2 * epsilon)

    np.testing.assert_allclose(autodiff, finite_difference, rtol=2e-7, atol=2e-9)
