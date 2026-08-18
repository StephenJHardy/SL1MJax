import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sl1mjax.sky import RegularGrid, physical_intensity, raw_from_intensity


def test_regular_grid_coordinates_are_centered_and_x64() -> None:
    grid = RegularGrid(size=3, pixel_size_rad=0.1)

    l, m = grid.coordinates

    np.testing.assert_array_equal(
        l.reshape(3, 3),
        np.array([[0.1, 0.0, -0.1], [0.1, 0.0, -0.1], [0.1, 0.0, -0.1]]),
    )
    np.testing.assert_array_equal(
        m.reshape(3, 3),
        np.array([[-0.1, -0.1, -0.1], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]),
    )
    assert l.dtype == np.float64
    assert m.dtype == np.float64


def test_physical_intensity_is_positive_softplus_in_x64() -> None:
    raw = jnp.array([-1000.0, -2.0, 0.0, 3.0, 1000.0], dtype=jnp.float32)

    intensity = physical_intensity(raw)

    assert intensity.dtype == jnp.float64
    assert np.all(np.asarray(intensity) >= 0.0)
    assert np.all(np.asarray(intensity[1:]) > 0.0)
    np.testing.assert_allclose(
        intensity[1:-1],
        np.logaddexp(np.asarray(raw[1:-1], dtype=np.float64), 0.0),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(intensity[-1], 1000.0, rtol=0.0, atol=0.0)


def test_positive_intensity_round_trips_through_raw_parameters() -> None:
    intensity = jnp.array([1e-10, 1e-4, 0.2, 1.0, 50.0], dtype=jnp.float64)

    recovered = physical_intensity(raw_from_intensity(intensity))

    np.testing.assert_allclose(recovered, intensity, rtol=2e-14, atol=1e-15)


def test_positivity_transform_has_sigmoid_derivative() -> None:
    raw = jnp.array([-3.0, 0.0, 2.5], dtype=jnp.float64)
    derivative = jax.vmap(jax.grad(lambda value: physical_intensity(value)))(raw)

    np.testing.assert_allclose(derivative, jax.nn.sigmoid(raw), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("size", "pixel_size"),
    [(1, 0.1), (3, 0.0), (3, np.inf), (4, 0.5)],
)
def test_regular_grid_rejects_nonphysical_geometry(
    size: int, pixel_size: float
) -> None:
    with pytest.raises(ValueError):
        RegularGrid(size=size, pixel_size_rad=pixel_size)
