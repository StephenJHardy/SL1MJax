import numpy as np
import pytest

from sl1mjax.data.synthetic import PointSource, correlations_for_basis, simulate_dataset
from sl1mjax.polarization import (
    Correlation,
    ReceptorBasis,
    stokes_i_to_correlations,
    validate_correlations,
)
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import RegularGrid


@pytest.mark.parametrize(
    ("basis", "correlations"),
    [
        (
            ReceptorBasis.LINEAR,
            (Correlation.XX, Correlation.XY, Correlation.YX, Correlation.YY),
        ),
        (
            ReceptorBasis.CIRCULAR,
            (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        ),
        (
            ReceptorBasis.STOKES,
            (Correlation.I, Correlation.Q, Correlation.U, Correlation.V),
        ),
        (ReceptorBasis.LINEAR, (Correlation.YY,)),
        (ReceptorBasis.CIRCULAR, (Correlation.LR, Correlation.RR)),
        (ReceptorBasis.STOKES, (Correlation.V, Correlation.I)),
    ],
)
def test_validate_correlations_accepts_valid_ordered_subsets(
    basis: ReceptorBasis, correlations: tuple[Correlation, ...]
) -> None:
    validate_correlations(basis, correlations)


@pytest.mark.parametrize(
    ("basis", "correlations", "message"),
    [
        (ReceptorBasis.LINEAR, (), "at least one correlation"),
        (
            ReceptorBasis.CIRCULAR,
            (Correlation.RR, Correlation.RR),
            "correlations must be unique",
        ),
        (
            ReceptorBasis.LINEAR,
            (Correlation.XX, Correlation.RR),
            "linear basis does not support",
        ),
        (
            ReceptorBasis.CIRCULAR,
            (Correlation.RR, Correlation.I),
            "circular basis does not support",
        ),
        (
            ReceptorBasis.STOKES,
            (Correlation.I, Correlation.XX),
            "stokes basis does not support",
        ),
    ],
)
def test_validate_correlations_rejects_empty_duplicate_and_mixed_products(
    basis: ReceptorBasis,
    correlations: tuple[Correlation, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_correlations(basis, correlations)


@pytest.mark.parametrize(
    ("correlations", "factors"),
    [
        (
            (Correlation.YY, Correlation.XY, Correlation.XX, Correlation.YX),
            (1.0, 0.0, 1.0, 0.0),
        ),
        (
            (Correlation.LR, Correlation.LL, Correlation.RR, Correlation.RL),
            (0.0, 1.0, 1.0, 0.0),
        ),
        (
            (Correlation.V, Correlation.I, Correlation.Q, Correlation.U),
            (0.0, 1.0, 0.0, 0.0),
        ),
    ],
)
def test_stokes_i_mapping_preserves_shape_order_and_complex_values(
    correlations: tuple[Correlation, ...], factors: tuple[float, ...]
) -> None:
    stokes_i = np.array([[1.0 + 2.0j, -3.0 + 0.5j], [4.0 - 1.0j, 2.5 + 0.0j]])

    actual = np.asarray(stokes_i_to_correlations(stokes_i, correlations))

    assert actual.shape == (2, 2, 4)
    np.testing.assert_allclose(actual, stokes_i[..., None] * np.asarray(factors))


@pytest.mark.parametrize("basis", list(ReceptorBasis))
def test_synthetic_dataset_contains_exact_multi_correlation_truth(
    basis: ReceptorBasis,
) -> None:
    grid = RegularGrid(size=7, pixel_size_rad=2.0e-4)
    sources = (
        PointSource(flux=1.25, l=0.0, m=0.0),
        PointSource(flux=0.4, l=4.0e-4, m=-2.0e-4),
    )

    dataset = simulate_dataset(
        grid,
        basis=basis,
        sources=sources,
        rows=9,
        channels=3,
        antennas=4,
        noise_std=0.0,
        seed=17,
    )
    block = dataset.blocks[0]
    expected = np.asarray(
        predict_stokes_i(
            np.asarray([source.flux for source in sources]),
            np.asarray([source.l for source in sources]),
            np.asarray([source.m for source in sources]),
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            correlations_for_basis(basis),
        )
    )

    assert block.receptor_basis is basis
    assert block.correlations == correlations_for_basis(basis)
    assert block.visibility.shape == (9, 3, 4)
    np.testing.assert_allclose(block.visibility, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(
        block.visibility[..., 0] == 0,
        np.broadcast_to(
            block.correlations[0]
            not in {
                Correlation.I,
                Correlation.XX,
                Correlation.YY,
                Correlation.RR,
                Correlation.LL,
            },
            block.visibility[..., 0].shape,
        ),
    )
    assert dataset.provenance == block.provenance
    assert dataset.provenance["truth"] == [
        {"flux": 1.25, "l": 0.0, "m": 0.0},
        {"flux": 0.4, "l": 4.0e-4, "m": -2.0e-4},
    ]
