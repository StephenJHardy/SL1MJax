import numpy as np
import pytest

from sl1mjax.data.synthetic import PointSource, correlations_for_basis, simulate_dataset
from sl1mjax.polarization import (
    Correlation,
    Receptor,
    ReceptorBasis,
    apply_jones_to_coherency,
    correlation_receptor_pair,
    invert_jones,
    pack_coherency,
    receptors_for_correlations,
    stokes_i_to_correlations,
    unpack_coherency,
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


@pytest.mark.parametrize(
    ("correlations", "receptors"),
    [
        ((Correlation.RR, Correlation.LL), (Receptor.R, Receptor.L)),
        (
            (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
            (Receptor.R, Receptor.L),
        ),
        ((Correlation.XX, Correlation.YY), (Receptor.X, Receptor.Y)),
        ((Correlation.RR,), (Receptor.R,)),
    ],
)
def test_receptors_follow_feeds_not_product_count(
    correlations: tuple[Correlation, ...], receptors: tuple[Receptor, ...]
) -> None:
    assert receptors_for_correlations(correlations) == receptors


def test_stokes_i_uses_a_scalar_jones_receptor() -> None:
    assert correlation_receptor_pair(Correlation.I) == (Receptor.I, Receptor.I)
    assert receptors_for_correlations((Correlation.I,)) == (Receptor.I,)


def test_stokes_v_is_not_a_feed_or_scalar_jones() -> None:
    with pytest.raises(ValueError, match="Stokes product"):
        correlation_receptor_pair(Correlation.V)
    with pytest.raises(ValueError, match="Stokes Q/U/V"):
        receptors_for_correlations((Correlation.I, Correlation.V))


def test_pack_unpack_round_trips_and_fills_missing_slots() -> None:
    correlations = (Correlation.RR, Correlation.LL)
    receptors = (Receptor.R, Receptor.L)
    visibility = np.array([[[1.0 + 0.0j, 2.0 + 3.0j]]])

    packed = pack_coherency(visibility, correlations, receptors)
    restored = unpack_coherency(packed, correlations, receptors)

    assert packed.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(packed[..., 0, 0], 1.0)
    np.testing.assert_allclose(packed[..., 1, 1], 2.0 + 3.0j)
    np.testing.assert_allclose(packed[..., 0, 1], 0.0)
    np.testing.assert_allclose(packed[..., 1, 0], 0.0)
    np.testing.assert_allclose(restored, visibility)


def test_circular_stokes_unpack_recovers_qu_from_rl_lr() -> None:
    from sl1mjax.polarization import (
        circular_stokes_from_correlations,
        electric_vector_position_angle_rad,
        fractional_linear_polarisation,
    )

    stokes_i, stokes_q, stokes_u = 7.5, -0.4, 0.7
    rr = stokes_i
    ll = stokes_i
    rl = stokes_q + 1j * stokes_u
    lr = stokes_q - 1j * stokes_u
    visibility = np.array([rr, rl, lr, ll])
    i, q, u, v = circular_stokes_from_correlations(
        visibility,
        (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
    )
    np.testing.assert_allclose(np.real(i), stokes_i)
    np.testing.assert_allclose(np.real(q), stokes_q)
    np.testing.assert_allclose(np.real(u), stokes_u)
    np.testing.assert_allclose(np.real(v), 0.0, atol=1e-12)
    assert fractional_linear_polarisation(q, u, i) == pytest.approx(
        np.hypot(stokes_q, stokes_u) / stokes_i
    )
    assert electric_vector_position_angle_rad(q, u) == pytest.approx(
        0.5 * np.arctan2(stokes_u, stokes_q)
    )


def test_leakage_jones_mixes_stokes_i_into_cross_hands() -> None:
    intensity = 4.0 + 0.0j
    sky = intensity * np.eye(2, dtype=np.complex128)
    leakage = 0.05 + 0.02j
    jones_p = np.array([[1.0, leakage], [leakage, 1.0]], dtype=np.complex128)
    jones_q = np.eye(2, dtype=np.complex128)

    observed = apply_jones_to_coherency(sky, jones_p, jones_q)
    products = unpack_coherency(
        observed,
        (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        (Receptor.R, Receptor.L),
    )

    expected = jones_p @ sky @ jones_q.conj().T
    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(products[1], intensity * leakage)
    np.testing.assert_allclose(products[2], intensity * leakage)
    corrected = apply_jones_to_coherency(
        observed, invert_jones(jones_p), invert_jones(jones_q)
    )
    np.testing.assert_allclose(corrected, sky, atol=1e-14)


def test_leakage_jones_matrices_match_casa_df_layout() -> None:
    from sl1mjax.polarization import leakage_jones_matrices, receptor_phase_jones

    d_r, d_l = 0.04 - 0.02j, -0.03 + 0.01j
    matrices = leakage_jones_matrices(np.array([d_r, d_l]))
    np.testing.assert_allclose(matrices, [[1.0, d_r], [d_l, 1.0]])
    phase = receptor_phase_jones(np.exp(1j * 0.3))
    np.testing.assert_allclose(phase[0, 0], np.exp(1j * 0.3))
    np.testing.assert_allclose(phase[1, 1], 1.0)
