import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from sl1mjax.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationSolution,
    align_solution_gauge,
    apply_calibration,
    baseline_jones,
    corrupt_model,
    identity_solution,
    read_calibration,
    write_calibration,
)
from sl1mjax.calibration_terms import geodetic_latitude_rad, parallactic_angle_rad
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, Receptor, ReceptorBasis


def _solution() -> CalibrationSolution:
    solution = identity_solution(
        antenna_count=3,
        correlations=(Correlation.RR, Correlation.LL),
        frequency_hz=np.array([1.0e9, 1.1e9]),
        time_s=np.array([0.0, 10.0]),
        reference_antenna=0,
    )
    gains = np.array(
        [
            [[1.0, 1.0], [1.2j, 0.8], [0.7 - 0.2j, 1.1j]],
            [[1.0, 1.0], [1.0 + 0.3j, 0.9 - 0.1j], [1.2, 0.8j]],
        ],
        dtype=np.complex128,
    )
    delays = np.array([[0.0, 0.0], [2.0e-9, -1.0e-9], [-3.0e-9, 4.0e-9]])
    bandpass = np.ones((3, 2, 2), dtype=np.complex128)
    bandpass[:, 1, :] = np.array([[1.0, 1.0], [0.9 + 0.1j, 1.1 - 0.1j], [1.2 - 0.2j, 0.8 + 0.2j]])
    return replace(solution, gains=gains, delays_s=delays, bandpass=bandpass)


def _block(visibility: np.ndarray) -> VisibilityBlock:
    return VisibilityBlock(
        uvw_m=np.zeros((6, 3)),
        frequency_hz=np.array([1.0e9, 1.1e9]),
        visibility=visibility,
        weight=np.full(visibility.shape, 4.0),
        flag=np.zeros(visibility.shape, dtype=bool),
        time_s=np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0]),
        antenna1=np.array([0, 0, 1, 0, 0, 1]),
        antenna2=np.array([1, 2, 2, 1, 2, 2]),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )


def test_corrupt_and_apply_recover_model_and_propagate_weights() -> None:
    solution = _solution()
    model = np.arange(24, dtype=np.float64).reshape(6, 2, 2) + 1j * np.linspace(
        0.0, 1.0, 24
    ).reshape(6, 2, 2)
    corrupted = np.asarray(
        corrupt_model(
            model,
            solution,
            time_s=_block(model).time_s,
            frequency_hz=np.array([1.0e9, 1.1e9]),
            antenna1=_block(model).antenna1,
            antenna2=_block(model).antenna2,
        )
    )
    block = _block(corrupted)

    corrected = apply_calibration(block, solution, propagate_weights=True)
    baseline, _ = baseline_jones(
        solution,
        block.time_s,
        block.frequency_hz,
        block.antenna1,
        block.antenna2,
    )

    np.testing.assert_allclose(corrected.visibility, model, atol=2e-14)
    np.testing.assert_allclose(corrected.weight, 4.0 * np.abs(baseline) ** 2)
    np.testing.assert_array_equal(block.visibility, corrupted)


def test_baseline_uses_jp_conjugate_jq_and_is_gauge_invariant() -> None:
    solution = replace(
        _solution(),
        delays_s=np.zeros((3, 2)),
        bandpass=np.ones((3, 2, 2), dtype=np.complex128),
    )
    block = _block(np.ones((6, 2, 2), dtype=np.complex128))
    baseline, _ = baseline_jones(
        solution,
        block.time_s,
        block.frequency_hz,
        block.antenna1,
        block.antenna2,
    )
    assert baseline[0, 0, 0] == pytest.approx(
        solution.gains[0, 0, 0] * np.conj(solution.gains[0, 1, 0])
    )

    phase = np.exp(1j * np.array([[0.3, -0.2], [1.1, 0.7]]))
    shifted = replace(solution, gains=solution.gains * phase[:, None, :])
    shifted_baseline, _ = baseline_jones(
        align_solution_gauge(shifted),
        block.time_s,
        block.frequency_hz,
        block.antenna1,
        block.antenna2,
    )
    np.testing.assert_allclose(shifted_baseline, baseline, atol=2e-14)


def test_validity_domain_requires_explicit_extrapolation() -> None:
    solution = _solution()
    block = replace(
        _block(np.ones((6, 2, 2), dtype=np.complex128)),
        time_s=np.full(6, 20.0),
    )
    with pytest.raises(ValueError, match="validity domain"):
        apply_calibration(block, solution)
    corrected = apply_calibration(block, solution, extrapolate=True)
    assert np.all(np.isfinite(corrected.visibility))


def test_linear_gain_interpolation_unwraps_phase_and_interpolates_amplitude() -> None:
    solution = identity_solution(
        antenna_count=2,
        correlations=(Correlation.RR, Correlation.LL),
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0, 10.0]),
    )
    gains = solution.gains.copy()
    gains[0, 1, :] = np.exp(1j * np.deg2rad(170.0))
    gains[1, 1, :] = 3.0 * np.exp(1j * np.deg2rad(-170.0))
    linear = replace(solution, gains=gains, interpolation="linear")

    baseline, valid = baseline_jones(
        linear,
        np.array([5.0]),
        np.array([1.0e9]),
        np.array([0]),
        np.array([1]),
        extrapolate=True,
    )

    np.testing.assert_allclose(baseline, -2.0 + 0.0j, atol=1e-14)
    assert np.asarray(valid).all()


def test_solution_is_pytree_and_round_trips(tmp_path: Path) -> None:
    solution = _solution()
    leaves = jax.tree_util.tree_leaves(solution)
    assert len(leaves) == 9
    path = tmp_path / "solution.npz"
    write_calibration(solution, path)

    restored = read_calibration(path)

    assert restored.correlations == solution.correlations
    assert restored.receptors == (Receptor.R, Receptor.L)
    assert restored.reference_antenna == solution.reference_antenna
    np.testing.assert_array_equal(restored.gains, solution.gains)
    np.testing.assert_array_equal(restored.bandpass, solution.bandpass)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert metadata["receptors"] == ["R", "L"]


def test_schema_v1_promotes_receptors_from_parallel_hands(tmp_path: Path) -> None:
    solution = _solution()
    path = tmp_path / "legacy.npz"
    write_calibration(solution, path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("receptors")
    metadata["schema_version"] = 1
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    restored = read_calibration(path)

    assert restored.receptors == (Receptor.R, Receptor.L)
    assert restored.receptor_count == 2
    assert restored.provenance["promoted_from_schema"] == 1
    np.testing.assert_array_equal(restored.gains, solution.gains)


def test_schema_v2_promotes_without_leakage_application(tmp_path: Path) -> None:
    solution = _solution()
    path = tmp_path / "schema_v2.npz"
    write_calibration(solution, path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("leakage_application")
    metadata["schema_version"] = 2
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    restored = read_calibration(path)

    assert restored.leakage_application == "exact"
    assert restored.provenance["promoted_from_schema"] == 2
    np.testing.assert_array_equal(restored.gains, solution.gains)


def test_identity_four_products_uses_two_receptors_not_four() -> None:
    solution = identity_solution(
        antenna_count=2,
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0]),
    )
    assert solution.receptors == (Receptor.R, Receptor.L)
    assert solution.gains.shape == (1, 2, 2)
    assert solution.delays_s.shape == (2, 2)
    assert solution.bandpass.shape == (2, 1, 2)


def test_diagonal_jones_corrects_cross_hands_on_four_product_data() -> None:
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    solution = identity_solution(
        antenna_count=2,
        correlations=correlations,
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0]),
    )
    gains = solution.gains.copy()
    gains[0, 0, :] = np.array([1.5 + 0.0j, 0.5 + 0.0j])
    gains[0, 1, :] = np.array([0.8 - 0.2j, 1.2 + 0.4j])
    solution = replace(solution, gains=gains)
    sky = np.array([[[3.0 + 0.0j, 0.4 + 0.1j, 0.2 - 0.3j, 2.5 + 0.0j]]])
    corrupted = np.asarray(
        corrupt_model(
            sky,
            solution,
            time_s=np.array([0.0]),
            frequency_hz=np.array([1.0e9]),
            antenna1=np.array([0]),
            antenna2=np.array([1]),
        )
    )
    g_p = gains[0, 0]
    g_q = gains[0, 1]
    expected = np.array(
        [
            [
                [
                    g_p[0] * sky[0, 0, 0] * np.conj(g_q[0]),
                    g_p[0] * sky[0, 0, 1] * np.conj(g_q[1]),
                    g_p[1] * sky[0, 0, 2] * np.conj(g_q[0]),
                    g_p[1] * sky[0, 0, 3] * np.conj(g_q[1]),
                ]
            ]
        ]
    )
    np.testing.assert_allclose(corrupted, expected)
    block = VisibilityBlock(
        uvw_m=np.zeros((1, 3)),
        frequency_hz=np.array([1.0e9]),
        visibility=corrupted,
        weight=np.ones_like(corrupted, dtype=np.float64),
        flag=np.zeros(corrupted.shape, dtype=bool),
        time_s=np.array([0.0]),
        antenna1=np.array([0]),
        antenna2=np.array([1]),
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    corrected = apply_calibration(block, solution)
    np.testing.assert_allclose(corrected.visibility, sky, atol=1e-14)


def test_two_receptor_solution_applies_to_four_product_block() -> None:
    solution = _solution()
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    n_row, n_chan = 6, 2
    sky = np.ones((n_row, n_chan, 4), dtype=np.complex128)
    sky[..., 1] = 0.1 + 0.02j
    sky[..., 2] = 0.05 - 0.04j
    four_product = replace(
        _block(np.ones((n_row, n_chan, 2), dtype=np.complex128)),
        visibility=sky,
        weight=np.ones((n_row, n_chan, 4)),
        flag=np.zeros((n_row, n_chan, 4), dtype=bool),
        correlations=correlations,
    )
    four_product = replace(
        four_product,
        visibility=np.asarray(
            corrupt_model(
                sky,
                replace(solution, correlations=correlations),
                time_s=four_product.time_s,
                frequency_hz=four_product.frequency_hz,
                antenna1=four_product.antenna1,
                antenna2=four_product.antenna2,
            )
        ),
    )
    corrected = apply_calibration(four_product, solution)
    np.testing.assert_allclose(corrected.visibility, sky, atol=2e-14)


def test_kcross_leakage_and_rl_phase_round_trip() -> None:
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    solution = identity_solution(
        antenna_count=2,
        correlations=correlations,
        frequency_hz=np.array([1.0e9, 1.1e9]),
        time_s=np.array([0.0]),
    )
    leakage = np.zeros((2, 2, 2), dtype=np.complex128)
    leakage[0, :, 0] = 0.05 + 0.01j
    leakage[1, :, 1] = -0.04 + 0.02j
    rl_phase = np.ones((2, 2), dtype=np.complex128)
    rl_phase[0] = np.exp(1j * 0.4)
    solution = replace(
        solution,
        cross_hand_delay_s=np.array([[2.0e-9, 0.0], [0.0, 0.0]]),
        cross_hand_delay_valid=np.ones((2, 2), dtype=bool),
        leakage=leakage,
        leakage_frequency_hz=np.array([1.0e9, 1.1e9]),
        leakage_valid=np.ones((2, 2, 2), dtype=bool),
        rl_phase=rl_phase,
        rl_phase_frequency_hz=np.array([1.0e9, 1.1e9]),
        rl_phase_valid=np.ones((2, 2), dtype=bool),
    )
    sky = np.array(
        [[[3.0, 0.4 + 0.2j, 0.4 - 0.2j, 3.0], [3.1, 0.35 + 0.1j, 0.35 - 0.1j, 3.1]]],
        dtype=np.complex128,
    )
    block = VisibilityBlock(
        uvw_m=np.zeros((1, 3)),
        frequency_hz=np.array([1.0e9, 1.1e9]),
        visibility=np.asarray(
            corrupt_model(
                sky,
                solution,
                time_s=np.array([0.0]),
                frequency_hz=np.array([1.0e9, 1.1e9]),
                antenna1=np.array([0]),
                antenna2=np.array([1]),
            )
        ),
        weight=np.ones_like(sky, dtype=np.float64),
        flag=np.zeros(sky.shape, dtype=bool),
        time_s=np.array([0.0]),
        antenna1=np.array([0]),
        antenna2=np.array([1]),
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    corrected = apply_calibration(block, solution)
    np.testing.assert_allclose(corrected.visibility, sky, atol=2e-12)


def test_flagged_leakage_flags_visibility_not_identity() -> None:
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    solution = identity_solution(
        antenna_count=2,
        correlations=correlations,
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0]),
    )
    junk = np.array([[[1.05 - 0.05j, 0.95 - 0.05j]], [[0.0, 0.0]]])
    solution = replace(
        solution,
        leakage=junk,
        leakage_frequency_hz=np.array([1.0e9]),
        leakage_valid=np.array([[[False, False]], [[True, True]]]),
    )
    sky = np.array([[[4.0, 0.0, 0.0, 4.0]]], dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.zeros((1, 3)),
        frequency_hz=np.array([1.0e9]),
        visibility=sky,
        weight=np.ones(sky.shape, dtype=np.float64),
        flag=np.zeros(sky.shape, dtype=bool),
        time_s=np.array([0.0]),
        antenna1=np.array([0]),
        antenna2=np.array([1]),
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    with pytest.raises(ValueError, match="validity domain"):
        apply_calibration(block, solution)
    corrected = apply_calibration(block, solution, extrapolate=True)
    assert np.all(corrected.flag)
    np.testing.assert_array_equal(corrected.visibility, np.zeros_like(sky))


def test_leakage_frequency_domain_is_not_silently_extrapolated() -> None:
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    solution = identity_solution(
        antenna_count=2,
        correlations=correlations,
        frequency_hz=np.array([1.0e9, 1.1e9]),
        time_s=np.array([0.0]),
    )
    solution = replace(
        solution,
        leakage=np.zeros((2, 1, 2), dtype=np.complex128),
        leakage_frequency_hz=np.array([1.0e9]),
        leakage_valid=np.ones((2, 1, 2), dtype=bool),
    )
    sky = np.ones((1, 2, 4), dtype=np.complex128)
    sky[..., 1:3] = 0.0
    block = VisibilityBlock(
        uvw_m=np.zeros((1, 3)),
        frequency_hz=np.array([1.0e9, 1.1e9]),
        visibility=sky,
        weight=np.ones(sky.shape, dtype=np.float64),
        flag=np.zeros(sky.shape, dtype=bool),
        time_s=np.array([0.0]),
        antenna1=np.array([0]),
        antenna2=np.array([1]),
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    with pytest.raises(ValueError, match="validity domain"):
        apply_calibration(block, solution)
    corrected = apply_calibration(block, solution, extrapolate=True)
    assert not np.any(corrected.flag[:, 0, :])
    assert np.all(corrected.flag[:, 1, :])


def test_propagate_weights_rejected_when_leakage_present() -> None:
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    solution = identity_solution(
        antenna_count=2,
        correlations=correlations,
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0]),
    )
    solution = replace(
        solution,
        leakage=np.zeros((2, 1, 2), dtype=np.complex128),
        leakage_frequency_hz=np.array([1.0e9]),
        leakage_valid=np.ones((2, 1, 2), dtype=bool),
    )
    sky = np.array([[[4.0, 0.0, 0.0, 4.0]]], dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.zeros((1, 3)),
        frequency_hz=np.array([1.0e9]),
        visibility=sky,
        weight=np.ones(sky.shape, dtype=np.float64),
        flag=np.zeros(sky.shape, dtype=bool),
        time_s=np.array([0.0]),
        antenna1=np.array([0]),
        antenna2=np.array([1]),
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    with pytest.raises(ValueError, match="propagate_weights"):
        apply_calibration(block, solution, propagate_weights=True)


def test_casa_parallel_preserving_keeps_parallel_hands() -> None:
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    solution = identity_solution(
        antenna_count=2,
        correlations=correlations,
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0]),
    )
    leakage = np.full((2, 1, 2), 0.08 + 0.02j)
    solution = replace(
        solution,
        leakage=leakage,
        leakage_frequency_hz=np.array([1.0e9]),
        leakage_valid=np.ones((2, 1, 2), dtype=bool),
    )
    sky = np.array([[[4.0, 0.4 + 0.2j, 0.4 - 0.2j, 4.0]]], dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.zeros((1, 3)),
        frequency_hz=np.array([1.0e9]),
        visibility=sky,
        weight=np.ones(sky.shape, dtype=np.float64),
        flag=np.zeros(sky.shape, dtype=bool),
        time_s=np.array([0.0]),
        antenna1=np.array([0]),
        antenna2=np.array([1]),
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    exact = apply_calibration(block, solution)
    casa = apply_calibration(
        block, replace(solution, leakage_application="casa_parallel_preserving")
    )
    np.testing.assert_allclose(casa.visibility[..., 0], sky[..., 0], atol=1e-12)
    np.testing.assert_allclose(casa.visibility[..., 3], sky[..., 3], atol=1e-12)
    assert np.max(np.abs(exact.visibility[..., 0] - sky[..., 0])) > 1e-3
    assert np.max(np.abs(casa.visibility[..., 1] - exact.visibility[..., 1])) < 1e-12


def _leakage_solution(
    correlations: tuple[Correlation, ...], *, casa: bool = False
) -> CalibrationSolution:
    solution = identity_solution(
        antenna_count=2,
        correlations=correlations,
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0]),
    )
    return replace(
        solution,
        leakage=np.full((2, 1, 2), 0.08 + 0.02j),
        leakage_frequency_hz=np.array([1.0e9]),
        leakage_valid=np.ones((2, 1, 2), dtype=bool),
        leakage_application="casa_parallel_preserving" if casa else "exact",
    )


def test_flagged_cross_hand_does_not_contaminate_exact_parallel_outputs() -> None:
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    solution = _leakage_solution(correlations)
    sky = np.array([[[4.0, 1.0e6 + 0.0j, 0.0, 4.0]]], dtype=np.complex128)
    flag = np.zeros(sky.shape, dtype=bool)
    flag[..., 1] = True
    block = VisibilityBlock(
        uvw_m=np.zeros((1, 3)),
        frequency_hz=np.array([1.0e9]),
        visibility=sky,
        weight=np.ones(sky.shape, dtype=np.float64),
        flag=flag,
        time_s=np.array([0.0]),
        antenna1=np.array([0]),
        antenna2=np.array([1]),
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    exact = apply_calibration(block, solution)
    assert np.all(exact.flag)
    np.testing.assert_array_equal(exact.visibility, np.zeros_like(sky))
    casa = apply_calibration(block, _leakage_solution(correlations, casa=True))
    assert np.all(casa.flag[..., 1])
    assert np.all(casa.flag[..., 2])
    assert not np.any(casa.flag[..., 0])
    assert not np.any(casa.flag[..., 3])
    np.testing.assert_allclose(casa.visibility[..., 0], 4.0, atol=1e-12)
    np.testing.assert_allclose(casa.visibility[..., 3], 4.0, atol=1e-12)


def test_exact_leakage_refuses_incomplete_coherency() -> None:
    four = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    two = (Correlation.RR, Correlation.LL)
    sky = np.array([[[4.0, 4.0]]], dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.zeros((1, 3)),
        frequency_hz=np.array([1.0e9]),
        visibility=sky,
        weight=np.ones(sky.shape, dtype=np.float64),
        flag=np.zeros(sky.shape, dtype=bool),
        time_s=np.array([0.0]),
        antenna1=np.array([0]),
        antenna2=np.array([1]),
        correlations=two,
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    with pytest.raises(ValueError, match="complete two-feed coherency"):
        apply_calibration(block, _leakage_solution(four))
    casa = apply_calibration(block, _leakage_solution(four, casa=True))
    np.testing.assert_allclose(casa.visibility, sky, atol=1e-12)
    assert not np.any(casa.flag)


def test_corrupt_model_refuses_casa_parallel_preserving() -> None:
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    sky = np.array([[[4.0, 0.4, 0.4, 4.0]]], dtype=np.complex128)
    with pytest.raises(ValueError, match="apply-only CASA oracle"):
        corrupt_model(
            sky,
            _leakage_solution(correlations, casa=True),
            time_s=np.array([0.0]),
            frequency_hz=np.array([1.0e9]),
            antenna1=np.array([0]),
            antenna2=np.array([1]),
        )


def test_circular_polarization_terms_require_rl_receptors() -> None:
    solution = identity_solution(
        antenna_count=2,
        correlations=(Correlation.XX, Correlation.YY),
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0]),
    )
    with pytest.raises(ValueError, match=r"receptors \(R, L\)"):
        replace(
            solution,
            leakage=np.zeros((2, 1, 2), dtype=np.complex128),
            leakage_frequency_hz=np.array([1.0e9]),
            leakage_valid=np.ones((2, 1, 2), dtype=bool),
        )
    with pytest.raises(ValueError, match=r"receptors \(R, L\)"):
        replace(
            solution,
            apply_parallactic_angle=True,
            antenna_position_m=np.array(
                [[-1601.0e3, -5042.0e3, 3554.0e3], [-1601.0e3, -5042.0e3, 3554.0e3]]
            ),
        )


def test_polarization_fields_round_trip(tmp_path: Path) -> None:
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    solution = identity_solution(
        antenna_count=2,
        correlations=correlations,
        frequency_hz=np.array([1.0e9]),
        time_s=np.array([0.0]),
    )
    solution = replace(
        solution,
        antenna_position_m=np.array(
            [[-1601.0e3, -5042.0e3, 3554.0e3], [-1601.0e3, -5042.0e3, 3554.0e3]]
        ),
        cross_hand_delay_s=np.array([[1.0e-9, 0.0], [1.0e-9, 0.0]]),
        cross_hand_delay_valid=np.ones((2, 2), dtype=bool),
        leakage=np.zeros((2, 1, 2), dtype=np.complex128),
        leakage_frequency_hz=np.array([1.0e9]),
        leakage_valid=np.ones((2, 1, 2), dtype=bool),
        rl_phase=np.ones((2, 1), dtype=np.complex128),
        rl_phase_frequency_hz=np.array([1.0e9]),
        rl_phase_valid=np.ones((2, 1), dtype=bool),
        apply_parallactic_angle=True,
    )
    path = tmp_path / "pol_solution.npz"
    write_calibration(solution, path)
    restored = read_calibration(path)
    np.testing.assert_array_equal(restored.cross_hand_delay_s, solution.cross_hand_delay_s)
    np.testing.assert_array_equal(restored.leakage, solution.leakage)
    np.testing.assert_array_equal(restored.rl_phase, solution.rl_phase)
    assert restored.apply_parallactic_angle is True
    assert restored.leakage_application == "exact"
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["apply_parallactic_angle"] is True
    assert metadata["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert metadata["leakage_application"] == "exact"


def test_parallactic_angle_uses_wgs84_geodetic_latitude() -> None:
    position = np.array([[-1_601_185.0, -5_041_977.0, 3_554_875.0]])
    geocentric = float(np.arctan2(position[0, 2], np.hypot(position[0, 0], position[0, 1])))
    geodetic = float(geodetic_latitude_rad(position)[0])
    assert np.rad2deg(geodetic) == pytest.approx(34.08, abs=0.05)
    assert np.rad2deg(geodetic - geocentric) == pytest.approx(0.18, abs=0.05)
    chi = parallactic_angle_rad(
        np.array([0.0]),
        (np.deg2rad(202.78), np.deg2rad(10.58)),
        position,
    )
    assert np.isfinite(chi).all()
