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
