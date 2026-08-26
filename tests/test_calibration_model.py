from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from sl1mjax.calibration import (
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
from sl1mjax.polarization import Correlation, ReceptorBasis


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
    assert restored.reference_antenna == solution.reference_antenna
    np.testing.assert_array_equal(restored.gains, solution.gains)
    np.testing.assert_array_equal(restored.bandpass, solution.bandpass)
