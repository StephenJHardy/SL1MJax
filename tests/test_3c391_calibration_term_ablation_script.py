from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.calibration import baseline_jones, identity_solution
from sl1mjax.data.synthetic import simulate_calibration_case
from sl1mjax.polarization import Correlation

SCRIPT = Path(__file__).parents[1] / "scripts" / "ablate_3c391_calibration_terms.py"


def _module():
    spec = importlib.util.spec_from_file_location("calibration_term_ablation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _solutions():
    base = identity_solution(
        antenna_count=3,
        correlations=(Correlation.RR, Correlation.LL),
        frequency_hz=np.array([1.0e9, 1.1e9]),
        time_s=np.array([0.0, 10.0]),
        reference_antenna=0,
    )
    donor = replace(
        base,
        gains=2.0 * np.ones((3, 3, 2), dtype=np.complex128),
        gain_time_s=np.array([0.0, 5.0, 10.0]),
        gain_valid=np.ones((3, 3, 2), dtype=bool),
        gain_interval_s=np.zeros((3, 3, 2)),
        delays_s=np.full_like(base.delays_s, 3e-9),
        bandpass=(1.0 + 0.2j) * base.bandpass,
    )
    return base, donor


def test_hybrid_solution_substitutes_only_selected_term_groups() -> None:
    module = _module()
    base, donor = _solutions()

    hybrid = module._hybrid_solution(
        base,
        donor,
        casa_gain=True,
        casa_delay=False,
        casa_bandpass=True,
    )

    np.testing.assert_array_equal(hybrid.gains, donor.gains)
    np.testing.assert_array_equal(hybrid.gain_time_s, donor.gain_time_s)
    np.testing.assert_array_equal(hybrid.delays_s, base.delays_s)
    np.testing.assert_array_equal(hybrid.bandpass, donor.bandpass)
    assert hybrid.provenance["ablation_terms"] == {
        "G": "casa",
        "K": "jax",
        "B": "casa",
    }


def test_factorial_candidates_cover_every_g_k_b_combination() -> None:
    module = _module()
    base, donor = _solutions()

    candidates = module._factorial_candidates(base, donor)
    labels = {candidate[0] for candidate in candidates}

    assert len(candidates) == 8
    assert len(labels) == 8
    assert "jax_G_jax_K_jax_B" in labels
    assert "casa_G_casa_K_casa_B" in labels


def test_hybrid_rejects_incompatible_reference_antenna() -> None:
    module = _module()
    base, donor = _solutions()

    with pytest.raises(ValueError, match="reference antennas"):
        module._hybrid_solution(
            base,
            replace(donor, reference_antenna=1),
            casa_gain=True,
            casa_delay=False,
            casa_bandpass=False,
        )


def test_reference_frequency_rebase_preserves_jones_terms() -> None:
    module = _module()
    solution, _ = _solutions()
    delays = np.array([[0.0, 0.0], [3e-9, -2e-9], [1e-9, 4e-9]])
    solution = replace(solution, delays_s=delays)
    rebased = module._rebase_reference_frequency(solution, 1.04e9)
    arguments = (
        np.array([2.0, 8.0]),
        np.array([1.0e9, 1.1e9]),
        np.array([0, 1]),
        np.array([1, 2]),
    )

    original, _ = baseline_jones(solution, *arguments, extrapolate=True)
    changed, _ = baseline_jones(rebased, *arguments, extrapolate=True)

    np.testing.assert_allclose(changed, original, atol=1e-14)


def test_gain_bandpass_canonicalization_preserves_jones_terms() -> None:
    module = _module()
    solution, _ = _solutions()
    factors = np.array([[1.3 + 0.2j, 0.8 - 0.1j], [0.7 + 0.3j, 1.2 + 0.4j], [1.1, 0.9j]])
    solution = replace(
        solution,
        bandpass=solution.bandpass * factors[:, None, :],
    )
    canonical = module._canonicalize_gain_bandpass_gauge(solution)
    arguments = (
        np.array([2.0, 8.0]),
        np.array([1.0e9, 1.1e9]),
        np.array([0, 1]),
        np.array([1, 2]),
    )

    original, _ = baseline_jones(solution, *arguments, extrapolate=True)
    changed, _ = baseline_jones(canonical, *arguments, extrapolate=True)

    np.testing.assert_allclose(changed, original, atol=1e-14)
    reference_index = np.argmin(
        np.abs(canonical.bandpass_frequency_hz - canonical.reference_frequency_hz)
    )
    np.testing.assert_allclose(canonical.bandpass[:, reference_index, :], 1.0)


def test_gain_bandpass_canonicalization_keeps_fully_invalid_term_invalid() -> None:
    module = _module()
    solution, _ = _solutions()
    bandpass_valid = solution.bandpass_valid.copy()
    bandpass_valid[2, :, 1] = False

    canonical = module._canonicalize_gain_bandpass_gauge(
        replace(solution, bandpass_valid=bandpass_valid)
    )

    np.testing.assert_array_equal(canonical.bandpass_valid, bandpass_valid)
    np.testing.assert_array_equal(canonical.bandpass[2, :, 1], solution.bandpass[2, :, 1])


def test_weight_control_preserves_frozen_coordinates_and_visibilities() -> None:
    module = _module()
    case = simulate_calibration_case(
        antenna_count=4,
        time_count=3,
        channel_count=5,
        noise_std=0.0,
        seed=7,
    )

    changed = module._propagate_aligned_weights((case.block,), case.truth)[0]
    baseline, valid = baseline_jones(
        case.truth,
        case.block.time_s,
        case.block.frequency_hz,
        case.block.antenna1,
        case.block.antenna2,
        extrapolate=True,
    )

    np.testing.assert_array_equal(changed.time_s, case.block.time_s)
    np.testing.assert_array_equal(changed.uvw_m, case.block.uvw_m)
    np.testing.assert_array_equal(changed.visibility, case.block.visibility)
    np.testing.assert_allclose(
        changed.weight,
        case.block.weight * np.where(valid, np.abs(baseline) ** 2, 0.0),
    )
    np.testing.assert_array_equal(changed.flag, case.block.flag | ~valid)
