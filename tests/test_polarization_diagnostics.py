from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.calibration import (
    apply_calibration,
    import_casa_polarization_solution,
    load_casa_calibration_golden,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.polarization_diagnostics import (
    calibrator_polarization_floor,
    deterministic_calibrator_cohort_split,
    deterministic_visibility_partitions,
    dirty_mosaic_stokes_images,
    dirty_stokes_images,
    evaluate_global_fractional_polarization,
    fit_global_fractional_polarization,
    fit_global_fractional_polarization_blocks,
    fit_partitioned_global_polarization,
)
from sl1mjax.polarization_inference import solve_cross_hand_delay, solve_rl_phase
from sl1mjax.sky import RegularGrid

POL_FIXTURE = Path(__file__).parent / "fixtures" / "3c391_polarization_golden.npz"
KBG_FIXTURE = Path(__file__).parent / "fixtures" / "3c391_calibration_golden.npz"


def _point_source_block(
    *,
    stokes_i: float,
    q: float,
    u: float,
    v: float,
    rows: int = 48,
    channels: int = 4,
    phase: np.ndarray | None = None,
) -> VisibilityBlock:
    frequency_hz = np.linspace(4.55e9, 4.65e9, channels)
    rng = np.random.default_rng(3)
    uvw_m = rng.normal(0.0, 200.0, size=(rows, 3))
    uvw_m[:, 2] = 0.0
    if phase is None:
        model_i = np.full((rows, channels), stokes_i, dtype=np.complex128)
    else:
        model_i = np.asarray(phase, dtype=np.complex128) * stokes_i
    visibility = np.zeros((rows, channels, 4), dtype=np.complex128)
    visibility[..., 0] = model_i * (1.0 + v)
    visibility[..., 3] = model_i * (1.0 - v)
    visibility[..., 1] = model_i * (q + 1j * u)
    visibility[..., 2] = model_i * (q - 1j * u)
    model_visibility = np.zeros_like(visibility)
    model_visibility[..., 0] = model_i
    model_visibility[..., 3] = model_i
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        visibility=visibility,
        model_visibility=model_visibility,
        weight=np.ones_like(visibility, dtype=np.float64),
        flag=np.zeros(visibility.shape, dtype=bool),
        time_s=np.repeat(np.arange(max(rows // 8, 2), dtype=np.float64), 8)[:rows],
        antenna1=np.arange(rows, dtype=np.int32) % 5,
        antenna2=(
            ((np.arange(rows, dtype=np.int32) % 5) + 1 + (np.arange(rows) // 5) % 3) % 6
        ).astype(np.int32),
        scan_id=np.where(np.arange(rows) < rows // 2, 1, 2).astype(np.int32),
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )


def test_fit_global_fractional_polarization_recovers_constant_quv() -> None:
    block = _point_source_block(stokes_i=4.0, q=0.08, u=-0.05, v=0.01)
    result = fit_global_fractional_polarization(block)
    assert result.q == pytest.approx(0.08, abs=1e-12)
    assert result.u == pytest.approx(-0.05, abs=1e-12)
    assert result.v == pytest.approx(0.01, abs=1e-12)
    assert result.polarized_linear_loss == pytest.approx(0.0, abs=1e-12)
    assert result.polarized_linear_loss < result.null_linear_loss
    assert result.polarized_v_loss < result.null_v_loss
    assert result.provenance["regressor"] == "complex_stokes_i_model"
    assert result.provenance["spatial_image"] is False
    assert result.provenance["rm"] is False


def test_complex_model_i_recovers_quv_through_visibility_nulls() -> None:
    rows, channels = 64, 6
    frequency_hz = np.linspace(4.55e9, 4.65e9, channels)
    rng = np.random.default_rng(11)
    uvw_m = rng.normal(0.0, 400.0, size=(rows, 3))
    uvw_m[:, 2] = 0.0
    l_rad = 4.0e-3
    wavelength_m = 299792458.0 / frequency_hz
    phase = np.exp(
        2j * np.pi * uvw_m[:, 0, None] * l_rad / wavelength_m[None, :]
    )
    q, u, v = 0.07, -0.04, 0.012
    block = _point_source_block(
        stokes_i=3.5, q=q, u=u, v=v, rows=rows, channels=channels, phase=phase
    )
    assert float(np.min(np.abs(np.real(phase)))) < 0.05
    result = fit_global_fractional_polarization(block)
    assert result.q == pytest.approx(q, abs=1e-12)
    assert result.u == pytest.approx(u, abs=1e-12)
    assert result.v == pytest.approx(v, abs=1e-12)

    observed_i = 0.5 * (block.visibility[..., 0] + block.visibility[..., 3])
    with pytest.raises(ValueError, match="Stokes-I model"):
        fit_global_fractional_polarization(replace(block, model_visibility=None))
    del observed_i


def test_fit_refuses_observed_intensity_when_model_is_missing() -> None:
    block = replace(
        _point_source_block(stokes_i=2.0, q=0.01, u=0.0, v=0.0),
        model_visibility=None,
    )
    with pytest.raises(ValueError, match="frozen complex Stokes-I"):
        fit_global_fractional_polarization(block)


def test_joint_and_partition_fits_are_deterministic() -> None:
    first = _point_source_block(stokes_i=4.0, q=0.08, u=-0.05, v=0.01)
    second = _point_source_block(stokes_i=4.0, q=0.08, u=-0.05, v=0.01, rows=40)
    joint = fit_global_fractional_polarization_blocks((first, second))
    assert joint.q == pytest.approx(0.08, abs=1e-12)
    partitions = deterministic_visibility_partitions(first)
    assert set(partitions) == {
        "baseline_even",
        "baseline_odd",
        "time_even",
        "time_odd",
        "channel_even",
        "channel_odd",
    }
    again = deterministic_visibility_partitions(first)
    for name, mask in partitions.items():
        np.testing.assert_array_equal(mask, again[name])
    fitted = fit_partitioned_global_polarization((first,))
    for result in fitted.values():
        assert result.q == pytest.approx(0.08, abs=1e-12)
        assert result.u == pytest.approx(-0.05, abs=1e-12)


def test_evaluate_global_fit_matches_direct_loss() -> None:
    block = _point_source_block(stokes_i=5.0, q=0.1, u=0.02, v=0.0)
    fitted = fit_global_fractional_polarization(block)
    scored = evaluate_global_fractional_polarization(
        block, fitted.q, fitted.u, fitted.v
    )
    assert scored["polarized_linear_loss"] == pytest.approx(
        fitted.polarized_linear_loss, abs=1e-12
    )
    null = evaluate_global_fractional_polarization(block, 0.0, 0.0, 0.0)
    assert null["polarized_linear_loss"] == pytest.approx(
        fitted.null_linear_loss, abs=1e-12
    )


def test_dirty_stokes_images_peak_at_phase_centre() -> None:
    block = _point_source_block(stokes_i=5.0, q=0.1, u=0.04, v=0.0)
    images = dirty_stokes_images(block, RegularGrid(17, np.deg2rad(2 / 3600)))
    centre = (8, 8)
    peak_index = np.unravel_index(np.argmax(images.stokes_i), images.stokes_i.shape)
    assert tuple(int(axis) for axis in peak_index) == centre
    assert images.peak_i == pytest.approx(5.0, rel=0.05)
    assert images.peak_q / images.peak_i == pytest.approx(0.1, abs=0.02)
    assert images.peak_u / images.peak_i == pytest.approx(0.04, abs=0.02)
    assert abs(images.peak_v / images.peak_i) < 0.02
    assert images.provenance["evidence_grade"] is False


def test_mosaic_dirty_stokes_is_invariant_to_duplicating_a_pointing() -> None:
    block = _point_source_block(stokes_i=3.0, q=0.05, u=-0.02, v=0.0, rows=24)
    grid = RegularGrid(9, np.deg2rad(4 / 3600))
    config = DirectDFTConfig(visibility_chunk_size=16, pixel_chunk_size=32)
    single = dirty_mosaic_stokes_images(
        (block,),
        grid,
        block.phase_centre_rad,
        labels=("C1",),
        config=config,
        minimum_sensitivity_fraction=0.0,
    )
    duplicated = dirty_mosaic_stokes_images(
        (block, block),
        grid,
        block.phase_centre_rad,
        labels=("C1", "C2"),
        config=config,
        minimum_sensitivity_fraction=0.0,
    )
    np.testing.assert_allclose(duplicated.stokes_q, single.stokes_q)
    np.testing.assert_allclose(duplicated.stokes_u, single.stokes_u)
    assert duplicated.recurrence["pairs"]["C1_C2"]["q"]["cosine"] == pytest.approx(1.0)
    assert duplicated.recurrence["pairs"]["C1_C2"]["u"]["cosine"] == pytest.approx(1.0)


def test_calibrator_floor_labels_apply_back_and_in_sample() -> None:
    if not POL_FIXTURE.is_file() or not KBG_FIXTURE.is_file():
        pytest.skip("polarisation or K/B/G golden is missing")
    flux = load_casa_calibration_golden(POL_FIXTURE, label="flux_angle")
    leakage = load_casa_calibration_golden(POL_FIXTURE, label="leakage_calibrator")
    casa_flux = replace(
        flux.block,
        visibility=flux.corrected_visibility,
        flag=flux.block.flag | flux.post_apply_flag,
    )
    casa_leakage = replace(
        leakage.block,
        visibility=leakage.corrected_visibility,
        flag=leakage.block.flag | leakage.post_apply_flag,
    )
    flux_floor = calibrator_polarization_floor(
        casa_flux, independence="apply_back", label="flux_angle"
    )
    leak_floor = calibrator_polarization_floor(
        casa_leakage, independence="in_sample", label="leakage_calibrator"
    )
    assert flux_floor.independence == "apply_back"
    assert leak_floor.independence == "in_sample"
    assert flux_floor.fractional_linear == pytest.approx(0.112, abs=0.02)
    assert np.rad2deg(flux_floor.casaguide_angle_rad) == pytest.approx(66.0, abs=5.0)
    assert abs(flux_floor.v) < 1.0e-3
    assert flux_floor.residual_v is not None
    assert abs(flux_floor.residual_v) < 1.0e-3
    assert leak_floor.fractional_linear < 0.03
    assert abs(leak_floor.v) < 5.0e-3

    fitted = fit_global_fractional_polarization(casa_flux)
    assert fitted.fractional_linear == pytest.approx(0.112, abs=0.02)
    assert np.rad2deg(fitted.casaguide_angle_rad) == pytest.approx(66.0, abs=5.0)
    assert abs(fitted.v) < 1.0e-3
    assert fitted.polarized_linear_loss < fitted.null_linear_loss

    solution = import_casa_polarization_solution(
        POL_FIXTURE, KBG_FIXTURE, label="flux_angle"
    )
    jax_corrected = apply_calibration(flux.block, solution, extrapolate=True)
    jax_floor = calibrator_polarization_floor(
        jax_corrected, independence="apply_back", label="flux_angle_jax"
    )
    assert jax_floor.fractional_linear == pytest.approx(
        flux_floor.fractional_linear, abs=0.01
    )
    assert abs(jax_floor.v) < 1.0e-3


def test_3c286_scan_split_is_held_out_not_independent() -> None:
    if not POL_FIXTURE.is_file() or not KBG_FIXTURE.is_file():
        pytest.skip("polarisation or K/B/G golden is missing")
    flux = load_casa_calibration_golden(POL_FIXTURE, label="flux_angle")
    split = deterministic_calibrator_cohort_split(flux.block)
    assert split.strategy == "scan_holdout_last"
    assert np.any(split.train) and np.any(split.holdout)
    assert not np.any(split.train & split.holdout)

    imported = import_casa_polarization_solution(
        POL_FIXTURE, KBG_FIXTURE, label="flux_angle"
    )
    kcross = solve_cross_hand_delay(flux.block, imported, split=split)
    with_leakage = replace(
        kcross.solution,
        leakage=imported.leakage,
        leakage_frequency_hz=imported.leakage_frequency_hz,
        leakage_valid=imported.leakage_valid,
        leakage_application=imported.leakage_application,
    )
    angle = solve_rl_phase(flux.block, with_leakage, split=split)
    corrected = apply_calibration(flux.block, angle.solution, extrapolate=True)
    holdout = calibrator_polarization_floor(
        corrected,
        independence="held_out_calibrator",
        label="flux_angle_holdout",
        sample_mask=split.holdout,
    )
    assert holdout.independence == "held_out_calibrator"
    assert holdout.fractional_linear == pytest.approx(0.112, abs=0.03)
    assert abs(holdout.v) < 2.0e-3
