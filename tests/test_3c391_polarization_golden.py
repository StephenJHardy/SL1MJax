from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.calibration import load_casa_calibration_golden
from sl1mjax.polarization import (
    Correlation,
    circular_stokes_from_correlations,
    electric_vector_position_angle_rad,
    fractional_linear_polarisation,
)

SCRIPT = Path(__file__).parents[1] / "scripts" / "export_3c391_polarization_golden.py"
FIXTURE = Path(__file__).parent / "fixtures" / "3c391_polarization_golden.npz"


def _module():
    spec = importlib.util.spec_from_file_location("export_3c391_polarization_golden", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _product_mask(selected: np.ndarray) -> np.ndarray:
    if selected.ndim != 3:
        raise ValueError(f"expected (row, channel, product) mask, got {selected.shape}")
    return np.all(selected, axis=-1)


def _median_stokes(
    visibility: np.ndarray, selected: np.ndarray
) -> tuple[float, float, float, float]:
    keep = _product_mask(selected)
    stokes_i, stokes_q, stokes_u, stokes_v = circular_stokes_from_correlations(
        visibility,
        (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
    )
    return (
        float(np.median(np.real(stokes_i[keep]))),
        float(np.median(np.real(stokes_q[keep]))),
        float(np.median(np.real(stokes_u[keep]))),
        float(np.median(np.real(stokes_v[keep]))),
    )


def test_polarization_export_defaults_are_local() -> None:
    module = _module()
    assert "BagOfWinds" not in str(module.default_measurement_set())
    assert module.default_pol_reference().name == "reference-pol"
    assert module.CORRELATIONS == ("RR", "RL", "LR", "LL")
    assert module.CORRELATION_INDICES == (0, 1, 2, 3)


def test_visibility_case_labels_do_not_collide_with_table_names() -> None:
    module = _module()
    module.assert_disjoint_export_labels()
    visibility_labels = {label for label, *_ in module.VISIBILITY_CASES}
    assert visibility_labels.isdisjoint(module.POL_TABLES)
    assert "leakage_calibrator" in visibility_labels
    assert "dterms" in module.POL_TABLES
    assert "leakage" not in visibility_labels
    assert "leakage" not in module.POL_TABLES


def test_polarization_golden_has_polarised_3c286_and_loadable_3c84() -> None:
    if not FIXTURE.is_file() or not FIXTURE.with_suffix(".json").is_file():
        pytest.skip("polarisation golden has not been exported yet")
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["correlations"] == ["RR", "RL", "LR", "LL"]
    assert metadata["visibility_cases"]["flux_angle"]["field_id"] == 0
    assert metadata["visibility_cases"]["leakage_calibrator"]["field_id"] == 9
    assert metadata["calibration_tables"]["kcross"]["viscal"] == "Kcross Jones"
    assert metadata["calibration_tables"]["dterms"]["viscal"] == "Df Jones"
    assert metadata["calibration_tables"]["angle"]["viscal"] == "Xf Jones"
    assert "flux_polarised_model" in metadata
    assert "constant IQUV" in metadata["flux_polarised_model"]["spectral_model"]

    flux = load_casa_calibration_golden(FIXTURE, label="flux_angle")
    leakage = load_casa_calibration_golden(FIXTURE, label="leakage_calibrator")
    assert flux.block.model_visibility is not None
    assert leakage.block.model_visibility is not None
    assert flux.block.flag.shape == flux.block.visibility.shape
    assert leakage.block.flag.shape == leakage.block.visibility.shape
    assert flux.block.visibility.shape[-1] == 4
    assert leakage.block.antenna1.shape == (leakage.block.visibility.shape[0],)
    assert np.any(np.abs(flux.block.model_visibility[..., 1]) > 0.1)
    assert np.any(np.abs(flux.block.model_visibility[..., 2]) > 0.1)

    model_selected = flux.block.active & np.isfinite(flux.block.model_visibility)
    model_i, model_q, model_u, model_v = _median_stokes(
        flux.block.model_visibility, model_selected
    )
    intended = metadata["flux_polarised_model"]
    assert model_i == pytest.approx(intended["stokes_i_jy"], rel=0.05)
    assert abs(model_q) > 0.1
    assert abs(model_u) > 0.1
    assert model_v == pytest.approx(0.0, abs=0.05)
    model_fraction = float(
        np.median(
            fractional_linear_polarisation(model_q, model_u, model_i)
        )
    )
    assert model_fraction == pytest.approx(0.112, abs=0.01)
    model_angle_deg = np.rad2deg(float(np.arctan2(model_u, model_q)))
    assert model_angle_deg == pytest.approx(66.0, abs=3.0)
    # Casaguide 66° is atan2(U, Q), not IAU EVPA χ = ½ atan2(U, Q).
    assert np.rad2deg(float(electric_vector_position_angle_rad(model_q, model_u))) == pytest.approx(
        33.0, abs=1.5
    )

    corrected_selected = ~flux.post_apply_flag & flux.block.active
    _, corrected_q, corrected_u, corrected_v = _median_stokes(
        flux.corrected_visibility, corrected_selected
    )
    corrected_i, _, _, _ = _median_stokes(flux.corrected_visibility, corrected_selected)
    corrected_fraction = float(
        np.hypot(corrected_q, corrected_u) / corrected_i
    )
    assert corrected_fraction == pytest.approx(0.112, abs=0.04)
    corrected_angle_deg = np.rad2deg(float(np.arctan2(corrected_u, corrected_q)))
    assert corrected_angle_deg == pytest.approx(66.0, abs=10.0)
    assert corrected_v / corrected_i == pytest.approx(0.0, abs=0.05)

    leakage_selected = ~leakage.post_apply_flag & leakage.block.active
    leak_i, leak_q, leak_u, leak_v = _median_stokes(
        leakage.corrected_visibility, leakage_selected
    )
    leak_fraction = float(np.hypot(leak_q, leak_u) / leak_i)
    assert leak_fraction < 0.03
    assert leak_v / leak_i == pytest.approx(0.0, abs=0.05)


def _normalized_rms(actual: np.ndarray, expected: np.ndarray, selected: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.sum(np.abs(actual[selected] - expected[selected]) ** 2)
            / np.sum(np.abs(expected[selected]) ** 2)
        )
    )


def test_imported_kcross_df_xf_reproduces_casa_corrected() -> None:
    from sl1mjax.calibration import apply_calibration, import_casa_polarization_solution

    kbg = Path(__file__).parent / "fixtures" / "3c391_calibration_golden.npz"
    if not FIXTURE.is_file() or not kbg.is_file():
        pytest.skip("polarisation or K/B/G golden is missing")

    for label, overall_limit, parallel_limit, cross_limit, vi_limit, min_keep, min_in_band in (
        ("flux_angle", 0.008, 0.003, 0.03, 1.0e-3, 0.80, 0.95),
        ("leakage_calibrator", 0.015, 0.01, 0.08, 1.0e-3, 0.84, 0.99),
    ):
        case = load_casa_calibration_golden(FIXTURE, label=label)
        solution = import_casa_polarization_solution(FIXTURE, kbg, label=label)
        assert solution.leakage_application == "casa_parallel_preserving"
        corrected = apply_calibration(case.block, solution, extrapolate=True)
        casa_active = ~case.post_apply_flag & case.block.active
        compare = corrected.active & casa_active
        assert np.any(compare)
        assert solution.leakage_frequency_hz is not None
        n_cal = solution.leakage_frequency_hz.size
        extra = case.block.frequency_hz.size - n_cal
        start = extra // 2
        edge = np.ones(case.block.frequency_hz.size, dtype=bool)
        edge[start : start + n_cal] = False
        in_band = ~edge
        retention = compare.sum() / casa_active.sum()
        in_band_casa = casa_active & in_band[None, :, None]
        in_band_keep = compare & in_band[None, :, None]
        assert retention >= min_keep
        assert in_band_keep.sum() / in_band_casa.sum() >= min_in_band
        previously_active_edge = case.block.active & edge[None, :, None]
        assert np.any(previously_active_edge)
        assert np.all(corrected.flag[previously_active_edge])
        rms = _normalized_rms(corrected.visibility, case.corrected_visibility, compare)
        assert rms < overall_limit
        slots = ((0, parallel_limit), (3, parallel_limit), (1, cross_limit), (2, cross_limit))
        for slot, limit in slots:
            assert (
                _normalized_rms(
                    corrected.visibility[..., slot],
                    case.corrected_visibility[..., slot],
                    compare[..., slot],
                )
                < limit
            )
        jax_i, jax_q, jax_u, jax_v = _median_stokes(corrected.visibility, compare)
        casa_i, casa_q, casa_u, casa_v = _median_stokes(case.corrected_visibility, compare)
        assert jax_v / jax_i == pytest.approx(casa_v / casa_i, abs=vi_limit)
        assert abs(jax_v / jax_i) < vi_limit
        if label == "flux_angle":
            assert jax_i == pytest.approx(casa_i, rel=0.02)
            assert jax_q == pytest.approx(casa_q, rel=0.05, abs=0.02)
            assert jax_u == pytest.approx(casa_u, rel=0.05, abs=0.02)
        assert solution.leakage is not None
        channel_keep = compare & in_band[None, :, None]
        jax_stokes = circular_stokes_from_correlations(
            corrected.visibility, case.block.correlations
        )
        casa_stokes = circular_stokes_from_correlations(
            case.corrected_visibility, case.block.correlations
        )
        for jax_term, casa_term in zip(jax_stokes, casa_stokes, strict=True):
            for channel in np.flatnonzero(in_band):
                keep = channel_keep[:, channel, :].all(axis=-1)
                assert np.any(keep)
                jax_median = float(np.median(np.real(jax_term[keep, channel])))
                casa_median = float(np.median(np.real(casa_term[keep, channel])))
                scale = max(abs(casa_median), 0.1)
                assert jax_median == pytest.approx(casa_median, abs=0.05 * scale)
        assert solution.apply_parallactic_angle is True
        assert solution.rl_phase is not None
        assert solution.cross_hand_delay_s is not None
