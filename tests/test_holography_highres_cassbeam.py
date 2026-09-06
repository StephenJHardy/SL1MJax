from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.cassbeam_highres import HighresCassbeamCatalog
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.holography import (
    AntennaPointingRole,
    HolographyObservation,
    ResolvedAntennaPointing,
)
from sl1mjax.holography_highres_cassbeam import (
    DEFAULT_CONVENTION,
    CassbeamConvention,
    antenna_frame_lm,
    apply_axis_convention,
    apply_jones_convention,
    bootstrap_complex_alpha,
    classify_highres_cassbeam_direct,
    convention_ladder,
    fit_complex_template_alpha,
    inject_cassbeam_template,
    lock_cassbeam_convention,
    manufactured_ep_s_eqh_closure,
    normalize_after_jones_convention,
    resample_cluster_indices,
    run_software_gates,
    software_gates_for_plane,
    template_injection_curve,
)
from sl1mjax.polarization import (
    Correlation,
    ReceptorBasis,
    circular_stokes_to_coherency,
    invert_jones,
)

_TEST_MODEL = "cassbeam_test_highres_schema_v1"
_TEST_RASTER = (9, 9, 2, 2)
_FREQ_HZ = 4.564e9
_PHASE = (0.0, 0.0)
_POSITIONS = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
    ],
    dtype=np.float64,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_test_catalog(root: Path, *, leak: complex = 0.02 + 0.01j) -> HighresCassbeamCatalog:
    (root / "reference").mkdir(parents=True)
    (root / "spw4").mkdir(parents=True)
    geom = root / "reference" / "vla_geom"
    geom.write_text("12.5\n", encoding="utf-8")
    params = root / "spw4" / "vla-cband-4564-g1024-p32.params"
    params.write_text(
        "freq = 4.564\ngridsize = 16\npixelsperbeam = 4\n",
        encoding="utf-8",
    )
    size = 9
    l_origin, m_origin = 3, 4
    gain = np.array([[2.0, 0.1], [0.05, 1.5]], dtype=np.complex128)
    native = np.zeros((size, size, 2, 2), dtype=np.complex128)
    for j in range(size):
        for i in range(size):
            dl = float(i - l_origin)
            dm = float(j - m_origin)
            norm = np.array(
                [
                    [1.0 - 0.01 * (dl * dl + dm * dm), leak * (dl + 1j * dm)],
                    [np.conjugate(leak) * (dm - 1j * dl), 1.0 - 0.008 * (dl * dl + dm * dm)],
                ],
                dtype=np.complex128,
            )
            native[j, i] = gain @ norm
    rows = []
    for j in range(size):
        for i in range(size):
            plane = native[j, i]
            rows.append(
                [
                    plane[0, 0].real,
                    plane[0, 0].imag,
                    plane[1, 0].real,
                    plane[1, 0].imag,
                    plane[0, 1].real,
                    plane[0, 1].imag,
                    plane[1, 1].real,
                    plane[1, 1].imag,
                ]
            )
    data = root / "spw4" / "vla-cband-4564-g1024-p32.jones.dat"
    np.savetxt(data, np.asarray(rows, dtype=np.float64))
    files = {
        "reference/vla_geom": _sha256(geom),
        "spw4/vla-cband-4564-g1024-p32.jones.dat": _sha256(data),
        "spw4/vla-cband-4564-g1024-p32.params": _sha256(params),
    }
    manifest = {
        "model_id": _TEST_MODEL,
        "raster": {"shape": [9, 9, 2, 2], "science_normalization": "inv(E(0)) @ E(s)"},
        "planes": [
            {
                "frequency_mhz": 4564,
                "shape": [9, 9, 2, 2],
                "native_columns": [
                    "Re_RR",
                    "Im_RR",
                    "Re_LR",
                    "Im_LR",
                    "Re_RL",
                    "Im_RL",
                    "Re_LL",
                    "Im_LL",
                ],
                "data": "spw4/vla-cband-4564-g1024-p32.jones.dat",
                "params": "spw4/vla-cband-4564-g1024-p32.params",
            }
        ],
        "files_sha256": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return HighresCassbeamCatalog(root, expected_model_id=_TEST_MODEL, expected_raster=_TEST_RASTER)


def test_highres_loader_normalizes_and_refuses_nearest_frequency(tmp_path: Path) -> None:
    catalog = _write_test_catalog(tmp_path)
    plane = catalog.plane(_FREQ_HZ)
    origin = plane.jones_norm[plane.m_origin_index, plane.l_origin_index]
    assert plane.l_origin_index == 3
    assert plane.m_origin_index == 4
    assert np.max(np.abs(origin - np.eye(2))) < 1.0e-12
    assert np.isclose(plane.jones_norm[4, 4, 0, 1], 0.02 + 0.01j, atol=1.0e-12)
    diag = plane.lookup([0.0], [0.0], off_diagonal=False)[0]
    assert diag[0, 0, 1] == 0.0
    assert diag[0, 1, 0] == 0.0
    with pytest.raises(ValueError, match="nearest-frequency"):
        catalog.plane(4.566e9)
    committed = Path(__file__).parents[1] / "src" / "sl1mjax" / "data" / "cassbeam_cband"
    with pytest.raises(ValueError, match="33×33"):
        HighresCassbeamCatalog(committed)


def test_software_gates_and_manufactured_closure(tmp_path: Path) -> None:
    catalog = _write_test_catalog(tmp_path)
    plane = catalog.plane(_FREQ_HZ)
    measured = np.stack(
        [
            np.array([plane.l_rad[4], plane.m_rad[5]]),
            np.array([plane.l_rad[5], plane.m_rad[4]]),
        ]
    )
    gates = run_software_gates(plane, measured_lm_rad=measured)
    assert gates["passed"] is True
    assert software_gates_for_plane(plane)["passed"] is True
    closure = manufactured_ep_s_eqh_closure(plane)
    assert closure["passed"] is True


def test_convention_ladder_is_finite_and_includes_default() -> None:
    ladder = convention_ladder()
    assert len(ladder) == 128
    assert DEFAULT_CONVENTION in ladder
    assert DEFAULT_CONVENTION.rotate_spatial is False
    assert any(item.rotate_spatial for item in ladder)
    assert any(item.swap_lm and item.l_sign == -1 and item.m_sign == -1 for item in ladder)
    assert all(abs(item.l_sign) == 1 and abs(item.m_sign) == 1 for item in ladder)


def test_template_alpha_and_injection_curve() -> None:
    rng = np.random.default_rng(0)
    t = np.zeros((200, 2, 2), dtype=np.complex128)
    t[:, 0, 1] = 0.03 + 0.01j
    t[:, 1, 0] = 0.025 - 0.008j
    noise = 0.002 * (rng.standard_normal(t.shape) + 1j * rng.standard_normal(t.shape))
    residual = 0.8 * t + noise
    weight = np.ones((200, 2, 2), dtype=np.float64)
    alpha = fit_complex_template_alpha(t, residual, weight)
    assert alpha.real == pytest.approx(0.8, abs=0.05)
    measured = noise.copy()
    diag = np.zeros_like(t)
    full = t
    observation = _tiny_observation(measured)
    geometry = {
        "time_index": np.zeros(200, dtype=np.int64),
        "moving_id": np.zeros(200, dtype=np.int64),
        "reference_id": np.ones(200, dtype=np.int64),
        "cell_l": np.arange(200, dtype=np.int64),
        "cell_m": np.zeros(200, dtype=np.int64),
    }
    train = np.zeros(200, dtype=bool)
    train[:120] = True
    hold = ~train
    curve = template_injection_curve(
        observation,
        full,
        diag,
        train_mask=train,
        holdout_mask=hold,
        geometry=geometry,
    )
    a1 = next(point for point in curve["points"] if point["a"] == 1.0)
    assert a1["holdout_increment"]["inconsistent_with_zero"] is True
    assert a1["holdout_increment"]["real"] == pytest.approx(1.0, abs=0.15)
    assert curve["a1_detectable"] is True
    assert "same template" in curve["matched_filter_identity"]
    injected = inject_cassbeam_template(measured, full, diag, 1.0)
    assert np.allclose(injected, measured + t)


def _tiny_observation(vis_2x2: np.ndarray) -> HolographyObservation:
    n = int(vis_2x2.shape[0])
    packed = np.stack(
        [vis_2x2[:, 0, 0], vis_2x2[:, 0, 1], vis_2x2[:, 1, 0], vis_2x2[:, 1, 1]],
        axis=-1,
    )[:, None, :]
    block = VisibilityBlock(
        uvw_m=np.ones((n, 3), dtype=np.float64),
        frequency_hz=np.array([_FREQ_HZ], dtype=np.float64),
        visibility=packed,
        weight=np.ones((n, 1, 4), dtype=np.float64),
        flag=np.zeros((n, 1, 4), dtype=bool),
        time_s=np.arange(n, dtype=np.float64),
        antenna1=np.zeros(n, dtype=np.int32),
        antenna2=np.ones(n, dtype=np.int32),
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )
    pointing = ResolvedAntennaPointing(
        unique_time_s=np.arange(n, dtype=np.float64),
        antenna_id=np.arange(2, dtype=np.int32),
        offset_lm_rad=np.zeros((n, 2, 2), dtype=np.float64),
        valid=np.ones((n, 2), dtype=bool),
        settled=np.ones((n, 2), dtype=bool),
        role=np.array(
            [["moving", "reference"]] * n,
            dtype="U16",
        ),
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )
    return HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=_POSITIONS[:2],
        calibration_state="casa_parang_true",
        phase_centre_rad=_PHASE,
        stokes_i=1.0,
    )


def test_classifier_outcomes() -> None:
    injection = {"a1_detectable": True}
    improves = {
        "bootstrap_improvement_over_i": {
            "rl": {"fraction_positive": 0.99},
            "lr": {"fraction_positive": 0.98},
        },
        "rr_ll_regression": False,
    }
    no_improve = {
        "bootstrap_improvement_over_i": {
            "rl": {"fraction_positive": 0.2},
            "lr": {"fraction_positive": 0.1},
        },
        "rr_ll_regression": False,
    }
    supported = classify_highres_cassbeam_direct(
        software_gates_passed=True,
        convention_status="locked",
        injection=injection,
        holdout_alpha={"consistent_with_zero": False, "consistent_with_one": True},
        holdout_paired=improves,
        holdout_fitted_paired=improves,
    )
    assert supported["outcome"] == "cassbeam_full_jones_supported"
    mismatch = classify_highres_cassbeam_direct(
        software_gates_passed=True,
        convention_status="locked",
        injection=injection,
        holdout_alpha={"consistent_with_zero": False, "consistent_with_one": False},
        holdout_paired=no_improve,
        holdout_fitted_paired=improves,
    )
    assert mismatch["outcome"] == "cassbeam_morphology_supported_scale_mismatch"
    nonzero_no_help = classify_highres_cassbeam_direct(
        software_gates_passed=True,
        convention_status="locked",
        injection=injection,
        holdout_alpha={"consistent_with_zero": False, "consistent_with_one": False},
        holdout_paired=no_improve,
        holdout_fitted_paired=no_improve,
    )
    assert nonzero_no_help["outcome"] == "cassbeam_offdiagonal_rejected"
    rejected = classify_highres_cassbeam_direct(
        software_gates_passed=True,
        convention_status="locked",
        injection=injection,
        holdout_alpha={"consistent_with_zero": True, "consistent_with_one": False},
        holdout_paired=no_improve,
        holdout_fitted_paired=no_improve,
    )
    assert rejected["outcome"] == "cassbeam_offdiagonal_rejected"
    limited = classify_highres_cassbeam_direct(
        software_gates_passed=True,
        convention_status="locked",
        injection={"a1_detectable": False},
        holdout_alpha=None,
    )
    assert limited["outcome"] == "sensitivity_limited"
    unresolved = classify_highres_cassbeam_direct(
        software_gates_passed=True,
        convention_status="convention_unresolved",
        injection=injection,
        holdout_alpha=None,
    )
    assert unresolved["outcome"] == "convention_unresolved"
    assert unresolved["scientific_decision"] is False
    failed = classify_highres_cassbeam_direct(
        software_gates_passed=False,
        convention_status="locked",
        injection=injection,
        holdout_alpha=None,
    )
    assert failed["outcome"] == "software_gate_failed"
    for report in (supported, mismatch, rejected, limited, unresolved, failed):
        assert report["full_jones_frozen"] is False
        assert report["spw5_closed"] is True


def test_training_only_convention_lock_recovers_manufactured_axis(tmp_path: Path) -> None:
    catalog = _write_test_catalog(tmp_path)
    plane = catalog.plane(_FREQ_HZ)
    n_time = 6
    n_ref = 2
    times = np.arange(n_time, dtype=np.float64)
    rows_t = []
    rows_p = []
    rows_q = []
    offsets = np.zeros((n_time, 1 + n_ref, 2), dtype=np.float64)
    for t_index, time_s in enumerate(times):
        offsets[t_index, n_ref] = np.array(
            [plane.l_rad[4 + (t_index % 2)], plane.m_rad[5]],
            dtype=np.float64,
        )
        for ref in range(n_ref):
            rows_t.append(time_s)
            rows_p.append(ref)
            rows_q.append(n_ref)
    n_row = len(rows_t)
    source = circular_stokes_to_coherency(2.0, 0.0, 0.0, 0.0)
    visibility = np.zeros((n_row, 1, 4), dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.ones((n_row, 3), dtype=np.float64),
        frequency_hz=np.array([_FREQ_HZ], dtype=np.float64),
        visibility=visibility,
        weight=np.ones((n_row, 1, 4), dtype=np.float64),
        flag=np.zeros((n_row, 1, 4), dtype=bool),
        time_s=np.asarray(rows_t, dtype=np.float64),
        antenna1=np.asarray(rows_p, dtype=np.int32),
        antenna2=np.asarray(rows_q, dtype=np.int32),
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )
    role = np.full((n_time, 1 + n_ref), AntennaPointingRole.REFERENCE.value, dtype="U16")
    role[:, n_ref] = AntennaPointingRole.MOVING.value
    pointing = ResolvedAntennaPointing(
        unique_time_s=times,
        antenna_id=np.arange(1 + n_ref, dtype=np.int32),
        offset_lm_rad=offsets,
        valid=np.ones((n_time, 1 + n_ref), dtype=bool),
        settled=np.ones((n_time, 1 + n_ref), dtype=bool),
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )
    observation = HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=_POSITIONS[: 1 + n_ref],
        calibration_state="casa_parang_true",
        phase_centre_rad=_PHASE,
        stokes_i=2.0,
        source_coherency_visibility=np.broadcast_to(source, (n_row, 1, 2, 2)).copy(),
    )
    from sl1mjax.holography_highres_cassbeam import predict_moving_reference_from_highres_cassbeam

    identity = {int(ant): np.eye(2, dtype=np.complex128) for ant in range(1 + n_ref)}
    chi = np.zeros((n_time, 1 + n_ref), dtype=np.float64)
    true = CassbeamConvention(-1, -1, False, "native", False)
    pred = predict_moving_reference_from_highres_cassbeam(
        observation,
        residual_jones=identity,
        catalog=catalog,
        convention=true,
        row_mask=np.ones(n_row, dtype=bool),
        parallactic_angle_rad=chi,
        off_diagonal=True,
    )
    packed_vis = np.stack(
        [pred[:, 0, 0, 0], pred[:, 0, 0, 1], pred[:, 0, 1, 0], pred[:, 0, 1, 1]],
        axis=-1,
    )[:, None, :]
    observation = replace(observation, block=replace(block, visibility=packed_vis))
    geometry = {
        "time_index": np.zeros(n_row, dtype=np.int64),
        "moving_id": np.full(n_row, n_ref, dtype=np.int64),
        "reference_id": np.asarray(rows_p, dtype=np.int64),
        "cell_l": np.zeros(n_row, dtype=np.int64),
        "cell_m": np.zeros(n_row, dtype=np.int64),
    }
    locked = lock_cassbeam_convention(
        observation,
        residual_jones=identity,
        catalog=catalog,
        train_mask=np.ones(n_row, dtype=bool),
        parallactic_angle_rad=chi,
        geometry=geometry,
        candidates=(
            true,
            CassbeamConvention(1, 1, False, "native", False),
            CassbeamConvention(-1, -1, False, "hermitian", True),
        ),
        max_rows=n_row,
    )
    assert locked["status"] == "locked"
    assert locked["selected"] == true.name
    assert "ea26" not in json.dumps(locked["candidates"])


def test_cluster_bootstrap_keeps_replacement_multiplicities() -> None:
    labels = np.array([1, 1, 2, 2, 2], dtype=np.int64)

    class _Fixed:
        def choice(self, unique, size, replace):
            assert replace is True
            return np.array([1, 1], dtype=np.int64)

    index = resample_cluster_indices(labels, _Fixed())
    assert index.tolist() == [0, 1, 0, 1]


def test_jones_convention_normalizes_after_transform(tmp_path: Path) -> None:
    catalog = _write_test_catalog(tmp_path)
    plane = catalog.plane(_FREQ_HZ)
    native = plane.jones_native[5, 4]
    origin = plane.origin_native()
    convention = CassbeamConvention(1, 1, False, "transpose", False, False)
    correct = normalize_after_jones_convention(native, origin, convention)
    already = invert_jones(origin) @ native
    wrong = apply_jones_convention(already, convention)
    assert not np.allclose(correct, wrong)


def test_mount_frame_does_not_spatially_rotate_azelgeo() -> None:
    offset = np.array([[0.003, 0.001]], dtype=np.float64)
    mount = CassbeamConvention(1, 1, False, "native", False, False)
    sky = CassbeamConvention(1, 1, False, "native", False, True)
    l_m, m_m = apply_axis_convention(offset, mount)
    l_s, m_s = apply_axis_convention(offset, sky)
    l_rot, m_rot = antenna_frame_lm(l_s, m_s, np.array([0.5 * np.pi]))
    assert l_m[0] == pytest.approx(0.003)
    assert m_m[0] == pytest.approx(0.001)
    assert not np.isclose(l_rot[0], l_m[0]) or not np.isclose(m_rot[0], m_m[0])


def test_score_template_predictions_uses_only_row_mask() -> None:
    from sl1mjax.holography_highres_cassbeam import score_template_predictions

    n = 6
    measured = np.zeros((n, 2, 2), dtype=np.complex128)
    measured[:, 0, 1] = np.array([1, 1, 1, 10, 10, 10], dtype=np.float64)
    diag = np.zeros_like(measured)
    full = np.zeros_like(measured)
    full[:, 0, 1] = 0.5
    observation = _tiny_observation(measured)
    geometry = {
        "time_index": np.arange(n, dtype=np.int64),
        "moving_id": np.zeros(n, dtype=np.int64),
        "reference_id": np.ones(n, dtype=np.int64),
        "cell_l": np.arange(n, dtype=np.int64),
        "cell_m": np.zeros(n, dtype=np.int64),
    }
    train = np.array([True, True, True, False, False, False])
    scored = score_template_predictions(
        observation,
        full,
        diag,
        row_mask=train,
        stokes_i=np.ones(n),
        geometry=geometry,
    )
    assert scored["n"] == 3
    assert scored["residual_power"]["alpha_0"]["rl"]["n"] == 3
    assert scored["residual_power"]["alpha_0"]["rl"]["median_abs_jy"] == pytest.approx(1.0)
    mixed = score_template_predictions(
        observation,
        full,
        diag,
        row_mask=np.ones(n, dtype=bool),
        stokes_i=np.ones(n),
        geometry=geometry,
    )
    assert mixed["residual_power"]["alpha_0"]["rl"]["median_abs_jy"] > 3.0


def test_complex_alpha_uncertainty_uses_real_and_imaginary_intervals() -> None:
    rng = np.random.default_rng(1)
    n = 240
    t = np.zeros((n, 2, 2), dtype=np.complex128)
    t[:, 0, 1] = 0.04 + 0.01j
    t[:, 1, 0] = 0.03 - 0.012j
    noise = 0.001 * (rng.standard_normal(t.shape) + 1j * rng.standard_normal(t.shape))
    residual = (0.0 + 0.7j) * t + noise
    weight = np.ones((n, 2, 2), dtype=np.float64)
    clusters = np.repeat(np.arange(40), 6)
    report = bootstrap_complex_alpha(t, residual, weight, clusters, n_boot=200, seed=2)
    assert report["consistent_with_zero"] is False
    assert report["consistent_with_one"] is False
    lo_i, hi_i = report["imag_ci95"]
    lo_r, hi_r = report["real_ci95"]
    assert lo_i < 0.7 < hi_i
    assert lo_r <= 0.0 <= hi_r or abs(report["real"]) < 0.15
