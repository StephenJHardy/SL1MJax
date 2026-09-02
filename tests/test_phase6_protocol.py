from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sl1mjax.beam_aware_imaging import sky_table_to_records
from sl1mjax.beam_conventions import require_beam_calibration_state
from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import ManufacturedVoltageBeam, integration_plan_from_table
from sl1mjax.inference import InferenceConfig
from sl1mjax.integration_planner import IntegrationTolerance
from sl1mjax.phase6_protocol import (
    HOLDOUT_FOLD,
    SEALED_FOLD,
    compare_operator_modes,
    phase6_folds,
    poison_sealed_visibilities,
    sky_and_plan_from_product,
    validate_phase6_masks,
    write_reconstruction_products,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.voltage_reconstruction import (
    VoltageReconstructionConfig,
    reconstruct_voltage_stokes_i,
    screen_virtual_splits,
    starting_central_table,
)

_ANTENNA_POSITION_M = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
        [-1_601_050.0, -5_042_200.0, 3_553_850.0],
    ]
)
_PHASE = (np.deg2rad(282.35), np.deg2rad(-0.93))
_IDENTITY = np.eye(2, dtype=np.complex128)
_ROOT = np.deg2rad(16.0 / 3600.0)


def _block() -> VisibilityBlock:
    times = 5.0e9 + 60.0 * np.arange(10)
    dummy = np.zeros((times.size, 1, 2), dtype=np.complex128)
    return VisibilityBlock(
        uvw_m=np.repeat(np.array([[40.0, -20.0, 3.0]]), times.size, axis=0),
        frequency_hz=np.array([4.536e9]),
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=times,
        antenna1=np.zeros(times.size, dtype=np.int32),
        antenna2=np.ones(times.size, dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )


def _config() -> VoltageReconstructionConfig:
    return VoltageReconstructionConfig(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        inference=InferenceConfig(
            solver="proximal_sgd",
            batch_grouping="times",
            steps=4,
            learning_rate=0.2,
            sparsity_weight=0.0,
            patience=4,
            validation_interval=2,
            min_delta=1e-12,
            batch_size_rows=8,
        ),
        tolerance=IntegrationTolerance(max_depth=1, forced_feature_depth=0),
        max_rounds=0,
        max_depth=1,
        leaf_penalty=0.0,
    )


def test_phase6_folds_are_disjoint_and_leave_fold_4_sealed() -> None:
    block = _block()
    folds = phase6_folds((block,))
    validate_phase6_masks((block,), folds.train, folds.holdout, folds.sealed)
    assert folds.holdout_fold == HOLDOUT_FOLD
    assert folds.sealed_fold == SEALED_FOLD
    assert not np.any(folds.train[0] & folds.holdout[0])
    assert not np.any(folds.train[0] & folds.sealed[0])
    assert not np.any(folds.holdout[0] & folds.sealed[0])
    assert np.any(folds.sealed[0])
    with pytest.raises(ValueError, match="disjoint"):
        validate_phase6_masks((block,), folds.train, folds.train, folds.sealed)


def test_phase6_fold_4_visibilities_cannot_affect_fit_or_splits() -> None:
    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=np.array([0.0, 0.0, 0.0, 0.8]),
    )
    block = _block()
    folds = phase6_folds((block,))
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, hess_ll=_IDENTITY * 12.0)
    config = _config()
    kwargs = dict(
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        train_masks=folds.train,
        holdout_masks=folds.holdout,
        config=config,
        beam_mode="manufactured",
    )
    clean = reconstruct_voltage_stokes_i(table, block, beam, **kwargs)
    poisoned_blocks = poison_sealed_visibilities((block,), folds.sealed)
    dirty = reconstruct_voltage_stokes_i(table, poisoned_blocks[0], beam, **kwargs)
    np.testing.assert_allclose(clean.fit.flux, dirty.fit.flux, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(clean.fit.train_loss, dirty.fit.train_loss, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(clean.fit.holdout_loss, dirty.fit.holdout_loss, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(clean.fit.kkt_residual, dirty.fit.kkt_residual, rtol=0.0, atol=1e-10)
    state = require_beam_calibration_state("casa_parang_true")
    clean_screen = screen_virtual_splits(
        clean.fit,
        (block,),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=state,
        train_masks=folds.train,
        config=config,
    )
    dirty_screen = screen_virtual_splits(
        dirty.fit,
        poisoned_blocks,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=state,
        train_masks=folds.train,
        config=config,
    )
    assert [item.leaf for item in clean_screen] == [item.leaf for item in dirty_screen]
    for left, right in zip(clean_screen, dirty_screen, strict=True):
        np.testing.assert_allclose(left.gradient, right.gradient, rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(
            left.predicted_improvement, right.predicted_improvement, rtol=0.0, atol=1e-10
        )


def test_phase6_products_omit_sealed_fold_samples(tmp_path) -> None:
    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=np.array([0.0, 0.0, 0.0, 0.8]),
    )
    block = _block()
    folds = phase6_folds((block,))
    result = reconstruct_voltage_stokes_i(
        table,
        block,
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        train_masks=folds.train,
        holdout_masks=folds.holdout,
        config=_config(),
        beam_mode="manufactured",
    )
    payload = write_reconstruction_products(
        tmp_path,
        result,
        (block,),
        pointing_ids=("C1",),
        antenna_position_m=_ANTENNA_POSITION_M,
        folds=folds,
        config={"stage": "test"},
        manifest={"commit": "test"},
    )
    assert payload["sealed_fold_unused"] is True
    assert payload["pointing_ids"] == ["C1"]
    product = np.load(tmp_path / "pointing_C1.npz")
    assert np.all(np.isnan(product["visibility"][folds.sealed[0]]))
    assert np.all(np.isnan(product["prediction"][folds.sealed[0]].real))
    image = np.load(tmp_path / "intrinsic_stokes_i.npz")
    assert "pixel_size_rad" in image
    assert "phase_centre_ra_rad" in image
    assert int(image["grid_size"]) == image["image"].shape[0]


def test_phase6_hysteresis_roundtrips_through_records() -> None:
    from sl1mjax.quadtree import QuadtreeLeaf
    from sl1mjax.refinement import MergeHysteresisState
    from sl1mjax.voltage_reconstruction import (
        merge_hysteresis_from_records,
        merge_hysteresis_to_records,
    )

    state = MergeHysteresisState(
        eligible_streak={QuadtreeLeaf(0, 1, 1): 1},
        split_cooldown={QuadtreeLeaf(0, 0, 0): 1},
    )
    restored = merge_hysteresis_from_records(merge_hysteresis_to_records(state))
    assert restored.eligible_streak[QuadtreeLeaf(0, 1, 1)] == 1
    assert restored.split_cooldown[QuadtreeLeaf(0, 0, 0)] == 1
    result = reconstruct_voltage_stokes_i(
        starting_central_table(
            root_size=2,
            root_pixel_size_rad=_ROOT,
            mosaic_phase_centre_rad=_PHASE,
            flux=np.full(4, 0.4),
        ),
        _block(),
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
        beam_mode="manufactured",
        hysteresis=state,
    )
    assert result.hysteresis is not None


def test_compare_operator_modes_passes_on_manufactured_batch() -> None:
    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=np.array([0.2, 0.0, 0.1, 0.4]),
    )
    plan = integration_plan_from_table(table)
    empty = _block()
    block = replace(empty, visibility=np.full(empty.visibility.shape, 0.15 + 0.02j))
    flux = np.array([item.stokes_i_jy for item in table.components if item.active])
    report = compare_operator_modes(
        flux,
        block,
        plan,
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=4, pixel_chunk_size=2),
        rtol=1e-8,
        atol=1e-10,
    )
    assert report["passed"]
    assert report["explicit_s"] >= 0.0
    assert report["parent_count"] == flux.size
    assert report["plan_sha256"]
    assert report["explicit_kernel_builds"] <= 1
    assert report["sampled_rows"] == block.uvw_m.shape[0]
    assert report["product"] is None
    assert report["source_hashes"] is None


def test_sky_and_plan_from_product_rebuilds_planner_depths(tmp_path) -> None:
    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=np.array([0.2, 0.0, 0.1, 0.4]),
    )
    depths = {
        component.component_id: 1
        for component in table.components
        if component.active
    }
    plan = integration_plan_from_table(table, depth_by_parent=depths)
    (tmp_path / "checkpoint.json").write_text(
        json.dumps(
            {
                "mosaic_phase_centre_rad": list(table.mosaic_phase_centre_rad),
                "source": table.source,
                "components": sky_table_to_records(table),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "integration_plan.json").write_text(
        json.dumps(
            {
                "parent_id": list(plan.parent_id),
                "parent_count": plan.parent_count,
                "node_count": plan.node_count,
                "depths": depths,
            }
        ),
        encoding="utf-8",
    )
    _table, flux, rebuilt = sky_and_plan_from_product(tmp_path)
    assert rebuilt.node_count == plan.node_count
    assert rebuilt.parent_count == plan.parent_count
    assert flux.size == plan.parent_count
    assert rebuilt.node_count > plan.parent_count


def _write_gate_product(directory, *, beam_mode="static_scalar", **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "beam_mode": beam_mode,
        "train_loss": 0.01,
        "holdout_loss": 0.012,
        "kkt_residual": 0.007,
        "audit_under_resolved": True,
        "sealed_fold_unused": True,
        "stop_reason": "maximum_rounds",
        "guard": {
            "status": "inactive",
            "field_expansion": False,
            "activated": [],
            "n_scored": 10,
            "accepted": False,
        },
        "config": {"sky_max_depth": 2, "integration_max_depth": 3},
        "manifest": {
            "train_folds": [0, 1, 2],
            "holdout_fold": 3,
            "sealed_fold": 4,
        },
    }
    payload.update(overrides)
    (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    for name in ("checkpoint.json", "component_table.json", "integration_plan.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    return directory


def test_review_phase6_product_accepts_recorded_gates(tmp_path) -> None:
    from sl1mjax.phase6_protocol import review_phase6_product

    directory = _write_gate_product(tmp_path / "commissioning" / "static_scalar")
    review = review_phase6_product(directory, stage="commissioning", beam_mode="static_scalar")
    assert review.present
    assert review.passed
    assert review.failures == ()


def test_review_phase6_product_rejects_field_expansion_and_full_jones(tmp_path) -> None:
    from sl1mjax.phase6_protocol import review_phase6_product

    expansion = _write_gate_product(
        tmp_path / "expand",
        guard={"status": "field_expansion_required", "field_expansion": True},
    )
    failed = review_phase6_product(expansion, stage="commissioning", beam_mode="static_scalar")
    assert failed.passed is False
    assert any("field-expansion" in item for item in failed.failures)

    jones = _write_gate_product(tmp_path / "jones", beam_mode="full_jones")
    refused = review_phase6_product(jones, stage="commissioning", beam_mode="full_jones")
    assert refused.passed is False
    assert any("full Jones" in item for item in refused.failures)


def test_guard_curvature_stays_within_predict_batches() -> None:
    from sl1mjax.beam_aware_imaging import ComponentFamily, SkyBasisType, SkyComponent
    from sl1mjax.phase6_protocol import _guard_curvature

    component = SkyComponent(
        component_id="outer_guard:guard:0:0:0",
        family=ComponentFamily.OUTER_GUARD,
        basis_type=SkyBasisType.UNIFORM_SQUARE,
        l_rad=0.01,
        m_rad=0.01,
        stokes_i_jy=0.0,
        width_rad=_ROOT,
        level=0,
        iy=0,
        ix=0,
    )
    block = _block()
    value = _guard_curvature(
        component,
        (block,),
        (np.asarray(block.active),),
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=require_beam_calibration_state("casa_parang_true"),
        mosaic_phase_centre_rad=_PHASE,
        batch_size_rows=2,
    )
    assert np.isfinite(value)
    assert value >= 0.0


def test_review_phase6_output_reports_missing_ladder_products(tmp_path) -> None:
    from sl1mjax.phase6_protocol import (
        phase6_commissioning_complete,
        phase6_ladder_complete,
        phase6_output_complete,
        review_phase6_output,
    )

    _write_gate_product(tmp_path / "commissioning" / "static_scalar")
    reviews = review_phase6_output(
        tmp_path, stages=("commissioning",), beams=("static_scalar", "streamed_scalar")
    )
    assert reviews[0].passed
    assert reviews[1].present is False
    assert phase6_output_complete(reviews) is False
    assert phase6_commissioning_complete(reviews) is False
    _write_gate_product(
        tmp_path / "commissioning" / "streamed_scalar", beam_mode="streamed_scalar"
    )
    _write_gate_product(
        tmp_path / "commissioning" / "diagonal_copolar", beam_mode="diagonal_copolar"
    )
    _write_gate_product(tmp_path / "commissioning-c4" / "static_scalar")
    _write_gate_product(
        tmp_path / "commissioning-c4" / "streamed_scalar", beam_mode="streamed_scalar"
    )
    _write_gate_product(
        tmp_path / "commissioning-c4" / "diagonal_copolar", beam_mode="diagonal_copolar"
    )
    commissioning = review_phase6_output(
        tmp_path,
        stages=("commissioning", "commissioning-c4"),
    )
    assert phase6_commissioning_complete(commissioning)
    assert phase6_output_complete(review_phase6_output(tmp_path)) is False
    assert phase6_ladder_complete(review_phase6_output(tmp_path)) is False


def test_select_topology_round_beams_keeps_airy_and_best_detailed(tmp_path) -> None:
    from sl1mjax.phase6_protocol import (
        commissioning_ready,
        select_topology_round_beams,
    )

    _write_gate_product(tmp_path / "commissioning" / "static_scalar", holdout_loss=0.014)
    _write_gate_product(
        tmp_path / "commissioning" / "streamed_scalar",
        beam_mode="streamed_scalar",
        holdout_loss=0.015,
    )
    _write_gate_product(
        tmp_path / "commissioning" / "diagonal_copolar",
        beam_mode="diagonal_copolar",
        holdout_loss=0.018,
    )
    for beam, mode in (
        ("static_scalar", "static_scalar"),
        ("streamed_scalar", "streamed_scalar"),
        ("diagonal_copolar", "diagonal_copolar"),
    ):
        _write_gate_product(tmp_path / "commissioning-c4" / beam, beam_mode=mode)
    report = commissioning_ready(tmp_path, require_compare=False)
    assert report["ready"]
    assert report["compare_passed"] is None
    no_dir = commissioning_ready(tmp_path, require_compare=True)
    assert no_dir["ready"] is False
    assert no_dir["compare_passed"] is False
    assert any("compare directory is required" in item for item in no_dir["compare_failures"])
    assert report["topology_beams"] == ("static_scalar", "streamed_scalar")
    assert select_topology_round_beams(report["reviews"]) == (
        "static_scalar",
        "streamed_scalar",
    )
    compare_dir = tmp_path / "compare"
    compare_dir.mkdir()
    (compare_dir / "operator_compare_static_scalar.json").write_text(
        '{"passed": false}', encoding="utf-8"
    )
    blocked = commissioning_ready(
        tmp_path, compare_dir=compare_dir, require_compare=True
    )
    assert blocked["ready"] is False
    assert blocked["compare_passed"] is False
    assert any("missing compare report" in item for item in blocked["compare_failures"])


def test_phase6_ladder_complete_requires_selected_topology_pair(tmp_path) -> None:
    from sl1mjax.phase6_protocol import (
        phase6_ladder_complete,
        phase6_output_complete,
        review_phase6_output,
    )

    for stage in ("commissioning", "commissioning-c4", "baseline"):
        for beam in ("static_scalar", "streamed_scalar", "diagonal_copolar"):
            holdout = {"streamed_scalar": 0.015, "diagonal_copolar": 0.018}.get(beam, 0.014)
            _write_gate_product(
                tmp_path / stage / beam,
                beam_mode=beam,
                holdout_loss=holdout,
            )
    reviews = review_phase6_output(tmp_path)
    assert phase6_ladder_complete(reviews) is False
    _write_gate_product(tmp_path / "full_round1" / "static_scalar", holdout_loss=0.014)
    _write_gate_product(
        tmp_path / "full_round1" / "streamed_scalar",
        beam_mode="streamed_scalar",
        holdout_loss=0.015,
    )
    reviews = review_phase6_output(tmp_path)
    assert phase6_ladder_complete(reviews) is True
    assert phase6_output_complete(reviews) is False


def _write_compare_report(directory, beam, product, *, passed=True, **overrides):
    from sl1mjax.phase6_protocol import (
        EXPLICIT_COMPARE_BEAM_MODEL_IDS,
        EXPLICIT_COMPARE_GEOMETRY,
        sha256_file,
    )

    geometry = EXPLICIT_COMPARE_GEOMETRY[beam]
    payload = {
        "passed": passed,
        "beam_mode": beam,
        "operator_mode": "explicit_jax",
        "product": str(product),
        "plan_sha256": "a" * 64,
        "source_hashes": {
            "checkpoint.json": sha256_file(product / "checkpoint.json"),
        },
        "batch_size_rows": geometry["batch_size_rows"],
        "pixel_chunk_size": geometry["pixel_chunk_size"],
        "visibility_chunk_size": geometry["visibility_chunk_size"],
        "beam_model_id": EXPLICIT_COMPARE_BEAM_MODEL_IDS[beam],
    }
    payload.update(overrides)
    path = directory / f"operator_compare_{beam}.json"
    path.write_text(
        json.dumps({"passed": passed, "reports": [payload]}),
        encoding="utf-8",
    )
    return path


def test_commissioning_ready_requires_sealed_compare_per_beam(tmp_path) -> None:
    from sl1mjax.phase6_protocol import (
        commissioning_ready,
        validate_explicit_compare_report,
    )

    for beam, mode in (
        ("static_scalar", "static_scalar"),
        ("streamed_scalar", "streamed_scalar"),
        ("diagonal_copolar", "diagonal_copolar"),
    ):
        _write_gate_product(tmp_path / "commissioning" / beam, beam_mode=mode)
        _write_gate_product(tmp_path / "commissioning-c4" / beam, beam_mode=mode)
    compare_dir = tmp_path / "compare"
    compare_dir.mkdir()
    (compare_dir / "operator_compare_static_scalar.json").write_text(
        '{"passed": true}', encoding="utf-8"
    )
    bare = commissioning_ready(
        tmp_path, compare_dir=compare_dir, require_compare=True
    )
    assert bare["ready"] is False
    assert any("missing compare report" in item for item in bare["compare_failures"])

    static = tmp_path / "commissioning" / "static_scalar"
    _write_compare_report(compare_dir, "static_scalar", static)
    one_beam = commissioning_ready(
        tmp_path, compare_dir=compare_dir, require_compare=True
    )
    assert one_beam["ready"] is False
    assert any(
        "missing compare report for streamed_scalar" in item
        for item in one_beam["compare_failures"]
    )
    assert any(
        "missing compare report for diagonal_copolar" in item
        for item in one_beam["compare_failures"]
    )

    _write_compare_report(
        compare_dir,
        "streamed_scalar",
        tmp_path / "commissioning" / "streamed_scalar",
        operator_mode="vjp",
    )
    _write_compare_report(
        compare_dir,
        "diagonal_copolar",
        tmp_path / "commissioning" / "diagonal_copolar",
    )
    stale_mode = commissioning_ready(
        tmp_path, compare_dir=compare_dir, require_compare=True
    )
    assert stale_mode["ready"] is False
    assert any("explicit_jax" in item for item in stale_mode["compare_failures"])

    _write_compare_report(
        compare_dir,
        "streamed_scalar",
        tmp_path / "commissioning" / "streamed_scalar",
    )
    sealed = commissioning_ready(
        tmp_path, compare_dir=compare_dir, require_compare=True
    )
    assert sealed["ready"]
    assert sealed["compare_passed"]
    assert sealed["compare_failures"] == ()

    failures = validate_explicit_compare_report(
        {"passed": True},
        beam_mode="static_scalar",
        product=static,
    )
    assert failures
    assert any("missing compare fields" in item for item in failures)

    _write_compare_report(
        compare_dir,
        "static_scalar",
        static,
        beam_model_id="wrong_beam",
    )
    wrong_id = commissioning_ready(
        tmp_path, compare_dir=compare_dir, require_compare=True
    )
    assert wrong_id["ready"] is False
    assert any("beam_model_id" in item for item in wrong_id["compare_failures"])

    _write_compare_report(compare_dir, "static_scalar", static)
    missing_chunk = dict(
        json.loads(
            (compare_dir / "operator_compare_static_scalar.json").read_text(encoding="utf-8")
        )
    )
    del missing_chunk["reports"][0]["visibility_chunk_size"]
    (compare_dir / "operator_compare_static_scalar.json").write_text(
        json.dumps(missing_chunk), encoding="utf-8"
    )
    no_chunk = commissioning_ready(
        tmp_path, compare_dir=compare_dir, require_compare=True
    )
    assert no_chunk["ready"] is False
    assert any("visibility_chunk_size" in item for item in no_chunk["compare_failures"])


def test_copy_phase6_products_skips_complete_dest(tmp_path) -> None:
    from sl1mjax.phase6_protocol import copy_phase6_products, product_is_complete

    live = tmp_path / "live"
    dest = tmp_path / "dest"
    _write_gate_product(live / "commissioning" / "static_scalar")
    (live / "commissioning" / "static_scalar" / "checkpoint.json").write_text(
        '{"source": "old-live"}', encoding="utf-8"
    )
    _write_gate_product(dest / "commissioning" / "static_scalar")
    (dest / "commissioning" / "static_scalar" / "checkpoint.json").write_text(
        '{"source": "resumed"}', encoding="utf-8"
    )
    (dest / "commissioning" / "static_scalar" / "summary.json").write_text(
        '{"source": "resumed-summary"}', encoding="utf-8"
    )
    incomplete = live / "commissioning" / "streamed_scalar"
    incomplete.mkdir(parents=True)
    (incomplete / "checkpoint.json").write_text('{"source": "partial"}', encoding="utf-8")
    copied = copy_phase6_products(live, dest, require_complete_source=True)
    assert copied == ()
    assert '"source": "resumed"' in (
        dest / "commissioning" / "static_scalar" / "checkpoint.json"
    ).read_text(encoding="utf-8")
    assert '"source": "resumed-summary"' in (
        dest / "commissioning" / "static_scalar" / "summary.json"
    ).read_text(encoding="utf-8")
    assert not (dest / "commissioning" / "streamed_scalar").exists()
    assert product_is_complete(dest / "commissioning" / "static_scalar")

    dest_incomplete = dest / "commissioning-c4" / "static_scalar"
    dest_incomplete.mkdir(parents=True)
    (dest_incomplete / "checkpoint.json").write_text('{"source": "stale"}', encoding="utf-8")
    _write_gate_product(live / "commissioning-c4" / "static_scalar")
    (live / "commissioning-c4" / "static_scalar" / "checkpoint.json").write_text(
        '{"source": "complete-live"}', encoding="utf-8"
    )
    copied = copy_phase6_products(live, dest, require_complete_source=True)
    assert copied == ("commissioning-c4/static_scalar",)
    assert '"source": "complete-live"' in (
        dest_incomplete / "checkpoint.json"
    ).read_text(encoding="utf-8")


def test_copy_phase6_products_can_fill_incomplete_dest(tmp_path) -> None:
    from sl1mjax.phase6_protocol import copy_phase6_products

    live = tmp_path / "live"
    dest = tmp_path / "dest"
    live_beam = live / "commissioning" / "static_scalar"
    dest_beam = dest / "commissioning" / "static_scalar"
    live_beam.mkdir(parents=True)
    dest_beam.mkdir(parents=True)
    (live_beam / "checkpoint.json").write_text('{"step": 12}', encoding="utf-8")
    (dest_beam / "checkpoint.json").write_text('{"step": 3}', encoding="utf-8")
    copied = copy_phase6_products(live, dest, require_complete_source=False)
    assert copied == ("commissioning/static_scalar",)
    assert '"step": 12' in (dest_beam / "checkpoint.json").read_text(encoding="utf-8")

    _write_gate_product(dest_beam)
    (live_beam / "checkpoint.json").write_text('{"step": 1}', encoding="utf-8")
    copied = copy_phase6_products(live, dest, require_complete_source=False)
    assert copied == ()
    assert (dest_beam / "summary.json").is_file()


def test_staged_source_manifest_identifies_dest_tree(tmp_path) -> None:
    from sl1mjax.phase6_protocol import (
        preserve_commissioning_source,
        sha256_file,
        staged_source_manifest,
        write_staged_source_manifest,
    )

    dest = tmp_path / "checkout"
    (dest / "src" / "sl1mjax").mkdir(parents=True)
    (dest / "uv.lock").write_text("lock-bytes\n", encoding="utf-8")
    (dest / "src" / "sl1mjax" / "phase6_protocol.py").write_text("proto\n", encoding="utf-8")
    (dest / "src" / "sl1mjax" / "voltage_operator_jax.py").write_text(
        "operator\n", encoding="utf-8"
    )
    payload = staged_source_manifest(dest)
    assert payload["tree"] == "explicit_staged"
    assert payload["dest"] == str(dest.resolve())
    assert payload["lock_sha256"] == sha256_file(dest / "uv.lock")
    assert payload["phase6_protocol_sha256"] == sha256_file(
        dest / "src" / "sl1mjax" / "phase6_protocol.py"
    )
    assert payload["voltage_operator_sha256"] == sha256_file(
        dest / "src" / "sl1mjax" / "voltage_operator_jax.py"
    )
    assert "commit" not in payload

    output = tmp_path / "out" / "explicit_source.json"
    write_staged_source_manifest(dest, output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["tree"] == "explicit_staged"

    live_source = tmp_path / "live" / "source.json"
    live_source.parent.mkdir()
    live_source.write_text('{"commit": "old-vjp"}', encoding="utf-8")
    dest_out = tmp_path / "out"
    first = preserve_commissioning_source(live_source, dest_out)
    assert first is not None
    assert json.loads(first.read_text(encoding="utf-8"))["commit"] == "old-vjp"
    first.write_text('{"commit": "kept"}', encoding="utf-8")
    live_source.write_text('{"commit": "newer-live"}', encoding="utf-8")
    again = preserve_commissioning_source(live_source, dest_out)
    assert json.loads(again.read_text(encoding="utf-8"))["commit"] == "kept"
    assert not (dest_out / "source.json").exists()


def test_integration_depth_ablation_helpers(tmp_path) -> None:
    from sl1mjax.phase6_protocol import (
        RANKING_HOLD_OUT_THRESHOLD,
        choose_seven_point_integration_depth,
        flux_weighted_audit_error,
        load_under_resolved_findings,
        locate_under_resolved_parents,
        raised_under_resolved_depths,
    )
    from sl1mjax.voltage_reconstruction import starting_central_table

    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=np.array([2.0, 0.1, 0.1, 0.1]),
    )
    findings = (
        {"component_id": table.components[0].component_id, "error": 1.0e-3},
        {"component_id": table.components[1].component_id, "error": 1.0e-4},
    )
    weighted = flux_weighted_audit_error(table, findings)
    assert weighted["n_under_resolved"] == 2.0
    assert weighted["flux_weighted_error"] == pytest.approx((2.0 * 1.0e-3 + 0.1 * 1.0e-4) / 2.1)
    raised = raised_under_resolved_depths(
        {item.component_id: 3 for item in table.components},
        [table.components[0].component_id],
        4,
    )
    assert raised[table.components[0].component_id] == 4
    assert raised[table.components[1].component_id] == 3
    assert choose_seven_point_integration_depth(0.01949, {4: 0.019490001}) == 3
    assert choose_seven_point_integration_depth(
        0.01949, {4: 0.01949 + RANKING_HOLD_OUT_THRESHOLD}
    ) == 4
    assert choose_seven_point_integration_depth(
        0.01949,
        {4: 0.01949 + RANKING_HOLD_OUT_THRESHOLD, 5: 0.01949 + 2 * RANKING_HOLD_OUT_THRESHOLD},
    ) == 5

    from dataclasses import replace

    from sl1mjax.phase6_protocol import _is_raster_boundary_component

    edge = replace(table.components[0], iy=0, ix=10, level=0)
    interior = replace(table.components[1], iy=50, ix=50, level=0)
    assert _is_raster_boundary_component(edge)
    assert not _is_raster_boundary_component(interior)
    located = locate_under_resolved_parents(
        replace(table, components=(edge, *table.components[1:])),
        ({"component_id": edge.component_id, "error": 1e-4},),
    )
    assert located["counts"]["raster_boundary"] == 1

    audit = tmp_path / "audit_findings.json"
    audit.write_text(
        json.dumps({"under_resolved": [{"component_id": "a", "error": 1e-4}]}),
        encoding="utf-8",
    )
    loaded = load_under_resolved_findings(tmp_path)
    assert loaded[0]["component_id"] == "a"
