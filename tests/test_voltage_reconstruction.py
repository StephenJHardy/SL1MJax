from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.beam_aware_imaging import ComponentFamily, SkyBasisType, SkyComponent
from sl1mjax.beam_conventions import require_beam_calibration_state
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import ManufacturedVoltageBeam, predict_voltage_from_plan
from sl1mjax.inference import InferenceConfig
from sl1mjax.integration_planner import IntegrationTolerance, integration_plan_from_planner
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import QuadtreeLeaf
from sl1mjax.voltage_reconstruction import (
    PRODUCTION_STOKES_I_BEAMS,
    VoltageReconstructionConfig,
    accept_ranked_splits,
    apply_central_splits,
    exact_virtual_child_scores,
    reconstruct_voltage_stokes_i,
    screen_virtual_splits,
    starting_central_table,
    stokes_i_beam,
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


def _config(**overrides) -> VoltageReconstructionConfig:
    values = dict(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        inference=InferenceConfig(
            solver="proximal_sgd",
            batch_grouping="times",
            steps=8,
            learning_rate=0.2,
            sparsity_weight=0.0,
            patience=8,
            validation_interval=2,
            min_delta=1e-12,
            batch_size_rows=8,
        ),
        tolerance=IntegrationTolerance(max_depth=1, forced_feature_depth=0),
        max_rounds=1,
        max_depth=1,
        max_splits_per_round=2,
        max_split_fraction=1.0,
        leaf_penalty=0.0,
    )
    values.update(overrides)
    return VoltageReconstructionConfig(**values)


def _block(table_flux: np.ndarray | None = None) -> VisibilityBlock:
    uvw = np.array(
        [
            [30.0, 8.0, 2.0],
            [180.0, -40.0, -6.0],
            [900.0, 120.0, 15.0],
            [40.0, -700.0, -12.0],
        ],
        dtype=np.float64,
    )
    frequency = np.array([4.536e9])
    dummy = np.zeros((uvw.shape[0], 1, 2), dtype=np.complex128)
    return VisibilityBlock(
        uvw_m=uvw,
        frequency_hz=frequency,
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=np.array([5.0e9, 5.0e9, 5.0e9 + 8_000.0, 5.0e9 + 8_000.0]),
        antenna1=np.array([0, 0, 1, 0], dtype=np.int32),
        antenna2=np.array([1, 2, 3, 2], dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )


def _structured_table(*, catalogue: tuple[SkyComponent, ...] = ()):
    flux = np.array([0.05, 0.05, 0.05, 1.4])
    return starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=flux,
        catalogue=catalogue,
    )


def _disjoint_masks(block: VisibilityBlock) -> tuple[np.ndarray, np.ndarray]:
    train = np.zeros(block.visibility.shape, dtype=bool)
    holdout = np.zeros(block.visibility.shape, dtype=bool)
    train[:2] = block.active[:2]
    holdout[2:] = block.active[2:]
    return train, holdout


def _fit(
    table,
    block: VisibilityBlock,
    beam,
    config: VoltageReconstructionConfig,
    *,
    train_mask: np.ndarray | None = None,
    holdout_mask: np.ndarray | None = None,
):
    from sl1mjax.voltage_reconstruction import _fit_table

    train = np.asarray(block.active) if train_mask is None else train_mask
    holdout = np.zeros(block.visibility.shape, dtype=bool) if holdout_mask is None else holdout_mask
    return _fit_table(
        table,
        (block,),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=require_beam_calibration_state("casa_parang_true"),
        train_masks=(train,),
        holdout_masks=(holdout,),
        config=config,
        pointing_ids=None,
    )


def test_flux_optimize_records_train_and_holdout_curve() -> None:
    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=np.full(4, 0.4),
    )
    block = _block()
    train, holdout = _disjoint_masks(block)
    fitted = _fit(
        table,
        block,
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        _config(max_rounds=0),
        train_mask=train,
        holdout_mask=holdout,
    )
    assert fitted.curve_steps == (2, 4, 6, 8)
    assert len(fitted.objective_history) == len(fitted.curve_steps)
    assert len(fitted.holdout_history) == len(fitted.curve_steps)
    assert np.all(np.isfinite(fitted.objective_history))
    assert np.all(np.isfinite(fitted.holdout_history))


def test_phase6_production_factory_refuses_full_jones() -> None:
    with pytest.raises(ValueError, match="not a production Stokes-I candidate"):
        stokes_i_beam("full_jones")
    assert tuple(PRODUCTION_STOKES_I_BEAMS) == (
        "static_scalar",
        "streamed_scalar",
        "diagonal_copolar",
    )
    for mode in PRODUCTION_STOKES_I_BEAMS:
        beam = stokes_i_beam(mode)
        assert beam.model_id


def test_phase6_starting_roots_refuse_overlapping_coarse_field() -> None:
    with pytest.raises(ValueError, match="coarse field"):
        starting_central_table(
            root_size=2,
            root_pixel_size_rad=_ROOT,
            mosaic_phase_centre_rad=_PHASE,
            catalogue=(
                SkyComponent(
                    component_id="coarse_field:coarse:0:0:0",
                    family=ComponentFamily.COARSE_FIELD,
                    basis_type=SkyBasisType.UNIFORM_SQUARE,
                    l_rad=0.0,
                    m_rad=0.0,
                    stokes_i_jy=1.0,
                    width_rad=_ROOT,
                    level=0,
                    iy=0,
                    ix=0,
                ),
            ),
        )


def test_phase6_splits_are_prefix_free_and_do_not_count_nodes() -> None:
    table = _structured_table()
    config = _config()
    parent = QuadtreeLeaf(0, 1, 1)
    split = apply_central_splits(table, (parent,), config)
    active = [item for item in split.components if item.active]
    assert len(active) == 7
    ids = {item.component_id for item in active}
    assert f"central_tree:central:0:{parent.iy}:{parent.ix}" not in ids
    children = {
        f"central_tree:central:1:{2 * parent.iy + dy}:{2 * parent.ix + dx}"
        for dy in (0, 1)
        for dx in (0, 1)
    }
    assert children <= ids
    result = reconstruct_voltage_stokes_i(
        table,
        _block(),
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=config,
        beam_mode="manufactured",
    )
    assert result.fit.plan.parent_count == sum(item.active for item in result.table.components)
    assert result.diagnostics["parent_count"] == result.fit.plan.parent_count
    assert result.fit.plan.node_count >= result.fit.plan.parent_count


def test_phase6_batched_screen_matches_single_parent_adjoint() -> None:
    table = _structured_table()
    config = _config()
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, hess_ll=_IDENTITY * 30.0)
    block = _block()
    fit = _fit(table, block, beam, config)
    batched = screen_virtual_splits(
        fit,
        (block,),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=require_beam_calibration_state("casa_parang_true"),
        train_masks=(np.asarray(block.active),),
        config=config,
    )
    assert batched
    for score in batched:
        single = screen_virtual_splits(
            fit,
            (block,),
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state=require_beam_calibration_state("casa_parang_true"),
            train_masks=(np.asarray(block.active),),
            config=config,
            candidates=(score.leaf,),
        )
        assert len(single) == 1
        np.testing.assert_allclose(score.gradient, single[0].gradient, rtol=0.0, atol=1e-8)


def test_phase6_screen_limits_virtual_children_to_parent_gradient_shortlist() -> None:
    table = _structured_table()
    config = _config(screen_parent_limit=1)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, hess_ll=_IDENTITY * 30.0)
    block = _block()
    fit = _fit(table, block, beam, config)
    limited = screen_virtual_splits(
        fit,
        (block,),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=require_beam_calibration_state("casa_parang_true"),
        train_masks=(np.asarray(block.active),),
        config=config,
    )
    assert len(limited) == 1
    assert fit.gradient is not None
    from sl1mjax.voltage_reconstruction import _central_leaf_parent_index

    parent_index = _central_leaf_parent_index(fit.table, fit.plan)
    expected = max(parent_index, key=lambda leaf: abs(float(fit.gradient[parent_index[leaf]])))
    assert limited[0].leaf == expected


def test_phase6_exact_shortlist_matches_virtual_child_predictions() -> None:
    table = _structured_table()
    config = _config()
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, hess_mm=_IDENTITY * 20.0)
    block = _block()
    fit = _fit(table, block, beam, config)
    screen = screen_virtual_splits(
        fit,
        (block,),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=require_beam_calibration_state("casa_parang_true"),
        train_masks=(np.asarray(block.active),),
        config=config,
    )
    leaves = tuple(item.leaf for item in screen)
    exact = exact_virtual_child_scores(
        fit,
        leaves,
        (block,),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=require_beam_calibration_state("casa_parang_true"),
        train_masks=(np.asarray(block.active),),
        config=config,
    )
    assert len(exact) == len(leaves)
    from sl1mjax.voltage_reconstruction import _flux_vector, _plan_table

    oracle = []
    for leaf in leaves:
        virtual = apply_central_splits(fit.table, (leaf,), config)
        report = _plan_table(
            virtual,
            (block,),
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state=require_beam_calibration_state("casa_parang_true"),
            config=config,
        )
        plan = integration_plan_from_planner(virtual, report)
        predicted = predict_voltage_from_plan(
            block,
            plan,
            _flux_vector(virtual),
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=config.operator,
        ).visibility
        residual = np.asarray(predicted) - block.visibility
        weight = np.where(block.active & (block.weight > 0), block.weight, 0.0)
        train_loss = float(np.sum(weight * np.abs(residual) ** 2) / np.sum(weight))
        oracle.append(fit.train_loss - train_loss)
    np.testing.assert_allclose(exact, oracle, rtol=0.0, atol=1e-12)


def test_phase6_rejected_batch_backtracks_to_ranked_prefix() -> None:
    table = _structured_table()
    config = _config(leaf_penalty=10.0, max_splits_per_round=4)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    block = _block()
    train, holdout = _disjoint_masks(block)
    fit = _fit(table, block, beam, config, train_mask=train, holdout_mask=holdout)
    ranked = (QuadtreeLeaf(0, 0, 0), QuadtreeLeaf(0, 1, 1))
    accepted, rejected, _next = accept_ranked_splits(
        fit,
        ranked,
        (block,),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=require_beam_calibration_state("casa_parang_true"),
        train_masks=(train,),
        holdout_masks=(holdout,),
        config=config,
        pointing_ids=None,
    )
    assert rejected[0] == ranked
    assert ranked[:1] in rejected
    assert accepted == ()


def test_phase6_reconstruct_constant_beam_and_candidates_share_geometry() -> None:
    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=np.full(4, 0.4),
    )
    config = _config(max_rounds=0)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    result = reconstruct_voltage_stokes_i(
        table,
        _block(),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=config,
        beam_mode="manufactured",
    )
    assert result.fit.plan.parent_count == 4
    assert result.diagnostics["parent_count"] == 4
    assert result.fit.plan.node_count >= 4
    assert result.audit is not None
    assert "optimizer_converged" in result.diagnostics
    assert result.fit.converged == (result.fit.kkt_residual <= config.inference.kkt_tolerance)


def test_phase6_default_geometry_is_the_declared_central_grid() -> None:
    assert VoltageReconstructionConfig().root_size == 104
    assert VoltageReconstructionConfig().root_pixel_size_rad == pytest.approx(
        np.deg2rad(16.0 / 3600.0)
    )


def test_phase6_overlapping_train_and_holdout_masks_are_refused() -> None:
    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=np.full(4, 0.4),
    )
    block = _block()
    with pytest.raises(ValueError, match="disjoint"):
        reconstruct_voltage_stokes_i(
            table,
            block,
            ManufacturedVoltageBeam(intercept=_IDENTITY),
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            train_masks=(np.asarray(block.active),),
            holdout_masks=(np.asarray(block.active),),
            config=_config(max_rounds=0),
            beam_mode="manufactured",
        )


def test_phase6_screen_ignores_holdout_and_zero_weight_samples() -> None:
    table = _structured_table()
    config = _config()
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, hess_ll=_IDENTITY * 30.0)
    block = _block()
    train, holdout = _disjoint_masks(block)
    fit = _fit(table, block, beam, config, train_mask=train, holdout_mask=holdout)
    poisoned_residuals = tuple(
        residual + 1.0e6 * holdout.astype(residual.dtype) for residual in fit.residuals
    )
    from sl1mjax.voltage_reconstruction import VoltageFitResult

    poisoned = VoltageFitResult(
        table=fit.table,
        plan=fit.plan,
        planner_report=fit.planner_report,
        flux=fit.flux,
        train_loss=fit.train_loss,
        holdout_loss=fit.holdout_loss,
        sparsity=fit.sparsity,
        topology_cost=fit.topology_cost,
        kkt_residual=fit.kkt_residual,
        steps=fit.steps,
        converged=fit.converged,
        predictions=fit.predictions,
        residuals=poisoned_residuals,
    )
    clean = screen_virtual_splits(
        fit,
        (block,),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=require_beam_calibration_state("casa_parang_true"),
        train_masks=(train,),
        config=config,
    )
    dirty = screen_virtual_splits(
        poisoned,
        (block,),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=require_beam_calibration_state("casa_parang_true"),
        train_masks=(train,),
        config=config,
    )
    assert clean
    for left, right in zip(clean, dirty, strict=True):
        np.testing.assert_allclose(left.gradient, right.gradient, rtol=0.0, atol=1e-10)


def test_phase6_screen_accepts_catalogue_atoms() -> None:
    catalogue = (
        SkyComponent(
            component_id="catalogue:nvss:delta:0",
            family=ComponentFamily.CATALOGUE,
            basis_type=SkyBasisType.DELTA,
            l_rad=0.04,
            m_rad=0.01,
            stokes_i_jy=0.8,
            width_rad=0.0,
        ),
    )
    table = _structured_table(catalogue=catalogue)
    config = _config()
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, hess_ll=_IDENTITY * 20.0)
    block = _block()
    fit = _fit(table, block, beam, config)
    assert fit.plan.parent_count == 5
    scores = screen_virtual_splits(
        fit,
        (block,),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=require_beam_calibration_state("casa_parang_true"),
        train_masks=(np.asarray(block.active),),
        config=config,
    )
    assert scores
    assert all(item.leaf.level == 0 for item in scores)


def test_phase6_convergence_gate_uses_kkt_not_step_exhaustion() -> None:
    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=np.full(4, 0.4),
    )
    block = _block()
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    exhausted = reconstruct_voltage_stokes_i(
        table,
        block,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(
            max_rounds=0,
            inference=InferenceConfig(
                solver="proximal_sgd",
                batch_grouping="times",
                steps=1,
                learning_rate=0.2,
                sparsity_weight=0.0,
                patience=8,
                validation_interval=1,
                min_delta=1e-12,
                kkt_tolerance=1e-30,
                batch_size_rows=8,
            ),
        ),
        beam_mode="manufactured",
    )
    assert exhausted.fit.steps == 1
    assert exhausted.fit.converged is False
    assert exhausted.fit.kkt_residual > 1e-30
    assert exhausted.diagnostics["optimizer_converged"] is False
    loose = reconstruct_voltage_stokes_i(
        table,
        block,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(
            max_rounds=0,
            inference=InferenceConfig(
                solver="proximal_sgd",
                batch_grouping="times",
                steps=1,
                learning_rate=0.2,
                sparsity_weight=0.0,
                patience=8,
                validation_interval=1,
                min_delta=1e-12,
                kkt_tolerance=1e6,
                batch_size_rows=8,
            ),
        ),
        beam_mode="manufactured",
    )
    assert loose.fit.converged is True
    assert loose.fit.kkt_residual <= 1e6


def test_phase6_reconstruction_rescored_shortlist_uses_exact_children() -> None:
    table = _structured_table()
    config = _config(max_rounds=1)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, hess_ll=_IDENTITY * 30.0)
    result = reconstruct_voltage_stokes_i(
        table,
        _block(),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=config,
        beam_mode="manufactured",
    )
    assert result.rounds
    if result.rounds[0].shortlist:
        assert {item.curvature_mode for item in result.rounds[0].shortlist} == {
            "voltage_exact_virtual_child"
        }


def test_phase6_training_batch_is_a_sliced_single_time_block() -> None:
    from sl1mjax.voltage_reconstruction import _sample_training_batch

    block = _block()
    config = _config()
    rng = np.random.default_rng(0)
    batch = _sample_training_batch((block,), (np.asarray(block.active),), config, rng)
    assert batch.block.uvw_m.shape[0] == config.inference.batch_size_rows
    assert np.unique(batch.block.time_s).size == 1
    assert batch.mask.shape == batch.block.visibility.shape
    assert int(np.sum(batch.mask)) <= int(np.sum(block.active))


def test_phase6_kkt_and_sgd_never_autodiff_a_full_pointing(monkeypatch) -> None:
    from sl1mjax import voltage_reconstruction as module

    seen_rows: list[int] = []
    seen_times: list[int] = []
    original = module.predict_voltage_from_plan_value_and_grad

    def wrapped(flux, block, *args, **kwargs):
        seen_rows.append(int(block.uvw_m.shape[0]))
        seen_times.append(int(np.unique(block.time_s).size))
        return original(flux, block, *args, **kwargs)

    monkeypatch.setattr(module, "predict_voltage_from_plan_value_and_grad", wrapped)
    reconstruct_voltage_stokes_i(
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
        config=_config(max_rounds=0),
        beam_mode="manufactured",
    )
    assert seen_rows
    assert max(seen_rows) <= 8
    assert max(seen_times) == 1


def test_kkt_max_batches_subsamples_official_gradient(monkeypatch) -> None:
    from sl1mjax import voltage_reconstruction as module
    from sl1mjax.finite_pixel import integration_plan_from_table

    seen: list[int] = []
    original = module.predict_voltage_from_plan_value_and_grad

    def wrapped(flux, block, *args, **kwargs):
        seen.append(int(block.uvw_m.shape[0]))
        return original(flux, block, *args, **kwargs)

    monkeypatch.setattr(module, "predict_voltage_from_plan_value_and_grad", wrapped)
    n_times = 8
    dummy = np.zeros((n_times, 1, 2), dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.repeat(np.array([[40.0, -20.0, 3.0]]), n_times, axis=0),
        frequency_hz=np.array([4.536e9]),
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=5.0e9 + 60.0 * np.arange(n_times),
        antenna1=np.zeros(n_times, dtype=np.int32),
        antenna2=np.ones(n_times, dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )
    table = starting_central_table(
        root_size=2,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
        flux=np.full(4, 0.4),
    )
    plan = integration_plan_from_table(table)
    config = _config(max_rounds=0, kkt_max_batches=2)
    module._loss_and_gradient(
        plan,
        np.full(plan.parent_count, 0.4),
        (block,),
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        train_masks=(np.asarray(block.active),),
        config=config,
    )
    assert len(seen) == 2


def test_real_shape_one_step_batch_stays_time_bounded() -> None:
    from sl1mjax.finite_pixel import (
        integration_plan_from_table,
        predict_voltage_from_plan_value_and_grad,
    )
    from sl1mjax.voltage_reconstruction import _sample_training_batch

    n_times = 32
    n_rows = 8
    times = np.repeat(5.0e9 + 60.0 * np.arange(n_times), n_rows)
    dummy = np.zeros((times.size, 1, 2), dtype=np.complex128)
    block = VisibilityBlock(
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
    table = starting_central_table(
        root_size=104,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
    )
    plan = integration_plan_from_table(table)
    assert plan.parent_count == 104 * 104
    config = VoltageReconstructionConfig(
        root_size=104,
        inference=InferenceConfig(
            solver="proximal_sgd",
            batch_grouping="times",
            steps=1,
            batch_size_rows=n_rows,
        ),
    )
    batch = _sample_training_batch(
        (block,),
        (np.asarray(block.active),),
        config,
        np.random.default_rng(0),
    )
    assert np.unique(block.time_s).size == n_times
    assert np.unique(batch.block.time_s).size == 1
    assert batch.block.uvw_m.shape[0] == n_rows
    value, gradient = predict_voltage_from_plan_value_and_grad(
        np.zeros(plan.parent_count),
        batch.block,
        plan,
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        train_mask=batch.mask,
    )
    assert np.isfinite(float(value))
    assert np.asarray(gradient).shape == (plan.parent_count,)


def test_predict_batch_size_can_exceed_sgd_rows() -> None:
    from dataclasses import replace

    from sl1mjax.voltage_reconstruction import (
        _iter_bounded_batches,
        _predict_row_capacity,
    )

    n_times = 2
    n_rows = 16
    times = np.repeat(5.0e9 + 60.0 * np.arange(n_times), n_rows)
    dummy = np.zeros((times.size, 1, 2), dtype=np.complex128)
    block = VisibilityBlock(
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
    config = _config(predict_batch_size_rows=16)
    sgd = _iter_bounded_batches((block,), (np.asarray(block.active),), config)
    predict_config = replace(
        config,
        inference=replace(
            config.inference, batch_size_rows=_predict_row_capacity(config)
        ),
    )
    predict = _iter_bounded_batches((block,), (np.asarray(block.active),), predict_config)
    assert config.inference.batch_size_rows == 8
    assert len(sgd) == n_times * (n_rows // 8)
    assert len(predict) == n_times
    assert all(int(batch.block.uvw_m.shape[0]) == 16 for batch in predict)


def test_skip_flux_optimize_evaluates_incoming_table_without_sgd() -> None:
    table = _structured_table()
    initial = np.array(
        [max(0.0, float(item.stokes_i_jy)) for item in table.components if item.active]
    )
    result = reconstruct_voltage_stokes_i(
        table,
        _block(),
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(max_rounds=0),
        beam_mode="manufactured",
        skip_flux_optimize=True,
    )
    assert result.fit.steps == 0
    np.testing.assert_allclose(result.fit.flux, initial)
    assert np.isfinite(result.fit.train_loss)
