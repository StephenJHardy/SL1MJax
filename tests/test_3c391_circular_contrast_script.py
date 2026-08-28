from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.circular_contrast import apply_global_circular_contrast
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import RegularGrid

SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_3c391_circular_contrast.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "diagnose_3c391_circular_contrast",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _circular_block(truth: float = 0.18) -> tuple[VisibilityBlock, np.ndarray]:
    grid = RegularGrid(3, 2e-3)
    l, m = grid.coordinates
    intensity = np.asarray([0.0, 0.4, 0.0, 0.2, 1.0, 0.15, 0.0, 0.3, 0.0])
    rng = np.random.default_rng(9)
    rows = 36
    uvw_m = rng.uniform(-900.0, 900.0, size=(rows, 3))
    antenna1 = rng.integers(0, 8, rows, dtype=np.int32)
    antenna2 = (antenna1 + rng.integers(1, 4, rows, dtype=np.int32)) % 8
    frequency_hz = np.asarray([4.6e9, 4.62e9])
    correlations = (Correlation.RR, Correlation.LL)
    model = np.asarray(
        predict_stokes_i(
            intensity,
            l,
            m,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
        )
    )
    visibility = apply_global_circular_contrast(model, correlations, truth)
    block = VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        visibility=visibility,
        weight=np.ones(visibility.shape),
        flag=np.zeros(visibility.shape, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
        model_visibility=model,
    )
    return block, model


def test_diagnose_recovers_injected_global_contrast_on_held_out_baselines() -> None:
    module = _module()
    truth = 0.18
    block, model = _circular_block(truth)
    payload = module.diagnose_circular_contrast(
        block,
        model,
        seed=4,
        selection_fraction=0.2,
        evaluation_fraction=0.2,
    )
    assert payload["global_contrast"] == pytest.approx(truth, abs=2e-3)
    assert payload["evaluation_relative_improvement"] > 0.9
    assert payload["rr_implied_contrast"] == pytest.approx(truth, abs=2e-3)
    assert payload["ll_implied_contrast"] == pytest.approx(truth, abs=2e-3)
    assert payload["implied_contrast_difference"] == pytest.approx(0.0, abs=4e-3)


def test_balanced_hand_scale_is_degenerate_with_global_contrast() -> None:
    module = _module()
    block, model = _circular_block(0.0)
    delta = 0.12
    visibility = model.copy()
    visibility[..., 0] *= 1.0 + delta
    visibility[..., 1] *= 1.0 - delta
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=visibility,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        model_visibility=model,
    )
    payload = module.diagnose_circular_contrast(
        block,
        model,
        seed=4,
        selection_fraction=0.2,
        evaluation_fraction=0.2,
    )
    assert payload["global_contrast"] == pytest.approx(delta, abs=2e-3)
    assert payload["rr_implied_contrast"] == pytest.approx(delta, abs=2e-3)
    assert payload["ll_implied_contrast"] == pytest.approx(delta, abs=2e-3)
    assert payload["implied_contrast_difference"] == pytest.approx(0.0, abs=4e-3)


def test_one_sided_hand_scale_is_not_a_single_sky_contrast() -> None:
    module = _module()
    block, model = _circular_block(0.0)
    visibility = model.copy()
    visibility[..., 1] *= 0.88
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=visibility,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        model_visibility=model,
    )
    payload = module.diagnose_circular_contrast(
        block,
        model,
        seed=4,
        selection_fraction=0.2,
        evaluation_fraction=0.2,
    )
    assert payload["rr_implied_contrast"] == pytest.approx(0.0, abs=2e-3)
    assert payload["ll_implied_contrast"] == pytest.approx(0.12, abs=2e-3)
    assert abs(payload["implied_contrast_difference"]) > 0.05


def test_hand_evaluation_power_uses_the_frozen_discovery_scale() -> None:
    module = _module()
    block, model = _circular_block(0.0)
    from sl1mjax.sky_recovery import split_search_baselines

    split = split_search_baselines(
        block, selection_fraction=0.2, evaluation_fraction=0.2, seed=4
    )
    visibility = model.copy()
    ll = block.correlations.index(Correlation.LL)
    visibility[..., ll] = np.where(
        split.discovery_mask[..., ll],
        0.92 * model[..., ll],
        visibility[..., ll],
    )
    visibility[..., ll] = np.where(
        split.evaluation_mask[..., ll],
        0.75 * model[..., ll],
        visibility[..., ll],
    )
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=visibility,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        model_visibility=model,
    )
    payload = module.diagnose_circular_contrast(
        block,
        model,
        seed=4,
        selection_fraction=0.2,
        evaluation_fraction=0.2,
    )
    frozen = module.parallel_hand_scale_residual_power(
        block,
        model,
        split.evaluation_mask,
        Correlation.LL,
        payload["ll_scale"],
    )
    refit_scale, refit_power = module.fit_parallel_hand_scale(
        block, model, split.evaluation_mask, Correlation.LL
    )
    assert payload["ll_evaluation_residual_power"] == pytest.approx(frozen)
    assert refit_scale != pytest.approx(payload["ll_scale"], abs=1e-4)
    assert refit_power < payload["ll_evaluation_residual_power"]


def test_hand_residual_power_excludes_the_other_parallel_product() -> None:
    module = _module()
    block, model = _circular_block(0.0)
    visibility = model.copy()
    visibility[..., 1] *= 0.7
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=visibility,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        model_visibility=model,
    )
    scale, power = module.fit_parallel_hand_scale(
        block, model, block.active, Correlation.RR
    )
    assert scale == pytest.approx(1.0, abs=1e-12)
    assert power == pytest.approx(0.0, abs=1e-18)


def test_hand_mask_keeps_only_the_requested_parallel_product() -> None:
    module = _module()
    block, _model = _circular_block()
    rr = module._hand_mask(block, block.active, Correlation.RR)
    ll = module._hand_mask(block, block.active, Correlation.LL)
    assert np.all(rr[..., 0])
    assert not np.any(rr[..., 1])
    assert not np.any(ll[..., 0])
    assert np.all(ll[..., 1])
