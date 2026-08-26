from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.data.synthetic import simulate_calibration_case
from sl1mjax.split import interleaved_time_folds

SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _module(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transfer_folds_excludes_candidate_only_time_bins() -> None:
    module = _module("build_3c391_recovery_policies")
    block = simulate_calibration_case(
        antenna_count=4, time_count=10, channel_count=1, seed=31
    ).block
    block = replace(block, time_s=np.arange(block.shape[0], dtype=float) * 60.0)
    folds = interleaved_time_folds((block,), bin_seconds=60.0)
    candidate = replace(
        block,
        time_s=np.concatenate((block.time_s[:-1], [10_000.0])),
    )

    transferred = module._transfer_folds(
        (block,), (candidate,), folds, bin_seconds=60.0
    )

    combined = np.logical_or.reduce([fold[0] for fold in transferred])
    assert not np.any(combined[-1])
    np.testing.assert_array_equal(combined[:-1], candidate.active[:-1])


def test_common_active_mask_never_selects_recovered_rows() -> None:
    module = _module("fit_3c391_recovery_policies")
    reference = simulate_calibration_case(
        antenna_count=4, time_count=4, channel_count=1, seed=32
    ).block
    policy = replace(
        reference,
        uvw_m=np.concatenate((reference.uvw_m, reference.uvw_m[:2])),
        visibility=np.concatenate((reference.visibility, reference.visibility[:2])),
        weight=np.concatenate((reference.weight, reference.weight[:2])),
        flag=np.concatenate((reference.flag, reference.flag[:2])),
        model_visibility=(
            None
            if reference.model_visibility is None
            else np.concatenate(
                (reference.model_visibility, reference.model_visibility[:2])
            )
        ),
        time_s=np.concatenate((reference.time_s, reference.time_s[:2])),
        antenna1=np.concatenate((reference.antenna1, reference.antenna1[:2])),
        antenna2=np.concatenate((reference.antenna2, reference.antenna2[:2])),
        field_id=np.concatenate((reference.field_id, reference.field_id[:2])),
        scan_id=np.concatenate((reference.scan_id, reference.scan_id[:2])),
        state_id=np.concatenate((reference.state_id, reference.state_id[:2])),
        observation_id=np.concatenate(
            (reference.observation_id, reference.observation_id[:2])
        ),
        feed1=np.concatenate((reference.feed1, reference.feed1[:2])),
        feed2=np.concatenate((reference.feed2, reference.feed2[:2])),
        interval_s=np.concatenate((reference.interval_s, reference.interval_s[:2])),
    )
    selected = np.zeros(reference.shape, dtype=bool)
    selected[0] = reference.active[0]

    embedded = module._common_active_masks((policy,), (reference,), (selected,))[0]

    np.testing.assert_array_equal(embedded[: reference.shape[0]], selected)
    assert not np.any(embedded[reference.shape[0] :])
