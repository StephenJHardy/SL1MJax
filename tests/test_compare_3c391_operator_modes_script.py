from __future__ import annotations

import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_3c391_operator_modes.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

from compare_3c391_operator_modes import build_parser  # noqa: E402
from run_3c391_phase6_bacchus import (  # noqa: E402
    _config_for_beam,
    checkpoint_resume,
    protocol_config,
)


def test_compare_script_requires_product() -> None:
    parser = build_parser()
    try:
        parser.parse_args([])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("--product must be required")
    parsed = parser.parse_args(["--product", "/tmp/c1_static_scalar"])
    assert parsed.product == Path("/tmp/c1_static_scalar")


def test_explicit_production_batch_uses_64_diagonal_rows() -> None:
    base = protocol_config(
        steps=1,
        max_rounds=0,
        max_splits_per_round=8,
        max_split_fraction=0.05,
        patience=1,
        sparsity_weight=0.0,
        strict_audit=False,
        operator_mode="explicit_jax",
    )
    diagonal = _config_for_beam(base, "diagonal_copolar")
    jones = _config_for_beam(base, "full_jones")
    airy = _config_for_beam(base, "static_scalar")
    assert base.operator_mode == "explicit_jax"
    assert airy.inference.batch_size_rows == 64
    assert diagonal.inference.batch_size_rows == 64
    assert diagonal.operator.pixel_chunk_size == 512
    assert jones.inference.batch_size_rows == 64
    assert jones.operator.pixel_chunk_size == 512
    vjp = _config_for_beam(
        protocol_config(
            steps=1,
            max_rounds=0,
            max_splits_per_round=8,
            max_split_fraction=0.05,
            patience=1,
            sparsity_weight=0.0,
            strict_audit=False,
            operator_mode="vjp",
        ),
        "diagonal_copolar",
    )
    assert vjp.inference.batch_size_rows == 32


def test_protocol_config_can_raise_integration_depth() -> None:
    default = protocol_config(
        steps=1,
        max_rounds=0,
        max_splits_per_round=8,
        max_split_fraction=0.05,
        patience=1,
        sparsity_weight=0.0,
        strict_audit=False,
    )
    raised = protocol_config(
        steps=1,
        max_rounds=0,
        max_splits_per_round=8,
        max_split_fraction=0.05,
        patience=1,
        sparsity_weight=0.0,
        strict_audit=False,
        integration_max_depth=5,
    )
    assert default.tolerance.max_depth == 3
    assert default.max_depth == 2
    assert raised.tolerance.max_depth == 5
    assert raised.max_depth == 2


def test_checkpoint_resume_keeps_one_topology_round() -> None:
    skip_flux, rounds, kind = checkpoint_resume(1, 1)
    assert skip_flux is True
    assert rounds == 1
    assert kind == "post-SGD"
    skip_flux, rounds, kind = checkpoint_resume(2, 1)
    assert skip_flux is True
    assert rounds == 0
    assert kind == "post-topology"
    skip_flux, rounds, kind = checkpoint_resume(0, 1)
    assert skip_flux is False
    assert rounds == 1
    assert kind == "warm-start"
