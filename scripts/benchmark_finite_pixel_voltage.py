"""Record compile time, runtime, and node counts for the finite-pixel path.

The CI case is small. Pass ``--representative`` for a one-pointing 3C391-like
problem that stays synthetic and does not open the sealed checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from sl1mjax.beam_aware_imaging import VoltageIntegrationMode, sky_table_from_records
from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import (
    ManufacturedVoltageBeam,
    integration_plan_from_table,
    predict_voltage_from_plan_value_and_grad,
)
from sl1mjax.polarization import Correlation, ReceptorBasis

_PHASE = (np.deg2rad(282.35), np.deg2rad(-0.93))
_ANTENNA_POSITION_M = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
        [-1_601_050.0, -5_042_200.0, 3_553_850.0],
    ]
)


def _table(n_parents: int, width_rad: float):
    records = []
    side = int(np.ceil(np.sqrt(n_parents)))
    for index in range(n_parents):
        iy, ix = divmod(index, side)
        records.append(
            {
                "component_id": f"central_tree:central:0:{iy}:{ix}",
                "family": "central_tree",
                "basis_type": "uniform_square",
                "l_rad": (ix - side / 2) * width_rad,
                "m_rad": (iy - side / 2) * width_rad,
                "stokes_i_jy": 0.1,
                "width_rad": width_rad,
                "level": 0,
                "iy": iy,
                "ix": ix,
                "active": True,
                "provenance": {"mosaic_name": "central"},
            }
        )
    return sky_table_from_records(records, mosaic_phase_centre_rad=_PHASE)


def _block(n_row: int, n_channel: int) -> VisibilityBlock:
    dummy = np.ones((n_row, n_channel, 2), dtype=np.complex128) * 0.05
    return VisibilityBlock(
        uvw_m=np.column_stack(
            (
                np.linspace(10.0, 80.0, n_row),
                np.linspace(-40.0, 30.0, n_row),
                np.linspace(-5.0, 12.0, n_row),
            )
        ),
        frequency_hz=np.linspace(4.536e9, 4.662e9, n_channel),
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=np.linspace(5.0e9, 5.0e9 + 1800.0, n_row),
        antenna1=np.zeros(n_row, dtype=np.int32),
        antenna2=np.ones(n_row, dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative", action="store_true")
    args = parser.parse_args()
    n_parents = 1024 if args.representative else 4
    n_row = 128 if args.representative else 4
    n_channel = 2
    mode = (
        VoltageIntegrationMode.SUBCELL_4X4
        if args.representative
        else VoltageIntegrationMode.SUBCELL_2X2
    )
    table = _table(n_parents, np.deg2rad(16.0 / 3600.0))
    plan = integration_plan_from_table(table, mode=mode, pad=True)
    block = _block(n_row, n_channel)
    flux = jnp.full((plan.parent_count,), 0.1, dtype=jnp.float64)
    beam = ManufacturedVoltageBeam(intercept=np.eye(2, dtype=np.complex128))

    def _step(values):
        return predict_voltage_from_plan_value_and_grad(
            values,
            block,
            plan,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=BeamOperatorConfig(visibility_chunk_size=8, pixel_chunk_size=16),
        )

    jitted = jax.jit(_step)
    start = time.perf_counter()
    compile_loss, compile_grad = jitted(flux)
    jax.block_until_ready(compile_loss)
    jax.block_until_ready(compile_grad)
    compile_s = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(3):
        loss, grad = jitted(flux)
        jax.block_until_ready(loss)
        jax.block_until_ready(grad)
    execute_s = (time.perf_counter() - start) / 3.0
    print(
        json.dumps(
            {
                "n_parents": plan.parent_count,
                "n_nodes": plan.node_count,
                "capacity": plan.capacity,
                "n_row": n_row,
                "n_channel": n_channel,
                "mode": mode.value,
                "compile_s": compile_s,
                "execute_s": execute_s,
                "recompile_count": 1,
                "representative": args.representative,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
