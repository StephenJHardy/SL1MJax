"""Diagnose the field-9 |RL|/I floor with complex coherent residuals.

Magnitude-only medians have a positive |RL| bias. Noise should average as
N^{-1/2}. A leftover that stays by antenna, channel, time or parallactic
angle is a calibration/model error, not a thermal floor.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_ms import _read_antennas, _tables
from sl1mjax.holography_reference_jones import (
    CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE,
    classify_crosshand_floor,
    coherent_residual_report,
)

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)


def _pol():
    path = Path(__file__).with_name("report_thol0001_scientific_pol_gates.py")
    spec = importlib.util.spec_from_file_location("thol0001_pol_gates", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bin_labels(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full(values.shape, "nan")
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1.0e-12
    edges[-1] += 1.0e-12
    index = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    return np.array([f"bin{int(i)}" for i in index])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    pol = _pol()
    tables = _tables()
    _ids, names, positions = _read_antennas(tables, arguments.measurement_set)
    positions = np.asarray(positions, dtype=np.float64)
    with tables.table(
        str(arguments.measurement_set / "FIELD"), readonly=True, ack=False
    ) as field_table:
        direction = np.asarray(field_table.getcell("PHASE_DIR", 9), dtype=np.float64).reshape(-1, 2)
        phase = (float(direction[0, 0]), float(direction[0, 1]))
    block = pol._load_field(arguments.measurement_set, 9, 4)
    inner, _ = pol._channel_masks(block["vis"].shape[1])
    i, _q, _u, _v = pol._stokes(block["vis"][:, inner])
    ii = np.nanmedian(i, axis=1)
    scans = np.unique(block["scan"])
    hold = np.isin(block["scan"], scans[max(1, scans.size // 2) :])
    good = hold & np.isfinite(ii) & (np.abs(ii) > 1.0)
    rl = block["vis"][:, inner, 1][good] / np.abs(ii[good, None])
    lr = block["vis"][:, inner, 2][good] / np.abs(ii[good, None])
    chi_ant = parallactic_angle_rad(block["time"], phase, positions)
    chi = 0.5 * (
        chi_ant[np.arange(block["time"].size), block["antenna1"]]
        + chi_ant[np.arange(block["time"].size), block["antenna2"]]
    )
    antenna_pairs = np.array(
        [
            f"{names[int(p)]}-{names[int(q)]}"
            for p, q in zip(block["antenna1"][good], block["antenna2"][good], strict=True)
        ]
    )
    channels = np.broadcast_to(np.arange(rl.shape[1]), rl.shape)
    times = np.broadcast_to(block["time"][good][:, None], rl.shape)
    scans_g = np.broadcast_to(block["scan"][good][:, None], rl.shape)
    pa = np.broadcast_to(chi[good][:, None], rl.shape)
    antenna1 = np.broadcast_to(block["antenna1"][good][:, None], rl.shape)
    report_rl = coherent_residual_report(rl)
    report_lr = coherent_residual_report(lr)
    stratified = {
        "antenna1": coherent_residual_report(rl, group_ids=np.array(names)[antenna1.reshape(-1)]),
        "scan": coherent_residual_report(rl, group_ids=scans_g.reshape(-1)),
        "channel": coherent_residual_report(rl, group_ids=channels.reshape(-1)),
        "time_quartile": coherent_residual_report(rl, group_ids=_bin_labels(times.reshape(-1), 4)),
        "parallactic_quartile": coherent_residual_report(
            rl, group_ids=_bin_labels(np.rad2deg(pa.reshape(-1)), 4)
        ),
        "baseline": coherent_residual_report(rl, group_ids=np.repeat(antenna_pairs, rl.shape[1])),
    }
    worst_groups = {
        axis: sorted(
            (
                {"group": name, **stats}
                for name, stats in (stratified[axis].get("by_group") or {}).items()
            ),
            key=lambda item: item.get("mean_abs") or 0.0,
            reverse=True,
        )[:8]
        for axis in ("antenna1", "scan", "channel", "baseline")
    }
    worst_baseline = max(
        (float(item.get("mean_abs") or 0.0) for item in worst_groups["baseline"]),
        default=float("nan"),
    )
    gate = classify_crosshand_floor(
        median_abs_rl_over_i=float(report_rl["median_abs"]),
        coherent_mean_abs=float(report_rl["coherent_mean_abs"]),
        averages_as_noise=bool(report_rl["averages_as_noise"]),
        max_group_coherent_abs=worst_baseline,
    )
    payload = {
        "schema": "thol0001_crosshand_floor_coherence_v1",
        "measurement_set": str(arguments.measurement_set),
        "field_id": 9,
        "data_desc_id": 4,
        "n": int(np.sum(good)),
        "n_samples": int(rl.size),
        "rl": report_rl,
        "lr": report_lr,
        "stratified": {
            key: {k: v for k, v in value.items() if k != "by_group"}
            | {"n_groups": len(value.get("by_group") or {})}
            for key, value in stratified.items()
        },
        "worst_groups": worst_groups,
        "gate": gate,
        "notes": [CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE],
    }
    path = arguments.output_dir / "crosshand_floor_coherence.json"
    write_json(payload, path)
    print(path)
    print("median_abs", report_rl["median_abs"], "coherent", report_rl["coherent_mean_abs"])
    print("averages_as_noise", report_rl["averages_as_noise"], "status", gate["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
