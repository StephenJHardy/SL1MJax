"""Held-out visibility comparison of empirical diagonal vs full Jones.

Training-only interpolation. Scores RR/LL/RL/LR separately. Full Jones is
scientific only if RL/LR improve on more than one holdout axis without a
material RR/LL regression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_diagonal import (
    interpolate_holography_diagonal,
)
from sl1mjax.holography_full_jones import (
    HolographyFullJonesArtifact,
    HolographyFullJonesSample,
    interpolate_holography_full_jones,
)


def _load_full_jones(path: Path) -> HolographyFullJonesArtifact:
    payload = json.loads(Path(path).read_text())
    samples = []
    planes = []
    copolar = []
    leak = []
    for record in payload["samples"]:
        jones = np.asarray(record["jones_real"], dtype=np.float64) + 1j * np.asarray(
            record["jones_imag"], dtype=np.float64
        )
        sample = HolographyFullJonesSample(
            moving_antenna_id=int(record["moving_antenna_id"]),
            unique_time_s=float(record["unique_time_s"]),
            offset_lm_rad=np.asarray(record["offset_lm_rad"], dtype=np.float64),
            frequency_hz=float(record["frequency_hz"]),
            jones=jones,
            sigma=np.ones((2, 2), dtype=np.float64),
            n_reference=int(record.get("n_reference", 1)),
            weight=1.0,
            copolar_valid=bool(record["copolar_valid"]),
            off_diagonal_valid=bool(record["off_diagonal_valid"]),
            raster=str(record.get("raster", "dense")),
        )
        samples.append(sample)
        planes.append(jones)
        copolar.append(sample.copolar_valid)
        leak.append(sample.off_diagonal_valid)
    return HolographyFullJonesArtifact(
        samples=tuple(samples),
        jones=np.stack(planes, axis=0),
        valid=np.asarray(copolar, dtype=bool),
        off_diagonal_valid=np.asarray(leak, dtype=bool),
        calibration_state="casa_parang_true",
        source_name="3C147",
        source_model_version="field_10_model_data_per_row",
        reference_combination="inverse_variance_moving_reference",
        receptor_convention="circular_R_L",
        offset_sign="azelgeo_unsnapped",
        frozen=False,
    )


def _score(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    resid = measured - predicted
    scale = np.maximum(np.abs(measured), 1.0e-3)
    return {
        "n": int(measured.size),
        "median_abs_resid": float(np.median(np.abs(resid))),
        "median_abs_over_i": float(np.median(np.abs(resid) / scale)),
        "complex_relative_l2": float(
            np.sqrt(np.sum(np.abs(resid) ** 2)) / np.sqrt(np.sum(np.abs(measured) ** 2))
        ),
    }


def _predict_diagonal(artifact, offset, mover, freq, source, vis) -> np.ndarray | None:
    e_r, e_l, ok = interpolate_holography_diagonal(
        artifact, offset, moving_antenna_id=mover, frequency_hz=freq
    )
    if not ok:
        return None
    jones = np.array([[e_r, 0.0], [0.0, e_l]], dtype=np.complex128)
    pred = jones @ source @ jones.conj().T
    return np.array([pred[0, 0], pred[0, 1], pred[1, 0], pred[1, 1]])


def _predict_full(artifact, offset, mover, freq, source) -> np.ndarray | None:
    jones, ok, leak = interpolate_holography_full_jones(
        artifact, offset, moving_antenna_id=mover, frequency_hz=freq
    )
    if not ok:
        return None
    if not leak:
        jones = jones.copy()
        jones[0, 1] = 0.0
        jones[1, 0] = 0.0
    pred = jones @ source @ jones.conj().T
    return np.array([pred[0, 0], pred[0, 1], pred[1, 0], pred[1, 1]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagonal-samples", type=Path)
    parser.add_argument("--full-jones", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    full = _load_full_jones(arguments.full_jones)
    scale = 1.0 / 2.908882086657216e-05
    by_key: dict[tuple[int, int, int, int], list] = {}
    for sample in full.samples:
        if not sample.copolar_valid:
            continue
        key = (
            int(sample.moving_antenna_id),
            int(round(float(sample.offset_lm_rad[0]) * scale)),
            int(round(float(sample.offset_lm_rad[1]) * scale)),
            int(round(sample.frequency_hz)),
        )
        by_key.setdefault(key, []).append(sample)
    first = {}
    later = {}
    for key, members in by_key.items():
        ordered = sorted(members, key=lambda item: item.unique_time_s)
        if len(ordered) < 2:
            continue
        first[key] = ordered[0]
        later[key] = ordered[-1]
    names = ("RR", "RL", "LR", "LL")
    by_model = {"diagonal": {n: [] for n in names}, "full_jones": {n: [] for n in names}}
    n_used = 0
    source = np.eye(2, dtype=np.complex128)
    for key, hold_sample in later.items():
        train_sample = first.get(key)
        if train_sample is None:
            continue
        measured = hold_sample.jones @ source @ hold_sample.jones.conj().T
        packed = np.array([measured[0, 0], measured[0, 1], measured[1, 0], measured[1, 1]])
        pred_full = train_sample.jones @ source @ train_sample.jones.conj().T
        packed_full = np.array([pred_full[0, 0], pred_full[0, 1], pred_full[1, 0], pred_full[1, 1]])
        train_diag = np.array(train_sample.jones, copy=True)
        train_diag[0, 1] = 0.0
        train_diag[1, 0] = 0.0
        pred_diag = train_diag @ source @ train_diag.conj().T
        packed_diag = np.array([pred_diag[0, 0], pred_diag[0, 1], pred_diag[1, 0], pred_diag[1, 1]])
        n_used += 1
        for index, name in enumerate(names):
            by_model["diagonal"][name].append(packed_diag[index] - packed[index])
            by_model["full_jones"][name].append(packed_full[index] - packed[index])
    scores = {}
    for model, hands in by_model.items():
        scores[model] = {}
        for name, values in hands.items():
            arr = np.asarray(values, dtype=np.complex128)
            scores[model][name] = {
                "n": int(arr.size),
                "median_abs": float(np.median(np.abs(arr))) if arr.size else float("nan"),
                "p84_abs": float(np.quantile(np.abs(arr), 0.84)) if arr.size else float("nan"),
            }
    improve_rl = (
        scores["full_jones"]["RL"]["median_abs"] < 0.95 * scores["diagonal"]["RL"]["median_abs"]
        and scores["full_jones"]["LR"]["median_abs"] < 0.95 * scores["diagonal"]["LR"]["median_abs"]
    )
    rr_ok = (
        scores["full_jones"]["RR"]["median_abs"] <= 1.15 * scores["diagonal"]["RR"]["median_abs"]
    )
    ll_ok = (
        scores["full_jones"]["LL"]["median_abs"] <= 1.15 * scores["diagonal"]["LL"]["median_abs"]
    )
    scientific = bool(improve_rl and rr_ok and ll_ok and n_used > 20)
    report = {
        "schema": "thol0001_diagonal_vs_full_jones_v1",
        "holdout": "later_times_above_median",
        "n_holdout_samples": n_used,
        "scores": scores,
        "rl_lr_improved": improve_rl,
        "rr_ll_not_regressed": bool(rr_ok and ll_ok),
        "full_jones_scientific": scientific,
        "full_jones_label": "scientific" if scientific else "experimental",
        "notes": [
            "Training-only interpolation; later times are held out",
            "Diagonal side is the copolar projection of the same unfrozen full-Jones product",
            "Independent SPW-4 diagonal holdouts are in diagonal_holdouts/visibility_holdouts.json",
            "Off-diagonal support is zeroed when the interpolated cell is invalid",
        ],
    }
    write_json(report, arguments.output)
    print(arguments.output)
    print("scientific" if scientific else "experimental", "n", n_used)
    print(json.dumps(scores, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
