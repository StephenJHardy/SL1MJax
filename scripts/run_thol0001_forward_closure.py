"""Close the SPW-4 empirical-beam forward-closure gate.

Stages 1–2 are compact fixtures. Stages 3–5 use sampled THOL0001 SPW-4
native-channel-32 rows. Exact-cell one-axis scoring is rerun only after
those stages pass. Gate failures are recorded; artifacts are always
written. Earlier scientific products are not overwritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np

from sl1mjax.beam_operator import unique_visibility_times
from sl1mjax.calibration_terms import parallactic_angle_rad as parallactic_angle
from sl1mjax.holography import HolographyObservation
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_forward_closure import (
    ALGEBRAIC_TOLERANCE,
    EMPIRICAL_COPOLAR_VISIBILITY_CLOSURE,
    FORWARD_CLOSURE_NOTE,
    FORWARD_CLOSURE_STAGES,
    cell_aggregation_closure,
    classify_forward_closure,
    compact_representative_rows,
    dummy_support_artifact,
    gauge_refactorization_closure,
    per_sample_map_closure,
    single_sample_algebraic_closure,
    spatial_support_contract,
)
from sl1mjax.holography_full_jones import (
    DEVELOPMENT_SET_NOTE,
    INTERPOLATOR_NEIGHBOR_K,
    INTERPOLATOR_RADIUS_SCALE,
    freeze_interpolation_support_from_rows,
    holography_one_axis_holdout_masks,
    interpolate_holography_full_jones,
    moving_reference_row_geometry,
    thin_training_rows,
)
from sl1mjax.holography_ms import _read_antennas, _tables, audit_holography_measurement_set
from sl1mjax.holography_reference_jones import (
    NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
    deserialize_reference_jones,
)
from sl1mjax.polarization import Receptor, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
NATIVE_CHANNEL = 32


def _diag():
    path = Path(__file__).with_name("run_thol0001_diagonal_recovery.py")
    spec = importlib.util.spec_from_file_location("thol0001_diagonal_recovery", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _one_axis():
    path = Path(__file__).with_name("run_thol0001_one_axis_visibility_holdouts.py")
    spec = importlib.util.spec_from_file_location("thol0001_one_axis_holdouts", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        if not (np.isfinite(number.real) and np.isfinite(number.imag)):
            return None
        return [number.real, number.imag]
    return value


def _safe_stage(name: str, callback):
    print("stage", name, flush=True)
    try:
        report = callback()
        print(
            " ",
            report.get("status"),
            report.get("first_failing_operation", ""),
            report.get("max_well_conditioned_error", report.get("max_complex_relative_error", "")),
            flush=True,
        )
        return report
    except Exception as error:
        print(" ", "error", error, flush=True)
        return {
            "stage": name,
            "status": "fail",
            "blocking": True,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def _load_observation(measurement_set: Path, channel: int):
    tables = _tables()
    ids, names, positions = _read_antennas(tables, measurement_set)
    names = np.asarray(names)
    positions = np.asarray(positions, dtype=np.float64)
    name_to_id = {str(name): int(ant) for ant, name in zip(ids, names, strict=True)}
    diag = _diag()
    audit = audit_holography_measurement_set(measurement_set)
    ddid = diag._ddid_for_spw(tables, measurement_set, 4)
    block, source = diag._holoraster_channel_block(
        tables,
        measurement_set,
        data_desc_id=ddid,
        channel=channel,
        data_column="CORRECTED_DATA",
        spectral_window_id=4,
    )
    observation = HolographyObservation(
        block=block,
        pointing=diag._subset_pointing(audit.resolved, block.time_s),
        antenna_position_m=positions,
        calibration_state="casa_parang_true",
        phase_centre_rad=block.phase_centre_rad,
        source_name="3C147",
        source_coherency_visibility=source,
        selected_spw_id=4,
    )
    unique_times, _ = unique_visibility_times(block.time_s)
    chi = parallactic_angle(unique_times, observation.phase_centre_rad, positions)
    return observation, chi, name_to_id, names


def _load_residual(output_dir: Path):
    family_cache = output_dir / "prediction_equivalent_family.json"
    stabilized = output_dir / "field9_stabilized_residual_jones.json"
    if family_cache.exists():
        return deserialize_reference_jones(json.loads(family_cache.read_text())["nominal"]["jones"])
    if stabilized.exists():
        return deserialize_reference_jones(json.loads(stabilized.read_text())["jones"])
    raise FileNotFoundError("need field9_stabilized_residual_jones.json or the family cache")


def _support_queries(support) -> dict[str, tuple[int, list[float]]]:
    queries = {
        "missing_antenna": (10_000, [0.0, 0.0]),
        "nan_query": (0, [float("nan"), 0.0]),
    }
    for antenna, offsets in support.offsets_by_antenna.items():
        points = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] < 2:
            continue
        order = np.argsort(np.hypot(points[:, 0], points[:, 1]))
        first = points[order[0]]
        second = points[order[min(1, order.size - 1)]]
        last = points[order[-1]]
        queries["exact"] = (int(antenna), [float(second[0]), float(second[1])])
        queries["interior"] = (
            int(antenna),
            [float(0.5 * (first[0] + second[0])), float(0.5 * (first[1] + second[1]))],
        )
        queries["boundary"] = (int(antenna), [float(last[0]), float(last[1])])
        queries["extrapolation"] = (
            int(antenna),
            [float(last[0] + 4.0 * (last[0] - first[0] + 1.0e-4)), float(last[1])],
        )
        break
    return queries


def _write_plots(output_dir: Path, stages: dict[str, dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    names = []
    rr = []
    ll = []
    rl = []
    lr = []
    for name in FORWARD_CLOSURE_STAGES:
        report = stages.get(name) or {}
        if "median_abs_rr_over_i" in report:
            names.append(name)
            rr.append(report.get("median_abs_rr_over_i"))
            ll.append(report.get("median_abs_ll_over_i"))
            rl.append(report.get("median_abs_rl_over_i"))
            lr.append(report.get("median_abs_lr_over_i"))
        for step_name, step in (report.get("steps") or {}).items():
            names.append(f"{name}:{step_name}")
            rr.append(step.get("median_abs_rr_over_i"))
            ll.append(step.get("median_abs_ll_over_i"))
            rl.append(step.get("median_abs_rl_over_i"))
            lr.append(step.get("median_abs_lr_over_i"))
    if names:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
        x = np.arange(len(names))
        axes[0].plot(x, rr, "o-", label="RR")
        axes[0].plot(x, ll, "s-", label="LL")
        axes[0].plot(x, rl, "^-", label="RL")
        axes[0].plot(x, lr, "v-", label="LR")
        axes[0].axhline(0.05, color="0.5", ls="--", lw=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("|residual| / I")
        axes[0].set_title("Amplitude residual by stage")
        axes[0].legend(fontsize=8)
        phases = []
        phase_names = []
        for name in FORWARD_CLOSURE_STAGES:
            report = stages.get(name) or {}
            if "median_arg_rr_rad" in report:
                phase_names.append(name)
                phases.append(
                    [
                        report.get("median_arg_rr_rad"),
                        report.get("median_arg_ll_rad"),
                        report.get("median_arg_rl_rad"),
                        report.get("median_arg_lr_rad"),
                    ]
                )
        if phases:
            arr = np.asarray(phases, dtype=np.float64)
            axes[1].plot(arr[:, 0], "o-", label="RR")
            axes[1].plot(arr[:, 1], "s-", label="LL")
            axes[1].plot(arr[:, 2], "^-", label="RL")
            axes[1].plot(arr[:, 3], "v-", label="LR")
            axes[1].set_xticks(np.arange(len(phase_names)))
            axes[1].set_xticklabels(phase_names, rotation=45, ha="right", fontsize=7)
            axes[1].set_ylabel("median arg residual (rad)")
            axes[1].set_title("Phase residual by stage")
        fig.tight_layout()
        path = output_dir / "residual_by_stage.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path.name))
    aggregation = stages.get("cell_aggregation_closure") or {}
    stratified = aggregation.get("stratified") or {}
    if stratified:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))
        panels = (
            ("mover", axes[0, 0], "Mover"),
            ("reference", axes[0, 1], "Reference"),
            ("raster", axes[1, 0], "Raster pass"),
            ("beam_radius", axes[1, 1], "Beam radius"),
        )
        for key, axis, title in panels:
            items = stratified.get(key) or {}
            labels = list(items)
            values = [items[label].get("median_abs_rr_over_i") for label in labels]
            axis.bar(np.arange(len(labels)), values)
            axis.set_xticks(np.arange(len(labels)))
            axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            axis.set_title(title)
            axis.set_ylabel("|RR residual| / I")
        fig.tight_layout()
        path = output_dir / "residual_stratified.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path.name))
    return written


def _readme(
    *,
    gate: dict[str, object],
    stages: dict[str, dict],
    one_axis: dict[str, object] | None,
    interpolator_changed: bool,
) -> str:
    first = gate.get("first_failure")
    aggregation = stages.get("cell_aggregation_closure") or {}
    operation = aggregation.get("first_failing_operation")
    per_sample = stages.get("per_sample_map_closure") or {}
    support = stages.get("spatial_support_contract") or {}
    algebraic = stages.get("single_sample_algebraic_closure") or {}
    lines = [
        "# THOL0001 SPW-4 forward closure",
        "",
        "This report isolates the ~40% exact-cell RR/LL residual that made the",
        "first one-axis full-Jones comparison uninterpretable.",
        "",
        f"First failed stage: `{first}`.",
        f"First failing aggregation operation: `{operation}`.",
        "",
    ]
    if algebraic.get("status") == "pass" and per_sample.get("status") == "pass":
        lines.append(
            "Single-sample algebraic closure and per-sample map closure both "
            "pass, so the RIME matrix order, conjugation, source coherency, "
            "reference Jones and parallactic-frame handling are not the cause."
        )
    elif algebraic.get("status") != "pass":
        lines.append(
            "Single-sample algebraic closure failed. The recover/predict "
            "operators themselves do not invert. No scientific comparison "
            "was advanced."
        )
    elif per_sample.get("status") != "pass":
        lines.append(
            f"Per-sample map closure failed with max complex relative error "
            f"{per_sample.get('max_complex_relative_error')}. A minimal "
            "fixture is written; scientific scoring stopped."
        )
    if operation:
        lines.extend(
            [
                "",
                f"The first map operation that creates a copolar residual "
                f"above threshold is `{operation}`.",
            ]
        )
    if support.get("status") == "pass":
        lines.extend(
            [
                "",
                "The interpolator support predicate and finite-output predicate "
                "now share one implementation. Duplicate raster-cell visits no "
                "longer collapse the neighbor radius to zero.",
            ]
        )
    elif support:
        lines.extend(
            [
                "",
                "The spatial support contract still fails; interpolation "
                "support labels and finite interpolator output disagree.",
            ]
        )
    interpretable = bool(gate.get("comparison_interpretable"))
    lines.extend(
        [
            "",
            (
                "The full-Jones versus diagonal comparison is now scientifically interpretable."
                if interpretable and one_axis
                else "The full-Jones versus diagonal comparison is not scientifically "
                "interpretable yet. No invalid pooled comparison was advanced."
            ),
        ]
    )
    if interpolator_changed:
        lines.extend(
            [
                "",
                DEVELOPMENT_SET_NOTE,
            ]
        )
    if one_axis is None:
        lines.extend(
            [
                "",
                "Exact-cell one-axis holdouts were not rerun because a earlier "
                "closure stage failed.",
            ]
        )
    else:
        lines.extend(
            ["", "Exact-cell one-axis results are in `one_axis_visibility_holdouts_v2.json`."]
        )
    lines.extend(
        [
            "",
            "SPW 5 stayed closed. The reserved outer fold was not scored. "
            "Axes were not pooled. The production full-Jones factory was not "
            "modified. No CASSBEAM acceptance is claimed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UNBLOCK / "forward_closure",
    )
    parser.add_argument("--spectral-window", type=int, default=4)
    parser.add_argument("--channel", type=int, default=NATIVE_CHANNEL)
    parser.add_argument("--max-compact-rows", type=int, default=96)
    parser.add_argument("--max-aggregation-rows", type=int, default=4000)
    parser.add_argument("--skip-one-axis", action="store_true")
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[str, dict] = {}
    stages["single_sample_algebraic_closure"] = _safe_stage(
        "single_sample_algebraic_closure",
        lambda: single_sample_algebraic_closure(tolerance=ALGEBRAIC_TOLERANCE),
    )
    stages["gauge_refactorization_closure"] = _safe_stage(
        "gauge_refactorization_closure",
        lambda: gauge_refactorization_closure(tolerance=ALGEBRAIC_TOLERANCE),
    )
    observation = None
    residual = None
    chi = None
    name_to_id = {}
    load_error = None
    try:
        if arguments.measurement_set.exists():
            observation, chi, name_to_id, _names = _load_observation(
                arguments.measurement_set, int(arguments.channel)
            )
            residual = _load_residual(arguments.product_dir)
        else:
            load_error = f"measurement set missing: {arguments.measurement_set}"
    except Exception as error:
        load_error = f"{type(error).__name__}: {error}"
        print("load_error", load_error, flush=True)

    if observation is None or residual is None or chi is None:
        for name in (
            "per_sample_map_closure",
            "cell_aggregation_closure",
            "spatial_support_contract",
        ):
            stages[name] = {
                "stage": name,
                "status": "fail",
                "blocking": True,
                "error": load_error or "observation not loaded",
            }
    else:
        compact = compact_representative_rows(observation, max_rows=int(arguments.max_compact_rows))
        stages["per_sample_map_closure"] = _safe_stage(
            "per_sample_map_closure",
            lambda: per_sample_map_closure(
                observation,
                residual_jones=residual,
                parallactic_angle_rad=chi,
                row_mask=compact,
            ),
        )
        if stages["per_sample_map_closure"].get("status") != "pass":
            packed = pack_coherency(
                observation.block.visibility,
                observation.block.correlations,
                (Receptor.R, Receptor.L),
            )
            rows = np.asarray(
                stages["per_sample_map_closure"].get("worst_row_indices")
                or np.flatnonzero(compact)[:8],
                dtype=np.int64,
            )
            fixture_path = arguments.output_dir / "per_sample_failure_fixture.npz"
            np.savez(
                fixture_path,
                rows=rows,
                antenna1=observation.block.antenna1[rows],
                antenna2=observation.block.antenna2[rows],
                time_s=observation.block.time_s[rows],
                visibility=packed[rows],
            )
            write_json(
                {
                    "n": int(rows.size),
                    "rows": rows.tolist(),
                    "npz": str(fixture_path.name),
                    "max_complex_relative_error": stages["per_sample_map_closure"].get(
                        "max_complex_relative_error"
                    ),
                    "max_well_conditioned_error": stages["per_sample_map_closure"].get(
                        "max_well_conditioned_error"
                    ),
                    "n_well_conditioned": stages["per_sample_map_closure"].get(
                        "n_well_conditioned"
                    ),
                    "note": "minimal fixture for the failed per-sample map closure",
                },
                arguments.output_dir / "per_sample_failure_fixture.json",
            )
        aggregation_mask = compact
        if stages["per_sample_map_closure"].get("status") == "pass":
            geometry = moving_reference_row_geometry(observation)
            usable = np.asarray(geometry["usable"], dtype=bool)
            selected = np.flatnonzero(usable)
            if selected.size > int(arguments.max_aggregation_rows):
                keep = np.zeros(usable.shape, dtype=bool)
                keep[
                    np.random.default_rng(32).choice(
                        selected, size=int(arguments.max_aggregation_rows), replace=False
                    )
                ] = True
                aggregation_mask = keep | compact
            else:
                aggregation_mask = usable
        stages["cell_aggregation_closure"] = _safe_stage(
            "cell_aggregation_closure",
            lambda: cell_aggregation_closure(
                observation,
                residual_jones=residual,
                parallactic_angle_rad=chi,
                row_mask=aggregation_mask,
            ),
        )
        train = thin_training_rows(
            observation,
            holography_one_axis_holdout_masks(
                observation,
                axis="leave_one_reference_out",
                held_reference_id=name_to_id.get("ea26"),
            )["train"]
            if "ea26" in name_to_id
            else compact,
            max_per_cell=4,
        )
        support = freeze_interpolation_support_from_rows(observation, train)
        stages["spatial_support_contract"] = _safe_stage(
            "spatial_support_contract",
            lambda: spatial_support_contract(support, _support_queries(support)),
        )
        missing_hand = dummy_support_artifact(
            {0: [[0.0, 0.0], [0.002, 0.0]]},
            copolar_valid=False,
        )
        _plane, ok, _leak = interpolate_holography_full_jones(
            missing_hand,
            np.array([0.0, 0.0]),
            moving_antenna_id=0,
            frequency_hz=4.564e9,
        )
        stages["spatial_support_contract"]["missing_hands_finite"] = bool(ok)
        if ok:
            stages["spatial_support_contract"]["status"] = "fail"
            stages["spatial_support_contract"]["blocking"] = True

    interpolator_changed = True
    one_axis_payload = None
    try:
        gate = classify_forward_closure(stages)
        if (
            gate.get("deterministic_closure_passed")
            and not arguments.skip_one_axis
            and observation is not None
        ):
            holdouts = _one_axis()
            exact_axes = ("leave_one_reference_out", "leave_one_visit_out")
            axis_reports = {}
            packed = pack_coherency(
                observation.block.visibility,
                observation.block.correlations,
                (Receptor.R, Receptor.L),
            )
            packed_plane = packed[:, 0] if packed.ndim == 4 else packed
            from sl1mjax.holography_diagonal import source_model_stokes_i

            intensity = source_model_stokes_i(observation)
            held_ref = int(name_to_id["ea26"])
            held_move = int(name_to_id["ea04"])
            for axis in exact_axes:
                axis_reports[axis] = holdouts._run_axis(
                    observation,
                    axis=axis,
                    residual_jones=residual,
                    chi=chi,
                    packed_plane=packed_plane,
                    intensity=intensity,
                    held_ref=held_ref,
                    held_move=held_move,
                    max_train_per_cell=4,
                    max_score_rows=15000,
                )
            exact_ok = all(
                item["gate"].get("status") != "fail"
                and not item["gate"].get("rr_ll_regression", False)
                for item in axis_reports.values()
            )
            stages["exact_cell_holdouts"] = {
                "stage": "exact_cell_holdouts",
                "status": "pass" if exact_ok else "fail",
                "blocking": not exact_ok,
                "axes": {name: item["gate"] for name, item in axis_reports.items()},
            }
            if exact_ok:
                for axis in ("spatial_checkerboard", "leave_one_mover_out"):
                    axis_reports[axis] = holdouts._run_axis(
                        observation,
                        axis=axis,
                        residual_jones=residual,
                        chi=chi,
                        packed_plane=packed_plane,
                        intensity=intensity,
                        held_ref=held_ref,
                        held_move=held_move,
                        max_train_per_cell=4,
                        max_score_rows=15000,
                    )
                stages["remaining_one_axis_diagnostics"] = {
                    "stage": "remaining_one_axis_diagnostics",
                    "status": "pass",
                    "blocking": False,
                    "axes": {
                        name: axis_reports[name]["gate"]
                        for name in ("spatial_checkerboard", "leave_one_mover_out")
                        if name in axis_reports
                    },
                }
            combined = holdouts.combine_one_axis_results(
                {name: item["gate"] for name, item in axis_reports.items()}
            )
            one_axis_payload = {
                "schema": "thol0001_one_axis_visibility_holdouts_v2",
                "status": "warn",
                "blocking": False,
                "outcome": "axes_reported_separately",
                "spectral_window_id": 4,
                "native_channel": int(arguments.channel),
                "development_evidence": interpolator_changed,
                "interpolator_unique_offsets": True,
                "reserved_outer_fold": True,
                "do_not_combine": True,
                "pooled_score": None,
                "axes": axis_reports,
                "combined": combined,
                "notes": [
                    FORWARD_CLOSURE_NOTE,
                    DEVELOPMENT_SET_NOTE,
                    NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
                ],
            }
            write_json(
                _jsonable(one_axis_payload),
                arguments.output_dir / "one_axis_visibility_holdouts_v2.json",
            )
            write_json(
                _jsonable(one_axis_payload),
                arguments.product_dir / "one_axis_visibility_holdouts_v2.json",
            )
        else:
            stages.setdefault(
                "exact_cell_holdouts",
                {
                    "stage": "exact_cell_holdouts",
                    "status": "skipped",
                    "blocking": False,
                    "reason": "deterministic closure did not pass",
                },
            )
            stages.setdefault(
                "remaining_one_axis_diagnostics",
                {
                    "stage": "remaining_one_axis_diagnostics",
                    "status": "skipped",
                    "blocking": False,
                    "reason": "exact-cell holdouts were not rerun",
                },
            )
    except Exception as error:
        print("one_axis_error", error, flush=True)
        traceback.print_exc()
        stages["exact_cell_holdouts"] = {
            "stage": "exact_cell_holdouts",
            "status": "fail",
            "blocking": True,
            "error": f"{type(error).__name__}: {error}",
        }
    gate = classify_forward_closure(stages)
    try:
        plots = _write_plots(arguments.output_dir, stages)
    except Exception as error:
        print("plot_error", error, flush=True)
        plots = []
    payload = {
        "schema": "thol0001_forward_closure_v1",
        "status": "pass" if gate.get("first_failure") is None else "fail",
        "blocking": gate.get("first_failure")
        not in {None, "exact_cell_holdouts", "remaining_one_axis_diagnostics"},
        "spectral_window_id": 4,
        "native_channel": int(arguments.channel),
        "original_failure": EMPIRICAL_COPOLAR_VISIBILITY_CLOSURE,
        "tolerance": ALGEBRAIC_TOLERANCE,
        "stages": stages,
        "gate": gate,
        "plots": plots,
        "one_axis_visibility_holdouts_v2": one_axis_payload is not None,
        "interpolator_neighbor_k": INTERPOLATOR_NEIGHBOR_K,
        "interpolator_radius_scale": INTERPOLATOR_RADIUS_SCALE,
        "interpolator_unique_offsets": True,
        "reserved_outer_fold_unused": True,
        "spw5_closed": True,
        "notes": (
            FORWARD_CLOSURE_NOTE,
            DEVELOPMENT_SET_NOTE,
            NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
        ),
        "provenance": {
            "measurement_set": str(arguments.measurement_set),
            "product_dir": str(arguments.product_dir),
            "load_error": load_error,
        },
    }
    write_json(_jsonable(payload), arguments.output_dir / "forward_closure_report.json")
    (arguments.output_dir / "README.md").write_text(
        _readme(
            gate=gate,
            stages=stages,
            one_axis=one_axis_payload,
            interpolator_changed=interpolator_changed,
        )
    )
    print(arguments.output_dir / "forward_closure_report.json")
    print("first_failure", gate.get("first_failure"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
