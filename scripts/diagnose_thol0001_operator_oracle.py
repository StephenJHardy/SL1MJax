"""Recover CASA 4x4 operators and diagnose Kcross and Df conventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_calibration_golden import (
    golden_visibility_block,
    import_thol0001_casa_tables,
)
from sl1mjax.holography_calibration_oracle import (
    BASIS_NAMES,
    classify_kcross_phase_fit,
    delay_reference_offset_hz,
    factor_kronecker_antenna_jones,
    fit_residual_phase_line,
    isolate_right_factor,
    operator_from_basis_outputs,
    parallel_hand_leak_terms,
    pin_jones_gauge,
    score_df_convention_ladder,
)
from sl1mjax.holography_ms import _tables
from sl1mjax.polarization import invert_jones

STAGES = (
    "K+B+G",
    "K+B+G+Kcross",
    "K+B+G+Kcross+Df",
    "K+B+G+Kcross+Df+Xf",
    "K+B+G+Kcross+Df+Xf+P",
)


def _casa_operator(
    oracle_root: Path, label: str, stage: str, channel: int, cases: list[dict]
) -> np.ndarray:
    tables = _tables()
    outputs = np.zeros((4, 4), dtype=np.complex128)
    by_basis = {case["basis"]: case for case in cases if case["label"] == label}
    applied_root = oracle_root / "applied" / stage.replace("+", "_")
    for index, name in enumerate(BASIS_NAMES):
        path = applied_root / Path(by_basis[name]["ms"]).name
        with tables.table(str(path), readonly=True, ack=False) as main:
            corrected = np.asarray(main.getcol("CORRECTED_DATA"))
        outputs[index] = corrected[0, int(channel)]
    return operator_from_basis_outputs(outputs)


def _sampled_d(solution, antenna: int, frequency_hz: float) -> np.ndarray:
    assert solution.leakage is not None
    assert solution.leakage_frequency_hz is not None
    index = int(np.argmin(np.abs(solution.leakage_frequency_hz - frequency_hz)))
    return np.asarray(solution.leakage[int(antenna), index])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument(
        "--product-root",
        type=Path,
        default=Path("/media/stephen/astro/vla/extracted/commissioning/products"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-desc-id", type=int, default=4)
    arguments = parser.parse_args()
    manifest = json.loads((arguments.oracle_root / "oracle_cases.json").read_text())
    channels = tuple(int(item) for item in manifest["channels"])
    cases = manifest["cases"]
    labels = sorted({case["label"] for case in cases})
    sample = next(Path(case["ms"]) for case in cases if case["label"] == labels[0])
    block, _, _ = golden_visibility_block(
        sample, data_column="DATA", data_desc_id=arguments.data_desc_id
    )
    root = arguments.product_root
    tables = {
        "antpos": root / "diagonal" / "antpos.cal",
        "G1": root / "diagonal" / "G1.cal",
        "K0": root / "diagonal" / "K0.cal",
        "B0": root / "diagonal" / "B0.cal",
        "Kcross": root / "fullpol" / "Kcross.cal",
        "Df": root / "fullpol" / "Df.cal",
        "Xf": root / "fullpol" / "Xf.cal",
    }
    full = import_thol0001_casa_tables(
        tables,
        measurement_set=sample,
        spectral_window_id=block.spectral_window_id,
        product="fullpol",
    )
    casa_tables = _tables()
    with casa_tables.table(str(tables["K0"] / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spw:
        k_ref = np.asarray(spw.getcol("REF_FREQUENCY"), dtype=np.float64)
    with casa_tables.table(str(sample / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spw:
        ms_ref = np.asarray(spw.getcol("REF_FREQUENCY"), dtype=np.float64)
    reports = []
    for label in labels:
        meta = next(case for case in cases if case["label"] == label)
        antenna1 = int(meta["antenna1"])
        antenna2 = int(meta["antenna2"])
        for channel in channels:
            frequency = float(block.frequency_hz[channel])
            stage_ops = {}
            for stage in STAGES:
                casa = _casa_operator(arguments.oracle_root, label, stage, channel, cases)
                factored = factor_kronecker_antenna_jones(casa)
                leaks = parallel_hand_leak_terms(casa)
                stage_ops[stage] = {
                    "casa": casa,
                    "factor_rel_l2": factored["relative_l2"],
                    "leaks": {
                        key: {"re": value.real, "im": value.imag} for key, value in leaks.items()
                    },
                }
            kbg = stage_ops["K+B+G"]["casa"]
            kcross = stage_ops["K+B+G+Kcross"]["casa"]
            df_op = stage_ops["K+B+G+Kcross+Df"]["casa"]
            isolate_right_factor(kcross, kbg)
            kcross_phase = {
                "rl_diag_phase_deg": float(
                    np.rad2deg(np.angle(kcross[1, 1] * np.conjugate(kbg[1, 1])))
                ),
                "lr_diag_phase_deg": float(
                    np.rad2deg(np.angle(kcross[2, 2] * np.conjugate(kbg[2, 2])))
                ),
            }
            prefix = factor_kronecker_antenna_jones(kcross)
            prefix_p, prefix_q = pin_jones_gauge(
                prefix["jones_p"],
                prefix["jones_q"],
                reference_is_p=antenna1 == full.reference_antenna,
            )
            # Factored Jones are the applied (inverse) Jones. Convert to forward prefix.
            forward_p = invert_jones(prefix_p)
            forward_q = invert_jones(prefix_q)
            d_p = _sampled_d(full, antenna1, frequency)
            d_q = _sampled_d(full, antenna2, frequency)
            ladder = score_df_convention_ladder(
                df_op, d_p, d_q, jones_p_prefix=forward_p, jones_q_prefix=forward_q
            )
            reports.append(
                {
                    "label": label,
                    "channel": channel,
                    "frequency_hz": frequency,
                    "antenna1": antenna1,
                    "antenna2": antenna2,
                    "contains_reference": meta.get("contains_reference"),
                    "kcross_phase": kcross_phase,
                    "df_leaks": stage_ops["K+B+G+Kcross+Df"]["leaks"],
                    "kronecker_rel_l2": {
                        stage: stage_ops[stage]["factor_rel_l2"] for stage in STAGES
                    },
                    "df_ladder_top": ladder[:6],
                    "imported_d_p": {
                        "re": complex(d_p[0]).real,
                        "im": complex(d_p[0]).imag,
                        "re1": complex(d_p[1]).real,
                        "im1": complex(d_p[1]).imag,
                    },
                    "imported_d_q": {
                        "re": complex(d_q[0]).real,
                        "im": complex(d_q[0]).imag,
                        "re1": complex(d_q[1]).real,
                        "im1": complex(d_q[1]).imag,
                    },
                }
            )
    # Kcross 8 vs 56
    kcross_pairs = {}
    for label in labels:
        pair = [item for item in reports if item["label"] == label]
        pair.sort(key=lambda item: item["channel"])
        if len(pair) >= 2:
            phases = [item["kcross_phase"]["rl_diag_phase_deg"] for item in pair]
            freqs = [item["frequency_hz"] for item in pair]
            fit = fit_residual_phase_line(np.asarray(freqs), np.deg2rad(np.asarray(phases)))
            delay = None
            if full.cross_hand_delay_s is not None:
                delay = float(full.cross_hand_delay_s[pair[0]["antenna1"], 0])
            kcross_pairs[label] = {
                "rl_phase_deg": phases,
                "fit": fit,
                "classification": classify_kcross_phase_fit(fit),
                "implied_ref_offset_hz": delay_reference_offset_hz(
                    float(fit["intercept_rad"]), delay or 0.0
                ),
                "imported_kcross_s": delay,
            }
    payload = {
        "passed": False,
        "jones_recovery_blocked": True,
        "jax_reference_frequency_hz": full.reference_frequency_hz,
        "ms_ref_frequency_hz": ms_ref.tolist(),
        "k_table_ref_frequency_hz": k_ref.tolist(),
        "kcross_by_label": kcross_pairs,
        "reports": reports,
        "notes": (
            "No empirical Df scale is applied",
            "Kronecker gauge pins the reference-antenna A_00 to 1",
        ),
    }
    print(
        json.dumps(
            {
                "jax_ref": full.reference_frequency_hz,
                "k_table_ref": k_ref.tolist(),
                "kcross_by_label": kcross_pairs,
                "df_winners": [
                    {
                        "label": item["label"],
                        "channel": item["channel"],
                        "top": item["df_ladder_top"][0],
                    }
                    for item in reports
                    if item["channel"] == channels[0]
                ],
            },
            indent=2,
            default=str,
        )
    )
    if arguments.output is not None:
        write_json(payload, arguments.output)
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
