"""CASA/JAX calibration-operator oracle and cumulative term bisection.

Holography conventions are not changed here. The oracle injects basis
visibilities and compares CASA's 4×4 correlation-space correction with
the operator induced by JAX antenna Jones. Bisection applies the same
raw rows after each cumulative stage so a failure can be classified
before pointing or beam recovery mix in.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sl1mjax.calibration import CalibrationSolution, apply_calibration
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.holography_calibration import JONES_RECOVERY_BLOCKED_NOTE, write_json
from sl1mjax.holography_calibration_golden import compare_casa_jax_visibilities
from sl1mjax.polarization import (
    Correlation,
    Receptor,
    apply_jones_to_coherency,
    invert_jones,
    pack_coherency,
    unpack_coherency,
)

BISECTION_STAGES = (
    "antpos",
    "K",
    "K+B",
    "K+B+G",
    "K+B+G+Kcross",
    "K+B+G+Kcross+Df",
    "K+B+G+Kcross+Df+Xf",
    "K+B+G+Kcross+Df+Xf+P",
)
CORRELATIONS = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
RECEPTORS = (Receptor.R, Receptor.L)
BASIS_NAMES = ("RR", "RL", "LR", "LL")
StageName = Literal[
    "antpos",
    "K",
    "K+B",
    "K+B+G",
    "K+B+G+Kcross",
    "K+B+G+Kcross+Df",
    "K+B+G+Kcross+Df+Xf",
    "K+B+G+Kcross+Df+Xf+P",
]


def classify_bisection_residual(
    *,
    relative_l2_by_corr: dict[str, float],
    phase_vs_frequency_slope_rad: float | None = None,
    mean_rl_phase_deg: float | None = None,
    mean_lr_phase_deg: float | None = None,
    resid_depends_on_antenna: bool = False,
    resid_vs_two_chi: float | None = None,
    resid_depends_on_baseline_order: bool = False,
) -> str:
    """Map a structured residual onto a convention hypothesis."""

    cross = max(relative_l2_by_corr.get("RL", 0.0), relative_l2_by_corr.get("LR", 0.0))
    parallel = max(relative_l2_by_corr.get("RR", 0.0), relative_l2_by_corr.get("LL", 0.0))
    if resid_depends_on_baseline_order:
        return "jq_hermitian_convention"
    if resid_vs_two_chi is not None and abs(resid_vs_two_chi) > 0.5:
        return "parallactic_angle_direction"
    if resid_depends_on_antenna and cross > parallel:
        return "d_placement_or_inversion"
    if (
        mean_rl_phase_deg is not None
        and mean_lr_phase_deg is not None
        and abs(mean_rl_phase_deg + mean_lr_phase_deg) < 20.0
        and abs(mean_rl_phase_deg) > 20.0
        and (phase_vs_frequency_slope_rad is None or abs(phase_vs_frequency_slope_rad) < 0.2)
    ):
        return "xf_sign_receptor_order_or_conjugation"
    if phase_vs_frequency_slope_rad is not None and abs(phase_vs_frequency_slope_rad) > 0.2:
        return "kcross_sign_units_ref_frequency_or_antenna_side"
    if cross > 0.1:
        return "unclassified_crosshand"
    if parallel > 1.0e-3:
        return "diagonal_phase_floor"
    return "consistent"


def solution_for_stage(solution: CalibrationSolution, stage: str) -> CalibrationSolution:
    """Strip later Jones terms. Does not invent new table values."""

    if stage not in BISECTION_STAGES:
        raise ValueError(f"unknown bisection stage {stage!r}")
    terms = stage.split("+")
    n_ant = solution.antenna_count
    n_rec = solution.receptor_count
    gains = solution.gains
    gain_valid = solution.gain_valid
    if "G" not in terms:
        gains = np.ones_like(solution.gains)
        gain_valid = np.ones_like(solution.gain_valid)
    delays = solution.delays_s
    delay_valid = solution.delay_valid
    if "K" not in terms:
        delays = np.zeros((n_ant, n_rec), dtype=np.float64)
        delay_valid = np.ones((n_ant, n_rec), dtype=bool)
    bandpass = solution.bandpass
    bandpass_valid = solution.bandpass_valid
    if "B" not in terms:
        bandpass = np.ones_like(solution.bandpass)
        bandpass_valid = np.ones_like(solution.bandpass_valid)
    offsets = solution.antenna_position_offset_m
    if "antpos" not in terms and "G" not in terms:
        offsets = (
            None
            if solution.antenna_position_offset_m is None
            else np.zeros_like(solution.antenna_position_offset_m)
        )
    kcross = solution.cross_hand_delay_s
    kcross_valid = solution.cross_hand_delay_valid
    if "Kcross" not in terms:
        kcross = None
        kcross_valid = None
    leakage = solution.leakage
    leakage_hz = solution.leakage_frequency_hz
    leakage_valid = solution.leakage_valid
    leakage_time = solution.leakage_time_s
    if "Df" not in terms:
        leakage = None
        leakage_hz = None
        leakage_valid = None
        leakage_time = None
    rl_phase = solution.rl_phase
    rl_hz = solution.rl_phase_frequency_hz
    rl_valid = solution.rl_phase_valid
    if "Xf" not in terms:
        rl_phase = None
        rl_hz = None
        rl_valid = None
    return replace(
        solution,
        gains=gains,
        gain_valid=gain_valid,
        delays_s=delays,
        delay_valid=delay_valid,
        bandpass=bandpass,
        bandpass_valid=bandpass_valid,
        antenna_position_offset_m=offsets,
        cross_hand_delay_s=kcross,
        cross_hand_delay_valid=kcross_valid,
        leakage=leakage,
        leakage_frequency_hz=leakage_hz,
        leakage_valid=leakage_valid,
        leakage_time_s=leakage_time,
        rl_phase=rl_phase,
        rl_phase_frequency_hz=rl_hz,
        rl_phase_valid=rl_valid,
        apply_parallactic_angle="P" in terms,
        leakage_application="casa_first_order" if "Df" in terms else "exact",
        provenance={**dict(solution.provenance), "bisection_stage": stage},
    )


def casa_gaintable_for_stage(stage: str, tables: dict[str, Any]) -> tuple[list[str], bool]:
    """CASA gaintable paths and parang flag for one cumulative stage."""

    order = []
    terms = stage.split("+")
    if "antpos" in terms or stage == "antpos" or "G" in terms:
        if "antpos" in tables:
            order.append(str(tables["antpos"]))
    mapping = {
        "K": "K0",
        "B": "B0",
        "G": "G1",
        "Kcross": "Kcross",
        "Df": "Df",
        "Xf": "Xf",
    }
    for term in stage.split("+"):
        if term in mapping:
            order.append(str(tables[mapping[term]]))
    return order, stage.endswith("+P")


def injected_basis_visibilities() -> NDArray[np.complex128]:
    """Four unit correlation-space basis visibilities, packed RR/RL/LR/LL."""

    return np.eye(4, dtype=np.complex128)


def correlation_correction_operator(
    jones_p: NDArray[np.complex128],
    jones_q: NDArray[np.complex128],
    *,
    invert: bool = True,
) -> NDArray[np.complex128]:
    """4×4 map on packed RR/RL/LR/LL induced by ``J_p C J_q^H``.

    ``invert=True`` matches CASA ``applycal`` / JAX ``apply_calibration``.
    """

    left = invert_jones(jones_p) if invert else np.asarray(jones_p)
    right = invert_jones(jones_q) if invert else np.asarray(jones_q)
    operator = np.zeros((4, 4), dtype=np.complex128)
    for column, basis in enumerate(injected_basis_visibilities()):
        coherency = pack_coherency(basis, CORRELATIONS, RECEPTORS)
        corrected = apply_jones_to_coherency(coherency, left, right)
        operator[:, column] = unpack_coherency(corrected, CORRELATIONS, RECEPTORS)
    return operator


def operator_from_basis_outputs(outputs: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Stack four corrected basis visibilities as columns of the 4×4 operator."""

    values = np.asarray(outputs, dtype=np.complex128)
    if values.shape != (4, 4):
        raise ValueError("basis outputs must have shape (4, 4) as RR/RL/LR/LL × basis")
    return values.T.copy()


# RR,RL,LR,LL is row-major. Column-major vec is RR,LR,RL,LL.
_COLMAJOR_PERM = np.array([0, 2, 1, 3], dtype=np.int32)


def factor_kronecker_antenna_jones(
    operator: NDArray[np.complex128],
) -> dict[str, object]:
    """Factor a packed 4x4 operator as A_q-star kronecker A_p.

    The scalar gauge is left free. Callers pin it with the reference antenna.
    """

    matrix = np.asarray(operator, dtype=np.complex128)
    if matrix.shape != (4, 4):
        raise ValueError("operator must be 4×4")
    colmajor = matrix[np.ix_(_COLMAJOR_PERM, _COLMAJOR_PERM)]
    unfolded = colmajor.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    left, singular, right = np.linalg.svd(unfolded, full_matrices=False)
    scale = np.sqrt(float(singular[0]))
    a_q_conj = left[:, 0].reshape(2, 2) * scale
    a_p = right[0].reshape(2, 2) * scale
    a_q = np.conjugate(a_q_conj)
    reconstructed = np.kron(np.conjugate(a_q), a_p)
    reconstructed = reconstructed[np.ix_(_COLMAJOR_PERM, _COLMAJOR_PERM)]
    residual = matrix - reconstructed
    denom = max(float(np.linalg.norm(matrix)), 1.0e-12)
    return {
        "jones_p": a_p,
        "jones_q": a_q,
        "singular_values": [float(value) for value in singular],
        "relative_l2": float(np.linalg.norm(residual) / denom),
        "reconstructed": reconstructed,
    }


def pin_jones_gauge(
    jones_p: NDArray[np.complex128],
    jones_q: NDArray[np.complex128],
    *,
    reference_is_p: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Fix the Kronecker scalar so the reference-antenna A_00 is 1."""

    reference = np.asarray(jones_p if reference_is_p else jones_q, dtype=np.complex128)
    scale = reference[0, 0]
    if abs(scale) < 1.0e-12:
        scale = 1.0 + 0j
    if reference_is_p:
        return jones_p / scale, jones_q * np.conjugate(scale)
    return jones_p * np.conjugate(scale), jones_q / scale


def isolate_right_factor(
    full: NDArray[np.complex128],
    prefix: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """If Jones compose as J = J_prefix J_extra, then M = M_extra M_prefix."""

    return np.linalg.solve(np.asarray(prefix).T, np.asarray(full).T).T


def fit_residual_phase_line(
    frequency_hz: NDArray[np.float64],
    phase_rad: NDArray[np.float64],
) -> dict[str, float]:
    """Fit residual phase as a + b * nu_GHz."""

    freq = np.asarray(frequency_hz, dtype=np.float64)
    phase = np.unwrap(np.asarray(phase_rad, dtype=np.float64))
    if freq.size < 2:
        raise ValueError("phase line fit needs at least two channels")
    ghz = freq / 1.0e9
    intercept, slope = np.polyfit(ghz, phase, 1)[::-1]
    model = intercept + slope * ghz
    residual = phase - model
    return {
        "intercept_rad": float(intercept),
        "slope_rad_per_ghz": float(slope),
        "rms_rad": float(np.sqrt(np.mean(residual**2))),
        "intercept_deg": float(np.rad2deg(intercept)),
        "slope_deg_per_ghz": float(np.rad2deg(slope)),
    }


def classify_kcross_phase_fit(
    fit: dict[str, float], *, slope_floor_rad_per_ghz: float = 0.05
) -> str:
    """Equal residuals mean a constant gauge; a slope means a delay error."""

    if abs(float(fit["slope_rad_per_ghz"])) > slope_floor_rad_per_ghz:
        return "delay_sign_units_reference_frequency_or_offset"
    if abs(float(fit["intercept_rad"])) > 0.005:
        return "constant_crosshand_gauge_or_receptor_placement"
    return "consistent"


def delay_reference_offset_hz(phase_rad: float, delay_s: float) -> float:
    """Constant residual phase 2 pi tau Delta nu_ref implied Delta nu_ref."""

    if abs(delay_s) < 1.0e-15:
        return float("nan")
    return float(phase_rad / (2.0 * np.pi * delay_s))


def leakage_jones_for_convention(
    d_terms: NDArray[np.complex128],
    *,
    cparam_swapped: bool = False,
    conjugate: bool = False,
    offdiag_exchanged: bool = False,
    first_order_inverse: bool = False,
    invert: bool = True,
) -> NDArray[np.complex128]:
    """Build one discrete Df Jones convention. No empirical scale factor."""

    terms = np.asarray(d_terms, dtype=np.complex128)
    if terms.shape[-1] != 2:
        raise ValueError("D terms must be (..., 2)")
    first, second = terms[..., 0], terms[..., 1]
    if cparam_swapped:
        first, second = second, first
    if conjugate:
        first, second = np.conjugate(first), np.conjugate(second)
    if offdiag_exchanged:
        first, second = second, first
    matrices = np.zeros(terms.shape[:-1] + (2, 2), dtype=np.complex128)
    matrices[..., 0, 0] = 1.0
    matrices[..., 1, 1] = 1.0
    matrices[..., 0, 1] = first
    matrices[..., 1, 0] = second
    if not invert:
        return matrices
    if first_order_inverse:
        inverse = np.zeros_like(matrices)
        inverse[..., 0, 0] = 1.0
        inverse[..., 1, 1] = 1.0
        inverse[..., 0, 1] = -matrices[..., 0, 1]
        inverse[..., 1, 0] = -matrices[..., 1, 0]
        return inverse
    return invert_jones(matrices)


def score_df_convention_ladder(
    casa_operator: NDArray[np.complex128],
    d_p: NDArray[np.complex128],
    d_q: NDArray[np.complex128],
    *,
    jones_p_prefix: NDArray[np.complex128] | None = None,
    jones_q_prefix: NDArray[np.complex128] | None = None,
) -> list[dict[str, object]]:
    """Score Df conventions against one CASA 4×4 operator.

    Prefix Jones are already-applied K/B/G/Kcross (forward, not inverted).
    No empirical D scale is scored.
    """

    identity = np.eye(2, dtype=np.complex128)
    prefix_p = identity if jones_p_prefix is None else np.asarray(jones_p_prefix)
    prefix_q = identity if jones_q_prefix is None else np.asarray(jones_q_prefix)
    ranked = []
    for cparam_swapped in (False, True):
        for conjugate in (False, True):
            for offdiag_exchanged in (False, True):
                for first_order in (False, True):
                    for invert in (True, False):
                        for q_hermitian in (True, False):
                            for d_before_prefix in (False, True):
                                d_p_j = leakage_jones_for_convention(
                                    d_p,
                                    cparam_swapped=cparam_swapped,
                                    conjugate=conjugate,
                                    offdiag_exchanged=offdiag_exchanged,
                                    invert=False,
                                )
                                d_q_j = leakage_jones_for_convention(
                                    d_q,
                                    cparam_swapped=cparam_swapped,
                                    conjugate=conjugate,
                                    offdiag_exchanged=offdiag_exchanged,
                                    invert=False,
                                )
                                if d_before_prefix:
                                    forward_p = d_p_j @ prefix_p
                                    forward_q = d_q_j @ prefix_q
                                else:
                                    forward_p = prefix_p @ d_p_j
                                    forward_q = prefix_q @ d_q_j
                                if invert and first_order:
                                    delta_p = leakage_jones_for_convention(
                                        d_p,
                                        cparam_swapped=cparam_swapped,
                                        conjugate=conjugate,
                                        offdiag_exchanged=offdiag_exchanged,
                                        first_order_inverse=True,
                                        invert=True,
                                    )
                                    delta_q = leakage_jones_for_convention(
                                        d_q,
                                        cparam_swapped=cparam_swapped,
                                        conjugate=conjugate,
                                        offdiag_exchanged=offdiag_exchanged,
                                        first_order_inverse=True,
                                        invert=True,
                                    )
                                    if d_before_prefix:
                                        apply_p = delta_p @ invert_jones(prefix_p)
                                        apply_q = delta_q @ invert_jones(prefix_q)
                                    else:
                                        apply_p = invert_jones(prefix_p) @ delta_p
                                        apply_q = invert_jones(prefix_q) @ delta_q
                                elif invert:
                                    apply_p = invert_jones(forward_p)
                                    apply_q = invert_jones(forward_q)
                                else:
                                    apply_p = forward_p
                                    apply_q = forward_q
                                if not q_hermitian:
                                    apply_q = np.conjugate(apply_q)
                                predicted = correlation_correction_operator(
                                    apply_p, apply_q, invert=False
                                )
                                report = compare_correction_operators(casa_operator, predicted)
                                ranked.append(
                                    {
                                        "cparam_swapped": cparam_swapped,
                                        "conjugate": conjugate,
                                        "offdiag_exchanged": offdiag_exchanged,
                                        "first_order_inverse": first_order,
                                        "invert": invert,
                                        "q_hermitian": q_hermitian,
                                        "d_before_prefix": d_before_prefix,
                                        "relative_l2": report["relative_l2"],
                                        "max_abs": report["max_abs"],
                                    }
                                )
    ranked.sort(key=lambda item: float(item["relative_l2"]))
    return ranked


def parallel_hand_leak_terms(operator: NDArray[np.complex128]) -> dict[str, complex]:
    """RR/LL columns isolate the two antenna-side leakage contributions."""

    matrix = np.asarray(operator, dtype=np.complex128)
    return {
        "rr_into_rl_q_side": complex(matrix[1, 0]),
        "rr_into_lr_p_side": complex(matrix[2, 0]),
        "ll_into_rl": complex(matrix[1, 3]),
        "ll_into_lr": complex(matrix[2, 3]),
        "rr_into_ll": complex(matrix[3, 0]),
        "ll_into_rr": complex(matrix[0, 3]),
    }


def compare_correction_operators(
    casa: NDArray[np.complex128],
    jax_op: NDArray[np.complex128],
    *,
    absolute_tolerance: float = 1.0e-6,
    relative_tolerance: float = 1.0e-5,
) -> dict[str, object]:
    """Compare CASA and JAX 4×4 correlation-space correction operators."""

    first = np.asarray(casa, dtype=np.complex128)
    second = np.asarray(jax_op, dtype=np.complex128)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("operators must be 4×4")
    residual = first - second
    scale = max(float(np.linalg.norm(first)), absolute_tolerance)
    rel_l2 = float(np.linalg.norm(residual) / scale)
    return {
        "relative_l2": rel_l2,
        "max_abs": float(np.max(np.abs(residual))),
        "passed": rel_l2 <= relative_tolerance
        or float(np.max(np.abs(residual))) <= absolute_tolerance,
        "residual": [[_complex_pair(value) for value in row] for row in residual],
        "casa_operator": [[_complex_pair(value) for value in row] for row in first],
        "jax_operator": [[_complex_pair(value) for value in row] for row in second],
        "jones_recovery_blocked": True,
        "notes": (
            "Injected-basis operators exclude source structure and small observed cross-hands",
            JONES_RECOVERY_BLOCKED_NOTE,
        ),
    }


def apply_basis_with_solution(
    block: VisibilityBlock,
    solution: CalibrationSolution,
    *,
    channel: int | None = None,
) -> NDArray[np.complex128]:
    """Apply one solution to the four injected bases on a single-row block."""

    if block.visibility.shape[0] != 1:
        raise ValueError("oracle basis apply uses a single visibility row")
    selected = block if channel is None else slice_block_channel(block, channel)
    outputs = np.zeros((4, 4), dtype=np.complex128)
    for index, basis in enumerate(injected_basis_visibilities()):
        visibility = np.zeros_like(selected.visibility)
        visibility[..., :] = basis
        injected = replace_block_visibility(selected, visibility)
        corrected = apply_calibration(injected, solution, extrapolate=True)
        outputs[index] = corrected.visibility[0, 0]
    return outputs


def slice_block_channel(block: VisibilityBlock, channel: int) -> VisibilityBlock:
    """Keep one channel so the oracle operator is a single 4×4 matrix."""

    index = int(channel)
    return replace(
        block,
        visibility=np.asarray(block.visibility[:, index : index + 1]),
        weight=np.asarray(block.weight[:, index : index + 1]),
        flag=np.asarray(block.flag[:, index : index + 1]),
        frequency_hz=np.asarray(block.frequency_hz[index : index + 1]),
    )


def replace_block_visibility(
    block: VisibilityBlock, visibility: NDArray[np.complex128]
) -> VisibilityBlock:
    return replace(
        block,
        visibility=np.asarray(visibility, dtype=np.complex128),
        provenance={**dict(block.provenance), "column": "DATA", "injected_basis": True},
    )


def compare_stage_visibilities(
    casa: NDArray[np.complex128],
    jax_vis: NDArray[np.complex128],
    valid: NDArray[np.bool_],
    *,
    stage: str,
    antenna1: NDArray[np.int32] | None = None,
    antenna2: NDArray[np.int32] | None = None,
    time_s: NDArray[np.float64] | None = None,
    frequency_hz: NDArray[np.float64] | None = None,
) -> dict[str, object]:
    """Golden metrics plus a convention classification for one stage."""

    report = compare_casa_jax_visibilities(
        casa,
        jax_vis,
        valid,
        antenna1=antenna1,
        antenna2=antenna2,
        time_s=time_s,
        frequency_hz=frequency_hz,
    )
    per_corr = report["per_correlation"]
    rel = {
        name: float(per_corr[name]["relative_l2"])
        for name in ("RR", "RL", "LR", "LL")
        if name in per_corr and "relative_l2" in per_corr[name]
    }
    rl_phase = per_corr.get("RL", {}).get("phase_err_deg_above_floor")
    lr_phase = per_corr.get("LR", {}).get("phase_err_deg_above_floor")
    orientation = report.get("breakdowns", {}).get("baseline_orientation", {})
    order_dep = False
    if orientation:
        left = orientation.get("antenna1_lt_antenna2", {}).get("RL", {}).get("relative_l2")
        right = orientation.get("antenna1_gt_antenna2", {}).get("RL", {}).get("relative_l2")
        if left is not None and right is not None and np.isfinite(left) and np.isfinite(right):
            order_dep = abs(float(left) - float(right)) > 0.05
    antenna_dep = False
    by_antenna = report.get("breakdowns", {}).get("antenna", {})
    if by_antenna:
        rl_l2 = [
            float(item["RL"]["relative_l2"])
            for item in by_antenna.values()
            if "RL" in item and np.isfinite(item["RL"].get("relative_l2", float("nan")))
        ]
        if len(rl_l2) >= 3 and (max(rl_l2) - min(rl_l2)) > 0.1:
            antenna_dep = True
    slope = None
    two_chi = None
    if frequency_hz is not None:
        slope = _residual_phase_frequency_slope(
            casa, jax_vis, valid, correlation=1, frequency_hz=frequency_hz
        )
    times = report.get("breakdowns", {}).get("time", {})
    if times:
        first = times.get("first", {}).get("RL", {}).get("median_resid_over_rr_ll")
        last = times.get("last", {}).get("RL", {}).get("median_resid_over_rr_ll")
        if first is not None and last is not None and np.isfinite(first) and np.isfinite(last):
            two_chi = float(last) - float(first)
    report["stage"] = stage
    report["phase_vs_frequency_slope_rad"] = slope
    report["resid_vs_two_chi"] = two_chi
    report["classification"] = classify_bisection_residual(
        relative_l2_by_corr=rel,
        phase_vs_frequency_slope_rad=slope,
        mean_rl_phase_deg=None if rl_phase is None else float(rl_phase),
        mean_lr_phase_deg=None if lr_phase is None else float(lr_phase),
        resid_depends_on_antenna=antenna_dep,
        resid_vs_two_chi=two_chi,
        resid_depends_on_baseline_order=order_dep,
    )
    return report


def _residual_phase_frequency_slope(
    casa: NDArray[np.complex128],
    jax_vis: NDArray[np.complex128],
    valid: NDArray[np.bool_],
    *,
    correlation: int,
    frequency_hz: NDArray[np.float64],
) -> float | None:
    """Median residual phase versus frequency, in rad per GHz."""

    freqs = np.asarray(frequency_hz, dtype=np.float64)
    if casa.shape[1] != freqs.size:
        return None
    phase = np.angle(
        np.asarray(casa)[..., correlation] * np.conj(np.asarray(jax_vis)[..., correlation])
    )
    usable = np.asarray(valid)[..., correlation] & np.isfinite(phase)
    per_chan = []
    used_freq = []
    for chan in range(freqs.size):
        selected = phase[:, chan][usable[:, chan]]
        if selected.size < 8:
            continue
        per_chan.append(float(np.median(selected)))
        used_freq.append(float(freqs[chan]))
    if len(per_chan) < 8:
        return None
    x = (np.asarray(used_freq) - used_freq[0]) / 1.0e9
    y = np.unwrap(np.asarray(per_chan))
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def stratify_df_delta(
    first: NDArray[np.complex128],
    second: NDArray[np.complex128],
    valid: NDArray[np.bool_],
    *,
    antenna: NDArray[np.int32],
    spectral_window_id: NDArray[np.int32],
    antenna_names: tuple[str, ...],
    reference_antennas: tuple[str, ...],
    moving_antennas: tuple[str, ...],
    snr: NDArray[np.float64] | None = None,
    tail_quantile: float = 0.9,
) -> dict[str, object]:
    """Explain a Df vs Df+QU tail. Do not adopt p90 as a floor yet."""

    delta = np.asarray(first) - np.asarray(second)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(delta)
    abs_delta = np.abs(delta)
    if not np.any(mask):
        return {"n": 0, "adopt_as_floor": False, "notes": "no overlapping valid D samples"}
    threshold = float(np.quantile(abs_delta[mask], tail_quantile))
    tail = mask & (abs_delta >= threshold)
    row_shape = (antenna.size,) + (1,) * (abs_delta.ndim - 1)
    by_antenna = []
    for index, name in enumerate(antenna_names):
        rows = np.reshape(antenna == index, row_shape)
        n_tail = int(np.sum(tail & rows))
        if n_tail == 0 and not np.any(rows):
            continue
        role = (
            "reference"
            if name in reference_antennas
            else "moving"
            if name in moving_antennas
            else "other"
        )
        selected = rows & mask
        by_antenna.append(
            {
                "antenna": name,
                "role": role,
                "n_tail": n_tail,
                "median_abs_delta": _median_or_nan(abs_delta[selected]),
                "median_real_delta": _median_or_nan(np.real(delta[selected])),
                "median_imag_delta": _median_or_nan(np.imag(delta[selected])),
            }
        )
    by_spw = {}
    for spw in np.unique(spectral_window_id):
        rows = np.reshape(spectral_window_id == spw, row_shape)
        by_spw[str(int(spw))] = {
            "n_tail": int(np.sum(tail & rows)),
            "median_abs_delta": _median_or_nan(abs_delta[rows & mask]),
        }
    edge = np.zeros(abs_delta.shape, dtype=bool)
    if abs_delta.ndim >= 2:
        edge[:, :5] = True
        edge[:, -5:] = True
    snr_tail = None
    if snr is not None:
        snr_tail = {
            "median_snr_all": _median_or_nan(np.asarray(snr)[mask]),
            "median_snr_tail": _median_or_nan(np.asarray(snr)[tail]),
        }
    concentrated = False
    if by_antenna:
        counts = np.asarray([item["n_tail"] for item in by_antenna], dtype=np.float64)
        if counts.sum() > 0 and float(np.max(counts) / counts.sum()) > 0.4:
            concentrated = True
    edge_fraction = float(np.sum(tail & edge) / np.sum(tail)) if np.any(tail) else 0.0
    if edge_fraction > 0.5:
        concentrated = True
    aligned = _align_leakage_gauge(first, second, mask, antenna, antenna_names, reference_antennas)
    return {
        "n": int(np.sum(mask)),
        "n_tail": int(np.sum(tail)),
        "tail_quantile": tail_quantile,
        "tail_threshold": threshold,
        "median_abs_delta": _median_or_nan(abs_delta[mask]),
        "p90_abs_delta": float(np.quantile(abs_delta[mask], 0.9)),
        "by_antenna": by_antenna,
        "by_spectral_window": by_spw,
        "edge_channel_tail_fraction": edge_fraction,
        "snr": snr_tail,
        "gauge_aligned": aligned,
        "concentrated_in_failed_subset": concentrated,
        "adopt_as_floor": (not concentrated) and threshold < 0.05,
        "notes": (
            "A p90 of 0.24 is not an ordinary leakage floor until the tail is explained",
            "Primary comparison is held-out visibility prediction, not raw D-table distance",
            "Df and Df+QU need not share a polarization gauge",
        ),
    }


def _align_leakage_gauge(
    first: NDArray[np.complex128],
    second: NDArray[np.complex128],
    mask: NDArray[np.bool_],
    antenna: NDArray[np.int32],
    antenna_names: tuple[str, ...],
    reference_antennas: tuple[str, ...],
) -> dict[str, float]:
    """Remove a common-mode D offset permitted by the source-polarization gauge."""

    common = np.zeros(np.asarray(first).shape[1:], dtype=np.complex128)
    usable = np.asarray(mask, dtype=bool)
    if np.any(usable):
        common = np.nanmean(np.where(usable, first - second, np.nan), axis=0)
    aligned = np.abs((first - second) - common)
    payload = {
        "median_abs_delta": _median_or_nan(aligned[usable]),
        "p90_abs_delta": float(np.quantile(aligned[usable], 0.9))
        if np.any(usable)
        else float("nan"),
    }
    if antenna_names and reference_antennas:
        try:
            ref_index = antenna_names.index(reference_antennas[0])
        except ValueError:
            return payload
        rows = np.reshape(antenna == ref_index, (antenna.size,) + (1,) * (aligned.ndim - 1))
        ref_offset = np.nanmean(np.where(rows & usable, first - second, np.nan), axis=0)
        ref_aligned = np.abs((first - second) - ref_offset)
        payload["median_abs_delta_after_refant"] = _median_or_nan(ref_aligned[usable])
        payload["p90_abs_delta_after_refant"] = (
            float(np.quantile(ref_aligned[usable], 0.9)) if np.any(usable) else float("nan")
        )
    return payload


def effective_chi_from_p_operators(
    with_p: NDArray[np.complex128],
    without_p: NDArray[np.complex128],
) -> dict[str, float]:
    """Recover CASA χ_p+χ_q from +P versus +Xf (or prefix) operators."""

    full = np.asarray(with_p, dtype=np.complex128)
    prefix = np.asarray(without_p, dtype=np.complex128)
    if full.shape != (4, 4) or prefix.shape != (4, 4):
        raise ValueError("operators must be 4x4")
    rl = full[1, 1] / prefix[1, 1]
    lr = full[2, 2] / prefix[2, 2]
    two_chi = float(np.angle(rl))
    return {
        "two_chi_rad": two_chi,
        "two_chi_deg": float(np.rad2deg(two_chi)),
        "chi_mean_rad": 0.5 * two_chi,
        "chi_mean_deg": float(np.rad2deg(0.5 * two_chi)),
        "lr_two_chi_rad": float(np.angle(lr)),
        "rl_lr_sum_rad": float(np.angle(rl * lr)),
        "rr_ratio_abs": float(np.abs(full[0, 0] / prefix[0, 0])),
        "ll_ratio_abs": float(np.abs(full[3, 3] / prefix[3, 3])),
    }


def classify_chi_residual(
    *,
    residual_deg: NDArray[np.float64],
    hour_angle_rad: NDArray[np.float64] | None = None,
    field_id: NDArray[np.int32] | None = None,
    antenna: NDArray[np.int32] | None = None,
    floor_deg: float = 0.05,
) -> str:
    """Map a χ residual onto feed, time-standard, field, or antenna causes."""

    values = np.asarray(residual_deg, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "no_samples"
    if float(np.ptp(values)) < floor_deg:
        return (
            "constant_feed_or_receptor_angle"
            if abs(float(np.median(values))) > floor_deg
            else "consistent"
        )
    if hour_angle_rad is not None and hour_angle_rad.size == residual_deg.size:
        usable = np.isfinite(hour_angle_rad) & np.isfinite(residual_deg)
        if int(np.count_nonzero(usable)) >= 4:
            slope = float(np.polyfit(hour_angle_rad[usable], residual_deg[usable], 1)[0])
            if abs(slope) > 0.2:
                return "hour_angle_apparent_or_time_standard"
    if field_id is not None and field_id.size == residual_deg.size:
        fields = np.unique(field_id[np.isfinite(residual_deg)])
        medians = [
            float(np.median(residual_deg[field_id == field]))
            for field in fields
            if np.any(field_id == field)
        ]
        if len(medians) >= 2 and float(np.ptp(medians)) > floor_deg:
            return "wrong_field_direction"
    if antenna is not None and antenna.size == residual_deg.size:
        antennas = np.unique(antenna[np.isfinite(residual_deg)])
        medians = [
            float(np.median(residual_deg[antenna == item]))
            for item in antennas
            if np.any(antenna == item)
        ]
        if len(medians) >= 2 and float(np.ptp(medians)) > floor_deg:
            return "observatory_centre_versus_antenna_itrf"
    return "unclassified_chi_residual"


def xf_operator_effect(
    with_x: NDArray[np.complex128],
    without_x: NDArray[np.complex128],
) -> dict[str, object]:
    """Say whether Xf changes the operator or is identity on this row."""

    full = np.asarray(with_x, dtype=np.complex128)
    prefix = np.asarray(without_x, dtype=np.complex128)
    rel = float(np.linalg.norm(full - prefix) / max(float(np.linalg.norm(prefix)), 1.0e-12))
    rl_phase = float(np.angle(full[1, 1] / prefix[1, 1])) if abs(prefix[1, 1]) else 0.0
    return {
        "relative_l2": rel,
        "rl_phase_deg": float(np.rad2deg(rl_phase)),
        "identity": rel < 1.0e-5,
    }


def score_interpolation_methods(
    query_times: NDArray[np.float64],
    cal_times: NDArray[np.float64],
    cal_values: NDArray[np.complex128],
    casa_values: NDArray[np.complex128],
    *,
    methods: tuple[str, ...] = (
        "nearest",
        "linear_complex",
        "linear_amp_phase",
        "casa_linear",
    ),
) -> list[dict[str, object]]:
    """Score discrete interpolation methods against CASA-implied Jones samples."""

    from sl1mjax.calibration import interpolate_complex_series

    ranked = []
    for method in methods:
        predicted = interpolate_complex_series(query_times, cal_times, cal_values, method=method)
        residual = np.asarray(casa_values) - predicted
        scale = max(float(np.linalg.norm(casa_values)), 1.0e-12)
        ranked.append(
            {
                "method": method,
                "relative_l2": float(np.linalg.norm(residual) / scale),
                "max_abs": float(np.max(np.abs(residual))),
            }
        )
    ranked.sort(key=lambda item: float(item["relative_l2"]))
    return ranked


def _complex_pair(value: complex) -> dict[str, float]:
    return {"re": float(np.real(value)), "im": float(np.imag(value))}


def _median_or_nan(values: NDArray[np.floating] | NDArray[np.complexfloating]) -> float:
    selected = np.asarray(values)
    selected = selected[np.isfinite(selected)]
    return float(np.median(selected)) if selected.size else float("nan")


def write_bisection_report(payload: dict[str, object], path: Any) -> Any:
    return write_json(payload, path)
