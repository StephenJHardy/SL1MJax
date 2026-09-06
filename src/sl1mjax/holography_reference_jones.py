"""Residual on-axis Jones for holography, including every antenna.

The identity-reference estimator is the failed product, not full-Jones
holography in principle. A residual on a moving antenna can be
misidentified as its direction-dependent beam if only the seven
holography references are solved.

    V_pq(t) = R_p(t) S(Q, U) R_q(t)^H          (on-axis field 9)
    R_p(t)  = P(χ_p)^H (I + ε_p) P(χ_p)        (CASA parang=True)
    V_mr(s) = R_m E_m(s) S R_r^H               (HOLORASTER)
    E_p(0)  = I                                (beam gauge)

S(Q, U) is a nuisance, not a fixed zero or a single point estimate.
Parallactic rotation separates sky-fixed Q/U from feed-fixed ε.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import CIRCULAR_P_JONES, RESIDUAL_JONES_SKY_FRAME
from sl1mjax.polarization import (
    circular_parallactic_jones,
    circular_stokes_to_coherency,
    invert_jones,
)

IDENTITY_REFERENCE_JONES_BLOCKS_OFFDIAG_NOTE = (
    "Per-reference identity gauge is adequate for |E|~1 copolar voltages. "
    "Reference scatter of ~2.5% is comparable to the cross-hand floor and "
    "can dominate off-diagonal voltages of a few percent. Estimate residual "
    "on-axis E_r(0) from reference-reference HOLORASTER rows and recover "
    "E_m(s) against those corrections."
)
CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE = (
    "A median |RL|/I of a few percent is comparable to the expected "
    "leakage. It is a blocking uncertainty until complex residuals are "
    "shown to be incoherent thermal noise. Magnitude-only medians have a "
    "positive |RL| bias."
)
THREE_C147_QU_IS_NUISANCE_NOTE = (
    "A measured field-9 P/I is not a passed Q=U=0 model. Treat Q/U as a "
    "nuisance with uncertainty and propagate it into recovered off-diagonals. "
    "Do not let either zero or the point estimate silently enter the beam."
)
JJH_PROXY_IS_NOT_DECISIVE_NOTE = (
    "J J^H later-visit agreement tests the present estimator. A baseline "
    "measures E_p S E_q^H with different antenna Jones matrices. The "
    "decisive comparison predicts held-out moving-reference visibilities."
)
THREE_C286_CIRCULAR_FLOOR_NOTE = (
    "Recovered 3C286 V/I is the current circular-polarisation calibration "
    "floor. It may be acceptable for this beam experiment but is not "
    "negligible for the eventual full-Stokes package."
)
ALL_ANTENNA_RESIDUAL_JONES_NOTE = (
    "Solve a smooth near-identity residual Jones R_p for every antenna "
    "from the on-axis field-9 track. Restricting the solve to the seven "
    "holography references can misidentify a moving-antenna residual as "
    "its direction-dependent beam. Coherent field-9 failures involve "
    "ea04, ea11 and ea25."
)
CONNECTED_HOLDOUT_NOTE = (
    "Validate R_p on held-out times, baselines, samples, and complete "
    "field-9 scan clusters while keeping the antenna graph connected. "
    "Holding out an entire antenna and predicting identity tests the "
    "antenna-population prior and is a separate gate."
)
RESIDUAL_JONES_VISIBILITY_GATES_NOTE = (
    "Full-Jones holography is blocked by whether a constrained residual "
    "antenna Jones can predict later field-9 data without absorbing source "
    "polarisation. The identity-gauge floor does not permanently block "
    "this estimator."
)
RESIDUAL_JONES_P_CONVENTION_NOTE = (
    "After CASA applycal(parang=True) the locked sky-frame residual is "
    "R = P^H (I+ε) P, matching J = … D P. Feed-frame RL then appears as "
    "e^{+2iχ} in the sky-frame cross-hand. Gauge invariance does not "
    "test this sign or order. The opposite sandwich P(I+ε)P^{-1} is refused."
)
ON_AXIS_BEAM_IDENTITY_NOTE = (
    "The beam gauge is E_p(0)=I. R_p(t) holds the complete on-axis "
    "residual. E_p(s) holds only direction-dependent variation. Otherwise "
    "constant leakage can move arbitrarily between the two factors."
)
CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE = (
    "Operator identifiability is not calibration adequacy. A median "
    "|RL|/I of a few percent is magnitude-biased and can sit at the "
    "leakage scale. Record the complex coherent residual and its "
    "N^{-1/2} behaviour for each holdout, especially by antenna and "
    "baseline."
)
ZERO_QU_IS_ABLATION_NOTE = (
    "The Q=U=0 field-9 model is an ablation. It must not silently "
    "become the 3C147 source model. Propagate the measured Q/U "
    "interval as a nuisance."
)
CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE = (
    "Field-9 residual-Jones holdouts are conditional on the existing CASA "
    "calibration. If those tables used all field-9 times, this is not an "
    "end-to-end calibration holdout."
)
CHANNEL_AVERAGING_CAN_FAKE_TIME_NOTE = (
    "Changing channel flags and weights can make frequency-dependent "
    "leakage look time-dependent. Fit native channels, or require "
    "identical channel support in every time bin, before adding time "
    "freedom."
)
SCAN_STATE_MAY_NEED_OFFSETS_NOTE = (
    "A raw per-scan ε jump is not itself evidence for a HOLORASTER "
    "state change. Confirm the jump in gauge-aligned predicted "
    "visibility operators, and by cross-applying the solutions, before "
    "adding piecewise scan offsets or a state term."
)
PER_SCAN_EPSILON_NOT_OBSERVABLE_NOTE = (
    "The scan-53→56 |Δε|~0.25 jump is absent from the visibilities and "
    "the cumulative CASA chain at channel 32. Fitted per-scan ε is not "
    "directly observable. Do not add piecewise scan offsets or a "
    "HOLORASTER-state term from that parameter jump."
)
ESTIMATOR_IDENTIFIABILITY_NOTE = (
    "The field-9 gate is visibility-operator identifiability, not unique "
    "Jones-factor identifiability. Compare predicted R_p S R_q^H "
    "operators, not raw ε. Individual ε_p remain gauge-dependent. A "
    "temporal GP on raw matrix entries follows weakly determined modes "
    "and extrapolates badly. Stabilize the all-antenna solve; do not add "
    "more time freedom."
)
COMPLETE_FIELD9_TRACK_NOTE = (
    "Once the estimator is stabilized, the residual Jones may use the "
    "complete field-9 track. Hold out baselines, samples, and complete "
    "scan clusters. There is no need to predict the entire later half "
    "from the early half if all field-9 calibration scans are legitimate "
    "inputs. HOLORASTER visibilities stay out of that fit."
)
PREDICTION_EQUIVALENT_BEAM_NOTE = (
    "Prediction-equivalent R_p solutions can produce the same held-out "
    "moving-reference visibilities and different RL/LR beam maps. Those "
    "maps are not uniquely physical until a beam gauge such as E_m(0)=I "
    "resolves the remaining Jones-factor freedom."
)
ONE_AXIS_VISIBILITY_HOLDOUT_NOTE = (
    "Score each visibility holdout axis separately. Do not combine them. "
    "The previous unioned holdout removed interpolation support. "
    "Leave-one-mover-out does not test a per-antenna empirical map."
)
NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE = (
    "Do not add piecewise scan offsets, a HOLORASTER-state term, a "
    "temporal GP, HOLORASTER fitting, or SPW 5 while the field-9 "
    "estimator is being stabilized."
)
SMOOTH_GP_REJECTED_NOTE = (
    "The early-to-late smooth residual-Jones GP is rejected. More knots or "
    "hyperparameter searches would only overfit the same training regime. "
    "That failure is evidence about calibration extrapolation, not a "
    "permanent block on full-Jones beam recovery."
)
INTERLEAVED_ONAXIS_TRANSFER_NOTE = (
    "THOL0001 interleaves on-axis scans so time-local instrumental changes "
    "can be removed before combining holography measurements. Early-to-late "
    "prediction without another calibrator visit is a stronger requirement "
    "than this beam experiment needs."
)
DO_NOT_FIT_UNCONSTRAINED_PER_SCAN_JONES_NOTE = (
    "Do not fit an unconstrained Jones independently for every scan. That "
    "would make field-9 Q/U unidentifiable and could remove real cross-hand "
    "beam response."
)
HIERARCHICAL_SCAN_STATE_MODEL_NOTE = (
    "A suitable next model is ε_{p,k}=ε_{p,0}+c_k+δε_{p,k}, where c_k is a "
    "low-dimensional shared scan or acquisition-state term and δε_{p,k} is a "
    "strongly shrunk antenna residual. Keep one global 3C147 Q/U nuisance."
)
INTERLEAVED_BEAM_TRANSFER_GATE_NOTE = (
    "The key gate is that interleaved on-axis calibration removes "
    "scan-dependent residual Jones and produces a direction-dependent beam "
    "that transfers across independent holography visits. For each "
    "HOLORASTER block use only its declared preceding or following field-9 "
    "calibration, not holography cross-hands."
)
CALIBRATION_CHAIN_STAGES = (
    "DATA",
    "K_B_G",
    "K_B_G_Kcross",
    "K_B_G_Kcross_Df",
    "K_B_G_Kcross_Df_Xf",
    "K_B_G_Kcross_Df_Xf_P",
)


def reference_jones_for_antenna(
    reference_jones: ArrayLike | Mapping[int, ArrayLike] | None,
    antenna_id: int,
) -> NDArray[np.complex128]:
    """Return the 2×2 on-axis Jones for one reference antenna."""

    if reference_jones is None:
        return np.eye(2, dtype=np.complex128)
    if isinstance(reference_jones, Mapping):
        if int(antenna_id) not in reference_jones:
            raise ValueError(
                f"no residual reference Jones for antenna {antenna_id}; "
                "do not silently substitute identity"
            )
        plane = np.asarray(reference_jones[int(antenna_id)], dtype=np.complex128)
    else:
        plane = np.asarray(reference_jones, dtype=np.complex128)
        if plane.ndim == 3:
            if int(antenna_id) >= plane.shape[0]:
                raise ValueError(f"reference_jones has no slot for antenna {antenna_id}")
            plane = plane[int(antenna_id)]
    if plane.shape != (2, 2):
        raise ValueError("reference Jones must have shape (2, 2) per antenna")
    return plane


def classify_crosshand_floor(
    *,
    median_abs_rl_over_i: float,
    coherent_mean_abs: float | None = None,
    averages_as_noise: bool | None = None,
    max_group_coherent_abs: float | None = None,
    leakage_scale: float = 0.02,
) -> dict[str, object]:
    """Magnitude floor is blocking until residuals are shown to be noise.

    A global complex mean can hide a few-percent leftover that is coherent
    on a subset of antennas or baselines. Stratified coherence at the
    leakage scale remains blocking.
    """

    median_abs = float(median_abs_rl_over_i)
    comparable = np.isfinite(median_abs) and median_abs >= leakage_scale
    group_coherent = (
        max_group_coherent_abs is not None
        and np.isfinite(max_group_coherent_abs)
        and float(max_group_coherent_abs) >= leakage_scale
    )
    globally_incoherent = averages_as_noise is True and (
        coherent_mean_abs is None or float(coherent_mean_abs) < 0.25 * median_abs
    )
    if group_coherent:
        status = "fail"
        blocking = True
    elif globally_incoherent:
        status = "warn" if comparable else "pass"
        blocking = False
    elif comparable:
        status = "fail"
        blocking = True
    else:
        status = "pass"
        blocking = False
    return {
        "status": status,
        "blocking": blocking,
        "blocks_identity_gauge": blocking,
        "blocks_residual_jones_estimator": False,
        "median_abs_rl_over_i": median_abs,
        "coherent_mean_abs": coherent_mean_abs,
        "averages_as_noise": averages_as_noise,
        "max_group_coherent_abs": max_group_coherent_abs,
        "notes": (
            CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE,
            RESIDUAL_JONES_VISIBILITY_GATES_NOTE,
        ),
    }


def classify_three_c147_qu_model(
    *,
    q_over_i: float,
    u_over_i: float,
    frac_pol: float | None = None,
    represented_as_exact_zero: bool,
) -> dict[str, object]:
    """A small measured P/I is not a passed exact-zero source model."""

    frac = float(frac_pol) if frac_pol is not None else float(np.hypot(q_over_i, u_over_i))
    if represented_as_exact_zero:
        status = "fail" if frac > 5.0e-4 else "warn"
    else:
        status = "warn"
    return {
        "status": status,
        "blocking": represented_as_exact_zero,
        "q_over_i": float(q_over_i),
        "u_over_i": float(u_over_i),
        "frac_pol": frac,
        "represented_as_exact_zero": represented_as_exact_zero,
        "relationship_to_model_data": (
            "multiply_resolved_stokes_i_by_measured_fractions_with_uncertainty"
        ),
        "notes": (THREE_C147_QU_IS_NUISANCE_NOTE,),
    }


def coherent_residual_report(
    values: ArrayLike,
    *,
    group_ids: ArrayLike | None = None,
    n_sizes: tuple[int, ...] = (4, 16, 64, 256),
) -> dict[str, object]:
    """Complex mean versus |·| bias, and whether averages follow N^{-1/2}.

    Thermal noise has a positive median |z|. A coherent leftover stays in
    the complex mean and does not fall as N^{-1/2} across independent draws.
    """

    z = np.asarray(values, dtype=np.complex128).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {"n": 0, "averages_as_noise": False, "status": "not_run"}
    median_abs = float(np.median(np.abs(z)))
    mean = complex(np.mean(z))
    mean_abs = float(np.abs(mean))
    rng = np.random.default_rng(0)
    by_n = []
    shuffled = z.copy()
    rng.shuffle(shuffled)
    for size in n_sizes:
        if z.size < 2 * size:
            continue
        n_groups = z.size // size
        means = np.array([np.mean(shuffled[i * size : (i + 1) * size]) for i in range(n_groups)])
        measured = float(np.median(np.abs(means)))
        expected = median_abs / np.sqrt(size)
        by_n.append(
            {
                "n": size,
                "median_abs_mean": measured,
                "noise_expectation": float(expected),
                "ratio": measured / expected if expected > 0 else float("nan"),
            }
        )
    ratios = [item["ratio"] for item in by_n if np.isfinite(item["ratio"])]
    averages_as_noise = (
        bool(ratios) and float(np.median(ratios)) < 2.0 and mean_abs < 0.25 * median_abs
    )
    by_group = {}
    if group_ids is not None:
        labels = np.asarray(group_ids).reshape(-1)
        if labels.size == z.size:
            for name in np.unique(labels):
                selected = z[labels == name]
                if selected.size == 0:
                    continue
                by_group[str(name)] = {
                    "n": int(selected.size),
                    "mean_abs": float(np.abs(np.mean(selected))),
                    "median_abs": float(np.median(np.abs(selected))),
                }
    return {
        "n": int(z.size),
        "median_abs": median_abs,
        "coherent_mean": [mean.real, mean.imag],
        "coherent_mean_abs": mean_abs,
        "mean_over_median_abs": mean_abs / median_abs if median_abs > 0 else float("nan"),
        "random_averaging": by_n,
        "averages_as_noise": averages_as_noise,
        "by_group": by_group,
        "notes": (CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE,),
    }


def predict_baseline_coherency(
    e_p: ArrayLike,
    source: ArrayLike,
    e_q: ArrayLike,
) -> NDArray[np.complex128]:
    """Return :math:`E_p S E_q^H`."""

    left = np.asarray(e_p, dtype=np.complex128)
    right = np.asarray(e_q, dtype=np.complex128)
    plane = np.asarray(source, dtype=np.complex128)
    return left @ plane @ np.conjugate(np.swapaxes(right, -1, -2))


def serialize_reference_jones(
    jones: Mapping[int, ArrayLike],
) -> dict[str, dict[str, list[list[float]]]]:
    """JSON-safe real/imag packing of a per-antenna Jones map."""

    packed = {}
    for antenna, plane in jones.items():
        arr = np.asarray(plane, dtype=np.complex128)
        packed[str(int(antenna))] = {
            "real": arr.real.tolist(),
            "imag": arr.imag.tolist(),
        }
    return packed


def deserialize_reference_jones(
    payload: Mapping[str, Mapping[str, list[list[float]]]],
) -> dict[int, NDArray[np.complex128]]:
    """Invert :func:`serialize_reference_jones`."""

    jones = {}
    for key, plane in payload.items():
        jones[int(key)] = np.asarray(plane["real"], dtype=np.float64) + 1j * np.asarray(
            plane["imag"], dtype=np.float64
        )
    return jones


def estimate_residual_reference_jones(
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    coherency: ArrayLike,
    *,
    source: ArrayLike | None = None,
    antenna_ids: ArrayLike | None = None,
    gauge_antenna_id: int | None = None,
    holdout_antenna_id: int | None = None,
    row_mask: ArrayLike | None = None,
    ridge: float = 1.0,
) -> dict[str, object]:
    """Least-squares :math:`E_r(0)\\approx I+\\varepsilon` from :math:`V_{rr'}=E_r S E_{r'}^H`.

    Corrections are ridge-shrunk toward identity. One antenna is fixed at
    identity as the remaining unitary/gauge pin. Held-out antennas and
    rows are not used in the fit. The linearisation drops the quadratic
    :math:`\\varepsilon_p S \\varepsilon_q^H` term.
    """

    p_ant = np.asarray(antenna1, dtype=np.int32).reshape(-1)
    q_ant = np.asarray(antenna2, dtype=np.int32).reshape(-1)
    vis = np.asarray(coherency, dtype=np.complex128)
    if vis.ndim != 3 or vis.shape[0] != p_ant.size or vis.shape[1:] != (2, 2):
        raise ValueError("coherency must have shape (row, 2, 2)")
    if source is None:
        s_planes = np.broadcast_to(np.eye(2, dtype=np.complex128), vis.shape)
    else:
        s_planes = np.broadcast_to(np.asarray(source, dtype=np.complex128), vis.shape)
    ids = (
        np.unique(np.concatenate([p_ant, q_ant]))
        if antenna_ids is None
        else np.asarray(antenna_ids, dtype=np.int32)
    )
    train = p_ant != q_ant
    if row_mask is not None:
        train = train & np.asarray(row_mask, dtype=bool).reshape(-1)
    if holdout_antenna_id is not None:
        train = train & (p_ant != int(holdout_antenna_id)) & (q_ant != int(holdout_antenna_id))
    gauge = int(ids[0] if gauge_antenna_id is None else gauge_antenna_id)
    if holdout_antenna_id is not None and int(holdout_antenna_id) == gauge:
        raise ValueError("holdout antenna cannot also be the gauge antenna")
    free = [int(ant) for ant in ids if int(ant) != gauge]
    index = {ant: slot for slot, ant in enumerate(free)}
    n_par = 8 * len(free)
    gram = float(ridge) * np.eye(max(n_par, 1), dtype=np.float64)
    proj = np.zeros(max(n_par, 1), dtype=np.float64)
    n_eq = 0

    def _eps_slot(antenna: int, row: int, col: int) -> int | None:
        if antenna == gauge or antenna not in index:
            return None
        return 8 * index[antenna] + 2 * (2 * row + col)

    def _add_complex(terms: list[tuple[int, complex, bool]], value: complex) -> None:
        real = np.zeros(n_par, dtype=np.float64)
        imag = np.zeros(n_par, dtype=np.float64)
        for slot, multiplier, conjugated in terms:
            mr = float(np.real(multiplier))
            mi = float(np.imag(multiplier))
            if conjugated:
                real[slot] += mr
                imag[slot] += mi
                real[slot + 1] += mi
                imag[slot + 1] -= mr
            else:
                real[slot] += mr
                imag[slot] += mi
                real[slot + 1] -= mi
                imag[slot + 1] += mr
        gram[:n_par, :n_par] += np.outer(real, real) + np.outer(imag, imag)
        proj[:n_par] += real * float(np.real(value)) + imag * float(np.imag(value))

    for row in np.flatnonzero(train):
        s = s_planes[row]
        leftover = vis[row] - s
        p_id = int(p_ant[row])
        q_id = int(q_ant[row])
        for a in range(2):
            for b in range(2):
                terms: list[tuple[int, complex, bool]] = []
                for k in range(2):
                    slot_p = _eps_slot(p_id, a, k)
                    if slot_p is not None:
                        terms.append((slot_p, s[k, b], False))
                    slot_q = _eps_slot(q_id, b, k)
                    if slot_q is not None:
                        terms.append((slot_q, s[a, k], True))
                _add_complex(terms, leftover[a, b])
                n_eq += 2

    if n_eq == 0:
        raise ValueError("no reference-reference rows for residual Jones")
    jones: dict[int, NDArray[np.complex128]] = {}
    if n_par == 0:
        for ant in ids:
            jones[int(ant)] = np.eye(2, dtype=np.complex128)
    else:
        solution = np.linalg.solve(gram[:n_par, :n_par], proj[:n_par])
        for ant in ids:
            eps = np.zeros((2, 2), dtype=np.complex128)
            if int(ant) in index:
                base = 8 * index[int(ant)]
                packed = solution[base : base + 8]
                eps[0, 0] = packed[0] + 1j * packed[1]
                eps[0, 1] = packed[2] + 1j * packed[3]
                eps[1, 0] = packed[4] + 1j * packed[5]
                eps[1, 1] = packed[6] + 1j * packed[7]
            jones[int(ant)] = np.eye(2, dtype=np.complex128) + eps
    holdout = None
    if holdout_antenna_id is not None:
        mask = (p_ant == int(holdout_antenna_id)) | (q_ant == int(holdout_antenna_id))
        residuals = [
            np.abs(
                vis[row]
                - predict_baseline_coherency(
                    jones[int(p_ant[row])], s_planes[row], jones[int(q_ant[row])]
                )
            )
            for row in np.flatnonzero(mask)
        ]
        holdout = {
            "antenna": int(holdout_antenna_id),
            "n": int(np.sum(mask)),
            "median_abs_resid": float(np.median(residuals)) if residuals else float("nan"),
            "used_identity_for_held_out_antenna": True,
        }
    deviations = [float(np.linalg.norm(plane - np.eye(2))) for plane in jones.values()]
    return {
        "jones": jones,
        "gauge_antenna_id": gauge,
        "ridge": float(ridge),
        "n_train": int(np.sum(train)),
        "n_equations": n_eq,
        "median_abs_epsilon": float(np.median(deviations)),
        "max_abs_epsilon": float(np.max(deviations)),
        "holdout": holdout,
        "notes": (IDENTITY_REFERENCE_JONES_BLOCKS_OFFDIAG_NOTE,),
    }


def sky_frame_residual_jones(
    feed_jones: ArrayLike,
    parallactic_angle_rad: ArrayLike,
    *,
    convention: str = RESIDUAL_JONES_SKY_FRAME,
) -> NDArray[np.complex128]:
    """Return the locked CASA sky-frame residual :math:`R=P^H(I+\\varepsilon)P`.

    The opposite sandwich ``P(I+ε)P^{-1}`` is accepted only as a negative
    control. It is not the scientific convention.
    """

    feed = np.asarray(feed_jones, dtype=np.complex128)
    para = circular_parallactic_jones(parallactic_angle_rad)
    if convention == RESIDUAL_JONES_SKY_FRAME:
        return invert_jones(para) @ feed @ para
    if convention == "P (I+eps) P^{-1}":
        return para @ feed @ invert_jones(para)
    raise ValueError(f"unsupported residual Jones sky-frame convention {convention!r}")


def injected_feed_leakage_rotation(
    parallactic_angle_rad: ArrayLike,
    *,
    eps_rl: complex = 0.03 + 0.0j,
) -> dict[str, object]:
    """Inject feed-frame RL leakage and recover the sky-frame 2χ slope.

    Locked ``P^H (I+ε) P`` must give slope +1 on unwrapped arg(RL)/ε versus
    2χ. The opposite sandwich must give slope −1. Gauge invariance is not
    this test.
    """

    angles = np.asarray(parallactic_angle_rad, dtype=np.float64).reshape(-1)
    feed = np.eye(2, dtype=np.complex128)
    feed[0, 1] = complex(eps_rl)
    source = np.eye(2, dtype=np.complex128)
    phases = {"locked": [], "opposite": []}
    for angle in angles:
        for name, convention in (
            ("locked", RESIDUAL_JONES_SKY_FRAME),
            ("opposite", "P (I+eps) P^{-1}"),
        ):
            sky = sky_frame_residual_jones(feed, angle, convention=convention)
            vis = sky @ source @ np.eye(2)
            phases[name].append(float(np.angle(vis[0, 1] / feed[0, 1])))
    two_chi = 2.0 * angles
    slopes = {}
    for name, series in phases.items():
        unwrap = np.unwrap(np.asarray(series, dtype=np.float64))
        slopes[name] = float(np.polyfit(two_chi, unwrap, 1)[0])
    locked_ok = abs(slopes["locked"] - 1.0) < 0.05
    opposite_ok = abs(slopes["opposite"] + 1.0) < 0.05
    return {
        "status": "pass" if locked_ok and opposite_ok else "fail",
        "circular_p_jones": CIRCULAR_P_JONES,
        "residual_jones_sky_frame": RESIDUAL_JONES_SKY_FRAME,
        "locked_slope_darg_over_d_two_chi": slopes["locked"],
        "opposite_slope_darg_over_d_two_chi": slopes["opposite"],
        "expected_locked_rl_phase": "exp(+2i chi)",
        "notes": (RESIDUAL_JONES_P_CONVENTION_NOTE,),
    }


def identical_channel_support(
    finite: ArrayLike,
    *,
    min_fraction: float = 1.0,
) -> tuple[NDArray[np.bool_], dict[str, object]]:
    """Channels that keep the same support in every time row.

    ``finite`` has shape (row, channel). A channel is kept only when the
    fraction of finite rows is at least ``min_fraction``. ``min_fraction=1``
    is identical support.
    """

    mask = np.asarray(finite, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("finite must have shape (row, channel)")
    fraction = np.mean(mask, axis=0)
    keep = fraction >= float(min_fraction)
    per_row = np.sum(mask, axis=1)
    time_dependent = bool(np.max(per_row) - np.min(per_row) > 0) if mask.size else False
    return keep, {
        "n_channel": int(mask.shape[1]),
        "n_kept": int(np.sum(keep)),
        "min_row_support": int(np.min(per_row)) if per_row.size else 0,
        "max_row_support": int(np.max(per_row)) if per_row.size else 0,
        "time_dependent_support": time_dependent,
        "notes": (CHANNEL_AVERAGING_CAN_FAKE_TIME_NOTE,),
    }


def residual_jones_for_antenna(
    residual_jones: ArrayLike | Mapping[int, ArrayLike] | None,
    antenna_id: int,
    *,
    parallactic_angle_rad: float | None = None,
) -> NDArray[np.complex128]:
    """Feed-frame residual, rotated into the sky frame when χ is given."""

    feed = reference_jones_for_antenna(residual_jones, antenna_id)
    if parallactic_angle_rad is None:
        return feed
    return sky_frame_residual_jones(feed, parallactic_angle_rad)


def antenna_graph_is_connected(
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    *,
    antenna_ids: ArrayLike | None = None,
    row_mask: ArrayLike | None = None,
) -> bool:
    """True when the selected baselines connect every listed antenna."""

    p_ant = np.asarray(antenna1, dtype=np.int32).reshape(-1)
    q_ant = np.asarray(antenna2, dtype=np.int32).reshape(-1)
    usable = p_ant != q_ant
    if row_mask is not None:
        usable = usable & np.asarray(row_mask, dtype=bool).reshape(-1)
    ids = (
        np.unique(np.concatenate([p_ant[usable], q_ant[usable]]))
        if antenna_ids is None
        else np.asarray(antenna_ids, dtype=np.int32)
    )
    if ids.size == 0:
        return False
    parent = {int(ant): int(ant) for ant in ids}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for p_id, q_id in zip(p_ant[usable], q_ant[usable], strict=True):
        if int(p_id) not in parent or int(q_id) not in parent:
            continue
        parent[find(int(p_id))] = find(int(q_id))
    roots = {find(int(ant)) for ant in ids}
    return len(roots) == 1


def connected_baseline_holdout_mask(
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    *,
    holdout_fraction: float = 0.2,
    rng: np.random.Generator | None = None,
) -> NDArray[np.bool_]:
    """Hold out a subset of unique pairs and keep the remaining graph connected.

    The returned mask is True for training rows.
    """

    p_ant = np.asarray(antenna1, dtype=np.int32).reshape(-1)
    q_ant = np.asarray(antenna2, dtype=np.int32).reshape(-1)
    pairs = np.stack([np.minimum(p_ant, q_ant), np.maximum(p_ant, q_ant)], axis=1)
    unique = np.unique(pairs, axis=0)
    generator = np.random.default_rng(0) if rng is None else rng
    order = generator.permutation(unique.shape[0])
    n_hold = int(np.floor(float(holdout_fraction) * unique.shape[0]))
    train_pairs = unique[order[n_hold:]]
    if not antenna_graph_is_connected(train_pairs[:, 0], train_pairs[:, 1]):
        train_pairs = unique
    allowed = {(int(a), int(b)) for a, b in train_pairs}
    return np.array([(int(a), int(b)) in allowed for a, b in pairs], dtype=bool)


def field9_scan_clusters(
    scan_id: ArrayLike,
    time_s: ArrayLike,
    row_mask: ArrayLike,
    *,
    predecessor_field: Mapping[int, int | None] | None = None,
    holoraster_field: int = 10,
    gap_s: float = 400.0,
) -> list[list[int]]:
    """Group field-9 scans into visit clusters.

    A new cluster starts after a HOLORASTER predecessor or after a time
    gap longer than ``gap_s``. Isolated scans are their own clusters.
    """

    scans = np.asarray(scan_id, dtype=np.int32).reshape(-1)
    times = np.asarray(time_s, dtype=np.float64).reshape(-1)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    unique = sorted({int(scan) for scan in scans[mask]})
    if not unique:
        return []
    medians = {scan: float(np.median(times[mask & (scans == scan)])) for scan in unique}
    ordered = sorted(unique, key=lambda scan: medians[scan])
    clusters: list[list[int]] = []
    current: list[int] = []
    previous_time = None
    for scan in ordered:
        predecessor = None if predecessor_field is None else predecessor_field.get(int(scan))
        new_visit = False
        if current:
            if predecessor is not None and int(predecessor) == int(holoraster_field):
                new_visit = True
            elif previous_time is not None and (medians[scan] - previous_time) > float(gap_s):
                new_visit = True
        if new_visit:
            clusters.append(current)
            current = [int(scan)]
        else:
            current.append(int(scan))
        previous_time = medians[scan]
    if current:
        clusters.append(current)
    return clusters


def held_out_scan_cluster_masks(
    scan_id: ArrayLike,
    row_mask: ArrayLike,
    clusters: list[list[int]],
    *,
    holdout_cluster_index: int | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], dict[str, object]]:
    """Hold out one complete visit cluster. Default: last cluster if ``n>=2``."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    scans = np.asarray(scan_id, dtype=np.int32).reshape(-1)
    if len(clusters) < 2:
        empty = np.zeros(mask.size, dtype=bool)
        return (
            mask.copy(),
            empty,
            {
                "n_clusters": len(clusters),
                "held_out_scans": [],
                "held_out_cluster_index": None,
                "status": "not_run",
                "notes": (COMPLETE_FIELD9_TRACK_NOTE,),
            },
        )
    index = len(clusters) - 1 if holdout_cluster_index is None else int(holdout_cluster_index)
    if index < 0 or index >= len(clusters):
        raise ValueError("holdout_cluster_index is outside the cluster list")
    held = [int(scan) for scan in clusters[index]]
    hold = mask & np.isin(scans, np.asarray(held, dtype=np.int32))
    train = mask & ~hold
    return (
        train,
        hold,
        {
            "n_clusters": len(clusters),
            "held_out_scans": held,
            "held_out_cluster_index": index,
            "n_train": int(np.sum(train)),
            "n_hold": int(np.sum(hold)),
            "status": "ok" if np.any(hold) and np.any(train) else "not_run",
            "notes": (COMPLETE_FIELD9_TRACK_NOTE,),
        },
    )


def sample_holdout_mask(
    row_mask: ArrayLike,
    *,
    holdout_fraction: float = 0.1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.bool_]:
    """Hold out a random subset of rows. True marks training rows."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    selected = np.flatnonzero(mask)
    train = mask.copy()
    if selected.size == 0:
        return train
    generator = np.random.default_rng(0) if rng is None else rng
    n_hold = int(np.floor(float(holdout_fraction) * selected.size))
    if n_hold <= 0:
        return train
    train[generator.choice(selected, size=n_hold, replace=False)] = False
    return train


def clustered_complex_scores(
    residuals: ArrayLike,
    cluster_ids: ArrayLike,
) -> dict[str, object]:
    """Score residuals as cluster means, not as independent samples."""

    z = np.asarray(residuals, dtype=np.complex128).reshape(-1)
    labels = np.asarray(cluster_ids).reshape(-1)
    if z.size != labels.size:
        raise ValueError("residuals and cluster_ids must have the same size")
    means = []
    for name in np.unique(labels):
        selected = z[labels == name]
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            continue
        means.append(complex(np.mean(selected)))
    if not means:
        return {"n_clusters": 0, "n": 0}
    packed = np.asarray(means, dtype=np.complex128)
    return {
        "n_clusters": int(packed.size),
        "n": int(z.size),
        "median_abs_cluster_mean": float(np.median(np.abs(packed))),
        "mean_abs_cluster_mean": float(np.mean(np.abs(packed))),
        "coherent_mean_abs": float(np.abs(np.mean(packed))),
    }


def classify_residual_jones_visibility(
    *,
    rl_improved_clusters: int,
    n_mover_clusters: int,
    n_reference_clusters: int,
    rr_ll_regression: bool,
    max_group_coherent_abs: float | None,
    qu_interval_spread: float | None,
    gauge_invariance_abs: float | None,
    leakage_scale: float = 0.02,
) -> dict[str, object]:
    """Held-out visibility gates for the residual-Jones estimator."""

    enough = n_mover_clusters >= 2 and n_reference_clusters >= 2
    improved = enough and rl_improved_clusters >= 2
    tail = (
        max_group_coherent_abs is not None
        and np.isfinite(max_group_coherent_abs)
        and float(max_group_coherent_abs) >= leakage_scale
    )
    qu_ok = qu_interval_spread is None or (
        np.isfinite(qu_interval_spread) and float(qu_interval_spread) < leakage_scale
    )
    gauge_ok = gauge_invariance_abs is None or (
        np.isfinite(gauge_invariance_abs) and float(gauge_invariance_abs) < 1.0e-6
    )
    passed = improved and not rr_ll_regression and not tail and qu_ok and gauge_ok
    return {
        "status": "pass" if passed else "fail" if enough else "not_run",
        "blocking": not passed if enough else False,
        "rl_improved_clusters": int(rl_improved_clusters),
        "n_mover_clusters": int(n_mover_clusters),
        "n_reference_clusters": int(n_reference_clusters),
        "rr_ll_regression": bool(rr_ll_regression),
        "max_group_coherent_abs": max_group_coherent_abs,
        "qu_interval_spread": qu_interval_spread,
        "gauge_invariance_abs": gauge_invariance_abs,
        "notes": (RESIDUAL_JONES_VISIBILITY_GATES_NOTE,),
    }


def _gram_condition_report(
    gram: NDArray[np.float64],
    free: list[int],
    index: Mapping[int, int],
) -> dict[str, object]:
    """Conditioning and off-diagonal variance from the last normal matrix."""

    if gram.size == 0 or gram.shape[0] == 0:
        return {"status": "not_run", "condition": float("nan")}
    evals = np.linalg.eigvalsh(np.asarray(gram, dtype=np.float64))
    evals = evals[np.isfinite(evals)]
    smallest = float(np.min(np.abs(evals))) if evals.size else float("nan")
    largest = float(np.max(np.abs(evals))) if evals.size else float("nan")
    condition = largest / smallest if smallest > 0 else float("inf")
    cov = np.linalg.pinv(gram, rcond=1.0e-12)
    offdiag = {}
    for ant in free:
        base = 8 * index[int(ant)]
        offdiag[int(ant)] = {
            "rl_variance": float(cov[base + 2, base + 2] + cov[base + 3, base + 3]),
            "lr_variance": float(cov[base + 4, base + 4] + cov[base + 5, base + 5]),
        }
    variances = [item["rl_variance"] + item["lr_variance"] for item in offdiag.values()]
    return {
        "status": "pass",
        "n_parameter": int(gram.shape[0]),
        "condition": float(condition),
        "min_abs_eigenvalue": smallest,
        "max_abs_eigenvalue": largest,
        "median_offdiag_variance": float(np.median(variances)) if variances else float("nan"),
        "max_offdiag_variance": float(np.max(variances)) if variances else float("nan"),
        "offdiag_variance": offdiag,
    }


def estimate_all_antenna_residual_jones(
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    coherency: ArrayLike,
    *,
    stokes_i: ArrayLike,
    chi1: ArrayLike,
    chi2: ArrayLike,
    antenna_ids: ArrayLike | None = None,
    gauge_antenna_id: int | None = None,
    row_mask: ArrayLike | None = None,
    q_over_i: float = 0.0,
    u_over_i: float = 0.0,
    fit_qu: bool = True,
    ridge: float = 1.0,
    qu_ridge: float = 1.0e-4,
    n_iter: int = 3,
    time_s: ArrayLike | None = None,
    offdiag_drift: bool = False,
    drift_ridge: float = 1.0e3,
    time_scale_s: float | None = None,
) -> dict[str, object]:
    """Joint :math:`R=P^H(I+\\varepsilon)P` and sky-fixed Q/U on field 9.

    Every antenna is solved. Q/U are free. Optional off-diagonal drift is
    :math:`\\varepsilon(t)=\\varepsilon_0+\\delta\\varepsilon\\,\\tau` with
    strong ridge on :math:`\\delta\\varepsilon`. One antenna is identity.
    """

    p_ant = np.asarray(antenna1, dtype=np.int32).reshape(-1)
    q_ant = np.asarray(antenna2, dtype=np.int32).reshape(-1)
    vis = np.asarray(coherency, dtype=np.complex128)
    intensity = np.asarray(stokes_i, dtype=np.float64).reshape(-1)
    angle_p = np.asarray(chi1, dtype=np.float64).reshape(-1)
    angle_q = np.asarray(chi2, dtype=np.float64).reshape(-1)
    if vis.ndim != 3 or vis.shape[0] != p_ant.size or vis.shape[1:] != (2, 2):
        raise ValueError("coherency must have shape (row, 2, 2)")
    if intensity.size != p_ant.size or angle_p.size != p_ant.size or angle_q.size != p_ant.size:
        raise ValueError("stokes_i, chi1 and chi2 must be per row")
    ids = (
        np.unique(np.concatenate([p_ant, q_ant]))
        if antenna_ids is None
        else np.asarray(antenna_ids, dtype=np.int32)
    )
    train = (p_ant != q_ant) & np.isfinite(intensity) & (np.abs(intensity) > 1.0e-3)
    if row_mask is not None:
        train = train & np.asarray(row_mask, dtype=bool).reshape(-1)
    if not antenna_graph_is_connected(p_ant, q_ant, antenna_ids=ids, row_mask=train):
        raise ValueError("training baselines do not keep the antenna graph connected")
    gauge = int(ids[0] if gauge_antenna_id is None else gauge_antenna_id)
    free = [int(ant) for ant in ids if int(ant) != gauge]
    index = {ant: slot for slot, ant in enumerate(free)}
    n_eps = 8 * len(free)
    n_drift = 4 * len(free) if offdiag_drift else 0
    n_par = n_eps + n_drift + (2 if fit_qu else 0)
    qu_base = n_eps + n_drift
    q_frac = float(q_over_i)
    u_frac = float(u_over_i)
    eps = {int(ant): np.zeros((2, 2), dtype=np.complex128) for ant in ids}
    drift = {int(ant): np.zeros((2,), dtype=np.complex128) for ant in ids}
    times = None if time_s is None else np.asarray(time_s, dtype=np.float64).reshape(-1)
    if offdiag_drift:
        if times is None or times.size != p_ant.size:
            raise ValueError("off-diagonal drift needs per-row time_s")
        span = float(np.ptp(times[train])) if np.any(train) else 1.0
        scale = float(time_scale_s) if time_scale_s else max(span, 1.0)
        tau = (times - float(np.median(times[train]))) / scale
    else:
        tau = np.zeros(p_ant.size, dtype=np.float64)
        scale = float("nan")

    def _eps_slot(antenna: int, row: int, col: int) -> int | None:
        if antenna == gauge or antenna not in index:
            return None
        return 8 * index[antenna] + 2 * (2 * row + col)

    train_rows = np.flatnonzero(train)
    para_p = circular_parallactic_jones(angle_p)
    para_q = circular_parallactic_jones(angle_q)
    # Locked R = P^H (I+ε) P ⇒ Ṽ = P V P^H = (I+ε)(P S P^H)(I+ε)^H
    feed_left = para_p
    feed_right = invert_jones(para_q)

    for _ in range(max(1, int(n_iter))):
        gram = float(ridge) * np.eye(max(n_par, 1), dtype=np.float64)
        if n_drift:
            gram[n_eps:qu_base, n_eps:qu_base] = float(drift_ridge) * np.eye(n_drift)
        if fit_qu:
            gram[qu_base:, qu_base:] = float(qu_ridge) * np.eye(2)
        proj = np.zeros(max(n_par, 1), dtype=np.float64)
        n_eq = 0
        for row in train_rows:
            source = circular_stokes_to_coherency(
                intensity[row],
                intensity[row] * q_frac,
                intensity[row] * u_frac,
                0.0,
            )
            feed_vis = feed_left[row] @ vis[row] @ feed_right[row]
            feed_s = feed_left[row] @ source @ feed_right[row]
            leftover = feed_vis - feed_s
            p_id = int(p_ant[row])
            q_id = int(q_ant[row])
            s = feed_s
            for a in range(2):
                for b in range(2):
                    real = np.zeros(n_par, dtype=np.float64)
                    imag = np.zeros(n_par, dtype=np.float64)

                    def _accum(
                        target_real: NDArray[np.float64],
                        target_imag: NDArray[np.float64],
                        slot: int,
                        multiplier: complex,
                        conjugated: bool,
                    ) -> None:
                        mr = float(np.real(multiplier))
                        mi = float(np.imag(multiplier))
                        if conjugated:
                            target_real[slot] += mr
                            target_imag[slot] += mi
                            target_real[slot + 1] += mi
                            target_imag[slot + 1] -= mr
                        else:
                            target_real[slot] += mr
                            target_imag[slot] += mi
                            target_real[slot + 1] -= mi
                            target_imag[slot + 1] += mr

                    for k in range(2):
                        slot_p = _eps_slot(p_id, a, k)
                        if slot_p is not None:
                            _accum(real, imag, slot_p, s[k, b], False)
                        slot_q = _eps_slot(q_id, b, k)
                        if slot_q is not None:
                            _accum(real, imag, slot_q, s[a, k], True)
                    if offdiag_drift:
                        # τ multiplies only the off-diagonal feed residuals
                        for hand, (row_i, col_i) in enumerate(((0, 1), (1, 0))):
                            slot_p = _eps_slot(p_id, row_i, col_i)
                            slot_a = None if slot_p is None else n_eps + 4 * index[p_id] + 2 * hand
                            if slot_a is not None and a == row_i:
                                _accum(
                                    real,
                                    imag,
                                    slot_a,
                                    tau[row] * s[col_i, b],
                                    False,
                                )
                            slot_q = _eps_slot(q_id, row_i, col_i)
                            slot_b = None if slot_q is None else n_eps + 4 * index[q_id] + 2 * hand
                            if slot_b is not None and b == row_i:
                                _accum(
                                    real,
                                    imag,
                                    slot_b,
                                    tau[row] * s[a, col_i],
                                    True,
                                )
                    if fit_qu:
                        d_q = (
                            feed_left[row]
                            @ np.array(
                                [[0.0, intensity[row]], [intensity[row], 0.0]],
                                dtype=np.complex128,
                            )
                            @ feed_right[row]
                        )
                        d_u = (
                            feed_left[row]
                            @ np.array(
                                [[0.0, 1j * intensity[row]], [-1j * intensity[row], 0.0]],
                                dtype=np.complex128,
                            )
                            @ feed_right[row]
                        )
                        real[qu_base] += float(np.real(d_q[a, b]))
                        imag[qu_base] += float(np.imag(d_q[a, b]))
                        real[qu_base + 1] += float(np.real(d_u[a, b]))
                        imag[qu_base + 1] += float(np.imag(d_u[a, b]))
                    gram[:n_par, :n_par] += np.outer(real, real) + np.outer(imag, imag)
                    proj[:n_par] += real * float(np.real(leftover[a, b])) + imag * float(
                        np.imag(leftover[a, b])
                    )
                    n_eq += 2
        if n_eq == 0:
            raise ValueError("no on-axis rows for residual Jones")
        solution = np.linalg.solve(gram[:n_par, :n_par], proj[:n_par])
        for ant in ids:
            plane = np.zeros((2, 2), dtype=np.complex128)
            if int(ant) in index:
                base = 8 * index[int(ant)]
                packed = solution[base : base + 8]
                plane[0, 0] = packed[0] + 1j * packed[1]
                plane[0, 1] = packed[2] + 1j * packed[3]
                plane[1, 0] = packed[4] + 1j * packed[5]
                plane[1, 1] = packed[6] + 1j * packed[7]
            eps[int(ant)] = plane
        if offdiag_drift:
            for ant in ids:
                if int(ant) not in index:
                    continue
                base = n_eps + 4 * index[int(ant)]
                packed = solution[base : base + 4]
                drift[int(ant)] = np.array(
                    [packed[0] + 1j * packed[1], packed[2] + 1j * packed[3]],
                    dtype=np.complex128,
                )
        if fit_qu:
            q_frac += float(solution[qu_base])
            u_frac += float(solution[qu_base + 1])

    jones = {ant: np.eye(2, dtype=np.complex128) + eps[ant] for ant in eps}
    deviations = [float(np.linalg.norm(eps[ant])) for ant in jones]
    return {
        "jones": jones,
        "drift_offdiag": drift if offdiag_drift else None,
        "q_over_i": q_frac,
        "u_over_i": u_frac,
        "frac_pol": float(np.hypot(q_frac, u_frac)),
        "gauge_antenna_id": gauge,
        "ridge": float(ridge),
        "drift_ridge": float(drift_ridge) if offdiag_drift else None,
        "time_scale_s": scale,
        "time_origin_s": float(np.median(times[train])) if offdiag_drift else None,
        "offdiag_drift": bool(offdiag_drift),
        "residual_jones_sky_frame": RESIDUAL_JONES_SKY_FRAME,
        "n_train": int(np.sum(train)),
        "n_equations": n_eq,
        "n_iter": int(n_iter),
        "median_abs_epsilon": float(np.median(deviations)),
        "max_abs_epsilon": float(np.max(deviations)),
        "graph_condition": _gram_condition_report(gram[:n_par, :n_par], free, index),
        "graph_connected": True,
        "notes": (
            ALL_ANTENNA_RESIDUAL_JONES_NOTE,
            THREE_C147_QU_IS_NUISANCE_NOTE,
            CONNECTED_HOLDOUT_NOTE,
            RESIDUAL_JONES_P_CONVENTION_NOTE,
            CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
        ),
    }


def feed_jones_at_time(
    residual_jones: Mapping[int, ArrayLike],
    antenna_id: int,
    *,
    time_s: float | None = None,
    drift_offdiag: Mapping[int, ArrayLike] | None = None,
    time_origin_s: float | None = None,
    time_scale_s: float | None = None,
) -> NDArray[np.complex128]:
    """Return :math:`I+\\varepsilon_0+\\delta\\varepsilon\\,\\tau` in the feed frame."""

    plane = np.array(reference_jones_for_antenna(residual_jones, antenna_id), copy=True)
    if drift_offdiag is None or time_s is None or time_origin_s is None or time_scale_s is None:
        return plane
    if int(antenna_id) not in drift_offdiag:
        return plane
    tau = (float(time_s) - float(time_origin_s)) / float(time_scale_s)
    delta = np.asarray(drift_offdiag[int(antenna_id)], dtype=np.complex128).reshape(-1)
    plane[0, 1] = plane[0, 1] + delta[0] * tau
    plane[1, 0] = plane[1, 0] + delta[1] * tau
    return plane


def predict_residual_visibilities(
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    *,
    stokes_i: ArrayLike,
    chi1: ArrayLike,
    chi2: ArrayLike,
    residual_jones: Mapping[int, ArrayLike],
    q_over_i: float,
    u_over_i: float,
    time_s: ArrayLike | None = None,
    drift_offdiag: Mapping[int, ArrayLike] | None = None,
    time_origin_s: float | None = None,
    time_scale_s: float | None = None,
) -> NDArray[np.complex128]:
    """Predict on-axis :math:`R_p S(Q,U) R_q^H` in the sky frame."""

    p_ant = np.asarray(antenna1, dtype=np.int32).reshape(-1)
    q_ant = np.asarray(antenna2, dtype=np.int32).reshape(-1)
    intensity = np.asarray(stokes_i, dtype=np.float64).reshape(-1)
    times = None if time_s is None else np.asarray(time_s, dtype=np.float64).reshape(-1)
    predicted = np.empty((p_ant.size, 2, 2), dtype=np.complex128)
    for row in range(p_ant.size):
        source = circular_stokes_to_coherency(
            intensity[row],
            intensity[row] * float(q_over_i),
            intensity[row] * float(u_over_i),
            0.0,
        )
        instant = None if times is None else float(times[row])
        left = sky_frame_residual_jones(
            feed_jones_at_time(
                residual_jones,
                int(p_ant[row]),
                time_s=instant,
                drift_offdiag=drift_offdiag,
                time_origin_s=time_origin_s,
                time_scale_s=time_scale_s,
            ),
            float(chi1.reshape(-1)[row]),
        )
        right = sky_frame_residual_jones(
            feed_jones_at_time(
                residual_jones,
                int(q_ant[row]),
                time_s=instant,
                drift_offdiag=drift_offdiag,
                time_origin_s=time_origin_s,
                time_scale_s=time_scale_s,
            ),
            float(chi2.reshape(-1)[row]),
        )
        predicted[row] = predict_baseline_coherency(left, source, right)
    return predicted


def residual_holdout_report(
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    coherency: ArrayLike,
    *,
    stokes_i: ArrayLike,
    chi1: ArrayLike,
    chi2: ArrayLike,
    residual_jones: Mapping[int, ArrayLike],
    q_over_i: float,
    u_over_i: float,
    row_mask: ArrayLike,
    cluster_ids: ArrayLike | None = None,
    time_s: ArrayLike | None = None,
    drift_offdiag: Mapping[int, ArrayLike] | None = None,
    time_origin_s: float | None = None,
    time_scale_s: float | None = None,
    group_ids: ArrayLike | None = None,
) -> dict[str, object]:
    """Held-out sky-frame residuals for a connected time or baseline split."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    vis = np.asarray(coherency, dtype=np.complex128)[mask]
    if vis.size == 0:
        return {"n": 0, "status": "not_run"}
    pred = predict_residual_visibilities(
        np.asarray(antenna1)[mask],
        np.asarray(antenna2)[mask],
        stokes_i=np.asarray(stokes_i)[mask],
        chi1=np.asarray(chi1)[mask],
        chi2=np.asarray(chi2)[mask],
        residual_jones=residual_jones,
        q_over_i=q_over_i,
        u_over_i=u_over_i,
        time_s=None if time_s is None else np.asarray(time_s)[mask],
        drift_offdiag=drift_offdiag,
        time_origin_s=time_origin_s,
        time_scale_s=time_scale_s,
    )
    resid = vis - pred
    intensity = np.asarray(stokes_i, dtype=np.float64).reshape(-1)[mask]
    scale = np.maximum(np.abs(intensity), 1.0e-3)
    rl = resid[:, 0, 1] / scale
    lr = resid[:, 1, 0] / scale
    rr = resid[:, 0, 0] / scale
    ll = resid[:, 1, 1] / scale
    report = {
        "n": int(np.sum(mask)),
        "graph_connected": antenna_graph_is_connected(antenna1, antenna2, row_mask=mask),
        "median_abs_rl_over_i": float(np.median(np.abs(rl[np.isfinite(rl)]))),
        "median_abs_lr_over_i": float(np.median(np.abs(lr[np.isfinite(lr)]))),
        "median_abs_rr_over_i": float(np.median(np.abs(rr[np.isfinite(rr)]))),
        "median_abs_ll_over_i": float(np.median(np.abs(ll[np.isfinite(ll)]))),
        "coherent_rl": coherent_residual_report(
            rl,
            group_ids=None if group_ids is None else np.asarray(group_ids).reshape(-1)[mask],
        ),
    }
    if cluster_ids is not None:
        labels = np.asarray(cluster_ids).reshape(-1)[mask]
        report["clustered_rl"] = clustered_complex_scores(rl, labels)
        report["clustered_lr"] = clustered_complex_scores(lr, labels)
    return report


def calibration_adequacy_report(
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    coherency: ArrayLike,
    *,
    stokes_i: ArrayLike,
    chi1: ArrayLike,
    chi2: ArrayLike,
    residual_jones: Mapping[int, ArrayLike],
    q_over_i: float,
    u_over_i: float,
    row_mask: ArrayLike,
    antenna_names: ArrayLike | None = None,
) -> dict[str, object]:
    """Complex coherent leftover and N^{-1/2} behaviour, by antenna and baseline."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    p_ant = np.asarray(antenna1, dtype=np.int32).reshape(-1)
    q_ant = np.asarray(antenna2, dtype=np.int32).reshape(-1)
    names = None if antenna_names is None else np.asarray(antenna_names)
    baseline = np.array(
        [f"{int(min(p, q))}-{int(max(p, q))}" for p, q in zip(p_ant, q_ant, strict=True)],
        dtype=object,
    )
    hold = residual_holdout_report(
        antenna1,
        antenna2,
        coherency,
        stokes_i=stokes_i,
        chi1=chi1,
        chi2=chi2,
        residual_jones=residual_jones,
        q_over_i=q_over_i,
        u_over_i=u_over_i,
        row_mask=mask,
        group_ids=baseline,
    )
    pred = predict_residual_visibilities(
        p_ant[mask],
        q_ant[mask],
        stokes_i=np.asarray(stokes_i)[mask],
        chi1=np.asarray(chi1)[mask],
        chi2=np.asarray(chi2)[mask],
        residual_jones=residual_jones,
        q_over_i=q_over_i,
        u_over_i=u_over_i,
    )
    vis = np.asarray(coherency, dtype=np.complex128)[mask]
    intensity = np.asarray(stokes_i, dtype=np.float64).reshape(-1)[mask]
    scale = np.maximum(np.abs(intensity), 1.0e-3)
    rl = (vis[:, 0, 1] - pred[:, 0, 1]) / scale
    by_antenna = {}
    for ant in np.unique(np.concatenate([p_ant[mask], q_ant[mask]])):
        selected = (p_ant[mask] == int(ant)) | (q_ant[mask] == int(ant))
        label = (
            str(names[int(ant)]) if names is not None and int(ant) < len(names) else str(int(ant))
        )
        by_antenna[label] = coherent_residual_report(rl[selected])
    coherent = hold.get("coherent_rl") or {}
    floor = classify_crosshand_floor(
        median_abs_rl_over_i=float(hold.get("median_abs_rl_over_i") or np.nan),
        coherent_mean_abs=coherent.get("coherent_mean_abs"),
        averages_as_noise=coherent.get("averages_as_noise"),
        max_group_coherent_abs=max(
            (
                float(item.get("mean_abs") or 0.0)
                for item in by_antenna.values()
                if item.get("n", 0)
            ),
            default=None,
        ),
    )
    return {
        "n": hold.get("n"),
        "median_abs_rl_over_i": hold.get("median_abs_rl_over_i"),
        "coherent_rl": coherent,
        "by_antenna": by_antenna,
        "by_baseline": coherent.get("by_group") or {},
        "averages_as_noise": coherent.get("averages_as_noise"),
        "calibration_adequate": (not bool(floor["blocking"]))
        and bool(coherent.get("averages_as_noise")),
        "floor": floor,
        "notes": (CALIBRATION_ADEQUACY_NOT_OPERATOR_NOTE, CROSSHAND_FLOOR_BLOCKS_FULL_JONES_NOTE),
    }


def gauge_invariance_abs(
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    *,
    stokes_i: ArrayLike,
    chi1: ArrayLike,
    chi2: ArrayLike,
    residual_jones: Mapping[int, ArrayLike],
    q_over_i: float,
    u_over_i: float,
    phase_rad: float = 0.3,
) -> float:
    """Max |ΔV| after a global phase, which must leave visibilities unchanged."""

    pred = predict_residual_visibilities(
        antenna1,
        antenna2,
        stokes_i=stokes_i,
        chi1=chi1,
        chi2=chi2,
        residual_jones=residual_jones,
        q_over_i=q_over_i,
        u_over_i=u_over_i,
    )
    rotated = {
        int(ant): np.exp(1j * float(phase_rad)) * np.asarray(plane, dtype=np.complex128)
        for ant, plane in residual_jones.items()
    }
    pred_rot = predict_residual_visibilities(
        antenna1,
        antenna2,
        stokes_i=stokes_i,
        chi1=chi1,
        chi2=chi2,
        residual_jones=rotated,
        q_over_i=q_over_i,
        u_over_i=u_over_i,
    )
    return float(np.max(np.abs(pred_rot - pred)))


def blocked_time_folds(
    time_s: ArrayLike,
    row_mask: ArrayLike,
    *,
    n_folds: int = 3,
) -> list[NDArray[np.bool_]]:
    """Contiguous time folds inside the training mask. Later holdout is excluded."""

    times = np.asarray(time_s, dtype=np.float64).reshape(-1)
    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return []
    order = selected[np.argsort(times[selected])]
    edges = np.linspace(0, order.size, int(n_folds) + 1, dtype=int)
    folds = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        hold = np.zeros(mask.size, dtype=bool)
        hold[order[start:stop]] = True
        folds.append(hold)
    return folds


def classify_field9_time_transfer(
    *,
    later_coherent_abs: float,
    later_averages_as_noise: bool,
    baseline_averages_as_noise: bool,
    qu_fold_spread: float | None,
    rr_ll_regression: bool,
    antenna_later_coherent: Mapping[str, float] | None = None,
    antenna_static_coherent: Mapping[str, float] | None = None,
    scale_coherent: Mapping[str, float] | None = None,
    previous_coherent_abs: float = 0.0099,
) -> dict[str, object]:
    """Later-time transfer gates. Hyperparameters must not be fit on this split."""

    material = float(later_coherent_abs) < 0.75 * float(previous_coherent_abs)
    qu_ok = qu_fold_spread is None or (
        np.isfinite(qu_fold_spread) and float(qu_fold_spread) < 0.003
    )
    later_ant = dict(antenna_later_coherent or {})
    static_ant = dict(antenna_static_coherent or {})
    antennas_ok = True
    if later_ant and static_ant:
        antennas_ok = all(
            name in static_ant
            and np.isfinite(later_ant[name])
            and float(later_ant[name]) <= float(static_ant[name]) + 1.0e-4
            for name in later_ant
        )
    scales = [float(value) for value in (scale_coherent or {}).values() if np.isfinite(value)]
    scale_ok = (not scales) or (max(scales) - min(scales) < 0.003)
    passed = (
        material
        and bool(later_averages_as_noise)
        and bool(baseline_averages_as_noise)
        and qu_ok
        and not rr_ll_regression
        and antennas_ok
        and scale_ok
    )
    return {
        "status": "pass" if passed else "fail",
        "blocking": not passed,
        "later_coherent_abs": float(later_coherent_abs),
        "previous_coherent_abs": float(previous_coherent_abs),
        "materially_below_previous": material,
        "later_averages_as_noise": bool(later_averages_as_noise),
        "baseline_averages_as_noise": bool(baseline_averages_as_noise),
        "qu_fold_spread": qu_fold_spread,
        "rr_ll_regression": bool(rr_ll_regression),
        "antenna_later_coherent": later_ant,
        "antenna_static_coherent": static_ant,
        "antennas_improved": antennas_ok,
        "scale_ok": scale_ok,
        "notes": (
            RESIDUAL_JONES_VISIBILITY_GATES_NOTE,
            CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
        ),
    }


def scan_state_jump_report(
    per_scan: Mapping[str, Mapping[int, ArrayLike]],
    *,
    jump_threshold: float = 0.015,
) -> dict[str, object]:
    """Detect scan-to-scan jumps in fitted feed-frame ε."""

    scans = sorted(per_scan, key=lambda name: int(name) if str(name).isdigit() else str(name))
    jumps = []
    for prev, cur in zip(scans[:-1], scans[1:], strict=True):
        ants = set(per_scan[prev]) & set(per_scan[cur])
        for ant in ants:
            delta = np.linalg.norm(np.asarray(per_scan[cur][ant]) - np.asarray(per_scan[prev][ant]))
            if float(delta) >= float(jump_threshold):
                jumps.append(
                    {
                        "from_scan": prev,
                        "to_scan": cur,
                        "antenna": int(ant),
                        "abs_delta": float(delta),
                    }
                )
    jumps.sort(key=lambda item: item["abs_delta"], reverse=True)
    return {
        "n_scan": len(scans),
        "n_jump": len(jumps),
        "jumps": jumps[:24],
        "piecewise_state_term_indicated": len(jumps) >= 3,
        "notes": (SCAN_STATE_MAY_NEED_OFFSETS_NOTE,),
    }


def select_drift_ridge_by_inner_folds(
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    coherency: ArrayLike,
    *,
    stokes_i: ArrayLike,
    chi1: ArrayLike,
    chi2: ArrayLike,
    time_s: ArrayLike,
    train_mask: ArrayLike,
    gauge_antenna_id: int,
    ridges: tuple[float, ...] = (1.0e2, 1.0e3, 1.0e4),
    n_folds: int = 3,
    ridge: float = 1.0,
    qu_ridge: float = 1.0e-4,
    n_iter: int = 2,
) -> dict[str, object]:
    """Choose :math:`\\delta\\varepsilon` ridge on inner training folds only."""

    mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    folds = blocked_time_folds(time_s, mask, n_folds=n_folds)
    by_ridge: dict[str, dict[str, object]] = {}
    for drift_ridge in ridges:
        fold_abs: list[float] = []
        fold_q: list[float] = []
        fold_u: list[float] = []
        for hold in folds:
            inner = mask & ~hold
            if not np.any(inner) or not antenna_graph_is_connected(
                antenna1, antenna2, row_mask=inner
            ):
                continue
            fit = estimate_all_antenna_residual_jones(
                antenna1,
                antenna2,
                coherency,
                stokes_i=stokes_i,
                chi1=chi1,
                chi2=chi2,
                gauge_antenna_id=int(gauge_antenna_id),
                row_mask=inner,
                ridge=ridge,
                qu_ridge=qu_ridge,
                n_iter=n_iter,
                time_s=time_s,
                offdiag_drift=True,
                drift_ridge=float(drift_ridge),
            )
            report = residual_holdout_report(
                antenna1,
                antenna2,
                coherency,
                stokes_i=stokes_i,
                chi1=chi1,
                chi2=chi2,
                residual_jones=fit["jones"],
                q_over_i=float(fit["q_over_i"]),
                u_over_i=float(fit["u_over_i"]),
                row_mask=hold,
                time_s=time_s,
                drift_offdiag=fit["drift_offdiag"],
                time_origin_s=fit["time_origin_s"],
                time_scale_s=fit["time_scale_s"],
            )
            coherent = report.get("coherent_rl") or {}
            fold_abs.append(float(coherent.get("coherent_mean_abs") or np.nan))
            fold_q.append(float(fit["q_over_i"]))
            fold_u.append(float(fit["u_over_i"]))
        if not fold_abs:
            continue
        by_ridge[f"{float(drift_ridge):g}"] = {
            "drift_ridge": float(drift_ridge),
            "median_coherent_abs": float(np.median(fold_abs)),
            "fold_coherent_abs": fold_abs,
            "q_over_i": fold_q,
            "u_over_i": fold_u,
        }
    if not by_ridge:
        raise ValueError("no inner folds for drift-ridge selection")
    chosen_key = min(by_ridge, key=lambda key: float(by_ridge[key]["median_coherent_abs"]))
    qs = [float(value) for item in by_ridge.values() for value in item["q_over_i"]]
    us = [float(value) for item in by_ridge.values() for value in item["u_over_i"]]
    return {
        "chosen_drift_ridge": float(by_ridge[chosen_key]["drift_ridge"]),
        "by_ridge": by_ridge,
        "qu_fold_spread": float(np.hypot(np.ptp(qs), np.ptp(us))) if qs else float("nan"),
        "n_folds": int(n_folds),
        "notes": (
            "Timescale / ridge chosen on inner blocked-time folds of the "
            "training half only. The later-time holdout is not used.",
            CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
        ),
    }


def classify_chain_jump_stage(
    stage_jumps: Mapping[str, float],
    *,
    threshold: float = 0.015,
) -> dict[str, object]:
    """First cumulative apply stage where the scan-to-scan jump appears."""

    previous = 0.0
    first_absolute = None
    first_increment = None
    increments = {}
    for stage in CALIBRATION_CHAIN_STAGES:
        if stage not in stage_jumps:
            continue
        value = float(stage_jumps[stage])
        increment = value - previous
        increments[stage] = increment
        if first_absolute is None and np.isfinite(value) and value >= float(threshold):
            first_absolute = stage
        if first_increment is None and np.isfinite(increment) and increment >= float(threshold):
            first_increment = stage
        previous = value
    present_in_data = first_absolute == "DATA"
    finite = [
        float(stage_jumps[stage])
        for stage in CALIBRATION_CHAIN_STAGES
        if stage in stage_jumps and np.isfinite(stage_jumps[stage])
    ]
    visibility_jump_absent = bool(finite) and first_absolute is None
    return {
        "status": "pass" if first_absolute is not None or visibility_jump_absent else "not_run",
        "first_stage_above_threshold": first_absolute,
        "first_increment_above_threshold": first_increment,
        "present_in_data": present_in_data,
        "visibility_jump_absent": visibility_jump_absent,
        "investigate_acquisition_state": present_in_data,
        "increments": increments,
        "threshold": float(threshold),
        "notes": (
            SMOOTH_GP_REJECTED_NOTE,
            "If the jump first appears after one calibration term, fix or "
            "model that term rather than calling it antenna drift.",
        ),
    }


def classify_common_mode_vs_antenna(
    per_antenna_delta: Mapping[str, complex | float],
) -> dict[str, object]:
    """Compare a shared scan jump to antenna-specific scatter."""

    values = np.asarray(
        [complex(item) for item in per_antenna_delta.values()],
        dtype=np.complex128,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"status": "not_run", "n": 0}
    common = complex(np.mean(values))
    scatter = float(np.std(np.abs(values)))
    common_abs = float(np.abs(common))
    common_mode = common_abs >= 2.0 * max(scatter, 1.0e-6)
    return {
        "status": "pass",
        "n": int(values.size),
        "common_mean": [common.real, common.imag],
        "common_abs": common_abs,
        "antenna_abs_scatter": scatter,
        "common_mode": common_mode,
        "antenna_specific": (not common_mode) and scatter >= 0.01,
    }


def hierarchical_residual_jones_model() -> dict[str, object]:
    """Document the hierarchical scan-state model. This is not a fit."""

    return {
        "schema": "thol0001_hierarchical_scan_state_residual_v0",
        "status": "specified_not_fit",
        "epsilon_p_k": "epsilon_p_0 + c_k + delta_epsilon_p_k",
        "c_k": "low-dimensional shared scan or acquisition-state term",
        "delta_epsilon_p_k": "strongly shrunk antenna residual",
        "q_u": "one global 3C147 nuisance distribution across all scans",
        "unconstrained_per_scan_jones": False,
        "validation": (
            "held-out baselines and times within each field-9 scan",
            "whole held-out field-9 scans from the bracketing/nearest-scan rule",
            "repeatability of independently recovered beam cells after only "
            "the associated on-axis calibration scans",
        ),
        "notes": (
            DO_NOT_FIT_UNCONSTRAINED_PER_SCAN_JONES_NOTE,
            HIERARCHICAL_SCAN_STATE_MODEL_NOTE,
            INTERLEAVED_ONAXIS_TRANSFER_NOTE,
        ),
    }


def classify_interleaved_onaxis_transfer(
    *,
    chain_stage: str | None,
    visit_repeatability_passed: bool | None = None,
    used_holography_crosshands_for_onaxis: bool = False,
    smooth_gp_rejected: bool = True,
) -> dict[str, object]:
    """Key remaining gate: interleaved on-axis calibration transfers."""

    if used_holography_crosshands_for_onaxis:
        status = "fail"
    elif visit_repeatability_passed is True:
        status = "pass"
    else:
        status = "not_run"
    return {
        "status": status,
        "blocking": status != "pass",
        "chain_jump_stage": chain_stage,
        "smooth_gp_rejected": bool(smooth_gp_rejected),
        "smooth_gp_permanently_blocks_full_jones": False,
        "used_holography_crosshands_for_onaxis": bool(used_holography_crosshands_for_onaxis),
        "notes": (
            INTERLEAVED_BEAM_TRANSFER_GATE_NOTE,
            SMOOTH_GP_REJECTED_NOTE,
            INTERLEAVED_ONAXIS_TRANSFER_NOTE,
            CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
        ),
    }


def matched_scan_row_masks(
    scan_id: ArrayLike,
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    usable: ArrayLike,
    scan_a: int,
    scan_b: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], dict[str, object]]:
    """Identical antenna, baseline and per-baseline time-sample support."""

    scans = np.asarray(scan_id, dtype=np.int32).reshape(-1)
    p_ant = np.asarray(antenna1, dtype=np.int32).reshape(-1)
    q_ant = np.asarray(antenna2, dtype=np.int32).reshape(-1)
    ok = np.asarray(usable, dtype=bool).reshape(-1)
    mask_a = np.zeros(scans.size, dtype=bool)
    mask_b = np.zeros(scans.size, dtype=bool)
    pairs = np.stack([np.minimum(p_ant, q_ant), np.maximum(p_ant, q_ant)], axis=1)
    n_kept = 0
    for left, right in np.unique(pairs, axis=0):
        if int(left) == int(right):
            continue
        rows_a = np.flatnonzero(
            ok & (scans == int(scan_a)) & (pairs[:, 0] == left) & (pairs[:, 1] == right)
        )
        rows_b = np.flatnonzero(
            ok & (scans == int(scan_b)) & (pairs[:, 0] == left) & (pairs[:, 1] == right)
        )
        take = min(rows_a.size, rows_b.size)
        if take == 0:
            continue
        mask_a[rows_a[:take]] = True
        mask_b[rows_b[:take]] = True
        n_kept += take
    report = {
        "n_matched_rows": int(n_kept),
        "n_matched_baselines": int(np.unique(pairs[mask_a], axis=0).shape[0])
        if np.any(mask_a)
        else 0,
        "identical_time_sampling": True,
        "notes": (PER_SCAN_EPSILON_NOT_OBSERVABLE_NOTE,),
    }
    return mask_a, mask_b, report


def align_residual_jones_maps(
    reference: Mapping[int, ArrayLike],
    other: Mapping[int, ArrayLike],
    *,
    gauge_antenna_id: int,
) -> tuple[dict[int, NDArray[np.complex128]], float]:
    """Global-phase align ``other`` to ``reference`` with the gauge antenna fixed."""

    products = []
    for ant, plane in reference.items():
        if int(ant) == int(gauge_antenna_id) or int(ant) not in other:
            continue
        ref = np.asarray(plane, dtype=np.complex128)
        alt = np.asarray(other[int(ant)], dtype=np.complex128)
        products.append(complex(np.vdot(ref, alt)))
    phase = float(np.angle(np.sum(products))) if products else 0.0
    aligned = {}
    for ant, plane in other.items():
        value = np.asarray(plane, dtype=np.complex128)
        if int(ant) == int(gauge_antenna_id):
            aligned[int(ant)] = np.array(value, copy=True)
        else:
            aligned[int(ant)] = np.exp(-1j * phase) * value
    return aligned, phase


def jones_parameter_jump(
    first: Mapping[int, ArrayLike],
    second: Mapping[int, ArrayLike],
) -> dict[str, object]:
    """Raw and per-antenna Frobenius |Δ(I+ε)| after both maps exist."""

    ants = sorted(set(int(ant) for ant in first) & set(int(ant) for ant in second))
    by_ant = {}
    norms = []
    for ant in ants:
        delta = np.linalg.norm(np.asarray(first[ant]) - np.asarray(second[ant]))
        by_ant[int(ant)] = float(delta)
        norms.append(float(delta))
    return {
        "n": len(ants),
        "median_abs_delta": float(np.median(norms)) if norms else float("nan"),
        "max_abs_delta": float(np.max(norms)) if norms else float("nan"),
        "by_antenna": by_ant,
    }


def predicted_operator_difference(
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    *,
    stokes_i: ArrayLike,
    chi1: ArrayLike,
    chi2: ArrayLike,
    first_jones: Mapping[int, ArrayLike],
    second_jones: Mapping[int, ArrayLike],
    q_over_i: float,
    u_over_i: float,
    row_mask: ArrayLike,
) -> dict[str, object]:
    """Compare predicted :math:`R_p S R_q^H` operators, not raw ε."""

    mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    pred_a = predict_residual_visibilities(
        np.asarray(antenna1)[mask],
        np.asarray(antenna2)[mask],
        stokes_i=np.asarray(stokes_i)[mask],
        chi1=np.asarray(chi1)[mask],
        chi2=np.asarray(chi2)[mask],
        residual_jones=first_jones,
        q_over_i=q_over_i,
        u_over_i=u_over_i,
    )
    pred_b = predict_residual_visibilities(
        np.asarray(antenna1)[mask],
        np.asarray(antenna2)[mask],
        stokes_i=np.asarray(stokes_i)[mask],
        chi1=np.asarray(chi1)[mask],
        chi2=np.asarray(chi2)[mask],
        residual_jones=second_jones,
        q_over_i=q_over_i,
        u_over_i=u_over_i,
    )
    intensity = np.asarray(stokes_i, dtype=np.float64).reshape(-1)[mask]
    scale = np.maximum(np.abs(intensity), 1.0e-3)
    delta = (pred_a - pred_b) / scale[:, None, None]
    rl = delta[:, 0, 1]
    return {
        "n": int(np.sum(mask)),
        "median_abs_rl_over_i": float(np.median(np.abs(rl[np.isfinite(rl)]))),
        "median_abs_all_over_i": float(np.median(np.abs(delta[np.isfinite(delta)]))),
        "notes": (ESTIMATOR_IDENTIFIABILITY_NOTE,),
    }


def classify_estimator_identifiability(
    *,
    parameter_max_abs_delta: float,
    operator_median_abs_rl: float,
    self_residual: float,
    cross_residual: float,
    gram_condition: float | None,
    qu_offdiag_spread: float | None,
    init_offdiag_spread: float | None,
    parameter_threshold: float = 0.05,
    operator_threshold: float = 0.005,
) -> dict[str, object]:
    """Parameter jumps that do not appear in operators are not physical."""

    parameter_large = float(parameter_max_abs_delta) >= float(parameter_threshold)
    operator_small = float(operator_median_abs_rl) < float(operator_threshold)
    cross_ok = float(cross_residual) <= 2.0 * max(float(self_residual), 1.0e-4)
    weak = (
        gram_condition is not None
        and np.isfinite(gram_condition)
        and float(gram_condition) >= 1.0e6
    )
    qu_deg = qu_offdiag_spread is not None and float(qu_offdiag_spread) >= 0.02
    init_split = init_offdiag_spread is not None and float(init_offdiag_spread) >= 0.02
    # Cross-apply is the physical test. Equivalent residuals mean the
    # parameter jump is gauge or weak identifiability, even if the raw
    # operator difference is larger than the tight 0.5% bar.
    physical = parameter_large and not cross_ok
    return {
        "status": "pass" if cross_ok else "warn",
        "blocking": not cross_ok,
        "gate_kind": "visibility_operator_identifiability",
        "jones_factors_unique": False,
        "epsilon_gauge_dependent": True,
        "parameter_jump_large": parameter_large,
        "operator_jump_small": operator_small,
        "cross_apply_equivalent": cross_ok,
        "weak_modes": weak,
        "qu_leakage_degeneracy": bool(qu_deg),
        "initialization_split": bool(init_split),
        "parameter_jump_is_physical": physical,
        "notes": (
            PER_SCAN_EPSILON_NOT_OBSERVABLE_NOTE,
            ESTIMATOR_IDENTIFIABILITY_NOTE,
            COMPLETE_FIELD9_TRACK_NOTE,
            PREDICTION_EQUIVALENT_BEAM_NOTE,
        ),
    }


def classify_stabilized_residual_jones(
    *,
    scan_cluster_residual: float | None,
    baseline_residual: float | None,
    sample_residual: float | None,
    family_operator_max_rl: float,
    family_cross_equivalent: bool,
    n_held_out_scans: int,
    operator_threshold: float = 0.02,
) -> dict[str, object]:
    """Complete-track static solve gated on operators, not unique ε."""

    cluster_ok = (
        scan_cluster_residual is not None
        and baseline_residual is not None
        and float(scan_cluster_residual) <= 2.0 * max(float(baseline_residual), 1.0e-4)
    )
    samples_ok = sample_residual is None or (
        baseline_residual is not None
        and float(sample_residual) <= 2.0 * max(float(baseline_residual), 1.0e-4)
    )
    family_ok = bool(family_cross_equivalent) or float(family_operator_max_rl) < float(
        operator_threshold
    )
    enough = int(n_held_out_scans) >= 1
    passed = enough and cluster_ok and samples_ok and family_ok
    return {
        "status": "pass" if passed else "warn" if enough else "not_run",
        "blocking": not passed if enough else False,
        "gate_kind": "visibility_operator_identifiability",
        "jones_factors_unique": False,
        "epsilon_gauge_dependent": True,
        "scan_cluster_holdout_ok": cluster_ok,
        "sample_holdout_ok": samples_ok,
        "prediction_equivalent_family": family_ok,
        "n_held_out_scans": int(n_held_out_scans),
        "notes": (
            ESTIMATOR_IDENTIFIABILITY_NOTE,
            COMPLETE_FIELD9_TRACK_NOTE,
            PREDICTION_EQUIVALENT_BEAM_NOTE,
            NO_SCAN_OFFSETS_OR_TIME_FREEDOM_NOTE,
        ),
    }
