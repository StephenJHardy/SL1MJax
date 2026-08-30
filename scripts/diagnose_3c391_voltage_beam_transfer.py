#!/usr/bin/env python3
"""Fixed-sky held-out loss for four streamed voltage beams on 3C391.

Uses the frozen sky, calibration, flags, weights, and native holdout
selection. Does not refit. Full Jones is constructed with
``allow_unfrozen=True`` in this diagnostic only; the production
``full_jones`` factory is unchanged.

The native holdout fixtures store RR/LL only. Cross-hand loss is reported
as predicted power, not as a data residual.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.beam_operator import (
    BeamOperatorConfig,
    SkyStokesPlanes,
    predict_voltage_beam,
)
from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.cassbeam_beam import (
    CassbeamCBandVoltageBeam,
    load_cassbeam_cband_artifact,
    voltage_beam_for_mode,
)
from sl1mjax.composite import MosaicSkyComponent
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock, read_dataset
from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.polarization import Correlation
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam
from sl1mjax.voltage_operator_jax import (
    off_diagonal_support_mask_jax,
    predict_voltage_beam_jax,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE = Path("outputs/3c391_native_averaging_ablation")
DEFAULT_PROTOCOL = Path("outputs/3c391_composite_catalogue_stage3/protocol.json")
DEFAULT_CHECKPOINT = Path("outputs/3c391_recovery_policy_fit_zero/sealed_active_only.npz")
DEFAULT_OUTPUT = Path("outputs/3c391_voltage_beam_transfer")
DEFAULT_POL_GOLDEN = ROOT / "tests" / "fixtures" / "3c391_polarization_golden.npz"
DEFAULT_MS = Path(
    "/home/stephen/checkouts/SL1MJax-frozen-20260824/data/3c391_ctm_mosaic_10s_spw0.ms"
)
BEAM_NAMES = (
    "static_scalar",
    "streamed_scalar",
    "diagonal_copolar",
    "full_jones_unfrozen",
)
OPERATOR_RELATIVE_TOLERANCE = 1.0e-5
OPERATOR_ABSOLUTE_TOLERANCE = 1.0e-8
PA_BIN_EDGES_RAD = np.deg2rad(np.linspace(-180.0, 180.0, 7))


def _to_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def flatten_positive_sky(
    components: tuple[MosaicSkyComponent, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mosaic-frame ``(l, m, I)`` for atoms with positive flux."""

    from sl1mjax.composite import MosaicPointComponent, MosaicQuadtreeComponent

    l_parts: list[np.ndarray] = []
    m_parts: list[np.ndarray] = []
    flux_parts: list[np.ndarray] = []
    for component in components:
        if isinstance(component, MosaicQuadtreeComponent):
            l_rad, m_rad = component.topology.centers()
        elif isinstance(component, MosaicPointComponent):
            l_rad = np.asarray(component.l_rad, dtype=np.float64).reshape(-1)
            m_rad = np.asarray(component.m_rad, dtype=np.float64).reshape(-1)
        else:
            raise TypeError(f"unsupported sky component {type(component)!r}")
        flux = np.asarray(component.flux, dtype=np.float64).reshape(-1)
        keep = flux > 0.0
        if not np.any(keep):
            continue
        l_parts.append(l_rad[keep])
        m_parts.append(m_rad[keep])
        flux_parts.append(flux[keep])
    if not flux_parts:
        raise ValueError("frozen sky has no positive-flux atoms")
    return (
        np.concatenate(l_parts),
        np.concatenate(m_parts),
        np.concatenate(flux_parts),
    )


def local_direction_cosines(
    l_mosaic: np.ndarray,
    m_mosaic: np.ndarray,
    mosaic_phase_centre_rad: tuple[float, float],
    block_phase_centre_rad: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate mosaic-frame directions into a pointing's tangent plane."""

    sky_ra, sky_dec = lmn_to_radec(
        mosaic_phase_centre_rad[0], mosaic_phase_centre_rad[1], l_mosaic, m_mosaic
    )
    local_l, local_m, _n = radec_to_lmn(
        block_phase_centre_rad[0], block_phase_centre_rad[1], sky_ra, sky_dec
    )
    return np.asarray(local_l, dtype=np.float64), np.asarray(local_m, dtype=np.float64)


def weighted_residual_power(
    residual: np.ndarray, weight: np.ndarray, mask: np.ndarray
) -> float:
    selected = np.asarray(mask, dtype=bool) & np.isfinite(weight) & (weight > 0)
    if not np.any(selected):
        return float("nan")
    values = residual[selected]
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if not np.any(finite):
        return float("nan")
    return float(np.sum(weight[selected][finite] * np.abs(values[finite]) ** 2))


def weighted_model_power(
    model: np.ndarray, weight: np.ndarray, mask: np.ndarray
) -> float:
    return weighted_residual_power(model, weight, mask)


def correlation_mask(
    block: VisibilityBlock, base: np.ndarray, correlation: Correlation
) -> np.ndarray:
    if correlation not in block.correlations:
        return np.zeros(block.shape, dtype=bool)
    mask = np.asarray(base, dtype=bool).copy()
    index = block.correlations.index(correlation)
    for other, _name in enumerate(block.correlations):
        if other != index:
            mask[..., other] = False
    return mask & block.active


def load_antenna_positions(
    *,
    polarization_golden: Path | None,
    measurement_set: Path | None,
    antenna_count: int,
) -> np.ndarray:
    """Load ITRF antenna positions from the polarisation golden or the MS."""

    if polarization_golden is not None and polarization_golden.is_file():
        with np.load(polarization_golden) as arrays:
            positions = np.asarray(arrays["antenna_position_m"], dtype=np.float64)
        if positions.shape[0] >= antenna_count:
            return positions[:antenna_count]
        raise ValueError(
            f"{polarization_golden} has {positions.shape[0]} antennas; need {antenna_count}"
        )
    if measurement_set is None or not measurement_set.is_dir():
        raise FileNotFoundError(
            "antenna positions require the polarisation golden or a MeasurementSet"
        )
    from casacore.tables import table

    with table(str(measurement_set / "ANTENNA")) as antennas:
        positions = np.asarray(antennas.getcol("POSITION"), dtype=np.float64)
    if positions.shape[0] < antenna_count:
        raise ValueError("MeasurementSet ANTENNA table is shorter than the block")
    return positions[:antenna_count]


def construct_beams(airy_max_radius_rad_at_1ghz: float) -> dict[str, Any]:
    """Return the four diagnostic beams. Full Jones is unfrozen here only."""

    catalog = VLABeamCatalog(airy_max_radius_rad_at_1ghz=airy_max_radius_rad_at_1ghz)
    diagonal = voltage_beam_for_mode("diagonal_copolar")
    return {
        "static_scalar": AnalyticAiryVoltageBeam(catalog=catalog),
        "streamed_scalar": voltage_beam_for_mode("streamed_scalar"),
        "diagonal_copolar": diagonal,
        "full_jones_unfrozen": CassbeamCBandVoltageBeam(
            load_cassbeam_cband_artifact(),
            off_diagonal=True,
            outer=diagonal.outer,
            allow_unfrozen=True,
        ),
    }


def explicit_airy_prediction(
    block: VisibilityBlock,
    l_rad: np.ndarray,
    m_rad: np.ndarray,
    flux: np.ndarray,
    *,
    airy_max_radius_rad_at_1ghz: float,
) -> np.ndarray:
    """Existing static-Airy predict on the same delta-function sky support."""

    beam = VLAPrimaryBeam(
        kind="airy",
        catalog=VLABeamCatalog(airy_max_radius_rad_at_1ghz=airy_max_radius_rad_at_1ghz),
    )
    weights = beam.power_weights(l_rad, m_rad, block.frequency_hz, receptor="I")
    return np.asarray(
        predict_stokes_i_explicit(
            flux,
            l_rad,
            m_rad,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            config=DirectDFTConfig(precision="float64"),
            beam_weights=weights,
        )
    )


def operator_reproduces_explicit_airy(
    streamed: np.ndarray, explicit: np.ndarray
) -> dict[str, float | bool]:
    """Compare streamed static Airy to the existing explicit Airy operator."""

    difference = streamed - explicit
    scale = float(np.max(np.abs(explicit)))
    peak = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(np.abs(difference) ** 2)))
    accepted = peak <= max(
        OPERATOR_ABSOLUTE_TOLERANCE, OPERATOR_RELATIVE_TOLERANCE * scale
    )
    return {
        "accepted": accepted,
        "peak_abs_difference": peak,
        "rms_difference": rms,
        "explicit_peak": scale,
    }


def leakage_support_mask(
    beam: CassbeamCBandVoltageBeam,
    l_rad: np.ndarray,
    m_rad: np.ndarray,
    frequency_hz: np.ndarray,
    parallactic_angle_rad_values: np.ndarray,
    *,
    backend: str = "jax",
    config: BeamOperatorConfig | None = None,
) -> np.ndarray:
    """True where every antenna/channel has valid off-diagonal Jones."""

    if backend == "jax":
        return off_diagonal_support_mask_jax(
            beam,
            l_rad,
            m_rad,
            frequency_hz,
            parallactic_angle_rad_values,
            calibration_state="casa_parang_true",
            config=config,
        )
    from sl1mjax.voltage_beam import beam_coordinates

    antennas = np.arange(parallactic_angle_rad_values.size, dtype=np.int32)
    evaluation = beam.evaluate(
        beam_coordinates(
            l_rad,
            m_rad,
            frequency_hz,
            parallactic_angle_rad=parallactic_angle_rad_values,
            antenna_id=antennas,
        ),
        calibration_state="casa_parang_true",
    )
    return np.asarray(np.all(evaluation.off_diagonal_valid, axis=(0, 2)), dtype=bool)


def predict_transfer_visibilities(
    block: VisibilityBlock,
    l_rad: np.ndarray,
    m_rad: np.ndarray,
    flux: np.ndarray,
    beam,
    *,
    antenna_position_m: np.ndarray,
    config: BeamOperatorConfig,
    backend: str,
):
    """Predict circular visibilities with the JAX or NumPy voltage operator."""

    kwargs = dict(
        block=block,
        l_rad=l_rad,
        m_rad=m_rad,
        sky=SkyStokesPlanes(stokes_i=flux),
        beam=beam,
        antenna_position_m=antenna_position_m,
        calibration_state="casa_parang_true",
        config=config,
    )
    if backend == "jax":
        return predict_voltage_beam_jax(**kwargs)
    return predict_voltage_beam(**kwargs)


def _bin_edges_mask(values: np.ndarray, edges: np.ndarray) -> list[tuple[str, np.ndarray]]:
    labels = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        if np.isfinite(high):
            name = f"[{low:.3g},{high:.3g})"
        else:
            name = f">={low:.3g}"
        labels.append((name, (values >= low) & (values < high)))
    return labels


def score_prediction(
    block: VisibilityBlock,
    prediction: np.ndarray,
    *,
    antenna_position_m: np.ndarray,
    pointing_radius_arcmin: float,
    leakage_atom_fraction: float | None,
) -> dict[str, Any]:
    residual = block.visibility - prediction
    mask = block.active
    row_pa = np.mean(
        parallactic_angle_rad(block.time_s, block.phase_centre_rad, antenna_position_m),
        axis=1,
    )
    unique_times, time_index = np.unique(block.time_s, return_inverse=True)
    payload: dict[str, Any] = {
        "total": weighted_residual_power(residual, block.weight, mask),
        "data_power": weighted_residual_power(block.visibility, block.weight, mask),
        "model_power": weighted_model_power(prediction, block.weight, mask),
        "pointing_radius_arcmin": pointing_radius_arcmin,
        "leakage_atom_fraction": leakage_atom_fraction,
        "correlations": {},
        "by_channel": [],
        "by_time": [],
        "by_parallactic_angle": [],
    }
    for correlation in (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL):
        if correlation in block.correlations:
            selected = correlation_mask(block, mask, correlation)
            payload["correlations"][correlation.value] = {
                "held_out_loss": weighted_residual_power(residual, block.weight, selected),
                "data_power": weighted_residual_power(
                    block.visibility, block.weight, selected
                ),
                "model_power": weighted_model_power(prediction, block.weight, selected),
                "in_data": True,
            }
        else:
            payload["correlations"][correlation.value] = {
                "held_out_loss": None,
                "data_power": None,
                "model_power": None,
                "in_data": False,
            }
    for channel in range(block.frequency_hz.size):
        channel_mask = np.zeros(block.shape, dtype=bool)
        channel_mask[:, channel, :] = mask[:, channel, :]
        payload["by_channel"].append(
            {
                "channel": channel,
                "frequency_hz": float(block.frequency_hz[channel]),
                "held_out_loss": weighted_residual_power(
                    residual, block.weight, channel_mask
                ),
            }
        )
    for index, time_s in enumerate(unique_times):
        time_mask = np.zeros(block.shape, dtype=bool)
        time_mask[time_index == index] = mask[time_index == index]
        payload["by_time"].append(
            {
                "time_s": float(time_s),
                "parallactic_angle_rad": float(np.mean(row_pa[time_index == index])),
                "held_out_loss": weighted_residual_power(residual, block.weight, time_mask),
            }
        )
    wrapped = (row_pa + np.pi) % (2.0 * np.pi) - np.pi
    for name, selected_rows in _bin_edges_mask(wrapped, PA_BIN_EDGES_RAD):
        bin_mask = np.zeros(block.shape, dtype=bool)
        bin_mask[selected_rows] = mask[selected_rows]
        payload["by_parallactic_angle"].append(
            {
                "bin": name,
                "held_out_loss": weighted_residual_power(residual, block.weight, bin_mask),
            }
        )
    return payload


def _resolve_protocol_paths(protocol: dict[str, Any], *roots: Path) -> dict[str, Any]:
    resolved = dict(protocol)
    frozen = resolved.get("frozen_directory")
    if not frozen:
        return resolved
    path = Path(frozen)
    for candidate in (path, *(root / path for root in roots if root is not None)):
        if (candidate / "summary.json").exists():
            resolved["frozen_directory"] = str(candidate)
            return resolved
    return resolved


def _load_components(
    protocol_path: Path, checkpoint: Path, mosaic_phase_centre_rad: tuple[float, float]
) -> tuple[MosaicSkyComponent, ...]:
    scripts_directory = str(Path(__file__).resolve().parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    from compare_3c391_composite_existing_flags import _components_from_checkpoint

    protocol = _resolve_protocol_paths(
        json.loads(protocol_path.read_text(encoding="utf-8")),
        Path.cwd(),
        protocol_path.resolve().parent.parent.parent,
        checkpoint.resolve().parent.parent.parent,
        ROOT,
    )
    return _components_from_checkpoint(checkpoint, protocol, mosaic_phase_centre_rad)


def _pointing_radius_arcmin(
    mosaic_phase_centre_rad: tuple[float, float],
    block_phase_centre_rad: tuple[float, float],
) -> float:
    l_rad, m_rad, _n = radec_to_lmn(
        mosaic_phase_centre_rad[0],
        mosaic_phase_centre_rad[1],
        np.asarray([block_phase_centre_rad[0]]),
        np.asarray([block_phase_centre_rad[1]]),
    )
    return float(np.rad2deg(np.hypot(l_rad[0], m_rad[0])) * 60.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--sky-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--sky-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL_GOLDEN)
    parser.add_argument("--measurement-set", type=Path, default=DEFAULT_MS)
    parser.add_argument(
        "--pointings",
        default="C1,C2,C3,C4,C5,C6,C7",
        help="Comma-separated native pointing labels",
    )
    parser.add_argument(
        "--beams",
        default=",".join(BEAM_NAMES),
        help="Comma-separated diagnostic beam names",
    )
    parser.add_argument(
        "--backend",
        choices=("jax", "numpy"),
        default="jax",
        help="Voltage operator backend. JAX is the default.",
    )
    arguments = parser.parse_args()
    selected_pointings = tuple(item.strip() for item in arguments.pointings.split(","))
    selected_beams = tuple(item.strip() for item in arguments.beams.split(","))
    unknown = [name for name in selected_beams if name not in BEAM_NAMES]
    if unknown:
        parser.error(f"unknown beams {unknown}; choose from {BEAM_NAMES}")

    first = read_dataset(arguments.native_directory / f"native_{selected_pointings[0]}.zarr")
    mosaic_phase_centre_rad = first.blocks[0].phase_centre_rad
    protocol = json.loads(arguments.sky_protocol.read_text(encoding="utf-8"))
    airy_radius = np.deg2rad(float(protocol["airy_max_radius_deg_at_1ghz"]))
    components = _load_components(
        arguments.sky_protocol, arguments.sky_checkpoint, mosaic_phase_centre_rad
    )
    l_mosaic, m_mosaic, flux = flatten_positive_sky(components)
    antenna_position_m = load_antenna_positions(
        polarization_golden=arguments.polarization_golden,
        measurement_set=arguments.measurement_set,
        antenna_count=first.blocks[0].antenna_count,
    )
    beams = construct_beams(airy_radius)
    # 26 antennas × 512 directions × 64 channels is about 55 MiB, under the
    # default 64 MiB Jones-tile budget. NumPy may use a larger pixel tile.
    config = (
        BeamOperatorConfig(visibility_chunk_size=256, pixel_chunk_size=512)
        if arguments.backend == "jax"
        else BeamOperatorConfig(visibility_chunk_size=256, pixel_chunk_size=2048)
    )
    arguments.output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "diagnostic": "3c391_voltage_beam_transfer",
        "backend": arguments.backend,
        "frozen": False,
        "full_jones_factory_unchanged": True,
        "calibration_state": "casa_parang_true",
        "sky_atoms_positive": int(flux.size),
        "sky_flux_jy": float(flux.sum()),
        "native_correlations": [c.value for c in first.blocks[0].correlations],
        "operator_gate": None,
        "pointings": {},
        "notes": (
            "Native holdout fixtures are RR/LL only. RL/LR held-out loss is "
            "absent, not zero. Full Jones is allow_unfrozen in this script "
            "only. Sky atoms are leaf centres; the stored mosaic model used "
            "quadtree pixel kernels."
        ),
    }

    operator_block = first.blocks[0]
    local_l, local_m = local_direction_cosines(
        l_mosaic, m_mosaic, mosaic_phase_centre_rad, operator_block.phase_centre_rad
    )
    explicit = explicit_airy_prediction(
        operator_block,
        local_l,
        local_m,
        flux,
        airy_max_radius_rad_at_1ghz=airy_radius,
    )
    streamed = predict_transfer_visibilities(
        operator_block,
        local_l,
        local_m,
        flux,
        beams["static_scalar"],
        antenna_position_m=antenna_position_m,
        config=config,
        backend=arguments.backend,
    )
    gate = operator_reproduces_explicit_airy(streamed.visibility, explicit)
    summary["operator_gate"] = {
        "pointing": selected_pointings[0],
        **gate,
    }
    (arguments.output / "operator_gate.json").write_text(
        json.dumps(_to_json(summary["operator_gate"]), indent=2, sort_keys=True) + "\n"
    )
    print("operator gate", json.dumps(_to_json(gate)), flush=True)
    if not gate["accepted"]:
        print("streamed static Airy does not reproduce explicit Airy; stop")
        (arguments.output / "summary.json").write_text(
            json.dumps(_to_json(summary), indent=2, sort_keys=True) + "\n"
        )
        return 1

    for pointing in selected_pointings:
        block = read_dataset(arguments.native_directory / f"native_{pointing}.zarr").blocks[0]
        local_l, local_m = local_direction_cosines(
            l_mosaic, m_mosaic, mosaic_phase_centre_rad, block.phase_centre_rad
        )
        radius = _pointing_radius_arcmin(mosaic_phase_centre_rad, block.phase_centre_rad)
        mid_time = float(np.median(block.time_s))
        mid_pa = parallactic_angle_rad(
            np.asarray([mid_time]), block.phase_centre_rad, antenna_position_m
        )[0]
        leakage = leakage_support_mask(
            beams["full_jones_unfrozen"],
            local_l,
            local_m,
            block.frequency_hz,
            mid_pa,
            backend=arguments.backend,
            config=config,
        )
        pointing_payload: dict[str, Any] = {
            "radius_arcmin": radius,
            "leakage_atom_fraction": float(np.mean(leakage)),
            "leakage_flux_fraction": float(flux[leakage].sum() / flux.sum()),
            "beams": {},
        }
        for beam_name in selected_beams:
            started = time.perf_counter()
            result = predict_transfer_visibilities(
                block,
                local_l,
                local_m,
                flux,
                beams[beam_name],
                antenna_position_m=antenna_position_m,
                config=config,
                backend=arguments.backend,
            )
            elapsed_s = time.perf_counter() - started
            scores = score_prediction(
                block,
                result.visibility,
                antenna_position_m=antenna_position_m,
                pointing_radius_arcmin=radius,
                leakage_atom_fraction=float(np.mean(leakage)),
            )
            scores["elapsed_s"] = elapsed_s
            pointing_payload["beams"][beam_name] = scores
            print(
                pointing,
                beam_name,
                "total",
                scores["total"],
                "RR",
                scores["correlations"]["RR"]["held_out_loss"],
                "LL",
                scores["correlations"]["LL"]["held_out_loss"],
                f"elapsed_s={elapsed_s:.1f}",
                flush=True,
            )
        summary["pointings"][pointing] = pointing_payload
        (arguments.output / "summary.json").write_text(
            json.dumps(_to_json(summary), indent=2, sort_keys=True) + "\n"
        )

    interpretation = _interpret(summary)
    summary["interpretation"] = interpretation
    (arguments.output / "summary.json").write_text(
        json.dumps(_to_json(summary), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(_to_json(interpretation), indent=2))
    return 0


def _interpret(summary: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, list[float]] = {name: [] for name in BEAM_NAMES}
    rr: dict[str, list[float]] = {name: [] for name in BEAM_NAMES}
    ll: dict[str, list[float]] = {name: [] for name in BEAM_NAMES}
    for pointing in summary["pointings"].values():
        for name, scores in pointing["beams"].items():
            totals[name].append(float(scores["total"]))
            rr[name].append(float(scores["correlations"]["RR"]["held_out_loss"]))
            ll[name].append(float(scores["correlations"]["LL"]["held_out_loss"]))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    means = {name: _mean(totals[name]) for name in totals if totals[name]}
    ranking = sorted(means, key=means.get)
    airy = means.get("static_scalar")
    composite = means.get("streamed_scalar")
    diagonal = means.get("diagonal_copolar")
    full = means.get("full_jones_unfrozen")
    return {
        "mean_held_out_loss": means,
        "mean_rr": {name: _mean(rr[name]) for name in rr if rr[name]},
        "mean_ll": {name: _mean(ll[name]) for name in ll if ll[name]},
        "ranking_best_first": ranking,
        "scalar_shape_matters": bool(
            airy is not None and composite is not None and composite < airy
        ),
        "squint_or_rl_structure_matters": bool(
            composite is not None and diagonal is not None and diagonal < composite
        ),
        "leakage_matters": bool(
            diagonal is not None and full is not None and full < diagonal
        ),
        "no_detailed_beam_beats_airy": bool(
            airy is not None
            and all(
                means.get(name, airy) >= airy
                for name in ("streamed_scalar", "diagonal_copolar", "full_jones_unfrozen")
                if name in means
            )
        ),
        "cross_hand_in_data": False,
        "do_not_freeze_full_jones": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
