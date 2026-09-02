"""CASA ``awp2`` beam oracles.

Stage 1 compares Stokes-I (and optional RR/LL) power images. Stage 2
compares complex baseline visibilities. Runtime evaluation never invokes
CASA. Frozen products must be generated on Bacchus and checksummed.

Freezing the Stage-1 ``.pb`` files records an immutable CASA measurement.
It does not accept CASSBEAM. ``casa_awp2_accepted`` and
``diagonal_copolar_is_casa_accepted()`` stay false until a later
visibility-domain or holography argument. Stage 2 requires a valid
frozen Stage-1 oracle, not CASSBEAM–CASA equality.

CASA ``awp2`` uses its internal ray-traced EVLA A-term, not a
``vpmanager`` / ``vptable`` beam. The exported ``.pb`` is an
image-domain PB / normalization product, not a complex per-receptor
Jones. Loading our CASSBEAM tables into CASA would test ingestion, not
beam-model accuracy. Do not remove CASSBEAM squint to match these
scalar ``.pb`` files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import ArrayLike, NDArray

from sl1mjax.cassbeam_beam import (
    CASA_AWP2_MAIN_LOBE_POWER_TOLERANCE,
    BeamImagingMode,
    diagonal_copolar_is_casa_accepted,
    voltage_beam_for_mode,
)
from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.polarization import (
    apply_jones_to_coherency,
    circular_stokes_to_coherency,
)
from sl1mjax.rime import SPEED_OF_LIGHT_M_S
from sl1mjax.voltage_beam import beam_coordinates, stokes_i_power_from_jones

POWER_ORACLE_SCHEMA_VERSION = 1
POLARIZATION_ORACLE_SCHEMA_VERSION = 1
POWER_ORACLE_VOLTAGE_PATTERN = "casa_default_evla_raytraced"
POWER_ORACLE_CONTOUR = 0.05
CORE_CONTOUR = 0.10
CENTRE_OFFSET_PIXEL_TOLERANCE = 0.25
FWHM_FRACTIONAL_TOLERANCE = 0.05
GateStatus = Literal["pass", "fail", "false", "not_run"]
_FROZEN_POWER_DIR = Path(__file__).with_name("data") / "casa_awp2_oracle"
_POWER_MANIFEST = "manifest.json"
POINT_SOURCE_ROLES = (
    "centre",
    "plus_l",
    "minus_l",
    "plus_m",
    "minus_m",
    "half_power",
    "main_lobe_edge",
)
STOKES_MODELS = ("I", "I+Q", "I+U", "I+V")


@dataclass(frozen=True)
class CasaAwp2PowerPlane:
    """One CASA ``.pb`` plane and the sky coordinates of its pixels."""

    power: NDArray[np.float64]
    l_rad: NDArray[np.float64]
    m_rad: NDArray[np.float64]
    frequency_hz: float
    parallactic_angle_rad: float
    stokes: str
    path: Path


@dataclass(frozen=True)
class PowerBeamComparison:
    """Stage-1 metrics for one CASA ``.pb`` plane.

    ``accepted`` is the historical 5% all-or-nothing flag. Named gates
    live on :class:`CasaAwp2Stage1Gates`. Do not treat ``accepted`` as
    CASSBEAM scientific acceptance.
    """

    centre_offset_l_arcmin: float
    centre_offset_m_arcmin: float
    centre_offset_pixels: float
    fwhm_casa_arcmin: float
    fwhm_sl1mjax_arcmin: float
    rms_residual: float
    max_abs_residual: float
    contour_pixel_count: int
    accepted: bool
    rms_residual_core: float
    max_abs_residual_core: float
    core_contour_pixel_count: int
    peak_residual_radius_arcmin: float
    casa_power_at_peak_residual: float
    sl1mjax_power_at_peak_residual: float
    centre_ok: bool
    fwhm_ok: bool
    core_pointwise_ok: bool
    five_percent_pointwise_ok: bool
    stokes: str = ""
    fits: str = ""


@dataclass(frozen=True)
class CasaAwp2Stage1Gates:
    """Split Stage-1 claims. Freezing the oracle is not CASSBEAM acceptance."""

    casa_awp2_scalar_core_compatible: GateStatus
    casa_awp2_scalar_5percent_equivalent: GateStatus
    casa_awp2_rrll_oracle_valid: GateStatus
    casa_full_jones_convention_accepted: GateStatus
    casa_awp2_accepted: bool
    diagonal_copolar_is_casa_accepted: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CasaAwp2Stage1Report:
    """Directory-level Stage-1 comparison plus the split gates."""

    gates: CasaAwp2Stage1Gates
    comparisons: tuple[PowerBeamComparison, ...]
    identical_stokes_groups: tuple[tuple[str, ...], ...]
    casa_version: str | None
    voltage_pattern: str
    frozen_products: bool


def power_oracle_dir() -> Path:
    """Return the directory that holds frozen Stage-1 FITS products."""

    return _FROZEN_POWER_DIR


def power_oracle_is_frozen() -> bool:
    """True only when a checksummed Stage-1 manifest marks the products frozen."""

    manifest_path = _FROZEN_POWER_DIR / _POWER_MANIFEST
    if not manifest_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != POWER_ORACLE_SCHEMA_VERSION:
        return False
    if not bool(payload.get("frozen")):
        return False
    if str(payload.get("voltage_pattern", "")) != POWER_ORACLE_VOLTAGE_PATTERN:
        return False
    for plane in payload.get("planes", ()):
        path = _FROZEN_POWER_DIR / str(plane["fits"])
        if not path.is_file():
            return False
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != plane["sha256"]:
            return False
    return bool(payload.get("planes"))


def polarization_oracle_is_frozen() -> bool:
    """Stage 2 visibility oracle. Requires a frozen Stage-1 product, not equality."""

    return False


def load_power_plane(
    path: Path,
    *,
    frequency_hz: float,
    parallactic_angle_rad: float,
    stokes: str,
) -> CasaAwp2PowerPlane:
    """Load a SIN-projected CASA ``.pb`` FITS and its pixel ``(l, m)``."""

    with fits.open(path) as hdus:
        data = np.asarray(np.squeeze(hdus[0].data), dtype=np.float64)
        header = hdus[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"{path} must squeeze to a 2-D primary-beam image")
    l_rad, m_rad = direction_cosines_from_primary_beam_header(header, data.shape)
    return CasaAwp2PowerPlane(
        power=data,
        l_rad=l_rad,
        m_rad=m_rad,
        frequency_hz=float(frequency_hz),
        parallactic_angle_rad=float(parallactic_angle_rad),
        stokes=str(stokes).upper(),
        path=path,
    )


def direction_cosines_from_primary_beam_header(
    header: fits.Header, shape: tuple[int, int]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(l, m)`` for every pixel of a celestial SIN WCS."""

    celestial = WCS(header).celestial
    rows, columns = np.indices(shape, dtype=np.float64)
    ra_deg, dec_deg = celestial.pixel_to_world_values(columns, rows)
    phase_ra, phase_dec = np.deg2rad(celestial.wcs.crval[:2])
    l_rad, m_rad, _n_rad = radec_to_lmn(
        float(phase_ra),
        float(phase_dec),
        np.deg2rad(ra_deg),
        np.deg2rad(dec_deg),
    )
    return np.asarray(l_rad, dtype=np.float64), np.asarray(m_rad, dtype=np.float64)


def sl1mjax_power_on_plane(plane: CasaAwp2PowerPlane) -> NDArray[np.float64]:
    """Evaluate the diagonal CASSBEAM mode on the FITS pixel coordinates."""

    beam = voltage_beam_for_mode(BeamImagingMode.DIAGONAL_COPOLAR)
    evaluation = beam.evaluate(
        beam_coordinates(
            plane.l_rad.reshape(-1),
            plane.m_rad.reshape(-1),
            plane.frequency_hz,
            parallactic_angle_rad=plane.parallactic_angle_rad,
        ),
        calibration_state="casa_parang_true",
    )
    if plane.stokes in {"I", "IQUV"}:
        power = stokes_i_power_from_jones(evaluation.jones)[0, :, 0]
    elif plane.stokes == "RR":
        power = np.square(np.abs(evaluation.jones[0, :, 0, 0, 0]))
    elif plane.stokes == "LL":
        power = np.square(np.abs(evaluation.jones[0, :, 0, 1, 1]))
    else:
        raise ValueError(f"unsupported oracle Stokes {plane.stokes!r}")
    return np.asarray(power.reshape(plane.power.shape), dtype=np.float64)


def compare_power_beams(
    casa: CasaAwp2PowerPlane,
    sl1mjax_power: ArrayLike,
    *,
    contour: float = POWER_ORACLE_CONTOUR,
    power_tolerance: float = CASA_AWP2_MAIN_LOBE_POWER_TOLERANCE,
) -> PowerBeamComparison:
    """Compare normalised power at the 5% and 10% CASA contours.

    Both images are divided by the pointing-centre value. The 5%
    all-or-nothing flag is ``accepted``. Core compatibility uses the
    10% contour and does not loosen the 5% residual tolerance.
    """

    model = np.asarray(sl1mjax_power, dtype=np.float64)
    if model.shape != casa.power.shape:
        raise ValueError("SL1MJax power must match the CASA image shape")
    casa_norm = _normalize_at_pointing(casa.power, casa.l_rad, casa.m_rad)
    model_norm = _normalize_at_pointing(model, casa.l_rad, casa.m_rad)
    residual = model_norm - casa_norm
    five = _contour_residual(casa_norm, residual, contour)
    core = _contour_residual(casa_norm, residual, CORE_CONTOUR)
    if five.count < 16:
        raise ValueError("CASA 5% contour is too small to compare")
    if core.count < 16:
        raise ValueError("CASA 10% contour is too small to compare")
    casa_l, casa_m = _mainlobe_centroid(casa_norm, casa.l_rad, casa.m_rad)
    model_l, model_m = _mainlobe_centroid(model_norm, casa.l_rad, casa.m_rad)
    pixel = _mean_pixel_scale_rad(casa.l_rad, casa.m_rad)
    offset = float(np.hypot(model_l - casa_l, model_m - casa_m))
    casa_fwhm = _mean_fwhm_arcmin(casa_norm, casa.l_rad, casa.m_rad)
    model_fwhm = _mean_fwhm_arcmin(model_norm, casa.l_rad, casa.m_rad)
    fwhm_ok = abs(model_fwhm / casa_fwhm - 1.0) <= FWHM_FRACTIONAL_TOLERANCE
    centre_ok = offset <= CENTRE_OFFSET_PIXEL_TOLERANCE * pixel
    five_pointwise_ok = (
        five.peak <= power_tolerance and five.rms <= power_tolerance
    )
    core_pointwise_ok = (
        core.peak <= power_tolerance and core.rms <= power_tolerance
    )
    accepted = centre_ok and fwhm_ok and five_pointwise_ok
    peak_index = five.peak_index
    radius_arcmin = float(
        np.rad2deg(np.hypot(casa.l_rad[peak_index], casa.m_rad[peak_index])) * 60.0
    )
    return PowerBeamComparison(
        centre_offset_l_arcmin=float(np.rad2deg(model_l - casa_l) * 60.0),
        centre_offset_m_arcmin=float(np.rad2deg(model_m - casa_m) * 60.0),
        centre_offset_pixels=offset / pixel,
        fwhm_casa_arcmin=casa_fwhm,
        fwhm_sl1mjax_arcmin=model_fwhm,
        rms_residual=five.rms,
        max_abs_residual=five.peak,
        contour_pixel_count=five.count,
        accepted=accepted,
        rms_residual_core=core.rms,
        max_abs_residual_core=core.peak,
        core_contour_pixel_count=core.count,
        peak_residual_radius_arcmin=radius_arcmin,
        casa_power_at_peak_residual=float(casa_norm[peak_index]),
        sl1mjax_power_at_peak_residual=float(model_norm[peak_index]),
        centre_ok=centre_ok,
        fwhm_ok=fwhm_ok,
        core_pointwise_ok=core_pointwise_ok,
        five_percent_pointwise_ok=five_pointwise_ok,
        stokes=casa.stokes,
        fits=casa.path.name,
    )


def compare_power_oracle_directory(directory: Path) -> tuple[PowerBeamComparison, ...]:
    """Compare Stage-1 FITS in a checksummed manifest directory.

    Does not require ``frozen: true``. Used after Bacchus generation.
    A disagreement is a measured beam difference, not a missing file.
    """

    return evaluate_casa_awp2_stage1(directory).comparisons


def evaluate_casa_awp2_stage1(directory: Path) -> CasaAwp2Stage1Report:
    """Compare checksummed Stage-1 planes and score the split gates."""

    payload = json.loads((directory / _POWER_MANIFEST).read_text(encoding="utf-8"))
    if str(payload.get("voltage_pattern", "")) != POWER_ORACLE_VOLTAGE_PATTERN:
        raise ValueError(
            "Stage-1 oracle must use CASA's default EVLA ray-traced A-term; "
            f"got voltage_pattern={payload.get('voltage_pattern')!r}"
        )
    planes = tuple(payload.get("planes", ()))
    loaded_planes: list[CasaAwp2PowerPlane] = []
    comparisons: list[PowerBeamComparison] = []
    for plane in planes:
        if plane.get("parallactic_angle_rad") is None:
            raise ValueError(f"{plane.get('fits')} is missing parallactic_angle_rad")
        path = directory / str(plane["fits"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != plane["sha256"]:
            raise ValueError(f"{path} checksum does not match the manifest")
        loaded = load_power_plane(
            path,
            frequency_hz=float(plane["frequency_hz"]),
            parallactic_angle_rad=float(plane["parallactic_angle_rad"]),
            stokes=str(plane["stokes"]),
        )
        loaded_planes.append(loaded)
        comparisons.append(compare_power_beams(loaded, sl1mjax_power_on_plane(loaded)))
    if not comparisons:
        raise ValueError(f"{directory} declares no Stage-1 planes")
    identical = _identical_stokes_groups(loaded_planes, planes)
    gates = stage1_gates_from_comparisons(
        tuple(comparisons),
        identical_stokes_groups=identical,
    )
    return CasaAwp2Stage1Report(
        gates=gates,
        comparisons=tuple(comparisons),
        identical_stokes_groups=identical,
        casa_version=payload.get("casa_version"),
        voltage_pattern=str(payload.get("voltage_pattern")),
        frozen_products=bool(payload.get("frozen")),
    )


def compare_frozen_power_oracle() -> tuple[PowerBeamComparison, ...]:
    """Load frozen Stage-1 FITS and compare every declared plane."""

    return evaluate_frozen_stage1().comparisons


def evaluate_frozen_stage1() -> CasaAwp2Stage1Report:
    """Score the committed Stage-1 oracle. Products may be frozen without acceptance."""

    if not power_oracle_is_frozen():
        raise FileNotFoundError("CASA awp2 power oracle is not frozen")
    return evaluate_casa_awp2_stage1(_FROZEN_POWER_DIR)


def stage1_gates_from_comparisons(
    comparisons: tuple[PowerBeamComparison, ...],
    *,
    identical_stokes_groups: tuple[tuple[str, ...], ...] = (),
) -> CasaAwp2Stage1Gates:
    """Split the historical all-or-nothing ``accepted`` flag into named claims."""

    stokes_i = tuple(item for item in comparisons if item.stokes in {"I", "IQUV", ""})
    if not stokes_i:
        stokes_i = comparisons
    core_pass = all(
        item.centre_ok and item.fwhm_ok and item.core_pointwise_ok for item in stokes_i
    )
    five_pass = all(item.accepted for item in stokes_i)
    rrll_present = any(item.stokes in {"RR", "LL"} for item in comparisons)
    if not rrll_present:
        rrll_status: GateStatus = "not_run"
        rrll_note = "no RR/LL planes were compared"
    elif identical_stokes_groups:
        rrll_status = "false"
        rrll_note = (
            "CASA I, RR, and LL .pb planes are identical; the export has no "
            "hand-dependent information"
        )
    else:
        rrll_status = "not_run"
        rrll_note = "RR/LL planes differ from I but are not a Jones export"
    notes = (
        "CASA .pb is an image-domain PB/normalization product, not a "
        "complex per-receptor Jones",
        rrll_note,
        "Do not remove CASSBEAM squint to match these scalar .pb files",
        "Stage 2 requires a frozen Stage-1 oracle, not CASSBEAM–CASA equality",
    )
    return CasaAwp2Stage1Gates(
        casa_awp2_scalar_core_compatible="pass" if core_pass else "fail",
        casa_awp2_scalar_5percent_equivalent="pass" if five_pass else "fail",
        casa_awp2_rrll_oracle_valid=rrll_status,
        casa_full_jones_convention_accepted="not_run",
        casa_awp2_accepted=False,
        diagonal_copolar_is_casa_accepted=diagonal_copolar_is_casa_accepted(),
        notes=notes,
    )


def stage1_report_as_dict(report: CasaAwp2Stage1Report) -> dict[str, Any]:
    """JSON-ready Stage-1 report. Does not mark CASSBEAM accepted."""

    return {
        "role": "casa_awp2_stage1_comparison",
        "frozen_products": report.frozen_products,
        "casa_awp2_accepted": report.gates.casa_awp2_accepted,
        "diagonal_copolar_is_casa_accepted": (
            report.gates.diagonal_copolar_is_casa_accepted
        ),
        "casa_version": report.casa_version,
        "voltage_pattern": report.voltage_pattern,
        "identical_stokes_groups": [list(group) for group in report.identical_stokes_groups],
        "gates": asdict(report.gates),
        "n_planes": len(report.comparisons),
        "planes": [asdict(item) for item in report.comparisons],
    }


def write_sine_projected_power_fits(
    path: Path,
    power: ArrayLike,
    *,
    phase_centre_rad: tuple[float, float],
    cell_rad: float,
    frequency_hz: float,
) -> None:
    """Write a CASA-like SIN primary-beam FITS for tests and scaffolding."""

    image = np.asarray(power, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 3:
        raise ValueError("power image must be a 2-D grid")
    if cell_rad <= 0.0:
        raise ValueError("cell_rad must be positive")
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = image.shape[1]
    header["NAXIS2"] = image.shape[0]
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = 0.5 * (image.shape[1] + 1.0)
    header["CRPIX2"] = 0.5 * (image.shape[0] + 1.0)
    header["CDELT1"] = -np.rad2deg(cell_rad)
    header["CDELT2"] = np.rad2deg(cell_rad)
    header["CRVAL1"] = np.rad2deg(phase_centre_rad[0])
    header["CRVAL2"] = np.rad2deg(phase_centre_rad[1])
    header["RESTFRQ"] = float(frequency_hz)
    fits.PrimaryHDU(image, header=header).writeto(path, overwrite=True)


def geometric_visibility_phase(
    uvw_m: ArrayLike,
    frequency_hz: float,
    l_rad: float,
    m_rad: float,
) -> NDArray[np.complex128]:
    """CASA geometric phase ``exp(+2πi [u l + v m + w(n-1)])``."""

    uvw = np.asarray(uvw_m, dtype=np.float64)
    if uvw.ndim != 2 or uvw.shape[-1] != 3:
        raise ValueError("uvw_m must have shape (row, 3)")
    n_rad = float(np.sqrt(max(0.0, 1.0 - l_rad * l_rad - m_rad * m_rad)))
    wavelengths = uvw * (float(frequency_hz) / SPEED_OF_LIGHT_M_S)
    phase = (
        wavelengths[:, 0] * l_rad
        + wavelengths[:, 1] * m_rad
        + wavelengths[:, 2] * (n_rad - 1.0)
    )
    return np.asarray(np.exp(2j * np.pi * phase), dtype=np.complex128)


def predict_point_source_visibilities(
    jones_p: ArrayLike,
    jones_q: ArrayLike,
    *,
    stokes_i: float,
    stokes_q: float,
    stokes_u: float,
    stokes_v: float,
    uvw_m: ArrayLike,
    frequency_hz: float,
    l_rad: float,
    m_rad: float,
) -> NDArray[np.complex128]:
    """Return ``E_p C E_q^H`` times the CASA geometric phase."""

    coherency = circular_stokes_to_coherency(stokes_i, stokes_q, stokes_u, stokes_v)
    apparent = apply_jones_to_coherency(coherency, jones_p, jones_q)
    phase = geometric_visibility_phase(uvw_m, frequency_hz, l_rad, m_rad)
    return np.asarray(apparent * phase[..., None, None], dtype=np.complex128)


def stokes_model_values(name: str) -> tuple[float, float, float, float]:
    """Return ``(I, Q, U, V)`` for a declared Stage-2 model."""

    if name == "I":
        return (1.0, 0.0, 0.0, 0.0)
    if name == "I+Q":
        return (1.0, 1.0, 0.0, 0.0)
    if name == "I+U":
        return (1.0, 0.0, 1.0, 0.0)
    if name == "I+V":
        return (1.0, 0.0, 0.0, 1.0)
    raise ValueError(f"unsupported Stokes model {name!r}")


@dataclass(frozen=True)
class _ContourResidual:
    rms: float
    peak: float
    count: int
    peak_index: tuple[int, ...]


def _contour_residual(
    casa_norm: np.ndarray,
    residual: np.ndarray,
    contour: float,
) -> _ContourResidual:
    inside = np.isfinite(casa_norm) & np.isfinite(residual) & (casa_norm >= contour)
    count = int(np.count_nonzero(inside))
    if count == 0:
        return _ContourResidual(rms=float("nan"), peak=float("nan"), count=0, peak_index=(0, 0))
    values = residual[inside]
    rms = float(np.sqrt(np.mean(np.square(values))))
    peak = float(np.max(np.abs(values)))
    masked = np.where(inside, np.abs(residual), -1.0)
    peak_index = tuple(int(item) for item in np.unravel_index(int(np.argmax(masked)), residual.shape))
    return _ContourResidual(rms=rms, peak=peak, count=count, peak_index=peak_index)


def _identical_stokes_groups(
    planes: list[CasaAwp2PowerPlane],
    records: tuple[dict[str, Any], ...],
) -> tuple[tuple[str, ...], ...]:
    groups: dict[tuple[float, str], list[int]] = {}
    for index, record in enumerate(records):
        hourangle = str(record.get("hourangle", ""))
        key = (float(record["frequency_hz"]), hourangle)
        groups.setdefault(key, []).append(index)
    identical: list[tuple[str, ...]] = []
    for indexes in groups.values():
        if len(indexes) < 2:
            continue
        first = np.asarray(planes[indexes[0]].power)
        names = [planes[index].path.name for index in indexes]
        if all(
            np.array_equal(first, planes[index].power, equal_nan=True)
            for index in indexes[1:]
        ):
            identical.append(tuple(names))
    return tuple(identical)


def _normalize_at_pointing(
    power: np.ndarray, l_rad: np.ndarray, m_rad: np.ndarray
) -> NDArray[np.float64]:
    centre = _sample_at_origin(power, l_rad, m_rad)
    if not np.isfinite(centre) or centre <= 0.0:
        raise ValueError("pointing-centre power must be finite and positive")
    return np.asarray(power / centre, dtype=np.float64)


def _sample_at_origin(
    power: np.ndarray, l_rad: np.ndarray, m_rad: np.ndarray
) -> float:
    radius = np.hypot(l_rad, m_rad)
    index = np.unravel_index(int(np.argmin(radius)), radius.shape)
    return float(power[index])


def _mainlobe_centroid(
    power: np.ndarray, l_rad: np.ndarray, m_rad: np.ndarray
) -> tuple[float, float]:
    finite = np.isfinite(power)
    if not np.any(finite):
        raise ValueError("power image has no main lobe")
    peak = float(np.max(power[finite]))
    weights = np.where(finite & (power >= 0.2 * peak), power, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("power image has no main lobe")
    return (
        float(np.sum(weights * l_rad) / total),
        float(np.sum(weights * m_rad) / total),
    )


def _mean_fwhm_arcmin(
    power: np.ndarray, l_rad: np.ndarray, m_rad: np.ndarray
) -> float:
    origin = np.unravel_index(int(np.argmin(np.hypot(l_rad, m_rad))), power.shape)
    width_l = _half_power_width_rad(power[origin[0], :], l_rad[origin[0], :])
    width_m = _half_power_width_rad(power[:, origin[1]], m_rad[:, origin[1]])
    return float(np.rad2deg(0.5 * (width_l + width_m)) * 60.0)


def _half_power_width_rad(strip: np.ndarray, coord: np.ndarray) -> float:
    finite = np.isfinite(strip)
    if not np.any(finite):
        raise ValueError("cannot measure half-power width")
    peak = float(np.max(strip[finite]))
    above = finite & (strip >= 0.5 * peak)
    if int(np.count_nonzero(above)) < 2:
        raise ValueError("cannot measure half-power width")
    return float(np.max(coord[above]) - np.min(coord[above]))


def _mean_pixel_scale_rad(l_rad: np.ndarray, m_rad: np.ndarray) -> float:
    dl = np.median(np.abs(np.diff(l_rad, axis=1)))
    dm = np.median(np.abs(np.diff(m_rad, axis=0)))
    scale = float(0.5 * (dl + dm))
    if scale <= 0.0:
        raise ValueError("FITS pixel scale must be positive")
    return scale
