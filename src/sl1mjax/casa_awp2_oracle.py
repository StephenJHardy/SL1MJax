"""CASA ``awp2`` beam oracles.

Stage 1 compares Stokes-I (and optional RR/LL) power images. Stage 2
compares complex baseline visibilities. Runtime evaluation never invokes
CASA. Frozen products must be generated on Bacchus and checksummed.

CASA ``awp2`` uses its internal ray-traced EVLA A-term, not a
``vpmanager`` / ``vptable`` beam. Stage 1 asks whether the committed
CASSBEAM evaluator reproduces that independent CASA model. Loading our
tables into CASA would test ingestion, not beam-model accuracy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import ArrayLike, NDArray

from sl1mjax.cassbeam_beam import (
    CASA_AWP2_MAIN_LOBE_POWER_TOLERANCE,
    BeamImagingMode,
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
CENTRE_OFFSET_PIXEL_TOLERANCE = 0.25
FWHM_FRACTIONAL_TOLERANCE = 0.05
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
    """Declared Stage-1 metrics inside CASA's 5% power contour."""

    centre_offset_l_arcmin: float
    centre_offset_m_arcmin: float
    centre_offset_pixels: float
    fwhm_casa_arcmin: float
    fwhm_sl1mjax_arcmin: float
    rms_residual: float
    max_abs_residual: float
    contour_pixel_count: int
    accepted: bool


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
    """Stage 2 is not frozen until Stage 1 is accepted and visibilities exist."""

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
    """Compare normalised power inside the CASA contour.

    Both images are divided by the interpolated pointing-centre value.
    The gate checks absolute centre location, FWHM, and residual power.
    """

    model = np.asarray(sl1mjax_power, dtype=np.float64)
    if model.shape != casa.power.shape:
        raise ValueError("SL1MJax power must match the CASA image shape")
    casa_norm = _normalize_at_pointing(casa.power, casa.l_rad, casa.m_rad)
    model_norm = _normalize_at_pointing(model, casa.l_rad, casa.m_rad)
    inside = np.isfinite(casa_norm) & np.isfinite(model_norm) & (casa_norm >= contour)
    if int(np.count_nonzero(inside)) < 16:
        raise ValueError("CASA 5% contour is too small to compare")
    residual = model_norm - casa_norm
    casa_l, casa_m = _mainlobe_centroid(casa_norm, casa.l_rad, casa.m_rad)
    model_l, model_m = _mainlobe_centroid(model_norm, casa.l_rad, casa.m_rad)
    pixel = _mean_pixel_scale_rad(casa.l_rad, casa.m_rad)
    offset = float(np.hypot(model_l - casa_l, model_m - casa_m))
    casa_fwhm = _mean_fwhm_arcmin(casa_norm, casa.l_rad, casa.m_rad)
    model_fwhm = _mean_fwhm_arcmin(model_norm, casa.l_rad, casa.m_rad)
    rms = float(np.sqrt(np.mean(np.square(residual[inside]))))
    peak = float(np.max(np.abs(residual[inside])))
    fwhm_ok = abs(model_fwhm / casa_fwhm - 1.0) <= FWHM_FRACTIONAL_TOLERANCE
    accepted = (
        peak <= power_tolerance
        and rms <= power_tolerance
        and offset <= CENTRE_OFFSET_PIXEL_TOLERANCE * pixel
        and fwhm_ok
    )
    return PowerBeamComparison(
        centre_offset_l_arcmin=float(np.rad2deg(model_l - casa_l) * 60.0),
        centre_offset_m_arcmin=float(np.rad2deg(model_m - casa_m) * 60.0),
        centre_offset_pixels=offset / pixel,
        fwhm_casa_arcmin=casa_fwhm,
        fwhm_sl1mjax_arcmin=model_fwhm,
        rms_residual=rms,
        max_abs_residual=peak,
        contour_pixel_count=int(np.count_nonzero(inside)),
        accepted=accepted,
    )


def compare_power_oracle_directory(directory: Path) -> tuple[PowerBeamComparison, ...]:
    """Compare Stage-1 FITS in a checksummed manifest directory.

    Does not require ``frozen: true``. Used after Bacchus generation and
    before acceptance. A disagreement is a measured beam difference.
    """

    payload = json.loads((directory / _POWER_MANIFEST).read_text(encoding="utf-8"))
    if str(payload.get("voltage_pattern", "")) != POWER_ORACLE_VOLTAGE_PATTERN:
        raise ValueError(
            "Stage-1 oracle must use CASA's default EVLA ray-traced A-term; "
            f"got voltage_pattern={payload.get('voltage_pattern')!r}"
        )
    comparisons = []
    for plane in payload.get("planes", ()):
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
        comparisons.append(compare_power_beams(loaded, sl1mjax_power_on_plane(loaded)))
    if not comparisons:
        raise ValueError(f"{directory} declares no Stage-1 planes")
    return tuple(comparisons)


def compare_frozen_power_oracle() -> tuple[PowerBeamComparison, ...]:
    """Load frozen Stage-1 FITS and compare every declared plane."""

    if not power_oracle_is_frozen():
        raise FileNotFoundError("CASA awp2 power oracle is not frozen")
    return compare_power_oracle_directory(_FROZEN_POWER_DIR)


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
