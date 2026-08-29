"""VLA C-band beam conventions and reference-artifact inventory.

Phase 1 records the scalar C-band inventory and internal coordinate
conventions. It does not evaluate a voltage Jones, introduce a cache, or
freeze a physical polarisation-orientation oracle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam import VLA_SQUINT_FWHM_FRACTION, VLABeamCatalog
from sl1mjax.polarization import Receptor

PERLEY2016_CATALOG_VERSION = "evla-memo-195-table-5"
PERLEY2016_SOURCE_URL = "https://library.nrao.edu/public/memos/evla/EVLAM_195.pdf"
PERLEY2016_MINIMUM_VALID_POWER = 0.05
# 3C391 C-band SPW0 used in the polarisation validation path.
OBSERVATION_3C391_FREQUENCY_HZ = (4.536e9, 4.662e9)

# EVLA Memo 195 §6.1: total RCP–LCP peak separation, not one receptor's offset.
EVLA195_TOTAL_SQUINT_ARCMIN_GHZ = 2.4
# Cotton & Uson 2008; Napier & Gustincic 1977; NVSS (Condon et al. 1998).
COTTON2008_TOTAL_SQUINT_FWHM_FRACTION = 0.06
NAPIER1977_TOTAL_SQUINT_FWHM_FRACTION = 0.053
NVSS_TOTAL_SQUINT_FWHM_FRACTION = 0.055

ON_AXIS_DI_JONES_ORDER = "GKB Kcross D X P"
CIRCULAR_P_JONES = "diag(exp(-i chi), exp(+i chi))"
CIRCULAR_STOKES = "RR=I+V, LL=I-V, RL=Q+iU, LR=Q-iU"
JONES_RECEPTOR_ORDER = (Receptor.R, Receptor.L)
ANALYTIC_SQUINT_RECEPTOR_HALF_OFFSET_FWHM = VLA_SQUINT_FWHM_FRACTION

CASSBEAM_UNPINNED_REQUIREMENTS = (
    "cassbeam source or binary version",
    "C-band feed and optical parameter file",
    "transmit-to-receive conversion",
    "native Jones basis and element ordering",
    "direction-axis orientation",
    "numerical settings",
    "input or output hash",
)
HOLOGRAPHY_UNPINNED_REQUIREMENTS = (
    "acquired holography artifact path",
    "immutable checksum",
    "correlation or Stokes convention in the files",
    "antenna-average recipe",
)

BeamKind = Literal["analytic", "empirical", "electromagnetic"]
SquintQuantity = Literal["receptor_half_offset", "total_rcp_lcp_separation"]


class BeamCalibrationState(StrEnum):
    """Declared on-axis calibration state of a visibility block.

    Unknown identifiers are rejected. The beam must not guess a Jones order.
    """

    CASA_PARANG_TRUE = "casa_parang_true"
    UNCALIBRATED = "uncalibrated"


class PerleyFrequencyPolicy(StrEnum):
    """How to choose Memo 195 coefficients as a function of frequency.

    ``casa_nearest`` matches CASA ``PBMath1DEVLA.nearestVPArray``. It is a
    CASA-parity oracle, not a spectral-discovery model. Interpolation is a
    separate policy and is not implemented.
    """

    CASA_NEAREST = "casa_nearest"
    INTERPOLATED = "interpolated"


class ConventionLock(StrEnum):
    """Whether a convention is locked internally or physically verified."""

    INTERNAL = "internal"
    PHYSICALLY_UNVERIFIED = "physically_unverified"
    PHYSICALLY_VERIFIED = "physically_verified"


PARALLACTIC_ANGLE_SIGN_LOCK = ConventionLock.INTERNAL
ANTENNA_FRAME_POLARIZATION_LOCK = ConventionLock.PHYSICALLY_UNVERIFIED


@dataclass(frozen=True)
class BeamReferenceArtifact:
    """One inventoried C-band or near-C-band beam artifact."""

    artifact_id: str
    kind: BeamKind
    band: str
    quantity: str
    frequency_hz: tuple[float, float] | None
    direction_support: str
    antenna_model: str
    epoch: str
    usable_for_cband: bool
    frozen_reference: bool
    notes: str
    unpinned_requirements: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Perley2016CBandWindow:
    """One EVLA Memo 195 Table 5 Stokes-I polynomial window."""

    frequency_hz: float
    a2: float
    a4: float
    a6: float
    hwhm_arcmin_ghz: float

    def normalized_radius_at_power(self, power: float) -> float:
        """Return the smallest positive ``r = θ_arcmin ν_GHz`` at ``power``."""

        if not np.isfinite(power) or power <= 0.0 or power > 1.0:
            raise ValueError("power must be in (0, 1]")
        roots = np.roots((self.a6, self.a4, self.a2, 1.0 - power))
        positive = [
            float(np.real(root))
            for root in roots
            if np.isclose(np.imag(root), 0.0, atol=1e-12) and np.real(root) > 0.0
        ]
        if not positive:
            raise ValueError(f"no positive radius where the polynomial equals {power}")
        return float(np.sqrt(min(positive)))

    def support_radius_arcmin(self, frequency_hz: float) -> float:
        """Return the 5% power radius at ``frequency_hz``."""

        _require_finite_positive_frequency(frequency_hz)
        return self.normalized_radius_at_power(PERLEY2016_MINIMUM_VALID_POWER) / (
            frequency_hz / 1.0e9
        )


def require_beam_calibration_state(
    state: BeamCalibrationState | str,
) -> BeamCalibrationState:
    """Accept a known calibration-state identifier or refuse to guess."""

    try:
        return BeamCalibrationState(state)
    except ValueError as error:
        raise ValueError(
            f"unknown beam calibration state {state!r}; refuse to guess Jones order"
        ) from error


def beam_requires_identity_on_axis(state: BeamCalibrationState | str) -> bool:
    """Return whether the voltage beam must satisfy ``E(0)=I``.

    After CASA ``applycal(parang=True)`` the on-axis product is
    ``J = J_GKB J_Kcross J_D J_X J_P``. A holographic Jones that still
    contains on-axis G, D, X, or P would double-apply those terms.
    """

    return require_beam_calibration_state(state) is BeamCalibrationState.CASA_PARANG_TRUE


def analytic_squint_is_evidence_grade() -> bool:
    """The dormant Airy squint is not evidence-grade until the audit closes."""

    return False


def analytic_squint_quantity() -> SquintQuantity:
    """Name the quantity stored as ``VLA_SQUINT_FWHM_FRACTION``."""

    return "receptor_half_offset"


def antenna_frame_polarization_is_physically_verified() -> bool:
    """Feed-frame PA and R/L sign have no external C-band oracle yet."""

    return ANTENNA_FRAME_POLARIZATION_LOCK is ConventionLock.PHYSICALLY_VERIFIED


def cband_reference_artifacts() -> tuple[BeamReferenceArtifact, ...]:
    """Return the Phase 1 C-band reference inventory."""

    return (
        BeamReferenceArtifact(
            artifact_id="sl1mjax_airy",
            kind="analytic",
            band="any",
            quantity="stokes_i_power",
            frequency_hz=None,
            direction_support=(
                "Blocked Airy with CASA setpbairy geometry; hard cutoff "
                "0.8 deg at 1 GHz, scaled as 1/frequency. Extended support "
                "is an ablation, not a measured far sidelobe."
            ),
            antenna_model="Ideal 25 m dish, 2.5 m circular blockage, array-average",
            epoch="geometric",
            usable_for_cband=True,
            frozen_reference=True,
            notes="Current 3C391 production path. Squint off.",
        ),
        BeamReferenceArtifact(
            artifact_id="perley2016_cband_stokes_i",
            kind="empirical",
            band="C",
            quantity="stokes_i_power",
            frequency_hz=(4.052e9, 7.948e9),
            direction_support=(
                "Azimuthally averaged power to the 5% level on the oversampled "
                "central grid. Sparse grid to the 4th null is archived, not fitted."
            ),
            antenna_model="Array average of 14 well-behaved antennas versus ea28",
            epoch="2015-2016 Jansky VLA holography",
            usable_for_cband=True,
            frozen_reference=True,
            notes=(
                "EVLA Memo 195 Table 5. CASA PBMath1DEVLA default since 5.0. "
                "Frequency-dependent coefficients are required at C-band. "
                "casa_nearest is a CASA-parity oracle, not a spectral model."
            ),
        ),
        BeamReferenceArtifact(
            artifact_id="casa_setpbairy",
            kind="analytic",
            band="any",
            quantity="stokes_i_power",
            frequency_hz=None,
            direction_support="Same dish, blockage, and 0.8 deg @ 1 GHz cutoff as sl1mjax_airy",
            antenna_model="CASA vpmanager.setpbairy VLA defaults",
            epoch="CASA voltage-pattern catalog",
            usable_for_cband=True,
            frozen_reference=True,
            notes="Geometry oracle for the current Airy path, not a C-band shape model.",
        ),
        BeamReferenceArtifact(
            artifact_id="nrao_gaussian_42_over_nu",
            kind="analytic",
            band="L-Q",
            quantity="stokes_i_fwhm",
            frequency_hz=(1.0e9, 5.0e10),
            direction_support="FWHM only; no sidelobes",
            antenna_model="Approximate array-average",
            epoch="VLA Observational Status Summary",
            usable_for_cband=True,
            frozen_reference=True,
            notes="θ_FWHM = 42/ν_GHz arcmin. Width check, not a predict model.",
        ),
        BeamReferenceArtifact(
            artifact_id="cassbeam_go",
            kind="electromagnetic",
            band="any",
            quantity="voltage_jones_2x2",
            frequency_hz=None,
            direction_support="Geometric-optics far-field including feed offset and struts",
            antenna_model="Parameterized VLA Cassegrain; may be array-average or per-antenna",
            epoch="Brisken 2003; CASA CASSBEAM adaptation",
            usable_for_cband=True,
            frozen_reference=False,
            unpinned_requirements=CASSBEAM_UNPINNED_REQUIREMENTS,
            notes=(
                "Identified full-Jones export route, not a frozen reference. "
                "Jagannathan et al. 2017 note that L/S/C need diffraction beyond GO."
            ),
        ),
        BeamReferenceArtifact(
            artifact_id="perley2016_cband_holography_grids",
            kind="empirical",
            band="C",
            quantity="correlation_grids_rr_rl_lr_ll",
            frequency_hz=(4.052e9, 7.948e9),
            direction_support=(
                "Central 17×17 oversampled grid plus sparse 23×23 grid to the 4th null"
            ),
            antenna_model="Calibrated holography; UVHOL averages of selected antennas",
            epoch="2015-2016 Jansky VLA holography",
            usable_for_cband=True,
            frozen_reference=False,
            unpinned_requirements=HOLOGRAPHY_UNPINNED_REQUIREMENTS,
            notes=(
                "Identified empirical full-polarisation route if reconstructed. "
                "Archive names CHOLO FITAB/FITTP and C-BEAM-ffffpp are recorded; "
                "no acquired artifact or checksum is pinned."
            ),
        ),
        BeamReferenceArtifact(
            artifact_id="jagannathan2017_asolver",
            kind="electromagnetic",
            band="L/S/C",
            quantity="voltage_jones_2x2",
            frequency_hz=(1.0e9, 8.0e9),
            direction_support="CASSBEAM far-field fitted to holography",
            antenna_model="CASSBEAM parameters perturbed against holography",
            epoch="Jagannathan et al. 2017",
            usable_for_cband=False,
            frozen_reference=False,
            notes="Methodology paper. No public ready-to-import C-band coefficient table.",
        ),
        BeamReferenceArtifact(
            artifact_id="jagannathan2021_atoz_plumber",
            kind="empirical",
            band="S",
            quantity="zernike_aperture_jones",
            frequency_hz=(2.0e9, 4.0e9),
            direction_support="Zernike AIP reconstructed to a full Mueller primary beam",
            antenna_model="Holography-derived array model",
            epoch="Jagannathan et al. 2021; ARDG plumber",
            usable_for_cband=False,
            frozen_reference=False,
            notes="Paper and plumber ship VLA S-band and MeerKAT L-band, not C-band.",
        ),
        BeamReferenceArtifact(
            artifact_id="iheanetu2019_lband",
            kind="empirical",
            band="L",
            quantity="holographic_jones",
            frequency_hz=(1.0e9, 2.0e9),
            direction_support="L-band holography",
            antenna_model="VLA L-band",
            epoch="Iheanetu et al. 2019",
            usable_for_cband=False,
            frozen_reference=False,
            notes="Must not be relabelled as a C-band beam.",
        ),
        BeamReferenceArtifact(
            artifact_id="sl1mjax_analytic_squint",
            kind="analytic",
            band="any",
            quantity="receptor_power_rr_ll",
            frequency_hz=None,
            direction_support="Opposite RR/LL displacements of the scalar power beam",
            antenna_model="Same as sl1mjax_airy",
            epoch="geometric",
            usable_for_cband=False,
            frozen_reference=False,
            notes=(
                "Stores a receptor half-offset of 0.06 FWHM, so the total "
                "RCP–LCP separation is 0.12 FWHM. Published totals are ~0.05–0.06 "
                "FWHM. Disabled until the C-band magnitude, feed PA, and sign "
                "are audited against CASA or Memo 195."
            ),
        ),
    )


def artifact_by_id(artifact_id: str) -> BeamReferenceArtifact:
    """Return one inventoried artifact or raise ``KeyError``."""

    for artifact in cband_reference_artifacts():
        if artifact.artifact_id == artifact_id:
            return artifact
    raise KeyError(artifact_id)


@lru_cache(maxsize=1)
def load_perley2016_cband_windows() -> tuple[Perley2016CBandWindow, ...]:
    """Load the pinned EVLA Memo 195 Table 5 Stokes-I coefficients."""

    path = Path(__file__).with_name("data") / "vla_cband_perley2016.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("catalog_version") != PERLEY2016_CATALOG_VERSION:
        raise ValueError("unexpected Perley 2016 C-band catalog version")
    if float(payload.get("minimum_valid_power", -1.0)) != PERLEY2016_MINIMUM_VALID_POWER:
        raise ValueError("unexpected Perley 2016 minimum valid power")
    windows = []
    for row in payload["windows"]:
        windows.append(
            Perley2016CBandWindow(
                frequency_hz=float(row["frequency_mhz"]) * 1.0e6,
                a2=float(row["a2"]),
                a4=float(row["a4"]),
                a6=float(row["a6"]),
                hwhm_arcmin_ghz=float(row["hwhm_arcmin_ghz"]),
            )
        )
    return tuple(windows)


def perley2016_frequency_support_hz() -> tuple[float, float]:
    """Return the closed frequency interval covered by Table 5."""

    windows = load_perley2016_cband_windows()
    return windows[0].frequency_hz, windows[-1].frequency_hz


def casa_nearest_switch_frequency_hz(
    lower: Perley2016CBandWindow, upper: Perley2016CBandWindow
) -> float:
    """Return the midpoint where ``casa_nearest`` switches windows."""

    if upper.frequency_hz <= lower.frequency_hz:
        raise ValueError("upper window must be higher in frequency than lower")
    return 0.5 * (lower.frequency_hz + upper.frequency_hz)


def _window_at_mhz(frequency_mhz: int) -> Perley2016CBandWindow:
    target = frequency_mhz * 1.0e6
    for window in load_perley2016_cband_windows():
        if abs(window.frequency_hz - target) < 0.5e6:
            return window
    raise KeyError(frequency_mhz)


def observation_3c391_crosses_casa_nearest_switch() -> bool:
    """True if the 3C391 C-band span includes the 4564/4692 MHz midpoint."""

    switch = casa_nearest_switch_frequency_hz(_window_at_mhz(4564), _window_at_mhz(4692))
    start, stop = OBSERVATION_3C391_FREQUENCY_HZ
    return start < switch < stop


def perley2016_stokes_i_validity(
    offset_arcmin: ArrayLike,
    frequency_hz: ArrayLike,
    window: Perley2016CBandWindow,
) -> NDArray[np.bool_]:
    """Return where the Memo 195 polynomial is inside its 5% fit domain."""

    offset = np.asarray(offset_arcmin, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    radius = offset * (frequency / 1.0e9)
    support = window.normalized_radius_at_power(PERLEY2016_MINIMUM_VALID_POWER)
    return np.asarray(
        np.isfinite(offset)
        & np.isfinite(frequency)
        & (frequency > 0.0)
        & np.isfinite(radius)
        & (radius >= 0.0)
        & (radius <= support),
        dtype=np.bool_,
    )


def perley2016_stokes_i_power(
    offset_arcmin: ArrayLike,
    frequency_hz: ArrayLike,
    window: Perley2016CBandWindow,
    *,
    require_valid: bool = True,
) -> NDArray[np.float64]:
    """Evaluate the inventoried Memo 195 Stokes-I polynomial.

    Samples outside the 5% power radius, or at non-finite frequencies, are
    unsupported. The default is fail-closed. This checks the catalog, not a
    voltage-beam evaluator.
    """

    offset = np.asarray(offset_arcmin, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    valid = perley2016_stokes_i_validity(offset, frequency, window)
    if require_valid and not np.all(valid):
        raise ValueError("Perley 2016 polynomial is unsupported at the requested samples")
    radius = offset * (frequency / 1.0e9)
    radius_sq = np.square(radius)
    power = np.asarray(
        1.0 + window.a2 * radius_sq + window.a4 * np.square(radius_sq) + window.a6 * radius_sq**3,
        dtype=np.float64,
    )
    return np.asarray(np.where(valid, power, np.nan), dtype=np.float64)


def select_perley2016_cband_window(
    frequency_hz: float,
    *,
    policy: PerleyFrequencyPolicy,
) -> Perley2016CBandWindow:
    """Select Table 5 coefficients under an explicit frequency policy."""

    if policy is PerleyFrequencyPolicy.INTERPOLATED:
        raise ValueError(
            "Perley frequency interpolation is not implemented; refuse to invent a spectrum"
        )
    if policy is not PerleyFrequencyPolicy.CASA_NEAREST:
        raise ValueError(f"unknown Perley frequency policy {policy!r}")
    _require_frequency_inside_perley_support(frequency_hz)
    windows = load_perley2016_cband_windows()
    return min(windows, key=lambda window: abs(window.frequency_hz - frequency_hz))


def nearest_perley2016_cband_window(frequency_hz: float) -> Perley2016CBandWindow:
    """CASA-parity nearest-window selection inside the Table 5 band."""

    return select_perley2016_cband_window(
        frequency_hz, policy=PerleyFrequencyPolicy.CASA_NEAREST
    )


def gaussian_fwhm_rad(frequency_hz: ArrayLike) -> NDArray[np.float64]:
    """Return the catalog Gaussian FWHM used by the current Airy path."""

    return VLABeamCatalog().gaussian_fwhm_rad(frequency_hz)


def evla195_total_squint_rad(frequency_hz: ArrayLike) -> NDArray[np.float64]:
    """Return the Memo 195 total RCP–LCP separation."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(frequency <= 0):
        raise ValueError("frequency must be positive")
    return np.asarray(
        np.deg2rad(EVLA195_TOTAL_SQUINT_ARCMIN_GHZ / 60.0) * (1.0e9 / frequency),
        dtype=np.float64,
    )


def evla195_receptor_half_offset_rad(frequency_hz: ArrayLike) -> NDArray[np.float64]:
    """Return half of the Memo 195 total RCP–LCP separation."""

    return 0.5 * evla195_total_squint_rad(frequency_hz)


def current_analytic_receptor_half_offset_rad(
    frequency_hz: ArrayLike,
) -> NDArray[np.float64]:
    """Return the unused Airy-path receptor offset (0.06 of Gaussian FWHM)."""

    return VLABeamCatalog().squint_offset_rad(frequency_hz)


def current_analytic_total_squint_rad(frequency_hz: ArrayLike) -> NDArray[np.float64]:
    """Return the unused Airy-path total RCP–LCP separation (0.12 of FWHM)."""

    return 2.0 * current_analytic_receptor_half_offset_rad(frequency_hz)


def current_versus_evla195_total_squint_ratio(frequency_hz: float) -> float:
    """Return how many times larger the unused Airy total is than Memo 195."""

    return float(
        current_analytic_total_squint_rad(frequency_hz)
        / evla195_total_squint_rad(frequency_hz)
    )


def nrao_gaussian_fwhm_arcmin(frequency_hz: ArrayLike) -> NDArray[np.float64]:
    """Return the Observational Status Summary FWHM ``42/ν_GHz`` arcmin."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(frequency <= 0):
        raise ValueError("frequency must be positive")
    return np.asarray(42.0 * (1.0e9 / frequency), dtype=np.float64)


def sky_east_is_positive_l() -> bool:
    """Sky-frame ``l`` increases toward increasing right ascension."""

    return True


def sky_north_is_positive_m() -> bool:
    """Sky-frame ``m`` increases toward increasing declination."""

    return True


def _require_finite_positive_frequency(frequency_hz: float) -> None:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency must be a finite positive value")


def _require_frequency_inside_perley_support(frequency_hz: float) -> None:
    _require_finite_positive_frequency(frequency_hz)
    low, high = perley2016_frequency_support_hz()
    if frequency_hz < low or frequency_hz > high:
        raise ValueError(
            f"frequency {frequency_hz} Hz is outside Perley 2016 C-band support "
            f"[{low}, {high}] Hz"
        )
