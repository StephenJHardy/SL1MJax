"""Opt-in EVLA-C diagonal survey voltage beam.

Exact native frequency only. Does not load the packaged generic CASSBEAM
tables and does not promote full Jones.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from sl1mjax.beam_conventions import (
    CIRCULAR_P_JONES,
    CIRCULAR_STOKES,
    JONES_RECEPTOR_ORDER,
    ON_AXIS_DI_JONES_ORDER,
    BeamCalibrationState,
    require_beam_calibration_state,
)
from sl1mjax.cassbeam_beam import (
    _antenna_frame_lm,
    _apply_parallactic_jones,
    _pointing_relative_lm,
)
from sl1mjax.cassbeam_highres import EXPECTED_RASTER, HighresCassbeamCatalog
from sl1mjax.evla_c_diagonal_survey import CATALOG_ID, refuse_frozen_write
from sl1mjax.evla_c_survey_compare import SURVEY_MODEL_ID
from sl1mjax.voltage_beam import JONES_AXES, BeamCoordinates, BeamEvaluation

OPT_IN_SURVEY_BEAM = CATALOG_ID
# Native 3C391 SPW0 is 4536–4662 MHz at 2 MHz. 4599 is the arithmetic
# midpoint but is not a native channel; use the lower neighbour 4598.
IMAGING_NODE_MHZ = (4536, 4598, 4662)
NATIVE_3C391_MHZ = tuple(range(4536, 4664, 2))


def survey_catalog_digest(root: Path) -> str:
    """Stable digest of the catalog manifest checksum table."""

    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    payload = json.dumps(manifest.get("files_sha256") or {}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def voltage_beam_for_survey_catalog(
    *,
    root: Path,
    digest: str,
    catalog_id: str = CATALOG_ID,
    expected_raster: tuple[int, int, int, int] = EXPECTED_RASTER,
) -> "EvlaCSurveyVoltageBeam":
    """Named opt-in selector. Wrong id or digest is a hard refuse."""

    refuse_frozen_write(root)
    destination = Path(root)
    if catalog_id != CATALOG_ID:
        raise ValueError(f"unknown survey catalog {catalog_id!r}")
    actual = survey_catalog_digest(destination)
    if actual != str(digest):
        raise ValueError(
            f"survey catalog digest mismatch: expected {digest}, got {actual}"
        )
    catalog = HighresCassbeamCatalog(
        destination,
        expected_model_id=SURVEY_MODEL_ID,
        expected_raster=expected_raster,
    )
    return EvlaCSurveyVoltageBeam(catalog, digest=actual)


def survey_catalog_record(
    *,
    root: Path,
    scored_slots: Sequence[Mapping[str, object]] = (),
    imaging_node_mhz: Sequence[int] = IMAGING_NODE_MHZ,
    native_3c391_mhz: Sequence[int] = NATIVE_3C391_MHZ,
) -> dict[str, object]:
    """Name generated, compared, and interpolation-unvalidated frequencies."""

    digest = survey_catalog_digest(root)
    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    compared = {
        int(round(float(slot["frequency_hz"]) / 1.0e6))
        for slot in scored_slots
        if slot.get("frequency_hz") is not None
    }
    imaging = {int(item) for item in imaging_node_mhz}
    native = {int(item) for item in native_3c391_mhz}
    planes: list[dict[str, object]] = []
    for plane in manifest.get("planes") or []:
        mhz = int(plane["frequency_mhz"])
        if mhz in compared:
            support = "empirically_compared"
        elif mhz in imaging:
            support = "imaging_node_interpolation_unvalidated"
        elif mhz in native:
            support = "imaging_native_interpolation_unvalidated"
        else:
            support = "generated_uncompared"
        planes.append(
            {
                "frequency_mhz": mhz,
                "frequency_hz": mhz * 1.0e6,
                "support": support,
                "interpolation_validated": False,
            }
        )
    return {
        "catalog_id": CATALOG_ID,
        "model_id": SURVEY_MODEL_ID,
        "digest": digest,
        "interpolation_policy": "refused",
        "nearest_plane_substitution": False,
        "airy_fallback": False,
        "residual_jones_policy": "not_applied",
        "full_jones_is_gate": False,
        "production_accepted": False,
        "imaging_node_mhz": list(imaging_node_mhz),
        "native_3c391_mhz": list(native_3c391_mhz),
        "planes": planes,
    }


class EvlaCSurveyVoltageBeam:
    """Diagonal EVLA-C high-resolution beam. Exact frequency, no Airy fallback."""

    antenna_planes_from_parallactic: bool = True

    def __init__(self, catalog: HighresCassbeamCatalog, *, digest: str) -> None:
        self.catalog = catalog
        self.digest = str(digest)
        self.model_id = f"{CATALOG_ID}:{self.digest[:12]}"
        self.off_diagonal = False

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation:
        state = require_beam_calibration_state(calibration_state)
        l_off, m_off = _pointing_relative_lm(coordinates)
        antennas = (
            np.array([0], dtype=np.int32)
            if coordinates.antenna_id is None
            else np.asarray(coordinates.antenna_id)
        )
        chi = np.asarray(coordinates.parallactic_angle_rad, dtype=np.float64)
        if chi.size == 1:
            chi = np.full(antennas.size, float(chi[0]), dtype=np.float64)
        if chi.size != antennas.size:
            raise ValueError("parallactic_angle_rad must be scalar or one value per antenna")
        n_dir = int(l_off.size)
        n_chan = int(coordinates.frequency_hz.size)
        jones = np.zeros((antennas.size, n_dir, n_chan, 2, 2), dtype=np.complex128)
        valid = np.zeros((antennas.size, n_dir, n_chan), dtype=bool)
        selected_hz: list[float] = []
        for channel, frequency_hz in enumerate(coordinates.frequency_hz):
            plane = self.catalog.plane(float(frequency_hz))
            selected_hz.append(float(plane.frequency_hz))
            for antenna_index, angle in enumerate(chi):
                l_ant, m_ant = _antenna_frame_lm(l_off, m_off, float(angle))
                sample, ok = plane.lookup(l_ant, m_ant, off_diagonal=False)
                sample = _apply_parallactic_jones(sample, float(angle), state)
                jones[antenna_index, :, channel] = sample
                valid[antenna_index, :, channel] = ok
        return BeamEvaluation(
            jones=jones,
            valid=valid,
            off_diagonal_valid=valid,
            provenance={
                "model_id": self.model_id,
                "catalog_id": CATALOG_ID,
                "catalog_digest": self.digest,
                "kind": "electromagnetic",
                "support_class": "survey_exact_frequency",
                "array_average": True,
                "jones_axes": list(JONES_AXES),
                "receptors": [receptor.value for receptor in JONES_RECEPTOR_ORDER],
                "receptor_basis": "circular",
                "direction_frame": "sky_direction_cosines",
                "frequency_policy": "exact_native_plane",
                "nearest_plane_substitution": False,
                "airy_fallback": False,
                "off_diagonal": False,
                "experimental": False,
                "calibration_state": state.value,
                "on_axis_di_jones_order": ON_AXIS_DI_JONES_ORDER,
                "circular_stokes": CIRCULAR_STOKES,
                "circular_p_jones": CIRCULAR_P_JONES,
                "selected_window_hz": selected_hz,
            },
        )
