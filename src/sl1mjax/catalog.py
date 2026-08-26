"""Radio catalogue snapshots and beam-aware out-of-field atom selection."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sl1mjax.beam import VLAPrimaryBeam
from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock


@dataclass(frozen=True)
class RadioCatalogSource:
    """One provenance-preserving radio catalogue component."""

    name: str
    ra_deg: float
    dec_deg: float
    reference_frequency_hz: float
    integrated_flux_jy: float
    catalog: str
    reference_url: str
    peak_flux_jy: float | None = None
    major_axis_arcsec: float | None = None
    minor_axis_arcsec: float | None = None
    position_angle_deg: float | None = None
    spectral_index: float | None = None
    epoch: str = ""


@dataclass(frozen=True)
class CatalogGuardAtom:
    """A catalogue source selected as an exact wide-field delta atom."""

    source: RadioCatalogSource
    l_rad: float
    m_rad: float
    offset_arcmin: float
    initial_flux_jy: float
    maximum_apparent_flux_jy: float
    maximum_beam_power: float


_OPTIONAL_FLOAT_FIELDS = (
    "peak_flux_jy",
    "major_axis_arcsec",
    "minor_axis_arcsec",
    "position_angle_deg",
    "spectral_index",
)
_CATALOG_FIELDS = (
    "name",
    "ra_deg",
    "dec_deg",
    "reference_frequency_hz",
    "integrated_flux_jy",
    "peak_flux_jy",
    "major_axis_arcsec",
    "minor_axis_arcsec",
    "position_angle_deg",
    "spectral_index",
    "catalog",
    "epoch",
    "reference_url",
)


def read_radio_catalog(path: str | Path) -> tuple[RadioCatalogSource, ...]:
    """Read the repository's stable radio-catalogue CSV schema."""

    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    sources: list[RadioCatalogSource] = []
    for row in rows:
        optional = {
            field: None if not row.get(field, "").strip() else float(row[field])
            for field in _OPTIONAL_FLOAT_FIELDS
        }
        sources.append(
            RadioCatalogSource(
                name=row["name"].strip(),
                ra_deg=float(row["ra_deg"]),
                dec_deg=float(row["dec_deg"]),
                reference_frequency_hz=float(row["reference_frequency_hz"]),
                integrated_flux_jy=float(row["integrated_flux_jy"]),
                catalog=row["catalog"].strip(),
                reference_url=row["reference_url"].strip(),
                epoch=row.get("epoch", "").strip(),
                **optional,
            )
        )
    _validate_catalog(tuple(sources))
    return tuple(sources)


def write_radio_catalog(
    path: str | Path,
    sources: tuple[RadioCatalogSource, ...],
) -> None:
    """Write a stable, reviewable catalogue snapshot as CSV."""

    _validate_catalog(sources)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CATALOG_FIELDS)
        writer.writeheader()
        for source in sources:
            writer.writerow(
                {
                    field: "" if getattr(source, field) is None else getattr(source, field)
                    for field in _CATALOG_FIELDS
                }
            )


def _validate_catalog(sources: tuple[RadioCatalogSource, ...]) -> None:
    if not sources:
        raise ValueError("radio catalogue must contain at least one source")
    names = [source.name for source in sources]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("radio catalogue source names must be non-empty and unique")
    for source in sources:
        required = (
            source.ra_deg,
            source.dec_deg,
            source.reference_frequency_hz,
            source.integrated_flux_jy,
        )
        if not np.all(np.isfinite(required)):
            raise ValueError(f"catalogue source {source.name!r} has non-finite values")
        if not 0 <= source.ra_deg < 360 or not -90 <= source.dec_deg <= 90:
            raise ValueError(f"catalogue source {source.name!r} has invalid coordinates")
        if source.reference_frequency_hz <= 0 or source.integrated_flux_jy < 0:
            raise ValueError(f"catalogue source {source.name!r} has invalid flux metadata")
        if not source.catalog or not source.reference_url:
            raise ValueError(f"catalogue source {source.name!r} lacks provenance")
        for field in _OPTIONAL_FLOAT_FIELDS:
            value = getattr(source, field)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"catalogue source {source.name!r} has invalid {field}")


def select_catalog_guard_atoms(
    sources: tuple[RadioCatalogSource, ...],
    blocks: tuple[VisibilityBlock, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    primary_beam: VLAPrimaryBeam,
    central_half_width_rad: tuple[float, float],
    minimum_apparent_flux_jy: float,
    default_spectral_index: float = -0.7,
    maximum_major_axis_arcsec: float | None = 45.0,
) -> tuple[CatalogGuardAtom, ...]:
    """Select compact outer sources that can matter through any pointing beam.

    The central exclusion is a rectangle in the mosaic tangent plane. Catalogue
    flux is extrapolated with the source spectral index when available and the
    supplied default otherwise. Selection uses maximum apparent flux over all
    pointing/channel pairs. Catalogue flux initializes the atom but remains a
    free non-negative parameter in imaging.
    """

    _validate_catalog(sources)
    if not blocks:
        raise ValueError("blocks must contain at least one visibility block")
    if len(central_half_width_rad) != 2 or np.any(~np.isfinite(central_half_width_rad)):
        raise ValueError("central_half_width_rad must contain two finite values")
    if any(value <= 0 for value in central_half_width_rad):
        raise ValueError("central half widths must be positive")
    if not np.isfinite(minimum_apparent_flux_jy) or minimum_apparent_flux_jy < 0:
        raise ValueError("minimum apparent flux must be finite and non-negative")
    if not np.isfinite(default_spectral_index):
        raise ValueError("default spectral index must be finite")
    if maximum_major_axis_arcsec is not None and (
        not np.isfinite(maximum_major_axis_arcsec) or maximum_major_axis_arcsec <= 0
    ):
        raise ValueError("maximum major axis must be finite and positive")

    ra = np.deg2rad([source.ra_deg for source in sources])
    dec = np.deg2rad([source.dec_deg for source in sources])
    reference_l, reference_m, _ = radec_to_lmn(
        mosaic_phase_centre_rad[0],
        mosaic_phase_centre_rad[1],
        ra,
        dec,
    )
    selected: list[CatalogGuardAtom] = []
    for index, source in enumerate(sources):
        if (
            abs(reference_l[index]) <= central_half_width_rad[0]
            and abs(reference_m[index]) <= central_half_width_rad[1]
        ):
            continue
        if (
            maximum_major_axis_arcsec is not None
            and source.major_axis_arcsec is not None
            and source.major_axis_arcsec > maximum_major_axis_arcsec
        ):
            continue
        spectral_index = (
            default_spectral_index if source.spectral_index is None else source.spectral_index
        )
        maximum_apparent = 0.0
        maximum_beam = 0.0
        initial_frequencies = []
        for block in blocks:
            local_l, local_m, _ = radec_to_lmn(
                block.phase_centre_rad[0],
                block.phase_centre_rad[1],
                ra[index : index + 1],
                dec[index : index + 1],
            )
            beam_power = primary_beam.power_weights(
                local_l,
                local_m,
                block.frequency_hz,
            )[0]
            intrinsic_flux = (
                source.integrated_flux_jy
                * (block.frequency_hz / source.reference_frequency_hz) ** spectral_index
            )
            maximum_apparent = max(
                maximum_apparent,
                float(np.max(beam_power * intrinsic_flux)),
            )
            maximum_beam = max(maximum_beam, float(np.max(beam_power)))
            initial_frequencies.extend(block.frequency_hz.tolist())
        if maximum_apparent < minimum_apparent_flux_jy:
            continue
        initial_frequency = float(np.median(initial_frequencies))
        selected.append(
            CatalogGuardAtom(
                source=source,
                l_rad=float(reference_l[index]),
                m_rad=float(reference_m[index]),
                offset_arcmin=float(
                    np.rad2deg(np.hypot(reference_l[index], reference_m[index])) * 60.0
                ),
                initial_flux_jy=float(
                    source.integrated_flux_jy
                    * (initial_frequency / source.reference_frequency_hz) ** spectral_index
                ),
                maximum_apparent_flux_jy=maximum_apparent,
                maximum_beam_power=maximum_beam,
            )
        )
    return tuple(
        sorted(
            selected,
            key=lambda atom: atom.maximum_apparent_flux_jy,
            reverse=True,
        )
    )
