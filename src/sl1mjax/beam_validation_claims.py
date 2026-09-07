"""Claim registry for the VLA C-band beam validation notebook.

Statuses come from the compact bundle. The notebook displays them. It does
not freeze a production beam or promote full Jones.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from sl1mjax.beam_validation_outputs import default_bundle_root

CLAIM_VERSION = "vla_c_band_beam_validation_v1"
CLAIM_STATUSES = ("pass", "warn", "fail", "blocked", "not_run")


def refuse_full_raster_squint(record: Mapping[str, object]) -> None:
    """Publication bundles must not treat the full-raster centroid as the answer."""

    if record.get("estimator") != "mainlobe_20pct_peak":
        raise ValueError("publication squint must use the 20%-of-peak main-lobe estimator")
    if record.get("full_raster_is_publication_estimator") in {True}:
        raise ValueError("full-raster squint is not a publication estimator")
    if record.get("publication_estimator") not in {True}:
        raise ValueError("squint record is not marked as the publication estimator")


def claim_registry(
    overrides: Mapping[str, Mapping[str, object]] | None = None,
    *,
    bundle_root: Path | None = None,
) -> dict[str, object]:
    """Load claims from the publication bundle when present."""

    root = Path(bundle_root) if bundle_root is not None else default_bundle_root()
    claims_path = root / "claims.json"
    if not claims_path.is_file():
        raise FileNotFoundError(
            "validation bundle claims.json is missing; this notebook is a draft "
            "scaffold until the compact bundle is built"
        )
    from sl1mjax.beam_validation_outputs import load_json

    registry = load_json(claims_path)
    extra = dict(overrides or {})
    claims = []
    for item in registry.get("claims") or ():
        row = dict(item)
        if row.get("id") in extra:
            row.update(extra[str(row["id"])])
        claims.append(row)
    registry = dict(registry)
    registry["claims"] = claims
    return registry
