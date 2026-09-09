from __future__ import annotations

from pathlib import Path

import pytest

from sl1mjax.beam_validation_claims import claim_registry, refuse_full_raster_squint
from sl1mjax.beam_validation_outputs import PUBLICATION_VERSION, default_bundle_root


def test_full_raster_squint_is_refused_for_publication() -> None:
    refuse_full_raster_squint(
        {
            "estimator": "mainlobe_20pct_peak",
            "publication_estimator": True,
            "full_raster_is_publication_estimator": False,
        }
    )
    with pytest.raises(ValueError, match="20%-of-peak"):
        refuse_full_raster_squint({"estimator": "full_raster_power_centroid"})


def test_claim_registry_requires_bundle_or_loads_publication() -> None:
    # Pin the frozen v1 statuses. After the v2 refresh, default_bundle_root()
    # prefers v2 and C05 becomes a development warn, not this historical block.
    root = Path(__file__).resolve().parents[1] / "src" / "sl1mjax" / "data" / PUBLICATION_VERSION
    if not (root / "claims.json").is_file():
        root = default_bundle_root()
    if not (root / "claims.json").is_file():
        with pytest.raises(FileNotFoundError, match="draft scaffold"):
            claim_registry(bundle_root=Path("/tmp/missing-vla-bundle"))
        return
    registry = claim_registry(bundle_root=root)
    by_id = {item["id"]: item for item in registry["claims"]}
    assert by_id["C01"]["status"] == "pass"
    assert by_id["C02"]["status"] == "warn"
    assert by_id["C05"]["status"] == "blocked"
    assert by_id["C06"]["status"] == "not_run"
    assert registry["convention_search"] is False
    assert registry["spw5"] == "sealed_diagonal_frequency_transfer"
