"""Canonical telescope-neutral visibility data."""

from sl1mjax.data.canonical import (
    VisibilityBlock,
    VisibilityDataset,
    read_dataset,
    write_dataset,
)
from sl1mjax.data.metadata import (
    AntennaRecord,
    CalibratorRole,
    FeedRecord,
    FieldRecord,
    ObservationMetadata,
    ObservationRecord,
    StateRecord,
)

__all__ = [
    "AntennaRecord",
    "CalibratorRole",
    "FeedRecord",
    "FieldRecord",
    "ObservationMetadata",
    "ObservationRecord",
    "StateRecord",
    "VisibilityBlock",
    "VisibilityDataset",
    "read_dataset",
    "write_dataset",
]
