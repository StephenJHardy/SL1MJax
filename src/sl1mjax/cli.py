"""SL1MJax command-line workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sl1mjax.data.canonical import read_dataset, write_dataset
from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.data.synthetic import simulate_dataset
from sl1mjax.imaging import ImagingConfig, reconstruct
from sl1mjax.inference import InferenceConfig
from sl1mjax.output import write_products
from sl1mjax.polarization import ReceptorBasis
from sl1mjax.sky import RegularGrid


def _ids(value: str | None) -> tuple[int, ...] | None:
    return None if value is None else tuple(int(item) for item in value.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sl1mjax")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate")
    simulate.add_argument("output", type=Path)
    simulate.add_argument(
        "--basis",
        choices=[value.value for value in ReceptorBasis],
        default="linear",
    )
    simulate.add_argument("--size", type=int, default=16)
    simulate.add_argument("--pixel-arcsec", type=float, default=5.0)
    simulate.add_argument("--rows", type=int, default=256)
    simulate.add_argument("--channels", type=int, default=1)
    simulate.add_argument("--noise-std", type=float, default=0.0)
    simulate.add_argument("--seed", type=int, default=0)

    ingest = commands.add_parser("ingest")
    ingest.add_argument("measurement_set", type=Path)
    ingest.add_argument("output", type=Path)
    ingest.add_argument("--column", default="CORRECTED_DATA")
    ingest.add_argument("--fields", help="comma-separated field IDs")
    ingest.add_argument("--data-description-ids", help="comma-separated data-description IDs")
    ingest.add_argument("--channels", help="comma-separated zero-based channel indices")
    ingest.add_argument("--row-stride", type=int, default=1)

    image = commands.add_parser("image")
    image.add_argument("input", type=Path)
    image.add_argument("output", type=Path)
    image.add_argument("--block", type=int, default=0)
    image.add_argument("--size", type=int, default=16)
    image.add_argument("--pixel-arcsec", type=float, default=5.0)
    image.add_argument("--steps", type=int, default=500)
    image.add_argument("--learning-rate", type=float, default=0.05)
    image.add_argument("--sparsity-weight", type=float, default=1e-4)
    image.add_argument("--smoothness-weight", type=float, default=0.0)
    image.add_argument("--chunk-size", type=int, default=4096)
    image.add_argument("--patience", type=int, default=100)
    image.add_argument("--holdout-fraction", type=float, default=0.2)
    image.add_argument("--split-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "simulate":
        grid = RegularGrid(arguments.size, np.deg2rad(arguments.pixel_arcsec / 3600))
        dataset = simulate_dataset(
            grid,
            basis=ReceptorBasis(arguments.basis),
            rows=arguments.rows,
            channels=arguments.channels,
            noise_std=arguments.noise_std,
            seed=arguments.seed,
        )
        write_dataset(dataset, arguments.output)
        print(arguments.output)
        return 0
    if arguments.command == "ingest":
        dataset = extract_measurement_set(
            arguments.measurement_set,
            data_column=arguments.column,
            fields=_ids(arguments.fields),
            data_description_ids=_ids(arguments.data_description_ids),
            channels=_ids(arguments.channels),
            row_stride=arguments.row_stride,
        )
        write_dataset(dataset, arguments.output)
        print(arguments.output)
        return 0
    if arguments.command == "image":
        dataset = read_dataset(arguments.input)
        if not 0 <= arguments.block < len(dataset.blocks):
            raise ValueError(f"block index {arguments.block} is out of range")
        inference = InferenceConfig(
            steps=arguments.steps,
            learning_rate=arguments.learning_rate,
            sparsity_weight=arguments.sparsity_weight,
            smoothness_weight=arguments.smoothness_weight,
            chunk_size=arguments.chunk_size,
            patience=arguments.patience,
        )
        configuration = ImagingConfig(
            size=arguments.size,
            pixel_size_rad=np.deg2rad(arguments.pixel_arcsec / 3600),
            inference=inference,
            holdout_fraction=arguments.holdout_fraction,
            split_seed=arguments.split_seed,
        )
        products = write_products(
            reconstruct(dataset.blocks[arguments.block], configuration),
            arguments.output,
        )
        print("\n".join(str(path) for path in products))
        return 0
    raise AssertionError(f"unhandled command {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
