"""SL1MJax command-line workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.data.canonical import read_dataset, write_dataset
from sl1mjax.data.metadata import CalibratorRole
from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.data.synthetic import simulate_dataset
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.imaging import ImagingConfig, reconstruct
from sl1mjax.inference import InferenceConfig
from sl1mjax.output import write_products
from sl1mjax.polarization import ReceptorBasis
from sl1mjax.sky import PIXEL_MODEL_NAMES, RegularGrid, pixel_basis_from_name


def _ids(value: str | None) -> tuple[int, ...] | None:
    return None if value is None else tuple(int(item) for item in value.split(","))


def _names(value: str | None) -> tuple[str, ...] | None:
    return None if value is None else tuple(item.strip() for item in value.split(","))


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
    simulate.add_argument("--pixel-model", choices=PIXEL_MODEL_NAMES, default="delta")
    simulate.add_argument("--gaussian-sigma-pixels", type=float, default=0.5)
    simulate.add_argument("--square-width-pixels", type=float, default=1.0)

    ingest = commands.add_parser("ingest")
    ingest.add_argument("measurement_set", type=Path)
    ingest.add_argument("output", type=Path)
    ingest.add_argument("--column", default="CORRECTED_DATA")
    ingest.add_argument("--model-column")
    ingest.add_argument("--fields", help="comma-separated field IDs")
    ingest.add_argument("--field-names", help="comma-separated exact field names")
    ingest.add_argument(
        "--roles",
        help="comma-separated normalized roles",
        choices=None,
    )
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
    image.add_argument(
        "--solver",
        choices=("softplus_adam", "fista", "proximal_sgd", "hybrid"),
        default="hybrid",
        help=(
            "Flux solver (default: hybrid). Saved Adam optimizer state can be "
            "continued only with softplus_adam."
        ),
    )
    image.add_argument("--batch-size-rows", type=int, default=1024)
    image.add_argument("--random-seed", type=int, default=0)
    image.add_argument("--hybrid-sgd-fraction", type=float, default=0.5)
    image.add_argument("--kkt-tolerance", type=float, default=1e-5)
    image.add_argument("--sparsity-weight", type=float, default=1e-4)
    image.add_argument("--smoothness-weight", type=float, default=0.0)
    image.add_argument("--chunk-size", type=int, default=4096)
    image.add_argument(
        "--operator-mode",
        choices=("autodiff", "explicit"),
        default="autodiff",
    )
    image.add_argument("--visibility-tile-size", type=int, default=256)
    image.add_argument("--pixel-tile-size", type=int, default=1024)
    image.add_argument("--precision", choices=("float32", "float64"), default="float64")
    image.add_argument("--patience", type=int, default=100)
    image.add_argument("--validation-interval", type=int, default=10)
    image.add_argument("--holdout-fraction", type=float, default=0.2)
    image.add_argument("--split-seed", type=int, default=0)
    image.add_argument(
        "--split-strategy",
        choices=("uv_cell", "random_row"),
        default="uv_cell",
    )
    image.add_argument("--pixel-model", choices=PIXEL_MODEL_NAMES, default="delta")
    image.add_argument("--gaussian-sigma-pixels", type=float, default=0.5)
    image.add_argument("--square-width-pixels", type=float, default=1.0)

    pointing_audit = commands.add_parser(
        "holography-pointing-audit",
        help="Reconstruct HOLORASTER pointing from MAIN metadata and POINTING",
    )
    pointing_audit.add_argument("measurement_set", type=Path)
    pointing_audit.add_argument("--output", type=Path)
    pointing_audit.add_argument("--field", default="HOLORASTER")
    pointing_audit.add_argument(
        "--frequencies-hz",
        default="4.564e9,4.692e9",
        help="comma-separated native channel target frequencies",
    )
    pointing_audit.add_argument("--archive", type=Path)
    pointing_audit.add_argument("--column", default="POINTING_OFFSET")
    pointing_audit.add_argument(
        "--bundle",
        type=Path,
        help="write versioned audit JSON, metadata fixture, occupancy maps, and transition plots",
    )

    commissioning_copy = commands.add_parser(
        "holography-commissioning-copy",
        help="Copy SPWs 4/5 and commissioning fields to an immutable derived MS",
    )
    commissioning_copy.add_argument("source", type=Path)
    commissioning_copy.add_argument("destination", type=Path)

    visibility_audit = commands.add_parser(
        "holography-visibility-audit",
        help="Pre-calibration visibility audit of a commissioning MS",
    )
    visibility_audit.add_argument("measurement_set", type=Path)
    visibility_audit.add_argument("--output", type=Path)

    source_audit = commands.add_parser(
        "holography-source-model-audit",
        help="Test the 3C147 point model on flux/bandpass scans",
    )
    source_audit.add_argument("measurement_set", type=Path)
    source_audit.add_argument("--output", type=Path)

    graph_audit = commands.add_parser(
        "holography-c286-graph",
        help="Audit surviving 3C286 antennas, channels, and cross-hands",
    )
    graph_audit.add_argument("measurement_set", type=Path)
    graph_audit.add_argument("--output", type=Path)

    coverage = commands.add_parser(
        "holography-term-coverage",
        help="Antenna and SPW coverage for K/B/G/Kcross/Df/Xf",
    )
    coverage.add_argument("measurement_set", type=Path)
    coverage.add_argument("--product-root", type=Path, required=True)
    coverage.add_argument("--output", type=Path)

    parang = commands.add_parser(
        "holography-field-parang",
        help="Parallactic-angle span for one field (field 9 by default)",
    )
    parang.add_argument("measurement_set", type=Path)
    parang.add_argument("--field-id", type=int, default=9)
    parang.add_argument("--output", type=Path)

    structure = commands.add_parser(
        "holography-structure-audit",
        help="3C147 model ratios and closure amplitudes versus UV",
    )
    structure.add_argument("measurement_set", type=Path)
    structure.add_argument("--scans", default="51")
    structure.add_argument("--holdout-kind", default="bandpass_ablation")
    structure.add_argument("--output", type=Path)

    golden_copy = commands.add_parser(
        "holography-calibration-golden-copy",
        help="Copy a small native-resolution HOLORASTER MS for the CASA/JAX golden",
    )
    golden_copy.add_argument("source", type=Path)
    golden_copy.add_argument("destination", type=Path)
    golden_copy.add_argument("--scans", default="18,58")
    golden_copy.add_argument("--times-per-scan", type=int, default=2)

    gate = commands.add_parser(
        "holography-jones-gate",
        help="Report the Jones-recovery validation gate",
    )
    gate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "simulate":
        grid = RegularGrid(arguments.size, np.deg2rad(arguments.pixel_arcsec / 3600))
        dataset = simulate_dataset(
            grid,
            basis=ReceptorBasis(arguments.basis),
            pixel_basis=pixel_basis_from_name(
                arguments.pixel_model,
                gaussian_sigma_pixels=arguments.gaussian_sigma_pixels,
                square_width_pixels=arguments.square_width_pixels,
            ),
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
            model_column=arguments.model_column,
            fields=_ids(arguments.fields),
            field_names=_names(arguments.field_names),
            roles=(
                None
                if arguments.roles is None
                else tuple(CalibratorRole(value) for value in _names(arguments.roles) or ())
            ),
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
            validation_interval=arguments.validation_interval,
            solver=arguments.solver,
            batch_size_rows=arguments.batch_size_rows,
            random_seed=arguments.random_seed,
            hybrid_sgd_fraction=arguments.hybrid_sgd_fraction,
            kkt_tolerance=arguments.kkt_tolerance,
            operator_mode=arguments.operator_mode,
            direct_dft=DirectDFTConfig(
                visibility_chunk_size=arguments.visibility_tile_size,
                pixel_chunk_size=arguments.pixel_tile_size,
                precision=arguments.precision,
            ),
        )
        configuration = ImagingConfig(
            size=arguments.size,
            pixel_size_rad=np.deg2rad(arguments.pixel_arcsec / 3600),
            pixel_basis=pixel_basis_from_name(
                arguments.pixel_model,
                gaussian_sigma_pixels=arguments.gaussian_sigma_pixels,
                square_width_pixels=arguments.square_width_pixels,
            ),
            inference=inference,
            holdout_fraction=arguments.holdout_fraction,
            split_seed=arguments.split_seed,
            split_strategy=arguments.split_strategy,
        )
        products = write_products(
            reconstruct(dataset.blocks[arguments.block], configuration),
            arguments.output,
        )
        print("\n".join(str(path) for path in products))
        return 0
    if arguments.command == "holography-pointing-audit":
        from sl1mjax.holography_ms import audit_holography_measurement_set
        from sl1mjax.holography_pointing_audit import (
            format_pointing_reconstruction_audit,
            pointing_reconstruction_audit_as_dict,
        )

        frequencies = tuple(
            float(item) for item in arguments.frequencies_hz.split(",") if item.strip()
        )
        audit = audit_holography_measurement_set(
            arguments.measurement_set,
            field_name=arguments.field,
            target_frequency_hz=frequencies,
            selected_column=arguments.column,
            archive_path=arguments.archive,
        )
        print(format_pointing_reconstruction_audit(audit))
        if arguments.bundle is not None:
            from sl1mjax.holography_pointing_maps import write_pointing_audit_bundle

            written = write_pointing_audit_bundle(audit, arguments.bundle)
            print("\n".join(str(path) for path in written.values()))
        if arguments.output is not None:
            arguments.output.write_text(
                json.dumps(pointing_reconstruction_audit_as_dict(audit), indent=2) + "\n"
            )
            print(arguments.output)
        return 0
    if arguments.command == "holography-commissioning-copy":
        from sl1mjax.holography_commissioning import copy_commissioning_measurement_set

        selection = copy_commissioning_measurement_set(arguments.source, arguments.destination)
        print(selection.dest_ms)
        return 0
    if arguments.command == "holography-visibility-audit":
        from sl1mjax.holography_commissioning import (
            audit_commissioning_visibilities,
            write_visibility_audit,
        )

        report = audit_commissioning_visibilities(arguments.measurement_set)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "n_rows",
                        "n_channels",
                        "n_correlations",
                        "d_scan",
                        "three_c286",
                        "baseline_kind_totals",
                        "flag_fraction_by_correlation",
                        "cross_hands",
                        "notes",
                    )
                    if key in report
                },
                indent=2,
            )
        )
        if arguments.output is not None:
            write_visibility_audit(report, arguments.output)
            print(arguments.output)
        return 0
    if arguments.command == "holography-source-model-audit":
        from sl1mjax.holography_calibration import three_c147_point_model_audit, write_json

        report = three_c147_point_model_audit(arguments.measurement_set)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ("accepted_as_point", "scans", "n_rows", "holdout_antenna", "notes")
                },
                indent=2,
            )
        )
        if arguments.output is not None:
            write_json(report, arguments.output)
            print(arguments.output)
        return 0
    if arguments.command == "holography-c286-graph":
        from sl1mjax.holography_calibration import three_c286_surviving_graph, write_json

        report = three_c286_surviving_graph(arguments.measurement_set)
        print(json.dumps(report, indent=2))
        if arguments.output is not None:
            write_json(report, arguments.output)
            print(arguments.output)
        return 0
    if arguments.command == "holography-term-coverage":
        from sl1mjax.holography_calibration import term_coverage_from_casa_tables, write_json

        root = arguments.product_root
        report = term_coverage_from_casa_tables(
            {
                "K": root / "diagonal" / "K0.cal",
                "B": root / "diagonal" / "B0.cal",
                "G": root / "diagonal" / "G1.cal",
                "Kcross": root / "fullpol" / "Kcross.cal",
                "Df": root / "fullpol" / "Df.cal",
                "Xf": root / "fullpol" / "Xf.cal",
            },
            measurement_set=arguments.measurement_set,
        )
        print(
            json.dumps(
                {
                    "three_c286_absent": report["three_c286_absent"],
                    "fullpol_support": {
                        key: report["fullpol_support"][key]
                        for key in (
                            "unsupported_full_pol",
                            "invalid_identity_inheritance",
                            "needs_justified_global_x",
                            "passed",
                        )
                    },
                    "jones_recovery_blocked": report["jones_recovery_blocked"],
                },
                indent=2,
            )
        )
        if arguments.output is not None:
            write_json(report, arguments.output)
            print(arguments.output)
        return 0
    if arguments.command == "holography-field-parang":
        from sl1mjax.holography_calibration import field_parallactic_report, write_json

        report = field_parallactic_report(arguments.measurement_set, arguments.field_id)
        print(json.dumps(report, indent=2))
        if arguments.output is not None:
            write_json(report, arguments.output)
            print(arguments.output)
        return 0
    if arguments.command == "holography-structure-audit":
        from sl1mjax.holography_calibration import three_c147_structure_audit, write_json

        scans = tuple(int(item) for item in arguments.scans.split(",") if item.strip())
        report = three_c147_structure_audit(
            arguments.measurement_set,
            scans=scans,
            holdout_kind=arguments.holdout_kind,
        )
        print(
            json.dumps(
                {
                    "accepted_as_point": report["accepted_as_point"],
                    "holdout_kind": report["holdout_kind"],
                    "closure_amplitudes": report["closure_amplitudes"],
                    "notes": report["notes"],
                },
                indent=2,
            )
        )
        if arguments.output is not None:
            write_json(report, arguments.output)
            print(arguments.output)
        return 0
    if arguments.command == "holography-calibration-golden-copy":
        from sl1mjax.holography_calibration_golden import (
            copy_golden_measurement_set,
            select_golden_holoraster_rows,
        )

        scans = tuple(int(item) for item in arguments.scans.split(",") if item.strip())
        rows = select_golden_holoraster_rows(
            arguments.source,
            scans=scans,
            times_per_scan=arguments.times_per_scan,
        )
        destination = copy_golden_measurement_set(arguments.source, arguments.destination, rows)
        print(destination)
        print(f"n_rows={rows.size}")
        return 0
    if arguments.command == "holography-jones-gate":
        from sl1mjax.holography_calibration import jones_recovery_gate, write_json

        report = jones_recovery_gate()
        print(json.dumps(report, indent=2))
        if arguments.output is not None:
            write_json(report, arguments.output)
            print(arguments.output)
        return 0
    raise AssertionError(f"unhandled command {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
