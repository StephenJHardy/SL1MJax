#!/usr/bin/env python3
"""Phase 5 maps, radial suitability, and null intervals from survey exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from sl1mjax.evla_c_survey_plots import write_plot_products


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("config/3c391_radio_guard_catalog.json"),
    )
    arguments = parser.parse_args()
    result = write_plot_products(
        arguments.channel_dir,
        arguments.output_dir,
        catalog_path=arguments.catalog,
    )
    print(result)


if __name__ == "__main__":
    main()
