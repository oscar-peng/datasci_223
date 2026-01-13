"""CLI wrapper for Assignment 02 pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Polars summary pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = pipeline.load_config(args.config)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.info("Loading data")
    encounters_lf, vitals_lf = pipeline.load_data(cfg)

    logging.info("Building summary LazyFrame")
    summary_lf = pipeline.build_summary(encounters_lf, vitals_lf, cfg)

    logging.info("Materializing outputs")
    df = pipeline.materialize(summary_lf, cfg)
    logging.info("Wrote %s rows", df.height)


if __name__ == "__main__":  # pragma: no cover
    main()
