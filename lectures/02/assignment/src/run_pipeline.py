"""CLI wrapper for Assignment 02 pipeline."""

from __future__ import annotations

import argparse
import logging

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
    data = pipeline.load_data(cfg)

    logging.info("Building summaries")
    dx_summary, hcpcs_summary = pipeline.build_summaries(data, cfg)

    logging.info("Materializing outputs")
    pipeline.materialize(dx_summary, hcpcs_summary, cfg)
    logging.info("Outputs written")


if __name__ == "__main__":  # pragma: no cover
    main()
