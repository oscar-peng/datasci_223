#!/usr/bin/env python3
"""Generate synthetic EHR data for Assignment 02 testing."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from faker import Faker

from src.ehr_generator import (
    CodebookConfig,
    generate_events,
    generate_patients,
    generate_sites,
    load_hcpcs_codes,
    load_icd10_codes,
    write_outputs,
)


CODEBOOK_CONFIG = CodebookConfig(
    icd10_prefixes=["E11", "I10", "E78", "N18", "E66"],
    icd10_category_map={
        "E11": "type_2_diabetes",
        "I10": "hypertension",
        "E78": "hyperlipidemia",
        "N18": "ckd",
        "E66": "obesity",
    },
    hcpcs_groups=[
        "CLINICAL LABORATORY SERVICES",
        "RADIOLOGY AND CERTAIN OTHER IMAGING SERVICES",
        "PHYSICAL THERAPY, OCCUPATIONAL THERAPY, AND OUTPATIENT SPEECH-LANGUAGE PATHOLOGY SERVICES",
    ],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Assignment 02 test data")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset size",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def generate_test_data(
    size: str = "small", output_dir: Path = Path("data"), seed: int = 42
) -> None:
    size_config = {
        "small": {"patients": 200, "events": (8, 15)},
        "medium": {"patients": 1000, "events": (10, 18)},
        "large": {"patients": 5000, "events": (12, 22)},
    }

    if size not in size_config:
        raise ValueError(f"Unsupported size: {size}")

    config = size_config[size]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    faker = Faker()
    faker.seed_instance(seed)
    rng = np.random.default_rng(seed)

    refs_dir = Path(__file__).parent / "refs" / "raw"
    icd10_path = refs_dir / "icd10cm-order-2026.txt"
    hcpcs_path = refs_dir / "2026_DHS_Code_List_Addendum_12_01_2025.txt"

    logging.info("Loading codebooks")
    icd10_df = load_icd10_codes(icd10_path, CODEBOOK_CONFIG)
    hcpcs_df = load_hcpcs_codes(hcpcs_path, CODEBOOK_CONFIG)

    logging.info("Generating sites")
    sites_df = generate_sites(config["patients"], faker, rng)

    logging.info("Generating patients")
    patients_df = generate_patients(config["patients"], sites_df, faker, rng)

    logging.info("Generating events")
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    events_df = generate_events(
        patients_df,
        sites_df,
        icd10_df,
        hcpcs_df,
        rng,
        start_date,
        end_date,
        min_events=config["events"][0],
        max_events=config["events"][1],
    )

    write_outputs(output_dir, patients_df, sites_df, events_df, icd10_df, hcpcs_df)

    logging.info("Wrote %s patients", patients_df.height)
    logging.info("Wrote %s sites", sites_df.height)
    logging.info("Wrote %s events", events_df.height)
    logging.info("Output directory: %s", output_dir)


def main() -> None:
    args = parse_args()
    generate_test_data(args.size, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
