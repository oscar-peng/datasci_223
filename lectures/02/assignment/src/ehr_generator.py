"""Generate synthetic EHR tables and code lookups for Assignment 02."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import polars as pl
from faker import Faker


@dataclass(frozen=True)
class CodebookConfig:
    icd10_prefixes: list[str]
    icd10_category_map: dict[str, str]
    hcpcs_groups: list[str]


def _icd10_with_dot(code_raw: str) -> str:
    code_raw = code_raw.strip()
    if len(code_raw) > 3:
        return f"{code_raw[:3]}.{code_raw[3:]}"
    return code_raw


def load_icd10_codes(path: Path, config: CodebookConfig) -> pl.DataFrame:
    """Load ICD-10 codes from the CDC order file and keep leaf codes."""
    rows: list[dict[str, str]] = []

    for line in path.read_text(encoding="latin-1").splitlines():
        if not line or line.startswith("#"):
            continue

        code_raw = line[6:13].strip()
        is_leaf = line[14:15].strip()
        short_desc = line[16:76].strip()

        if not code_raw or is_leaf != "1" or not short_desc:
            continue

        code = _icd10_with_dot(code_raw)
        if config.icd10_prefixes and not any(
            code.startswith(prefix) for prefix in config.icd10_prefixes
        ):
            continue

        category = "other"
        for prefix, label in config.icd10_category_map.items():
            if code.startswith(prefix):
                category = label
                break

        rows.append({"code": code, "short_desc": short_desc, "category": category})

    if not rows:
        raise ValueError("No ICD-10 codes matched filters. Check prefixes.")

    return pl.DataFrame(rows).unique(subset=["code"]).sort("code")


def _is_group_header(line: str) -> bool:
    if not line.isupper():
        return False
    if line.startswith("INCLUDE") or line.startswith("EXCLUDE"):
        return False
    if line.startswith("LIST OF") or line.startswith("THIS CODE LIST"):
        return False
    return len(line) <= 80


def load_hcpcs_codes(path: Path, config: CodebookConfig) -> pl.DataFrame:
    """Load HCPCS/CPT codes and attach the most recent group header."""
    rows: list[dict[str, str]] = []
    group = "UNSPECIFIED"

    for line in path.read_text(encoding="latin-1").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        code_match = re.match(r"^[A-Z0-9]{5}\b", stripped)
        if code_match:
            code = code_match.group(0)
            desc = stripped[len(code) :].strip().strip('"')
            if not desc:
                continue
            if config.hcpcs_groups and group not in config.hcpcs_groups:
                continue
            rows.append({"code": code, "short_desc": desc, "group": group})
            continue

        if _is_group_header(stripped):
            group = stripped.strip('\"')

    if not rows:
        raise ValueError("No HCPCS codes matched filters. Check group names.")

    return pl.DataFrame(rows).unique(subset=["code"]).sort("code")


def _weighted_choice(rng: np.random.Generator, items: list[str], weights: Iterable[float]) -> str:
    weights_array = np.array(list(weights), dtype=float)
    weights_array = weights_array / weights_array.sum()
    return rng.choice(items, p=weights_array)


def generate_sites(num_patients: int, faker: Faker, rng: np.random.Generator) -> pl.DataFrame:
    """Generate site dimension table scaled by patient count."""
    if num_patients < 500:
        site_count = 3
    elif num_patients < 2000:
        site_count = 5
    else:
        site_count = 8

    site_types = [
        ("Community Clinic", 0.45),
        ("Regional Hospital", 0.35),
        ("Specialty Center", 0.2),
    ]

    sites: list[dict[str, str | int]] = []
    for idx in range(site_count):
        site_type = _weighted_choice(rng, [t[0] for t in site_types], [t[1] for t in site_types])
        city = faker.city()
        site_id = f"SITE-{idx + 1:03d}"

        if site_type == "Community Clinic":
            panel_size = rng.integers(200, 600)
        elif site_type == "Specialty Center":
            panel_size = rng.integers(400, 900)
        else:
            panel_size = rng.integers(700, 1800)

        sites.append(
            {
                "site_id": site_id,
                "site_name": f"{city} {site_type}",
                "site_type": site_type,
                "zip_code": faker.postcode()[:5],
                "panel_size": int(panel_size),
            }
        )

    return pl.DataFrame(sites)


def generate_patients(
    num_patients: int, sites_df: pl.DataFrame, faker: Faker, rng: np.random.Generator
) -> pl.DataFrame:
    """Generate patient table with demographics and home site."""
    site_ids = sites_df["site_id"].to_list()
    site_weights = sites_df["panel_size"].to_list()

    patients: list[dict[str, str]] = []
    genders = ["Female", "Male", "Nonbinary"]

    for idx in range(num_patients):
        patient_id = f"PAT-{idx + 1:06d}"
        dob = faker.date_of_birth(minimum_age=18, maximum_age=90).isoformat()
        home_site_id = _weighted_choice(rng, site_ids, site_weights)
        gender = rng.choice(genders, p=[0.52, 0.45, 0.03])

        patients.append(
            {
                "patient_id": patient_id,
                "dob": dob,
                "gender": gender,
                "zip_code": faker.postcode()[:5],
                "home_site_id": home_site_id,
            }
        )

    return pl.DataFrame(patients)


def _random_timestamp(
    rng: np.random.Generator, start_dt: datetime, end_dt: datetime
) -> datetime:
    total_seconds = int((end_dt - start_dt).total_seconds())
    offset = int(rng.integers(0, total_seconds))
    return start_dt + timedelta(seconds=offset)


def generate_events(
    patients_df: pl.DataFrame,
    sites_df: pl.DataFrame,
    icd10_df: pl.DataFrame,
    hcpcs_df: pl.DataFrame,
    rng: np.random.Generator,
    start_date: datetime,
    end_date: datetime,
    min_events: int,
    max_events: int,
) -> pl.DataFrame:
    """Generate a long table of diagnosis and procedure events."""
    icd10_primary = icd10_df.filter(pl.col("category") == "type_2_diabetes")["code"].to_list()
    icd10_secondary = icd10_df.filter(pl.col("category") != "type_2_diabetes")["code"].to_list()
    hcpcs_codes = hcpcs_df["code"].to_list()

    site_ids = sites_df["site_id"].to_list()
    site_weights = sites_df["panel_size"].to_list()

    events: list[dict[str, str]] = []
    event_counter = 1

    for patient in patients_df.iter_rows(named=True):
        patient_id = patient["patient_id"]
        home_site_id = patient["home_site_id"]

        has_diabetes = rng.random() < 0.7
        total_events = int(rng.integers(min_events, max_events + 1))

        diabetes_events_remaining = 1 if has_diabetes else 0

        for _ in range(total_events):
            record_type = rng.choice(["ICD10", "HCPCS"], p=[0.45, 0.55])
            site_id = home_site_id if rng.random() < 0.7 else _weighted_choice(
                rng, site_ids, site_weights
            )
            event_ts = _random_timestamp(rng, start_date, end_date)

            if record_type == "ICD10":
                if diabetes_events_remaining > 0:
                    code = rng.choice(icd10_primary)
                    diabetes_events_remaining -= 1
                else:
                    code_pool = icd10_secondary or icd10_primary
                    code = rng.choice(code_pool)
            else:
                code = rng.choice(hcpcs_codes)

            events.append(
                {
                    "event_id": f"EVT-{event_ts.year}-{event_counter:08d}",
                    "patient_id": patient_id,
                    "site_id": site_id,
                    "event_ts": event_ts.isoformat(timespec="seconds"),
                    "record_type": record_type,
                    "code": code,
                }
            )
            event_counter += 1

    return pl.DataFrame(events)


def write_outputs(
    output_dir: Path,
    patients_df: pl.DataFrame,
    sites_df: pl.DataFrame,
    events_df: pl.DataFrame,
    icd10_df: pl.DataFrame,
    hcpcs_df: pl.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    patients_df.write_parquet(output_dir / "patients.parquet")
    sites_df.write_parquet(output_dir / "sites.parquet")
    events_df.write_parquet(output_dir / "events.parquet")
    icd10_df.write_parquet(output_dir / "icd10_codes.parquet")
    hcpcs_df.write_parquet(output_dir / "hcpcs_codes.parquet")
