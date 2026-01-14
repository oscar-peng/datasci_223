"""Polars pipeline helpers for Assignment 02."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import polars as pl
import yaml


def load_config(path: str | Path) -> dict:
    """Load YAML configuration."""
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text())


def _parse_datetime(column: str) -> pl.Expr:
    return pl.col(column).str.strptime(pl.Datetime, strict=False)


def load_data(cfg: dict) -> Dict[str, pl.LazyFrame]:
    """Return LazyFrames for patients, sites, events, and code lookups."""
    data_cfg = cfg["data"]
    data = {
        "patients": pl.scan_parquet(data_cfg["patients_path"]),
        "sites": pl.scan_parquet(data_cfg["sites_path"]),
        "events": pl.scan_parquet(data_cfg["events_path"]).with_columns(
            _parse_datetime("event_ts").alias("event_ts")
        ),
        "icd10": pl.scan_parquet(data_cfg["icd10_path"]),
        "hcpcs": pl.scan_parquet(data_cfg["hcpcs_path"]),
    }
    return data


def build_summaries(data: Dict[str, pl.LazyFrame], cfg: dict) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Construct lazy summaries for diagnosis prevalence and HCPCS counts."""
    start_dt = datetime.fromisoformat(cfg["data"]["start_date"])
    prefix = cfg["data"]["diabetes_prefix"]

    events = data["events"].filter(pl.col("event_ts") >= pl.lit(start_dt))
    dx_events = events.filter(pl.col("record_type") == "ICD10")
    proc_events = events.filter(pl.col("record_type") == "HCPCS")

    dx_diabetes = dx_events.filter(pl.col("code").str.starts_with(prefix))

    patients_by_site = (
        events.select(["site_id", "patient_id"]).unique().group_by("site_id").agg(
            pl.len().alias("patients_seen")
        )
    )

    dx_patients_by_site = (
        dx_diabetes.select(["site_id", "patient_id"]).unique().group_by("site_id").agg(
            pl.len().alias("diabetes_patients")
        )
    )

    dx_summary = (
        patients_by_site
        .join(dx_patients_by_site, on="site_id", how="left")
        .with_columns(pl.col("diabetes_patients").fill_null(0))
        .with_columns(
            (pl.col("diabetes_patients") / pl.col("patients_seen"))
            .round(3)
            .alias("diabetes_prevalence")
        )
        .join(data["sites"].select(["site_id", "site_name", "site_type"]), on="site_id", how="left")
        .select(
            [
                "site_id",
                "site_name",
                "site_type",
                "patients_seen",
                "diabetes_patients",
                "diabetes_prevalence",
            ]
        )
        .sort("diabetes_prevalence", descending=True)
    )

    hcpcs_summary = (
        proc_events
        .join(data["hcpcs"].select(["code", "group"]), on="code", how="left")
        .group_by(["site_id", "group"])
        .agg(pl.len().alias("procedure_count"))
        .join(data["sites"].select(["site_id", "site_name"]), on="site_id", how="left")
        .select(["site_id", "site_name", "group", "procedure_count"])
        .sort(["site_id", "procedure_count"], descending=[False, True])
    )

    return dx_summary, hcpcs_summary


def materialize(dx_summary: pl.LazyFrame, hcpcs_summary: pl.LazyFrame, cfg: dict) -> None:
    """Execute summaries and write artifacts."""
    outputs_cfg = cfg["outputs"]

    dx_df = dx_summary.collect(engine="streaming")
    hcpcs_df = hcpcs_summary.collect(engine="streaming")

    output_paths = [
        Path(outputs_cfg["dx_summary_parquet"]),
        Path(outputs_cfg["dx_summary_csv"]),
        Path(outputs_cfg["hcpcs_summary_parquet"]),
        Path(outputs_cfg["hcpcs_summary_csv"]),
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    dx_df.write_parquet(outputs_cfg["dx_summary_parquet"])
    dx_df.write_csv(outputs_cfg["dx_summary_csv"])

    hcpcs_df.write_parquet(outputs_cfg["hcpcs_summary_parquet"])
    hcpcs_df.write_csv(outputs_cfg["hcpcs_summary_csv"])
