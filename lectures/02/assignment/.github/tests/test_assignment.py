"""Autograder tests for Assignment 02: Polars EHR event logs."""

from pathlib import Path

import polars as pl
import pytest
import yaml


@pytest.fixture(scope="module")
def config():
    config_path = Path("config.yaml")
    assert config_path.exists(), "config.yaml not found"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_data_files_exist(config):
    data_paths = [
        config["data"]["patients_path"],
        config["data"]["sites_path"],
        config["data"]["events_path"],
        config["data"]["icd10_path"],
        config["data"]["hcpcs_path"],
    ]
    missing = [path for path in data_paths if not Path(path).exists()]
    assert not missing, f"Missing data files: {missing}"


def test_outputs_exist(config):
    output_paths = [
        config["outputs"]["dx_summary_parquet"],
        config["outputs"]["dx_summary_csv"],
        config["outputs"]["hcpcs_summary_parquet"],
        config["outputs"]["hcpcs_summary_csv"],
    ]
    missing = [path for path in output_paths if not Path(path).exists()]
    assert not missing, f"Missing output files: {missing}"


def test_dx_summary_schema(config):
    df = pl.read_parquet(config["outputs"]["dx_summary_parquet"])
    required_columns = {
        "site_id",
        "site_name",
        "site_type",
        "patients_seen",
        "diabetes_patients",
        "diabetes_prevalence",
    }
    missing = required_columns - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"


def test_hcpcs_summary_schema(config):
    df = pl.read_parquet(config["outputs"]["hcpcs_summary_parquet"])
    required_columns = {"site_id", "site_name", "group", "procedure_count"}
    missing = required_columns - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"


def test_dx_summary_values(config):
    df = pl.read_parquet(config["outputs"]["dx_summary_parquet"])
    assert df.height > 0, "Diagnosis summary is empty"

    assert df["patients_seen"].min() >= config["processing"]["min_patients_per_site"], (
        "patients_seen below expected minimum"
    )
    assert df["diabetes_patients"].min() >= 0, "diabetes_patients should be non-negative"

    prevalence = df["diabetes_prevalence"]
    assert prevalence.min() >= 0, "diabetes_prevalence should be >= 0"
    assert prevalence.max() <= 1, "diabetes_prevalence should be <= 1"


def test_hcpcs_summary_values(config):
    df = pl.read_parquet(config["outputs"]["hcpcs_summary_parquet"])
    assert df.height > 0, "HCPCS summary is empty"
    assert df["procedure_count"].min() > 0, "procedure_count should be positive"
    assert df["group"].null_count() == 0, "HCPCS group should not be null"


def test_parquet_csv_match(config):
    dx_parquet = pl.read_parquet(config["outputs"]["dx_summary_parquet"])
    dx_csv = pl.read_csv(config["outputs"]["dx_summary_csv"])
    assert dx_parquet.height == dx_csv.height, "DX parquet/csv row counts differ"

    hcpcs_parquet = pl.read_parquet(config["outputs"]["hcpcs_summary_parquet"])
    hcpcs_csv = pl.read_csv(config["outputs"]["hcpcs_summary_csv"])
    assert hcpcs_parquet.height == hcpcs_csv.height, "HCPCS parquet/csv row counts differ"
