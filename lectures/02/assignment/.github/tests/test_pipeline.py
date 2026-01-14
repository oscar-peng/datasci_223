from pathlib import Path

import polars as pl
import pytest

from src import pipeline


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    cfg = pipeline.load_config("config.yaml")

    tmp_outputs = tmp_path_factory.mktemp("outputs")
    cfg["outputs"]["summary_parquet"] = str(tmp_outputs / "summary.parquet")
    cfg["outputs"]["summary_csv"] = str(tmp_outputs / "summary.csv")
    cfg["outputs"]["chart_png"] = str(tmp_outputs / "summary.png")

    tmp_data = tmp_path_factory.mktemp("data")
    encounters_dir = tmp_data / "encounters"
    vitals_dir = tmp_data / "vitals"
    encounters_dir.mkdir()
    vitals_dir.mkdir()

    encounters = pl.DataFrame(
        {
            "encounter_id": ["ENC-0001", "ENC-0002", "ENC-0003"],
            "patient_id": ["PAT-0001", "PAT-0002", "PAT-0001"],
            "facility": ["UCSF", "ZSFG", "UCSF"],
            "admit_ts": ["2023-01-05", "2023-02-10", "2023-03-01"],
            "discharge_ts": ["2023-01-06", "2023-02-10", "2023-03-02"],
        }
    )
    encounters.write_csv(encounters_dir / "encounters_1.csv")

    vitals = pl.DataFrame(
        {
            "patient_id": ["PAT-0001", "PAT-0001", "PAT-0002", "PAT-0003"],
            "timestamp": ["2023-01-05", "2023-01-06", "2023-02-10", "2023-01-15"],
            "heart_rate": [70, 74, 80, 65],
            "systolic_bp": [120, 122, 130, 110],
            "diastolic_bp": [80, 82, 85, 70],
            "bmi": [24.0, 24.3, 29.1, 23.5],
        }
    )
    vitals.write_csv(vitals_dir / "vitals_1.csv")

    cfg["data"]["encounters_glob"] = str(encounters_dir / "*.csv")
    cfg["data"]["vitals_glob"] = str(vitals_dir / "*.csv")
    cfg["data"]["start_date"] = "2023-01-01"
    cfg["data"]["facilities"] = ["UCSF", "ZSFG"]

    return cfg


def test_load_data(cfg):
    encounters_lf, vitals_lf = pipeline.load_data(cfg)
    assert isinstance(encounters_lf, pl.LazyFrame)
    assert isinstance(vitals_lf, pl.LazyFrame)
    encounters_schema = encounters_lf.collect_schema()
    vitals_schema = vitals_lf.collect_schema()
    assert "patient_id" in encounters_schema
    assert "heart_rate" in vitals_schema


def test_build_summary_schema(cfg):
    encounters_lf, vitals_lf = pipeline.load_data(cfg)
    summary_lf = pipeline.build_summary(encounters_lf, vitals_lf, cfg)
    assert isinstance(summary_lf, pl.LazyFrame)
    schema = summary_lf.collect_schema()
    for col in ["facility", "year", "month", "num_vitals", "avg_hr", "avg_bmi"]:
        assert col in schema


def test_materialize_creates_outputs(cfg):
    encounters_lf, vitals_lf = pipeline.load_data(cfg)
    summary_lf = pipeline.build_summary(encounters_lf, vitals_lf, cfg)
    df = pipeline.materialize(summary_lf, cfg)
    assert df.height > 0
    parquet_path = Path(cfg["outputs"]["summary_parquet"])
    csv_path = Path(cfg["outputs"]["summary_csv"])
    assert parquet_path.exists()
    assert csv_path.exists()

    parquet_df = pl.read_parquet(parquet_path)
    csv_df = pl.read_csv(csv_path)
    assert parquet_df.height == csv_df.height
    for col in ["facility", "year", "month", "num_vitals"]:
        assert col in parquet_df.columns
