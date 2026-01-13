from pathlib import Path

import polars as pl
import pytest

from src import pipeline


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    cfg = pipeline.load_config("config.yaml")
    tmp = tmp_path_factory.mktemp("outputs")
    cfg["outputs"]["summary_parquet"] = str(tmp / "summary.parquet")
    cfg["outputs"]["summary_csv"] = str(tmp / "summary.csv")
    cfg["outputs"]["chart_png"] = str(tmp / "summary.png")
    return cfg


def test_load_data(cfg):
    encounters_lf, vitals_lf = pipeline.load_data(cfg)
    assert isinstance(encounters_lf, pl.LazyFrame)
    assert isinstance(vitals_lf, pl.LazyFrame)
    assert "patient_id" in encounters_lf.columns
    assert "heart_rate" in vitals_lf.columns


def test_build_summary_schema(cfg):
    encounters_lf, vitals_lf = pipeline.load_data(cfg)
    summary_lf = pipeline.build_summary(encounters_lf, vitals_lf, cfg)
    schema = summary_lf.schema
    assert schema["facility"] == pl.Utf8
    assert "num_vitals" in schema


def test_materialize_creates_outputs(cfg):
    encounters_lf, vitals_lf = pipeline.load_data(cfg)
    summary_lf = pipeline.build_summary(encounters_lf, vitals_lf, cfg)
    df = pipeline.materialize(summary_lf, cfg)
    assert df.height >= 0
    assert Path(cfg["outputs"]["summary_parquet"]).exists()
    assert Path(cfg["outputs"]["summary_csv"]).exists()
