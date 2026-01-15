# Assignment 02: Polars on EHR Event Logs - SOLUTION

Build a Polars pipeline that summarizes diagnosis prevalence and procedure activity from synthetic EHR events.

## Setup

```python
import polars as pl
import yaml
from pathlib import Path
from datetime import datetime

print(f"Polars version: {pl.__version__}")
print("Environment ready!")
```

## Generate data

```python
from pathlib import Path

from generate_test_data import generate_test_data

generate_test_data(size="small", output_dir=Path("data"))
```

## Configuration

```python
with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("Config loaded:")
print(f"  Patients: {config['data']['patients_path']}")
print(f"  Sites: {config['data']['sites_path']}")
print(f"  Events: {config['data']['events_path']}")
print(f"  ICD-10 lookup: {config['data']['icd10_path']}")
print(f"  HCPCS lookup: {config['data']['hcpcs_path']}")
```

## Part 1: Lazy Data Loading

```python
patients = pl.scan_parquet(config["data"]["patients_path"])
sites = pl.scan_parquet(config["data"]["sites_path"])

events = (
    pl.scan_parquet(config["data"]["events_path"])
    .with_columns(
        pl.col("event_ts").str.strptime(pl.Datetime, strict=False).alias("event_ts")
    )
)

icd10 = pl.scan_parquet(config["data"]["icd10_path"]).select(
    ["code", "short_desc", "category"]
)

hcpcs = pl.scan_parquet(config["data"]["hcpcs_path"]).select(
    ["code", "short_desc", "group"]
)

print("Patients schema:")
print(patients.collect_schema())

print("Events schema:")
print(events.collect_schema())
```

## Part 2: Filter and Prep Events

```python
start_date = datetime.fromisoformat(config["data"]["start_date"])

events_filtered = events.filter(pl.col("event_ts") >= pl.lit(start_date))

dx_events = events_filtered.filter(pl.col("record_type") == "ICD10")
proc_events = events_filtered.filter(pl.col("record_type") == "HCPCS")
```

## Part 3: Diagnosis Prevalence by Site

```python
prefix = config["data"]["diabetes_prefix"]

dx_diabetes = dx_events.filter(pl.col("code").str.starts_with(prefix))

patients_by_site = (
    events_filtered.select(["site_id", "patient_id"])
    .unique()
    .group_by("site_id")
    .agg(pl.len().alias("patients_seen"))
)

dx_patients_by_site = (
    dx_diabetes.select(["site_id", "patient_id"])
    .unique()
    .group_by("site_id")
    .agg(pl.len().alias("diabetes_patients"))
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
    .join(sites.select(["site_id", "site_name", "site_type"]), on="site_id", how="left")
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
```

## Part 4: Procedure Activity by Site and Group

```python
hcpcs_summary = (
    proc_events
    .join(hcpcs.select(["code", "group"]), on="code", how="left")
    .group_by(["site_id", "group"])
    .agg(pl.len().alias("procedure_count"))
    .join(sites.select(["site_id", "site_name"]), on="site_id", how="left")
    .select(["site_id", "site_name", "group", "procedure_count"])
    .sort(["site_id", "procedure_count"], descending=[False, True])
)
```

## Part 5: Collect and Export

```python
dx_df = dx_summary.collect(engine="streaming")
hcpcs_df = hcpcs_summary.collect(engine="streaming")

Path("outputs").mkdir(exist_ok=True)

dx_df.write_parquet(config["outputs"]["dx_summary_parquet"])
dx_df.write_csv(config["outputs"]["dx_summary_csv"])

hcpcs_df.write_parquet(config["outputs"]["hcpcs_summary_parquet"])
hcpcs_df.write_csv(config["outputs"]["hcpcs_summary_csv"])

print("Outputs written:")
print(dx_df.head())
print(hcpcs_df.head())
```

## Validation

```python
outputs = [
    config["outputs"]["dx_summary_parquet"],
    config["outputs"]["dx_summary_csv"],
    config["outputs"]["hcpcs_summary_parquet"],
    config["outputs"]["hcpcs_summary_csv"],
]

missing = [path for path in outputs if not Path(path).exists()]
if missing:
    print("Missing outputs:", missing)
else:
    print("All outputs created")
```
