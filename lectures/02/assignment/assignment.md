# Assignment 02: Polars on EHR Event Logs

Build a Polars pipeline that summarizes diagnosis prevalence and procedure activity from synthetic EHR events. Use lazy scans, filtering, joins, and group-bys.

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
import sys
import subprocess

subprocess.run(
    [sys.executable, "generate_test_data.py", "--size", "small", "--output-dir", "data"],
    check=True,
)
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

Use `pl.scan_parquet()` to create LazyFrames without loading data into memory.

```python
# TODO: Scan patients, sites, events, and code lookups
patients = None
sites = None
events = None
icd10 = None
hcpcs = None

# Check schemas (fast, still lazy)
if patients is not None:
    print("Patients schema:")
    print(patients.collect_schema())

if events is not None:
    print("Events schema:")
    print(events.collect_schema())
```

## Part 2: Filter and Prep Events

Filter to the assignment date window and split ICD-10 vs HCPCS events.

```python
start_date = datetime.fromisoformat(config["data"]["start_date"])

# TODO: parse event_ts to Datetime
# TODO: filter events to event_ts >= start_date
# TODO: split into dx_events and proc_events by record_type

events_filtered = None

dx_events = None
proc_events = None
```

## Part 3: Diagnosis Prevalence by Site

Compute the percent of patients per site with a type 2 diabetes diagnosis.

```python
prefix = config["data"]["diabetes_prefix"]

# TODO: Filter dx_events to ICD-10 codes starting with prefix
# TODO: total patients per site (unique patient_id from events_filtered)
# TODO: diabetes patients per site (unique patient_id from filtered dx)
# TODO: join counts + site names, calculate prevalence

dx_summary = None

if dx_summary is not None:
    print(dx_summary.collect_schema())
```

## Part 4: Procedure Activity by Site and Group

Summarize HCPCS procedures by site and DHS group.

```python
# TODO: join proc_events to hcpcs lookup on code
# TODO: group by site_id + group, count procedures
# TODO: join site names

hcpcs_summary = None
```

## Part 5: Collect and Export

```python
# TODO: collect both summaries using streaming engine
# TODO: create outputs directory
# TODO: write Parquet + CSV outputs using config paths

if dx_summary is not None and hcpcs_summary is not None:
    print("Outputs ready")
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

## Next Steps

Run `python -m pytest .github/tests/test_assignment.py -v` in your terminal.
