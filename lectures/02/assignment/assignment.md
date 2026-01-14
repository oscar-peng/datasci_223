# Assignment 02: Out-of-Core Analytics with Polars (Notebook)

This notebook walks you through the same workflow as `README.md`, but in a notebook-friendly format.
You will still implement the core logic in `src/pipeline.py`.

## Goals

- Build a lazy Polars pipeline over CSV inputs
- Join encounters + vitals and compute monthly summaries
- Materialize outputs with streaming and export Parquet/CSV
- Confirm outputs and pass the tests

## Setup

If the `data/` folder is missing, generate the synthetic dataset from the assignment root:

```bash
uv run python generate_assignment_data.py --size small --output-dir data
```

If you want to scale up later, rerun with `--size large`.

```python
from src import pipeline

cfg = pipeline.load_config("config.yaml")
cfg
```

## Part 1: Configure the Pipeline

Open `config.yaml` and confirm these sections are filled out:

- `data.encounters_glob`
- `data.vitals_glob`
- `data.start_date`
- `data.facilities`
- `outputs.summary_parquet`
- `outputs.summary_csv`
- (optional) `outputs.chart_png`

Keep paths relative so the autograder can run anywhere.

## Part 2: Implement `pipeline.py`

Open `src/pipeline.py` and complete the TODOs for:

- `load_data(cfg)`
- `build_summary(encounters_lf, vitals_lf, cfg)`
- `materialize(summary_lf, cfg)`

Return to this notebook after you implement each section.

### Checkpoint: load_data

```python
encounters_lf, vitals_lf = pipeline.load_data(cfg)
encounters_lf.collect_schema()
```

```python
vitals_lf.collect_schema()
```

### Checkpoint: build_summary

```python
summary_lf = pipeline.build_summary(encounters_lf, vitals_lf, cfg)
summary_lf.collect_schema()
```

```python
summary_lf.explain()
```

### Checkpoint: materialize

```python
df = pipeline.materialize(summary_lf, cfg)
df.head()
```

## Part 3: Validate Outputs

```python
from pathlib import Path

output_parquet = Path(cfg["outputs"]["summary_parquet"])
output_csv = Path(cfg["outputs"]["summary_csv"])

output_parquet.exists(), output_csv.exists()
```

```python
import polars as pl

parquet_df = pl.read_parquet(output_parquet)
csv_df = pl.read_csv(output_csv)
parquet_df.height, csv_df.height
```

## Part 4: Tests (run in terminal)

```bash
uv run pytest .github/tests -q
```

## Submission Checklist

- `src/pipeline.py` TODOs completed
- `outputs/README.md` documents artifacts
- Parquet + CSV created in `outputs/`
- All tests pass locally
