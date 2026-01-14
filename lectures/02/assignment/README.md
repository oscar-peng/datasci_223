# Assignment 02: Out-of-Core Analytics with Polars

**Due:** Before Lecture 03
**Points:** Pass/Fail (autograded)
**Skills:** Columnar formats, Polars lazy pipelines, reproducible batch scripts

## Overview

This assignment extends Lecture 02 demos. You will:

1. Convert tall CSVs into Parquet to reduce memory pressure
2. Build a Polars lazy pipeline that joins encounters with vitals
3. Execute the pipeline via script/CLI using streaming mode
4. Emit both Parquet + CSV artifacts and pass automated tests

All tests live under `.github/tests` inside this folder. Push frequently to see GitHub Classroom feedback.

Starter code is intentionally incomplete. You will fill in TODOs in `src/pipeline.py`.

## Assignment Structure

```
assignment/
├── assignment.md                 # Notebook-friendly instructions
├── assignment.ipynb              # Generated from assignment.md
├── config.yaml                     # Centralized settings
├── src/
│   ├── __init__.py
│   ├── pipeline.py                 # You implement functions here
│   └── run_pipeline.py             # CLI entrypoint (uses pipeline.py)
├── data/
│   ├── encounters/*.csv            # Synthetic encounter logs (provided)
│   └── vitals/*.csv                # Synthetic vitals data (provided)
├── outputs/
│   ├── README.md                   # Describe generated artifacts
│   └── (created by you)
├── hints.md                        # Optional hints
└── .github/
    ├── tests/test_pipeline.py      # Autograder tests
    └── workflows/classroom.yml     # Do not modify
```

## Data setup

If `data/` is missing, generate the synthetic dataset:

```bash
uv run python generate_assignment_data.py --size small --output-dir data
```

For faster iteration you can regenerate a smaller dataset by adjusting `--size` and re-running the command above.

## Part 1: Configure the Pipeline

**File:** `config.yaml`

Define inputs, filters, and outputs declaratively.

```yaml
data:
  encounters_glob: "data/encounters/*.csv"
  vitals_glob: "data/vitals/*.csv"
  start_date: "2023-01-01"
  facilities: ["UCSF", "ZSFG"]

outputs:
  summary_parquet: "outputs/facility_month_summary.parquet"
  summary_csv: "outputs/facility_month_summary.csv"
  chart_png: "outputs/facility_month_summary.png"

processing:
  bmi_floor: 12
  bmi_ceiling: 70
```

You may expand the config with additional knobs (columns to keep, chart titles, etc.). Autograder reads file paths from here.

## Part 2: Implement `pipeline.py`

**File:** `src/pipeline.py`

Provide three key functions:

1. `load_data(cfg) -> tuple[pl.LazyFrame, pl.LazyFrame]`
   - Use `pl.scan_csv` with glob patterns from config
   - Apply column projection + type casting

2. `build_summary(encounters_lf, vitals_lf, cfg) -> pl.LazyFrame`
   - Filter by facility + `start_date`
   - Join on `patient_id`
   - Compute monthly aggregates (counts + means)
   - Return a lazy frame ready for collection

3. `materialize(summary_lf, cfg) -> pl.DataFrame`
   - Execute with `.collect(engine="streaming")`
   - Write Parquet + CSV paths from config
   - Optionally emit a quick Altair/Matplotlib visualization (save to `chart_png`)

Keep functions pure (inputs → outputs). Logging lives in the CLI wrapper.

**Required methods (from lecture):**

- `pl.scan_csv`, `.select`, `.filter`, `.with_columns`, `.cast`
- `.group_by(...).agg([...])`, `.join(...)`
- `.collect(engine="streaming")`, `.write_parquet(...)`, `.write_csv(...)`
- Optional: `.to_pandas()` for chart export

## Part 3: CLI Wrapper

**File:** `src/run_pipeline.py`

- Parse `--config` argument (default `config.yaml`)
- Load YAML, configure logging
- Call `load_data`, `build_summary`, `materialize`
- Print row counts + file locations

Example usage:

```bash
uv run python src/run_pipeline.py --config config.yaml
```

## Part 4: Validation + Artifacts

Populate `outputs/README.md` with:

- Artifact descriptions (shape, columns)
- How to regenerate (`uv run ...` command)
- Any caveats (e.g., data simulated, chart uses aggregated data)

Ensure generated files are ignored appropriately (add to `.gitignore` under this folder if needed) **but** autograder expects them present when tests run locally.

## Assignment notebook (optional)

This assignment is also authored as a notebook-friendly Markdown file:

- `assignment.md` (source)
- `assignment.ipynb` (generated via jupytext)

To regenerate the notebook:

```bash
uv run python -m jupytext --to notebook assignment.md -o assignment.ipynb
```

## Tests

Run locally from this assignment folder:

```bash
uv run pytest .github/tests -q
```

Tests check that:

- `load_data` returns LazyFrames with required columns
- `build_summary` produces a LazyFrame with expected schema
- `materialize` writes both Parquet + CSV and returns a DataFrame matching fixtures
- Optional chart file exists if specified in config

Passing locally mirrors GitHub Classroom.

## Submission Checklist

- [ ] `config.yaml` filled with dataset + output paths
- [ ] `pipeline.py` implements load/build/materialize functions
- [ ] `run_pipeline.py` accepts `--config` and logs progress
- [ ] `outputs/README.md` explains generated artifacts
- [ ] All tests in `.github/tests` pass locally
- [ ] Push to GitHub Classroom and confirm CI success

## Hints

- Peek at Lecture 02 demos (`demo/01a_streaming_filter.md`, `demo/02a_lazy_join.md`, `demo/03a_batch_report.md`) for patterns
- Use `.explain()` to confirm filters/projectors push down before collecting
- For easier testing, work on small CSVs first, then scale up
- Keep config paths relative so autograder can run anywhere

Good luck! Streaming > swap files 😄
