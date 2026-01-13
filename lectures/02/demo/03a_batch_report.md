# 03a_batch_report: Reproducible batch pipeline (CLI)

Goal: run a small, reproducible Polars batch job that reads the generated wearable tables, joins them, and writes a report artifact.

This is the same mental model as a real lab pipeline: config-driven inputs, lazy scans, and streaming output.

## Files

- `03a_batch_report.py` (script)
- `03a_config.yaml` (config)

## Data setup

From `lectures/02/demo/`:

```bash
mkdir -p data outputs
uv run python generate_demo_data.py --size small --output-dir data
```

## 1) Configure inputs/filters/outputs (YAML)

Open `03a_config.yaml` and notice it defines:

- input globs (sensor parquet parts + the small dimension tables)
- filter knobs (`start_date`, `missingness_max`, night-hour window)
- output paths (Parquet + CSV)

For the demo, change `filters.start_date` (e.g., move it earlier/later) and re-run the pipeline to see the outputs change.

## 2) Run the batch pipeline (CLI)

```bash
uv run python 03a_batch_report.py --config 03a_config.yaml
```

Watch the logs:

- “Scanning inputs lazily…” (load phase)
- “Plan summary…” (transform phase)
- “Collecting with streaming engine…” (materialize phase)
- “Writing …parquet / …csv” (artifact emission)

## 3) Confirm artifacts exist

```bash
ls -lh outputs/
```

Expected:

- `outputs/sleep_hrv_report.parquet`
- `outputs/sleep_hrv_report.csv`

## 4) Validate the output (fast sanity checks)

Run this immediately after the pipeline finishes:

```bash
uv run python - <<'PY'
import polars as pl

df = pl.read_parquet("outputs/sleep_hrv_report.parquet")
print(df.head())

assert df.height > 0
assert {"occupation", "gender", "n_nights"}.issubset(set(df.columns))
assert df["n_nights"].min() > 0

# Basic plausibility checks (not strict scientific claims)
assert df["avg_sleep_efficiency"].is_finite().all()
assert df["avg_night_sdnn"].is_finite().all()

print("OK: output looks sane")
PY
```

## Checkpoints

- The script logs the query plan and output paths.
- The parquet output has one row per `(occupation, gender)` group.
- The report includes both sleep and physiology columns.
