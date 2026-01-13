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

## Run the batch report

```bash
uv run python 03a_batch_report.py --config 03a_config.yaml
```

## Outputs

- `outputs/sleep_hrv_report.parquet`
- `outputs/sleep_hrv_report.csv`

## Checkpoints

- The script logs row counts and output paths
- The parquet output has one row per `(occupation, gender)` group
- The report includes both sleep and physiology columns
