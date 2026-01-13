# 01a_streaming_filter: Polars scan + streaming collect

Goal: generate a realistic (but synthetic) wearable dataset and use Polars *lazy scans* to filter and aggregate without ever materializing the full table in memory.

This demo is intentionally boring-data-science: scan → filter → select → group_by → collect(streaming=True) → write Parquet.

## Prereqs

- From repo root: `uv pip install -r requirements.txt`

## Data setup

From `lectures/02/demo/`:

```bash
mkdir -p data outputs
uv run python generate_demo_data.py --size small --output-dir data
```

You should now have:

- `data/sensor_hrv/part-*.parquet`
- `data/sleep_diary.parquet`
- `data/user_profile.parquet`

## Steps

```python
from pathlib import Path

sensor_dir = Path("data/sensor_hrv")
required_files = [
    Path("data/sleep_diary.parquet"),
    Path("data/user_profile.parquet"),
]

missing_files = [p for p in required_files if not p.exists()]
sensor_parts = list(sensor_dir.glob("*.parquet"))

if missing_files or not sensor_parts:
    raise FileNotFoundError(
        "Missing demo data.\n"
        f"- Missing files: {', '.join(str(p) for p in missing_files) if missing_files else '(none)'}\n"
        f"- Sensor parts found: {len(sensor_parts)}\n"
        "Run: uv run python generate_demo_data.py --size small --output-dir data"
    )
```

### 1) Scan the big table lazily

```python
import polars as pl

sensor = pl.scan_parquet("data/sensor_hrv/*.parquet")
print(sensor.schema)
```

Key point: `scan_parquet` builds a query plan; it does not load the full table.

### 2) Filter + project early (predicate/projection pushdown)

```python
import polars as pl

query = (
    pl.scan_parquet("data/sensor_hrv/*.parquet")
    .filter(pl.col("missingness_score") <= 0.35)
    .filter(pl.col("ts_start") >= pl.datetime(2024, 1, 15))
    .select([
        "device_id",
        pl.col("ts_start").dt.date().alias("date"),
        "heart_rate",
        "hrv_sdnn",
        "steps",
    ])
)

print(query.explain())
```

### 3) Aggregate to a daily summary and stream the collect

```python
import polars as pl
from pathlib import Path

daily = (
    query
    .group_by(["device_id", "date"])
    .agg([
        pl.len().alias("num_segments"),
        pl.mean("heart_rate").alias("mean_hr"),
        pl.mean("hrv_sdnn").alias("mean_sdnn"),
        pl.sum("steps").alias("steps_sum"),
    ])
    .sort(["device_id", "date"])
)

result = daily.collect(engine="streaming")
Path("outputs").mkdir(parents=True, exist_ok=True)
result.write_parquet("outputs/daily_device_summary.parquet")
print(result.head())
```

## Checkpoints

- `outputs/daily_device_summary.parquet` exists
- `result.height > 0`
- `num_segments` looks like "a day of 5-min windows" (usually a couple hundred)

## Stretch (optional)

- Change the filter to keep only nighttime segments and compare `mean_sdnn`.
- Try `.collect()` without `streaming=True` and watch memory/time differences on `--size large`.
