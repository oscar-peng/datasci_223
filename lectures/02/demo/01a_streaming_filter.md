# 01a_streaming_filter: Polars scan + streaming collect

Goal: generate a realistic (but synthetic) wearable dataset and use Polars *lazy scans* to filter and aggregate without ever materializing the full table in memory.

This demo is intentionally boring-data-science: scan → filter → select → group_by → collect(engine="streaming") → write Parquet.

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

## 4) Polars vs pandas (same task)

We’ll compute the *same* daily summary in two ways:

- **Polars**: scan Parquet parts lazily, then `collect(engine="streaming")`
- **pandas**: read Parquet via `pyarrow` (with pushdown) and groupby in memory

This isn’t to “dunk on pandas” — it’s to show what changes when you move from eager, row-wise CSV workflows to columnar scans + query planning.

### 4a) Polars timing (Parquet scan → streaming aggregate)

```python
import time
import polars as pl

polars_t0 = time.perf_counter()
polars_out = daily.collect(engine="streaming")
polars_t1 = time.perf_counter()

print(f"polars: {polars_out.height:,} rows in {polars_t1 - polars_t0:.2f}s")
print(f"polars result size (estimated): {polars_out.estimated_size('mb'):.2f} MB")
```

### 4b) pandas timing (Parquet read via pyarrow → in-memory groupby)

This shows what pandas can do today when it uses `pyarrow` for Parquet I/O:

- **projection pushdown** via `columns=[...]`
- **predicate pushdown** via `filters=[...]`

…but once the filtered data is in a pandas `DataFrame`, the groupby is still an in-memory operation.

```python
import time
import pandas as pd

pd_t0 = time.perf_counter()

# Note: pandas delegates Parquet reading to pyarrow.
# Using filters/columns keeps the materialized pandas DataFrame smaller.
df = pd.read_parquet(
    "data/sensor_hrv",
    engine="pyarrow",
    columns=["device_id", "ts_start", "heart_rate", "hrv_sdnn", "steps", "missingness_score"],
    filters=[
        ("missingness_score", "<=", 0.35),
        ("ts_start", ">=", pd.Timestamp("2024-01-15")),
    ],
)

df["date"] = df["ts_start"].dt.date

pd_out = (
    df
    .groupby(["device_id", "date"], as_index=False)
    .agg(
        num_segments=("heart_rate", "size"),
        mean_hr=("heart_rate", "mean"),
        mean_sdnn=("hrv_sdnn", "mean"),
        steps_sum=("steps", "sum"),
    )
    .sort_values(["device_id", "date"])
)

pd_t1 = time.perf_counter()

print(f"pandas: {len(pd_out):,} rows in {pd_t1 - pd_t0:.2f}s")
print(f"pandas input memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
```

## Checkpoints

- You can explain why Polars reads Parquet parts efficiently (`scan_parquet` + pushdown)
- You can explain why pandas-on-CSV is inherently eager (must materialize rows)
- The two outputs agree on shape/columns (minor float differences are fine)
