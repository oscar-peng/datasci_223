# 02a_lazy_join: Join multiple tables + inspect the plan

Goal: use Polars `LazyFrame` joins to connect the wearable tables and compute a *sleep ↔ physiology* summary.

Data relationships:

- `user_profile.user_id` joins to `sleep_diary.user_id`
- `sensor_hrv.device_id` encodes the user id suffix (e.g., `DEV-00012` → `USER-00012`)

## Data setup

From `lectures/02/demo/`:

```bash
mkdir -p data outputs
uv run python generate_demo_data.py --size small --output-dir data
```

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

### 1) Create LazyFrames

```python
import polars as pl

users = pl.scan_parquet("data/user_profile.parquet")
sleep = pl.scan_parquet("data/sleep_diary.parquet")
sensor = pl.scan_parquet("data/sensor_hrv/*.parquet")
```

### 2) Normalize keys (derive `user_id` from `device_id`)

```python
import polars as pl

sensor_keyed = sensor.with_columns(
    pl.concat_str([pl.lit("USER-"), pl.col("device_id").str.split("-").list.get(1)]).alias("user_id"),
    pl.col("ts_start").dt.date().alias("date"),
)
```

### 3) Join sleep diary to nightly physiology

We join on `user_id` + `date`, then summarize by occupation.

```python
import polars as pl

night_segments = (
    sensor_keyed
    .filter(pl.col("missingness_score") <= 0.35)
    .filter(pl.col("ts_start").dt.hour().is_between(22, 23) | pl.col("ts_start").dt.hour().is_between(0, 6))
    .select(["user_id", "date", "hrv_sdnn", "hrv_rmssd", "heart_rate", "steps"])
)

nightly = (
    night_segments
    .group_by(["user_id", "date"])
    .agg([
        pl.len().alias("num_segments"),
        pl.mean("heart_rate").alias("night_mean_hr"),
        pl.mean("hrv_sdnn").alias("night_mean_sdnn"),
        pl.mean("hrv_rmssd").alias("night_mean_rmssd"),
        pl.sum("steps").alias("night_steps"),
    ])
)

joined = (
    sleep
    .join(nightly, on=["user_id", "date"], how="inner")
    .join(users.select(["user_id", "age", "gender", "occupation"]), on="user_id", how="left")
)

summary = (
    joined
    .group_by(["occupation", "gender"])
    .agg([
        pl.len().alias("n_nights"),
        pl.mean("sleep_efficiency").alias("avg_sleep_efficiency"),
        pl.mean("night_mean_sdnn").alias("avg_night_sdnn"),
        pl.corr("sleep_efficiency", "night_mean_sdnn").alias("corr_sleep_sdnn"),
    ])
    .sort(["occupation", "gender"])
)

print(summary.explain())
out = summary.collect(engine="streaming")
out.write_parquet("outputs/sleep_hrv_by_group.parquet")
print(out)
```

## Checkpoints

- `outputs/sleep_hrv_by_group.parquet` exists
- `summary.explain()` shows projection/predicate pushdown before joins
- You can explain *why* we derived `user_id` from `device_id`
