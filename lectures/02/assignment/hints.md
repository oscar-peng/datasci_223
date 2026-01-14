# Assignment 02 Hints

## Lazy scans

- Use `pl.scan_parquet(config["data"]["events_path"])` for events.
- Call `.collect_schema()` to verify columns without loading data.

## Distinct patients by site

```python
patients_by_site = (
    events_filtered
    .select(["site_id", "patient_id"])
    .unique()
    .group_by("site_id")
    .agg(pl.len().alias("patients_seen"))
)
```

## Diabetes filter

```python
prefix = config["data"]["diabetes_prefix"]
dx_diabetes = dx_events.filter(pl.col("code").str.starts_with(prefix))
```

## Procedure summary

```python
proc_with_group = proc_events.join(hcpcs.select(["code", "group"]), on="code", how="left")
```

## Output writing

- Create `outputs/` before writing.
- Use config paths so tests can find files.
