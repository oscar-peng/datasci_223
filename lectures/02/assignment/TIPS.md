# Assignment 02 Tips

## Polars patterns

- Use `pl.scan_parquet(...)` for lazy scans and `collect_schema()` to verify columns.
- Convert `event_ts` to `Datetime` once in a `.with_columns(...)` call.
- For distinct patients per site, use `.select([...]).unique().group_by(...).agg(pl.len())`.
- Use `.str.starts_with("E11")` for ICD-10 prefixes.

## Join flow

- `events` already includes `site_id`; join `sites` only for names/types.
- Join `hcpcs_codes` to procedures on `code` to get the `group` column.

## Streaming tips

- `.collect(engine="streaming")` works well after filtering and grouping.
- Avoid `.sort()` until after aggregation to keep streaming efficient.

## Debugging

- Print `LazyFrame.explain()` if filters seem ignored.
- If outputs are empty, check your date filter and prefix filter first.
