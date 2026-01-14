# Assignment 02 Hints

## Data sizing

- Start with a subset of CSVs (copy a few files into `data/sample/`) to iterate quickly.
- Once the pipeline works, point config back to the full glob.

## Polars tips

- Use `pl.scan_csv(..., dtypes={"patient_id": pl.Int32})` to control parsing.
- Chain `.select()` immediately to limit columns; the rest stay untouched.
- Call `.explain()` on the LazyFrame to ensure filters appear before joins.
- Use `.collect_schema()` to preview dtypes without triggering an expensive scan.

## Streaming behavior

- `.collect(engine="streaming")` supports aggregations and joins but not sorts that require full materialization. If you must sort, do it on a reduced dataset.
- When streaming, avoid `.with_columns(pl.col("..."))` that depend on entire table (e.g., cumulative sums) unless necessary.

## Testing strategy

1. `load_data` should not materialize anything—use `.head().collect()` only within tests.
2. `build_summary` can be validated by checking `.collect_schema()` and `.explain()`.
3. `materialize` should be idempotent; you can delete outputs and rerun without stale state.

## Debugging failures

- If tests complain about missing columns, print `LazyFrame.collect_schema()` to confirm names.
- For date filters, ensure they are parsed as `Datetime` (use `pl.datetime` helper or string parsing).
- Altair export requires `altair` + a renderer (vega-lite). If missing, wrap in try/except (already done in starter).
