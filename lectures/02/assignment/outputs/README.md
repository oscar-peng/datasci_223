# Assignment 02 Outputs

Artifacts created by `uv run python src/run_pipeline.py --config config.yaml`:

| File | Description |
| ---- | ----------- |
| `facility_month_summary.parquet` | Machine-friendly table with facility/year/month aggregates (counts + averages) |
| `facility_month_summary.csv` | Same data for quick spreadsheet review |
| `facility_month_summary.png` | Optional chart (monthly BMI trend per facility) |

Regenerate by deleting the files and rerunning the command above. Tests will fail if these files are missing locally when autograder executes.
