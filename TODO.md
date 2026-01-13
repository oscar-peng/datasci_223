# Lecture 02 Follow-up Tasks

- [x] Build lecture visuals (#FIXME graphics: memory vs dataset chart, row/column diagram, updated Polars benchmark, lazy-plan diagram, monitoring screenshot).
- [x] Select and fetch new XKCDs (Data Pipeline/Workflow/etc.) via `scripts/fetch_xkcd_2x.py` and embed them in the lecture.
- [x] Verify lecture media paths (`02/media/...`) and update `mkdocs.yml` nav if new files are introduced.
- [ ] Ensure demo artifacts/data generators exist and align with the written instructions (e.g., big CSV generator, dimension tables, pipeline configs).
- [ ] Ensure assignment sample data plus README instructions match the shipped fixtures.
- [ ] Convert each `lectures/02/demo/*.md` via Jupytext, execute the notebooks end-to-end, and capture key outputs.
- [ ] Check in the required Jupytext partners (`.ipynb` or percent-format `.py`) after execution so the demos stay synced.
- [ ] Validate the assignment from a scratch directory using the existing `.venv` (`uv run pytest .github/tests -q`).
- [ ] Document demo and assignment validation results for future instructors (logs or summary notes).
- [x] Prep git staging for the updated `lectures/02` tree once validations pass (commit/push requested).
- [ ] (Optional later) Add `lectures/02/NOTES.md` once lecture content is fully locked.
- [ ] (Optional later) Update `lectures/planned_lectures.md` if Lecture 02 scope diverges from the plan.
