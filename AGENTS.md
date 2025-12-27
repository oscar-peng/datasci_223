# AGENTS

Applied Data Science with Python course materials (UCSF DataSci 223, Spring 2025). Documentation-heavy repo for beginners; lectures are long-form Markdown with speaking notes and demos.

## Repository layout
- `lectures_25/`: MkDocs content root. Contains `index.md`, `syllabus.md`, `project.md`, cheat sheets, references, and lecture folders `01`-`10` with `lecture_XX.md` plus images/assets.
- `mkdocs.yml`: MkDocs Material config; nav lists lectures/exercises. Note `docs_dir` currently points to `lectures` (does not exist); update to `lectures_25` or add a `lectures` symlink before builds.
- `build.sh`: Installs Playwright Chromium for PDF export buttons, then runs `mkdocs build`.
- `overrides/`: Theme customizations (favicon, fonts, `extra.css`, template `main.html`).
- `requirements.txt`: MkDocs plugins (material, exporter, charts, callouts, minify, macros) plus teaching libs (altair, pandas, dash, plotly, selenium, playwright helpers).
- `planned_lectures.md`: Outline/backlog of lecture topics and datasets.
- `refs/`: Supporting notes, data links, and meta authoring guide (`refs/instructions.md` sets tone/structure for lectures). Also includes PhysioNet notes, SQL guide, debugging tips, etc.
- `all_xkcd.html`: Convenience page with XKCD strips.

## Build and preview
- Recommended: Python 3.11/3.12, create a virtual environment (`uv venv .venv` or `python -m venv .venv`) and activate it.
- `pip install -r requirements.txt` (mkdocs-material stack + exporter).
- Fix docs root first: either set `docs_dir: lectures_25` in `mkdocs.yml` or `ln -s lectures_25 lectures`; otherwise `mkdocs serve/build` will fail because `lectures` is missing.
- Local preview: `mkdocs serve -f mkdocs.yml`.
- Static build: `mkdocs build` or `./build.sh` (runs `playwright install chromium` for PDF export). For GitHub Pages, standard `mkdocs gh-deploy` after build (not yet wired in CI).

## Authoring workflow
- Follow `refs/instructions.md`: audience is beginner health data science students; 90-minute lecture pacing with 3 demo breaks; include HTML comment speaking notes under each `###`; balance concept/reference/examples.
- New lecture: mirror existing folder pattern (`lectures_25/0X/lecture_0X.md` + assets). Keep `use_directory_urls: false` in mind when adding links.
- Update `mkdocs.yml` nav when adding/moving lectures or course info pages.
- Assets live alongside their lecture folder; fonts/images for theme live under `overrides/assets/`.
- Exercises section in nav points to GitHub Classroom links; verify/update URLs each term.
- Content style (additive): each subsection should include (1) a short, matter-of-fact summary of what is covered, (2) a visual or `#FIXME` placeholder, (3) a function/signature or API reference, and (4) a minimal code example; sprinkle humor and comics throughout (use existing assets like `xkcd_git.png` or other repo images—never invent links).
- Comics/visual sourcing: prefer images local to the lecture folder; if reusing from elsewhere, copy into the lecture’s `media/` subdir first. `all_xkcd.html` lists available XKCD panels—pick from there and copy locally instead of hotlinking.
- XKCD helper: use `scripts/fetch_xkcd_2x.py` to download comics via explainxkcd file pages (2x “Original file” links). Extend `COMICS` in the script and run `./scripts/fetch_xkcd_2x.py` to refresh local copies into `lectures/01/media/`.

## Dependencies and data
- Core deps: mkdocs >=1.6, mkdocs-material >=9.6, mkdocs-exporter (PDF), mkdocs-callouts, mkdocs-charts, mkdocs-minify, mkdocs-macros.
- Optional/demo deps: pandas, numpy, altair, dash/plotly, selenium + webdriver-manager, Playwright. Install Playwright browsers before exporting PDF.
- No large datasets committed; see `refs/physionet.md` and other `refs/` docs for data sources used in examples.

## Engineering guardrails (per ~/.claude/CLAUDE.md)
- Configuration-first: do not hardcode values that belong in config; prefer centralized settings and utility modules over duplication.
- DRY and separation of concerns: extract shared logic into helpers; keep orchestration separate from business logic and from data/visual layers.
- Simplicity and clear data flow: favor explicit, minimal abstractions with named inputs/outputs; avoid hidden side effects.
- Naming and comments: descriptive function/variable names; brief contextual comments only when logic is non-obvious.
- Testing/validation: never declare work done without running code; document assumptions if testing is impossible and provide user-run steps.
- Environment discipline: use venv/uv/conda, stable working dir, relative paths; avoid complex one-liners—script it instead.
- Git hygiene: professional commit messages; no co-author tags for automation; conventional commits preferred.

## Known issues / TODO
- Fix `docs_dir` mismatch before any build/serve; validate nav paths after the change.
- Re-run a full `mkdocs build` + exporter to confirm PDF buttons still work after Playwright install.
- Review `planned_lectures.md` (sections 9/10 merged) against actual lecture notes for consistency.
- Periodically check external exercise links and syllabus details (dates/rooms) for freshness.
- Consider adding CI for mkdocs build/deploy once docs_dir is corrected.

## Operational notes
- Current session is read-only, so builds/tests were not executed here. Dependencies are pinned to 2025-era versions; recreate the venv when upgrading.
