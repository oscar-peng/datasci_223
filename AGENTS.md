# AGENTS

Applied Data Science with Python course materials (UCSF DataSci 223, Spring 2025). Documentation-heavy repo for beginners; lectures are long-form Markdown with speaking notes and demos. Adapt/update previous year's content from `lectures_25/` when possible.

**Course structure**: Applied survey course—each week covers a single topic, either standalone (e.g., SQL) or as part of a series building complexity (classification → neural networks → LLMs → LLM API/agentic/workflows).

**GitHub Pages site URL**: `https://christopherseaman.github.io/datasci_223/` (configured in `mkdocs.yml` as `site_url`; images use relative paths like `01/media/...` WITHOUT leading slash so MkDocs prepends site_url correctly)

**IDE**: VS Code is the common IDE for this class.

## Repository layout
- `lectures/`: Current year's content (MkDocs source). Being built out to 11 lectures. `index.md` symlinks to repo `README.md`.
- `lectures_25/`: Last year's content (10 lectures). Reference material to adapt/update from.
- `docs/`: MkDocs build output (generated, gitignored for local dev but used for GitHub Pages).
- `mkdocs.yml`: MkDocs Material config; nav lists lectures/exercises.
- `build.sh`: Installs Playwright Chromium for PDF export buttons, then runs `mkdocs build`.
- `overrides/`: Theme customizations (favicon, fonts, `extra.css`, template `main.html`).
- `requirements.txt`: MkDocs plugins (material, exporter, charts, callouts, minify, macros) plus teaching libs (altair, pandas, dash, plotly, selenium, playwright helpers).
- `lectures/planned_lectures.md`: Current year's 11-lecture plan.
- `lectures_25/planned_lectures.md`: Last year's outlines (reference).
- `refs/`: Supporting notes, data links, and meta authoring guide (`refs/instructions.md` sets tone/structure for lectures). Also includes PhysioNet notes, SQL guide, debugging tips, etc.
- `all_xkcd.html` / `all_xkcd.csv`: Index of available XKCD comics for reference.

## Build and preview
- Python 3.11/3.12. **Prefer `uv`** for local dev (`uv venv .venv && uv pip install -r requirements.txt`); student default is `python -m venv`.
- `pip install -r requirements.txt` (mkdocs-material stack + exporter).
- Local preview: `mkdocs serve -f mkdocs.yml`.
- Static build: `mkdocs build` or `./build.sh` (runs `playwright install chromium` for PDF export).
- Deploy: `mkdocs gh-deploy --force` (CI auto-deploys on push to main via `.github/workflows/deploy.yml`).

## Authoring workflow
- Follow `refs/instructions.md`: audience is beginner health data science students; 90-minute lectures with additional demo time at 1/3, 2/3, and end; include HTML comment speaking notes under each `###`; balance concept/reference/examples.
- **Notion format quirk**: Lectures served from Notion allow no page title and multiple H1s (`#`) in a single document.
- New lecture: create in `lectures/0X/lecture_0X.md`, adapting from `lectures_25/` when applicable. YAML frontmatter: `lecture_number` and `pdf: true`. Keep `use_directory_urls: false` in mind when adding links.
- Update `mkdocs.yml` nav when adding/moving lectures or course info pages.
- Assets live alongside their lecture folder; fonts/images for theme live under `overrides/assets/`.
- Exercises section in nav points to GitHub Classroom links; verify/update URLs each term.
- Content style: each section/subsection should include (1) brief prose intro/explanation, (2) optional visual or `#FIXME` placeholder, (3) reference (function signature, common parameters), and (4) minimal code example. Demos should have more complexity with real/realistic data. Sprinkle humor/comics throughout (use existing assets—never invent links).
- Comics/visual sourcing: prefer images local to the lecture folder; if reusing from elsewhere, copy into the lecture’s `media/` subdir first. `all_xkcd.html` lists available XKCD panels—pick from there and copy locally instead of hotlinking.
- XKCD helper: use `scripts/fetch_xkcd_2x.py` to download comics via explainxkcd file pages (2x "Original file" links). Usage: `./scripts/fetch_xkcd_2x.py 1597:Git 1722:Debugging:xkcd_debugging.png`
- Demo format: Markdown notebooks in `demo/` folder, convert via `jupytext --to notebook demo.md`. Use realistic health data examples.

## Assignment structure
Weekly assignments are **pass/fail** and should be straightforward for students who understand the lecture content. Coursework uses GitHub Classroom with pytest-based autograding via GitHub Actions.

```
lectures/XX/assignment/
├── README.md                 # Instructions
├── {source_files}.py         # Starter/solution code
├── hints.md                  # Optional hints
└── .github/
    ├── tests/test_*.py       # pytest test suite
    └── workflows/classroom.yml
```

Grading workflow downloads latest tests from template repo and runs pytest. References:
- `lectures_25/06/assignment/` - last year's multi-part notebook assignment example
- `../datasci_217/07/assignment/.github` - grading workflow examples from prerequisite course

## Dependencies and data
- Core deps: mkdocs >=1.6, mkdocs-material >=9.6, mkdocs-exporter (PDF), mkdocs-callouts, mkdocs-charts, mkdocs-minify, mkdocs-macros.
- Optional/demo deps: pandas, numpy, altair, dash/plotly, selenium + webdriver-manager, Playwright. Install Playwright browsers before exporting PDF.
- No large datasets committed; see `refs/physionet.md` and other `refs/` docs for data sources used in examples.

## Engineering guardrails (per ~/.claude/CLAUDE.md)
- Configuration-first: do not hardcode values that belong in config; prefer centralized settings and utility modules over duplication.
- DRY and separation of concerns: extract shared logic into helpers; keep orchestration separate from business logic and from data/visual layers.
- Simplicity and clear data flow: favor explicit, minimal abstractions with named inputs/outputs; avoid hidden side effects.
- Naming and comments: descriptive function/variable names; brief contextual comments only when logic is non-obvious.
- **Testing/validation required**: never declare work done without actually running the code AND validating output. Run scripts (e.g., `./build.sh`), then verify outputs exist and are correct (e.g., check generated files, inspect content). If testing is impossible, document assumptions and provide user-run verification steps.
- Environment discipline: prefer `uv` for speed, fallback to venv/conda; stable working dir, relative paths; avoid complex one-liners—script it instead.
- Git hygiene: direct, professional commit messages—no cute co-author tags (e.g., "Co-Authored-By: Claude"), no emoji, no "Generated with" footers. Conventional commits preferred.

