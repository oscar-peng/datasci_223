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
- Follow `refs/instructions.md`: audience is beginner health data science students; 90-minute lectures with additional demo time at 1/3, 2/3, and end; balance concept/reference/examples; no inline HTML comment speaking notes in the lecture text.
- **Notion format quirk**: Lectures served from Notion allow no page title and multiple H1s (`#`) in a single document.
- **Lecture formatting**: title line is plain text (no leading `#`); main headings use a single `#` with real sub-sections via `##`/`###` (never bolded fake headings); prefer concise bullets over long prose with 4-space indents for nesting; each section combines brief intro, a visual/table/output or `#FIXME` placeholder, a `### Reference` table, and a short `### Code` snippet (demos carry the complexity); place visuals before code; mark demos only with `# LIVE DEMO!`; keep tone non-conversational; skip “Summary” sections; sprinkle comics spaced through the lecture (use existing assets).
- New lecture: create in `lectures/0X/lecture_0X.md`, adapting from `lectures_25/` when applicable. YAML frontmatter: `lecture_number` and `pdf: true`. Keep `use_directory_urls: false` in mind when adding links.
- Update `mkdocs.yml` nav when adding/moving lectures or course info pages.
- Assets live alongside their lecture folder; fonts/images for theme live under `overrides/assets/`.
- Exercises section in nav points to GitHub Classroom links; verify/update URLs each term.
- Content style: keep lecture code blocks brief and focused on single concepts; demos should add realistic health-data complexity and clear checkpoints.
- Comics/visual sourcing: prefer images local to the lecture folder; if reusing from elsewhere, copy into the lecture’s `media/` subdir first. `all_xkcd.html` lists available XKCD panels—pick from there and copy locally instead of hotlinking.
- XKCD helper: use `scripts/fetch_xkcd_2x.py` to download comics via explainxkcd file pages (2x "Original file" links). Usage: `./scripts/fetch_xkcd_2x.py 1597:Git 1722:Debugging:xkcd_debugging.png`

## Demo structure and conventions

Lectures include 3 hands-on demos at ~1/3, ~2/3, and end of 90-minute session. Demos build on lecture content with realistic complexity.

**Content philosophy:**
- **Lecture code blocks:** Short, simple, minimal examples demonstrating single concepts
- **Demo code:** Realistic complexity with health data, multiple steps, edge cases—mirrors real-world usage
- Demos should be completable in 10-15 minutes with clear success checkpoints

**File naming convention:** `0Xy_description.suffix`
- `X` = demo number (1, 2, 3)
- `y` = order within demo (a, b, c, ...; omit if single file)
- `description` = very short descriptor (1-3 words)
- `suffix` = file type (`.md`, `.py`, `.ipynb`, `.yaml`, etc.)

**Examples:**
```
demo/
├── DEMO_GUIDE.md              # Brief walkthrough for all demos
├── 01_setup_resources.md      # Demo 1: single file, no letter needed
├── 02a_brittle_cleaning.md    # Demo 2: starter notebook (before)
├── 02b_hardened_cleaning.md   # Demo 2: solution notebook (after)
├── 02_config.yaml             # Demo 2: config file (single file, no letter)
├── 03a_buggy_bmi.py           # Demo 3: script to debug
├── 03b_buggy_analysis.md      # Demo 3: notebook to debug
└── data/                      # Shared data for all demos
```

**Markdown → Jupyter conversion:**
- Write demos as `.md` files (easier to review, git-friendly)
- Convert with `jupytext --to notebook demo/*.md` before class
- Use jupytext percent format (`#%%`) or markdown format for cells

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

### Assignment testing philosophy

**Test behaviors and artifacts, not implementation details.** Students may solve problems in many valid ways (Gödel incompleteness applies to grading too).

**Good tests check:**
1. **Code execution**: Does the code run without errors?
2. **Artifacts generated**: Are output files created with correct format/content?
3. **Function behavior**: Do imported functions produce correct results with known inputs?
4. **Known input/output pairs**: Test with fixtures, verify expected outputs

**Bad tests check:**
- Specific code patterns ("import logging" string matching)
- Function names or variable names (students may name differently)
- Code structure or style (unless that's the learning objective)

**Testing patterns:**

```python
# GOOD: Test execution and output
def test_script_produces_correct_output():
    result = subprocess.run(["python", "script.py", "input.csv"], capture_output=True)
    assert result.returncode == 0
    assert Path("output.csv").exists()
    df = pd.read_csv("output.csv")
    assert len(df) == 50  # Expected number of rows
    assert df["bmi"].between(15, 50).all()  # Sanity check values

# GOOD: Test imported function behavior
def test_validation_function():
    from student_module import validate_bounds  # Import their function
    valid_df = pd.DataFrame({"weight_kg": [70], "height_cm": [175]})
    assert validate_bounds(valid_df) is not None  # Should pass

    invalid_df = pd.DataFrame({"weight_kg": [5], "height_cm": [175]})
    with pytest.raises(ValueError):
        validate_bounds(invalid_df)  # Should fail

# BAD: Test code patterns
def test_has_logging():  # Too brittle!
    source = notebook_source()
    assert "import logging" in source  # Students might use print() or custom logger
```

**For notebooks:**
- Execute with `nbconvert --execute` and check exit code
- Read outputs from executed notebook cells
- Or import functions from converted `.py` and test directly

**References:**
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
