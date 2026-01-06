---
lecture_number: 01
pdf: true
---

# Lecture 1: Reliable Notebooks and Debugging First 🚦🐛

## Quick hits: setup + hygiene

**Demo: Accept the course repo via GitHub Classroom + enable GitHub Education perks.**  
Links: [GitHub Education](https://education.github.com/pack) · `#FIXME` Classroom invite · [DS-217 Lecture 01 deep-dive on tooling](https://www.notion.so/1-Python-the-Command-Line-and-VS-Code-271d9fdd1a1a805784e1fe68dc985696?pvs=21).

### What carries over from the prereq
<!---
Reassure students they already know the basics; emphasize this session is about reliability. Mention we'll move faster than last year and lean on links for depth.
--->
Summary: Quick reminder that Python, git, Markdown, and VS Code basics already exist so we can focus on reliability and debugging.
Visual: ![XKCD: Git](media/xkcd_git.png)
Signature: `python -m venv .venv && source .venv/bin/activate`
*venv = virtual environment (isolated Python packages for this project)*
Example:
```bash
git status && git commit -am "chore: warm up"
```
- Assume familiarity with Python syntax, basic git, Markdown, and VS Code navigation.
- Skip long installs: choose one path for local (venv + VS Code) or cloud (Codespaces).
- Detailed setup lives in DS-217 Lecture 01 (link above) and prior notes in `lectures_25/01`.
- **PHI (Protected Health Information)**: Patient data requiring special security—never commit to public repos or log in plain text.

### Fast local/cloud workflow
<!---
Explain that choosing a single environment reduces friction. Encourage Codespaces for consistency, local venv for performance/PHI. Mention WSL briefly for Windows. For Windows activation: `.venv\Scripts\activate` instead of `source .venv/bin/activate`.
--->
Summary: Pick one workflow (local venv or Codespaces) and stick to it for predictable grading and fewer surprises.
Visual: ![Codespaces debug icon](media/debug-icon.png)
Signature: `codespace.create(repo, machine="small")`
Example:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- Local: `python -m venv .venv && source .venv/bin/activate` then `pip install -r requirements.txt`. (Windows: `.venv\Scripts\activate`)
  - **pip**: Python's package installer (installs libraries listed in requirements.txt)
- Cloud: GitHub Codespaces → select the repo → pick a small machine → reuse **devcontainer** if provided (pre-configured development environment).
- Windows users: consider WSL2 for a Linux environment inside Windows—matches Codespaces and avoids path headaches.
- VS Code essentials: Python + Jupyter extensions; Command Palette for everything; auto-format on save.
- The `-m` flag runs a module as a script (e.g., `python -m venv` runs the venv module).

### Notebook hygiene and reproducibility
<!---
Highlight “run-all ready” notebooks, cleared outputs on commit, deterministic runs, and config separation. Mention why this matters for grading and team science.
--->
Summary: Notebooks must be run-all ready, deterministic, and free of stray outputs or secret paths.
Visual: ![XKCD: Data Trap](media/data_trap_2x.png)
Signature: `def run_all(notebook_path: Path) -> None`
Example:
```python
# Clear outputs before commit
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace lecture.ipynb
```
- Notebooks should run top-to-bottom without manual tweaks; add clear `# TODO` if not.
- Clear outputs before committing unless the output is the point; keep plots lightweight.
- Capture environment: pin deps in `requirements.txt`, store configs in `.env` or YAML, never hardcode secrets/paths.
- Use relative paths and small sample data for demos; document larger data sources.

### Git/GitHub/Markdown in 5 minutes
<!---
Offer the minimal command set and when to use GUI. Encourage short commits and descriptive messages. Note Markdown basics for README and notebooks.
--->
Summary: Minimal git/Markdown toolkit for fast, clean commits and readable docs.
Visual: ![XKCD: Git Commit](media/xkcd_git_commit.png)
Signature: `git commit -m "feat: summary"`
Example:
```markdown
# Title
## Section
- bullet
`code`
```
- Git flow: `git status` → `git add` → `git commit -m "feat: short message"` → `git push`.
- Use GitHub Desktop or VS Code Source Control if the CLI slows you down.
- Markdown recap: one `#` title per doc, headings for structure, fenced code blocks with language tags, link with `[text](url)`.

### Demo (~30 min): Accept and open the starter repo
<!---
Walkthrough: open the classroom link, create repo, clone or open in Codespaces, run `pip install -r requirements.txt`, verify notebooks open. Note this sets up grading later.
--->
Summary: Accept Classroom, clone/open, install deps, and prove Run All works before coding.
Visual: #FIXME Add screenshot of Classroom acceptance flow (clone or Codespaces).
Signature: `gh classroom accept <invite-url>`
Example:
```bash
gh repo clone <classroom-repo>
cd <classroom-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && jupyter nbconvert --execute starter.ipynb
```
- Accept the GitHub Classroom invite (`#FIXME` link) and enable the GitHub Education pack if not already.
- Open the repo locally or in Codespaces; verify `.venv` or devcontainer activation.
- Run `pip install -r requirements.txt`; open the starter notebook and confirm it runs `Run All` without edits.

## Defensive programming for data science

**Demo: Hardening a tiny data-cleaning notebook before it breaks.**  
Links: [`defensive_programming_notebook.ipynb`](./demo/defensive_programming_notebook.ipynb) · [`logging` docs](https://docs.python.org/3/library/logging.html).

### Common failure modes in health data projects
<!---
Frame debugging as risk management: data drift, messy inputs, environment drift, bad assumptions. Use light humor (ghost of missing values).
--->
Summary: How health datasets fail—missing columns, unit drift, stale environments, and silent PHI leaks.
Visual: ![XKCD: Error Types](media/xkcd_error_types.png)
Signature: `def assert_expected_columns(df: pd.DataFrame, expected: list[str]) -> None`
Example:
```python
def assert_expected_columns(df, expected):
    missing = [c for c in expected if c not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
```
- Data surprises: missing columns, unexpected units, **schema drift** between hospitals (schema = expected structure/types of data columns).
- Environment drift: different Python versions, missing packages, stale virtualenvs.
- Hidden assumptions: hardcoded file paths, magic numbers, unseeded randomness.
- Security/ethics: avoid PHI in logs, public clouds, or screenshots.

### Guardrails: DRY/KISS, linters, and configs over hardcoding
<!---
Show how small helpers/configs reduce bugs. Emphasize clarity and minimal abstractions. Mention `.env` + `pydantic` or `dotenv` as optional. Define DRY/KISS for beginners.
--->
Summary: Centralize settings and keep helpers tiny so you can reuse them and spot side effects.

**Quick definitions:**
- **DRY** (Don't Repeat Yourself): Extract repeated code into functions—one place to fix bugs.
- **KISS** (Keep It Simple): Prefer clear, obvious code over clever one-liners.
- **Pure functions**: Functions that always return the same output for the same input and don't modify external state—easier to test and debug.
Visual: ![Linter reminder](media/linter.png)
Signature: `def load_settings(config_path: Path) -> dict`
Example:
```python
import yaml
from pathlib import Path

def load_settings(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())  # YAML: human-readable config format

SETTINGS = load_settings(Path("config.yaml"))
```
- Centralize settings (e.g., `config.yaml` or `.env`) and load them once.
- Keep helper functions small and named for intent; avoid clever one-liners that hide side effects.
- Validate inputs early: **assert** expected columns, value ranges, and units (assertions = checks that raise errors if assumptions violated).
- Prefer pure functions where possible so tests are easy.
- **Use a linter** (e.g., `ruff`, `flake8`): catches typos, unused imports, and style issues before you run the code. VS Code can run linters on save.

### Exceptions, logging, and safe exits
<!---
Students often overuse bare except. Show structured exceptions, logging levels, and graceful fallbacks. Mention PHI-safe logging.
--->
Summary: Raise specific exceptions, log clearly, and fail fast without leaking PHI.
Visual: ![XKCD: Compiler Complaint](media/xkcd_compiler_complaint.png)
Signature: `def load_clean_data(path: str) -> list[dict]`
Example:
```python
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def load_clean_data(path: str) -> list[dict]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input: {csv_path}")
    logging.info("Reading %s", csv_path)
    # TODO: add schema validation
    return csv_path.read_text().splitlines()

try:
    rows = load_clean_data("data/intake.csv")
except FileNotFoundError as err:
    logging.error("Check your path or fetch the sample data: %s", err)
```

### Demo (~60 min): Make the notebook harder to break
<!---
Practice adding assertions, config loading, and logging to an existing notebook. Show before/after of a failing cell now giving actionable errors. Encourage students to try with bad input.
--->
Summary: Run the cleaning notebook with broken inputs, add schema/bounds checks and logging, rerun to see actionable errors.
Visual: ![Debugging gif](media/debugging.gif)
Signature: `def validate_values(df: pd.DataFrame) -> pd.DataFrame`
Example:
```python
import pandas as pd

def validate_values(df: pd.DataFrame) -> pd.DataFrame:
    bounds = {"weight_kg": (30, 250), "height_cm": (120, 230)}
    for col, (lower, upper) in bounds.items():
        bad = ~df[col].between(lower, upper)
        if bad.any():
            raise ValueError(f"{col} out of bounds: {df.loc[bad, ['patient_id', col]]}")
    return df

df = pd.read_csv("demo/data/patient_intake_bad_values.csv")
validate_values(df)
```
- Start with a brittle cleaning notebook ([demo notebook](./demo/defensive_programming_notebook.ipynb)); run it with a missing column to see the failure.
- Add config-driven file paths, schema checks, and logging statements.
- Re-run with both good and bad inputs; confirm errors are now descriptive and logged.

## Debugging in VS Code + Jupyter

**Demo: Step-through debugging in VS Code for scripts and notebooks.**  
Links: [VS Code Python debugging](https://code.visualstudio.com/docs/python/debugging) · [`vscode_debug_sample.py`](./demo/vscode_debug_sample.py) · [`vscode_debug_walkthrough.md`](./demo/vscode_debug_walkthrough.md) · screenshots below.

### Print debugging: start here
<!---
Cover print debugging first—it's the most accessible technique for beginners. Show strategic print placement and f-string tricks.
--->
Summary: Print statements are your first debugging tool—fast, simple, and works everywhere.
Visual: ![When code works first time (suspicious)](media/code-worked-the-first-time-suspicious.jpg)
Signature: `print(f"{var=}")`  # Python 3.8+ shows variable name and value
Example:
```python
def calculate_bmi(weight_kg, height_m):
    print(f"{weight_kg=}, {height_m=}")  # See inputs
    bmi = weight_kg / (height_m ** 2)
    print(f"{bmi=}")  # See output
    return bmi
```
- Start with `print()` statements at key points: function entry, before/after transforms, inside loops.
- Use f-string `{var=}` syntax (Python 3.8+) to print both name and value: `print(f"{df.shape=}")`.
- For persistent debugging, switch to `logging` so you can toggle verbosity without removing code.
- Remove or comment out prints before committing—or graduate to logging.

### When prints aren't enough: pdb and VS Code debugger
<!---
Outline when to use print/logging vs. debugger. Mention pdb for terminal-only cases and VS Code for visual learners. Keep tone light (detective reference).
--->
Summary: Move to pdb or VS Code debugger when you need state inspection and call stacks.
Visual: ![XKCD: Debugging](media/xkcd_debugging.png)
Signature: `import pdb; pdb.set_trace()`  # or just `breakpoint()` in Python 3.7+
Example:
```python
try:
    risky_fn()
except Exception:
    import pdb; pdb.set_trace()  # Drop into debugger on error
```
- **pdb** (Python Debugger): built-in, terminal-based, works anywhere. Commands: `n` (next), `s` (step into), `c` (continue), `p var` (print).
- **ipdb**: like pdb but with IPython features (tab completion, colors). Install: `pip install ipdb`.
- **breakpoint()**: Python 3.7+ builtin that drops into pdb (replaces `import pdb; pdb.set_trace()`).
- Switch to the VS Code debugger for call stacks, watches, and conditional breakpoints.
- Debugging is like being a detective in a crime movie where you're also the culprit.

**Which debugging tool should you use?**

- **Quick variable check?** → `print(f"{var=}")`
- **See how state changes over time?** → VS Code debugger with breakpoints
- **Debugging via SSH/remote terminal?** → `pdb` or `breakpoint()`
- **Want persistent debug messages?** → `logging` module (can toggle on/off)

### Runtime variable inspection in VS Code
<!---
Explicitly show how to inspect variables at runtime using VS Code's debugging panels. Many students know how to set breakpoints but don't leverage the Variables/Watch/Debug Console panels effectively.
--->
Summary: Use Variables panel, Watch expressions, and Debug Console to inspect state without littering code with print statements.
Visual: ![VS Code debug panels](media/debug_view.png)
Signature: Variables panel auto-populates when paused at breakpoint
Example:
```python
# When paused at breakpoint, inspect:
# - Variables panel: see all locals/globals
# - Watch: add expressions like `df.shape`, `len(results)`
# - Debug Console: evaluate `df.head()`, `type(variable)`
```
- **Variables panel:** Auto-shows all local/global variables when paused; expand DataFrames to see shape/dtypes/head
- **Watch expressions:** Add custom expressions (e.g., `len(patients)`, `patients["age"].max()`) that update at each step
- **Debug Console:** Evaluate any Python expression while paused: `df.describe()`, `type(variable)`, `variable.keys()`
- **Call Stack:** Shows function call chain—click frames to inspect variables at different levels
- **Hover inspection:** Hover over variables in code to see current values inline

**Key workflow for scripts:**
1. Set breakpoint → run debugger → execution pauses
2. Check Variables panel for unexpected values
3. Add Watch for calculated expressions (e.g., `row["bmi"] > 30`)
4. Use Debug Console to run methods: `df.info()`, `df["col"].value_counts()`
5. Step through and watch expressions update

**Key workflow for notebooks:**
1. Click debug icon on cell → debugger starts
2. Set breakpoint inside cell code
3. Same Variables/Watch/Debug Console features as scripts
4. After debugging, restart kernel to clear state

### VS Code debugger basics (scripts)
<!---
Describe setting breakpoints, inspecting variables, and stepping controls. Note launch.json is optional with the Python extension. Add placeholder for a screenshot.
--->
Summary: Set breakpoints, run under the debugger, and use call stack/watches to trace BMI bugs.
Visual: ![VS Code debug view](media/debug_view.png)
![Debug run button](media/debug-run.png)
Signature: `"request": "launch"` entry in `.vscode/launch.json`
Example:
```json
{
  "version": "0.2.0",
  "configurations": [{
    "name": "Debug BMI",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/demo/vscode_debug_sample.py"
  }]
}
```
- Open the script, click in the gutter to set breakpoints, run with the debug play button.
- Watch/Variables panels reveal state; Call Stack shows frames; use Step Into/Over/Out.
- Conditional breakpoints: right-click breakpoint → add expression (e.g., `row["bmi"] is None`).

### Debugging notebooks in VS Code
<!---
Students may not know notebook debugging exists. Show the debug cell button and how it maps to standard debugger controls. Warn about state carried between cells.
--->
Summary: Use the debug button per cell, set breakpoints inside cells, and restart kernels before Run All.
Visual: ![XKCD: Debugger](media/xkcd_debugger.png)
Signature: `#%%` cell debug blocks in Python files map to notebook-style debugging.
Example:
```python
#%% Debug this cell
from demo.vscode_debug_sample import calculate_bmi
calculate_bmi(80, 1.75)
```
- In VS Code, use the debug icon beside a cell to enter a debug session for that cell.
- Breakpoints work inside notebook cells; continue/step behaves like scripts.
- Restart kernel + Run All after debugging to confirm clean state.

### Debugging checklist for messy data
<!---
Provide a reusable flow: reproduce, minimize, inspect assumptions, add guards, re-run. Emphasize saving small failing fixtures for tests.
--->
Summary: Reproduce with tiny fixtures, check assumptions, add assertions/logging, and rerun until stable.
Visual: ![XKCD: Existential Bug Reports](media/xkcd_existential_bug_reports.png)
Signature: `def reproduce_bug(input_path: Path) -> None`
Example:
```python
fixture = Path("demo/data/patient_intake_missing_height.csv")
try:
    load_intake_data(fixture)
except Exception as err:
    print("Reproduced:", err)
```
- Reproduce the bug with the smallest possible input; save that fixture for future tests.
- Confirm assumptions (data types, units, ranges) before blaming the code.
- Add assertions and logging near the failure; rerun with breakpoints to inspect.
- Once fixed, add a minimal test or notebook cell that proves the fix stays fixed.

### Demo (~90 min): Walk through a VS Code debug session
<!---
Guide students through setting a breakpoint, stepping through a loop, and fixing a logic bug. Use a BMI calculator or similar from lecture_02. End by rerunning tests/notebook.
--->
Summary: Step through the BMI script, fix the formula/typo/indexing bugs, and rerun to verify clean output.
Visual: ![Rubber duck debugging](media/ducky.jpg)
Signature: `breakpoint()` builtin for quick stops
Example:
```python
from demo.vscode_debug_sample import calculate_bmi

breakpoint()
print(calculate_bmi(70, 1.75))
```
- Open the [`vscode_debug_sample.py`](./demo/vscode_debug_sample.py) (adapted from `lectures_25/02` BMI example).
- Set a breakpoint inside a loop, run the debugger, inspect variables, and adjust a faulty condition.
- Re-run the cell/notebook with Run All to confirm the fix and clean state.

## Assignment (auto-graded)
<!---
Clarify scaffold + grading so students know what to submit after the debugging focus.
--->
Summary: Auto-graded Classroom repo mirroring datasci_217 scaffold; prove logging/assertions and one VS Code debug walkthrough.
Visual: ![XKCD: New Bug](media/xkcd_new_bug.png)
Signature: `.github/tests/test_*` executed by GitHub Actions
Example:
```bash
pytest .github/tests -q
```

 - Lightweight, auto-gradable via GitHub Actions (mirrors the datasci_217 layout with `.github/tests`, `assignment.ipynb` + `assignment.md`, `requirements.txt`, `data/`, and `output/` folders).
- Focus: add logging + assertions to a small notebook, plus one VS Code debug walkthrough.
- Submission: accept the Classroom repo, push changes, verify Actions pass; rubric is fully automated.

## Resources
<!---
Point to deeper references and remind about tone; mention humor break to keep energy up.
--->
Summary: Where to dig deeper on tooling/debugging plus a comic to keep morale up.
Visual: ![XKCD: Compiling](media/xkcd_compiling.png)
Signature: `open("refs/instructions.md").read()`
Example:
```bash
sed -n '1,40p' refs/instructions.md
```

- Deep dives: DS-217 Lecture 01 (tooling) and `lectures_25/02` (debugging demos) plus the local `./demo` assets above.
- References: Python `logging`, VS Code debugging guide, and MkDocs notes in `refs/`.
- When in doubt, explain your code to a rubber duck. If it still doesn't make sense, the bug is in your assumptions.

![Read the docs](media/read-the-docs.jpeg)
