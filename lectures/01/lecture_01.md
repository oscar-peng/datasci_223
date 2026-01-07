---
lecture_number: 01
pdf: true
---

01: Defensive Programming and Debugging 🐛

# Links & Self-guided Review

- `#FIXME` Classroom invite
- [GitHub Education](https://education.github.com/pack)
- [DS-217 Lecture 01](https://www.notion.so/1-Python-the-Command-Line-and-VS-Code-271d9fdd1a1a805784e1fe68dc985696?pvs=21)
- [Markdown Tutorial](https://www.markdownguide.org/basic-syntax/)
- [Shell Basics](https://swcarpentry.github.io/shell-novice/)
- [Exercism Python Basics](https://exercism.org/tracks/python)
- [GitHub Hello World](https://docs.github.com/en/get-started/start-your-journey/hello-world)

# But First, A Blast from the Past

## Carryovers from DataSci-217

![XKCD: Git](media/xkcd_git.png)

- You already know Python, git/Markdown, and VS Code basics—this lecture focuses on reliability and debugging.
- Pick one workflow (local venv or Codespaces) and stick with it to reduce surprises.
- PHI reminder: never log or commit identifiable patient data.

### Reference: DS-217 carryovers

| Topic         | What to reuse                       |
| ------------- | ----------------------------------- |
| Python basics | Functions, imports, venv activation |
| Git hygiene   | Small commits, meaningful messages  |
| Markdown      | Headings, fenced code blocks, links |

### Code Snippet: Warmup commands

```bash
python -m venv .venv && source .venv/bin/activate
git status && git commit -am "chore: warm up"
```

## Command line quick hits

- Same commands everywhere: use the CLI for speed and reproducibility.
- Shell in Jupyter works too (`!ls`, `!pwd`), but keep paths relative.

### Reference: Workflow commands

| Command          | Purpose                   |
| ---------------- | ------------------------- |
| `pwd`            | Show current directory    |
| `ls -la`         | List files (long, hidden) |
| `cd <path>`      | Change directory          |
| `cp <src> <dst>` | Copy files                |
| `mv <src> <dst>` | Move/rename               |
| `rm <file>`      | Remove file (careful)     |

### Code Snippet: Shell basics

```bash
pwd
ls -la
cd lectures/01
```

## Workflow: local venv or Codespaces

![Codespaces debug icon](media/debug-icon.png)
![Python import tips](media/python_import.webp)

- Try the different ways of doing things and pick one workflow (local venv or Codespaces) and stick to it for fewer surprises.
- Local venv for performance/PHI; Codespaces for consistency and easy onboarding.
- Windows: WSL2 + `.venv/Scripts/activate` mirrors Linux/Codespaces.
- VS Code: Python + Jupyter extensions, format-on-save, debugger panel.

### Reference: Workflow setup

| Command                           | Purpose                     |
| --------------------------------- | --------------------------- |
| `python -m venv .venv`            | Create isolated environment |
| `source .venv/bin/activate`       | Activate venv (Linux/macOS) |
| `.venv\\Scripts\\activate`        | Activate venv (Windows)     |
| `pip install -r requirements.txt` | Install course dependencies |

### Code Snippet: venv + install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Notebook hygiene and reproducibility

![Notebook cleared outputs](media/jupyter_clear.png)
![XKCD: Data Trap](media/data_trap_2x.png)

- Run-all ready, deterministic, and no stray outputs or secrets.
- Clear outputs before commits unless the output is the point.
- Keep configs/paths in YAML or `.env`; avoid hardcoded secrets or PHI.

### Reference: Notebook hygiene

| Practice       | Why it matters                    |
| -------------- | --------------------------------- |
| Clear outputs  | Prevent stale screenshots/results |
| Pin deps       | Reproducible environments         |
| Relative paths | Portability across machines       |

### Code Snippet: Clear outputs

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace lecture.ipynb
```

## Jupyter magics & shell in notebooks

![Notebook cleared outputs](media/jupyter_clear.png)

- Magics speed up debugging and profiling; shell commands help inspect files without leaving the notebook.

### Reference: Jupyter magics

| Magic            | Purpose                     |
| ---------------- | --------------------------- |
| `%pwd`, `%ls`    | Where am I / list files     |
| `%run script.py` | Run another script/notebook |
| `%timeit expr`   | Quick timing                |
| `%%bash`         | Run a bash cell             |
| `!ls data`       | Shell command from a cell   |

### Code Snippet: Magics

```python
%pwd
%timeit [x**2 for x in range(1000)]
!ls data
```

## Git/GitHub/Markdown in 5 minutes

![Git branches at a glance](media/git_branches.png)

- Minimal loop: status → add → commit → push.
- Markdown: one `#` title, structured headings, fenced code blocks.
- GUI (VS Code Source Control) is fine if it keeps you moving.

### Reference: Git/Markdown cheatsheet

| Command                         | Purpose                     |
| ------------------------------- | --------------------------- |
| `git status`                    | See staged/unstaged changes |
| `git add <path>`                | Stage files                 |
| `git commit -m "feat: message"` | Save a snapshot             |
| `git push`                      | Sync to GitHub              |
| `git config user.email`         | Set author email            |

| Markdown      | Purpose                   |
| ------------- | ------------------------- |
| `# Heading`   | Section titles            |
| `- bullet`    | Lists                     |
| ` ` `lang`    | Code fences with language |
| `[text](url)` | Links                     |

### Code Snippet: Git loop

```bash
git status
git add README.md
git commit -m "chore: refresh setup notes"
git push
```

# LIVE DEMO!

# Defensive programming for data science

![Linter reminder](media/linter.png)

## Guardrails overview

- Guard against schema drift, unit mismatches, and bad inputs.
- Centralize config; keep helpers small; prefer pure functions.

### Reference: Guardrails

| Guardrail                   | Purpose                       |
| --------------------------- | ----------------------------- |
| Assertions on schema/units  | Fail fast on bad data         |
| Config files (`.env`, YAML) | Avoid hardcoded paths/secrets |
| Linters (`ruff`, `flake8`)  | Catch bugs before runtime     |

### Code Snippet: Load settings

```python
import yaml
from pathlib import Path

def load_settings(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())

SETTINGS = load_settings(Path("config.yaml"))
```

## Common failure modes in health data projects

![XKCD: Error Types](media/xkcd_error_types.png)

- Missing columns, unexpected units, unseeded randomness.
- Environment drift: different Python versions or stale venvs.
- PHI leaks via logs or screenshots.

### Reference: Failure modes

| Risk            | Quick guard                           |
| --------------- | ------------------------------------- |
| Missing columns | `assert_expected_columns`             |
| Unit drift      | Normalize units + validate ranges     |
| Stale env       | Recreate venv from `requirements.txt` |

### Code Snippet: Assert schema

```python
def assert_expected_columns(df, expected):
    missing = [c for c in expected if c not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
```

## DRY/KISS, linters, configs

![Linter reminder](media/linter.png)

- DRY/KISS: one helper beats repeated snippets.
- Pure functions simplify testing.
- Linters catch typos and unused imports early.

### Reference: DRY/KISS tools

| Tool           | What it prevents        |
| -------------- | ----------------------- |
| Config files   | Hardcoded paths/secrets |
| Linters        | Typos, unused code      |
| Pure functions | Hidden side effects     |

### Code Snippet: Config helper

```python
import yaml
from pathlib import Path

def load_settings(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())
```


## Code quality tools (quick sweep)

![Linter reminder](media/linter.png)

- Formatters (`black`) and linters (`ruff`) keep code clean; tests catch regressions.
- Run before committing or wire into a pre-commit hook.

### Reference: Quality tools

| Tool     | Purpose                |
| -------- | ---------------------- |
| `ruff`   | Lint/format fast       |
| `black`  | Consistent formatting  |
| `pytest` | Execute tests/fixtures |

### Code Snippet: Lint/format/test

```bash
ruff check .
black .
pytest -q
```

## Exceptions, logging, and safe exits

![XKCD: Compiler Complaint](media/xkcd_compiler_complaint.png)

- Raise specific exceptions; avoid bare `except`.
- Log at the right level; no PHI in logs.
- Fail fast with actionable messages.

### Reference: Logging/exceptions

| Logging level | Use for                    |
| ------------- | -------------------------- |
| INFO          | High-level progress        |
| WARNING       | Non-blocking issues        |
| ERROR         | Failures needing attention |

### Code Snippet: Logging with checks

```python
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def load_clean_data(path: str) -> list[dict]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input: {csv_path}")
    logging.info("Reading %s", csv_path)
    return csv_path.read_text().splitlines()
```

## Data security and ethics (quick hits)

![XKCD: Data Trap](media/data_trap_2x.png)

- No PHI in logs, screenshots, or public clouds; strip identifiers early.
- Use least-privilege access; encrypt at rest if handling real data.
- Keep audit trails for data pulls; document IRB/DUAs where relevant.

### Reference: PHI/ethics

| Risk                | Guardrail                                |
| ------------------- | ---------------------------------------- |
| PHI exposure        | De-identify; avoid logging identifiers   |
| Unauthorized access | Least privilege; access reviews          |
| Data copies         | Centralize storage; avoid ad-hoc exports |

### Code Snippet: Redact PHI

```python
def redact_phi(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in {"name", "mrn", "dob"}}
```

# LIVE DEMO!!

# Debugging in VS Code + Jupyter

## Debugging toolkit overview

![When code works first time (suspicious)](media/code-worked-the-first-time-suspicious.jpg)

- Start simple with prints/logging; move to pdb/VS Code for deeper inspection.
- Breakpoints + Variables/Watch/Debug Console = see state without littering prints.

### Reference: Debugging toolkit

| Tool               | Use case                   |
| ------------------ | -------------------------- |
| `print(f"{var=}")` | Quick value checks         |
| `breakpoint()`     | Drop into pdb              |
| VS Code debugger   | Visual stepping/inspection |

### Code Snippet: Print + calc

```python
def calculate_bmi(weight_kg, height_m):
    print(f"{weight_kg=}, {height_m=}")
    bmi = weight_kg / (height_m ** 2)
    print(f"{bmi=}")
    return bmi
```

## Print debugging: start here

![XKCD: Debugging](media/xkcd_debugging.png)
![#FIXME: screenshot of VS Code gutter breakpoint set on BMI script](media/xkcd_debugging.png)

- Use f-strings with `{var=}` to see names + values.
- Remove prints before commit or migrate to logging.

### Reference: Print patterns

| Pattern                 | Purpose            |
| ----------------------- | ------------------ |
| `print(f"{df.shape=}")` | Check dimensions   |
| `print(f"{row=}")`      | Inspect loop state |
| `print(f"{result=}")`   | Verify outputs     |

### Code Snippet: Print debugging

```python
def calculate_bmi(weight_kg, height_m):
    print(f"{weight_kg=}, {height_m=}")
    return weight_kg / (height_m ** 2)
```

## pdb and VS Code debugger

![VS Code debug panels](media/debug_view.png)

- pdb/ipdb for terminal; VS Code for visuals and conditional breakpoints.
- Break on exception with `breakpoint()` inside `except`.

### Reference: Breakpoints & commands

| Tool/command   | Purpose                      |
| -------------- | ---------------------------- |
| `n / s / c`    | Next, step into, continue    |
| `p var`        | Print variable               |
| Conditional BP | Pause when expression true   |
| Logpoint       | Print a message without stop |
| `breakpoint()` | Drop into pdb on exception   |

### Code Snippet: Conditional + logpoint

```python
try:
    risky_fn()
except Exception:
    breakpoint()  # pdb session

# VS Code logpoint: right-click breakpoint -> "Add Logpoint"
# Message example: "value={value}"
```

## Runtime variable inspection in VS Code

![VS Code debug panels](media/debug_view.png)
![#FIXME: screenshot of VS Code conditional breakpoint dialog](media/debug_view.png)

- Variables panel shows locals/globals; expand DataFrames.
- Watch expressions track custom values.
- Debug Console evaluates code while paused.

### Reference: VS Code panels

| Panel         | Purpose                         |
| ------------- | ------------------------------- |
| Variables     | Inspect state at breakpoint     |
| Watch         | Track expressions (`df.shape`)  |
| Debug Console | Run ad-hoc checks (`df.head()`) |

### Code Snippet: Inspect while paused

```python
# Pause at breakpoint, then:
# - Check Variables panel
# - Add Watch: df.shape
# - Debug Console: df.dtypes
```

## VS Code debugger basics (scripts)

![Debug run button](media/debug-run.png)
![#FIXME: Screenshot of clicking the gutter to set a breakpoint in vscode_debug_sample.py](media/debug-run.png)

- Click the gutter to set breakpoints (red dot), then use Run and Debug (F5) or the play button to start the Python debugger.
- In Run and Debug, pick the Python config or accept the default; make sure the correct interpreter/venv is selected.
- Inspect call stack, Variables, and Watch panels while stepping; launch.json is optional because the Python extension supplies defaults.

### Reference: launch.json fields

| Field     | Meaning              |
| --------- | -------------------- |
| `program` | Entry script         |
| `request` | `launch` vs `attach` |
| `type`    | `python`             |

### Code Snippet: launch config

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug BMI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/demo/vscode_debug_sample.py"
        }
    ]
}
```

## Debugging notebooks in VS Code

![XKCD: Debugger](media/xkcd_debugger.png)
![#FIXME: Screenshot highlighting the notebook cell debug button and an inline breakpoint](media/xkcd_debugger.png)

- Click the debug icon on the cell; set breakpoints inside.
- Restart kernel before Run All after debugging.

### Reference: Notebook debugging steps

| Step                | Purpose                        |
| ------------------- | ------------------------------ |
| Debug cell button   | Start a notebook debug session |
| Breakpoints in cell | Pause where needed             |
| Restart kernel      | Clear state after debugging    |

### Code Snippet: Debug a cell

```python
#%% Debug this cell
from demo.vscode_debug_sample import calculate_bmi
calculate_bmi(80, 1.75)
```

## Debugging checklist for messy data

![XKCD: Existential Bug Reports](media/xkcd_existential_bug_reports.png)

- Reproduce with the smallest failing fixture.
- Check assumptions (types, units, nulls) before changing code.
- Add assertions/logging near the failure and rerun.

### Reference: Debugging checklist

| Step      | Goal                            |
| --------- | ------------------------------- |
| Reproduce | Confirm the failure             |
| Minimize  | Small fixture for fast loops    |
| Guard     | Assertions/logging close to bug |

### Code Snippet: Reproduce bug

```python
from pathlib import Path

fixture = Path("demo/data/patient_intake_missing_height.csv")
try:
    load_intake_data(fixture)
except Exception as err:
    print("Reproduced:", err)
```

## Tests to lock in fixes

![Debug run button](media/debug-run.png)

- Save failing fixtures and add tiny tests so bugs stay fixed.
- Prefer small, deterministic inputs; avoid brittle expectations.

### Reference: Tests & fixtures

| Tool       | Use                   |
| ---------- | --------------------- |
| `pytest`   | Run quick checks      |
| `tmp_path` | Temp dirs for outputs |
| Fixtures   | Reuse failing inputs  |

### Code Snippet: Bounds test

```python
import pandas as pd

def test_bmi_bounds():
    from demo.vscode_debug_sample import calculate_bmi
    assert 15 < calculate_bmi(70, 1.75) < 50
```

# Read the docs

![Read the docs](media/read-the-docs.jpeg)
Rubber ducking is still undefeated for finding your own bugs.

# LIVE DEMO!!!
