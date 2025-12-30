# Demo 1: Setup and Environment Verification (~30 min)

This demo walks through accepting the GitHub Classroom assignment, setting up your development environment, and verifying everything works before writing code.

## Part 1: GitHub Classroom Setup

### Accept the assignment
1. Click the GitHub Classroom invite link (provided in class)
2. Authorize GitHub Classroom if this is your first time
3. Accept the assignment—this creates a private repo for you

### Enable GitHub Education benefits (if not already done)
- Visit [GitHub Education Pack](https://education.github.com/pack)
- Sign up with your UCSF email
- Benefits include: Copilot, advanced Codespaces hours, free hosting, etc.

## Part 2: Environment Options

Choose **one** workflow and stick with it throughout the course:

### Option A: GitHub Codespaces (recommended for consistency)
**Pros:** Same environment for everyone, nothing to install locally, works on any device
**Cons:** Requires internet, limited free hours (but plenty with Education pack)

1. From your assignment repo page, click **Code** → **Codespaces** → **Create codespace on main**
2. Wait for container to build (uses `.devcontainer` if provided)
3. Terminal opens automatically—you're ready to code

### Option B: Local development (recommended for performance/PHI work)
**Pros:** Faster, works offline, keeps PHI local
**Cons:** Environment differences across machines, more setup

**Mac/Linux:**
```bash
git clone <your-repo-url>
cd <repo-name>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
git clone <your-repo-url>
cd <repo-name>
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows users:** Consider [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) for a Linux environment that matches Codespaces.

## Part 3: Verify Your Setup

### Test 1: Python and packages
```bash
python --version          # Should be 3.11 or 3.12
pip list                  # Check pandas, jupyter, pytest installed
```

### Test 2: Jupyter notebooks
```bash
jupyter notebook         # Should open browser
# OR in VS Code: Open a .ipynb file, select kernel, run a cell
```

### Test 3: Run All on starter notebook
1. Open `starter.ipynb` (or similar provided notebook)
2. Click **Run All** (or **Restart Kernel and Run All**)
3. All cells should execute without errors
4. If errors appear, check requirements.txt installation

### Test 4: Git workflow
```bash
git status                              # Should show clean or untracked files
echo "# Test" > test.md                 # Create test file
git add test.md
git commit -m "test: verify git works"
git push
```

Check your GitHub repo—the commit should appear.

## Part 4: VS Code Essentials

Install these extensions if using local VS Code:
- **Python** (Microsoft): Linting, debugging, IntelliSense
- **Jupyter** (Microsoft): Notebook support in VS Code
- **Pylance** (Microsoft): Fast type checking and autocomplete

**Recommended settings:**
- Format on save: `File → Preferences → Settings → "format on save"`
- Auto-save: `File → Auto Save`

**Command Palette** (`Cmd+Shift+P` / `Ctrl+Shift+P`): Your best friend
- `Python: Select Interpreter` → choose your `.venv`
- `Format Document` → auto-format code
- `Developer: Reload Window` → restart if things get weird

## Success Checklist

- [ ] Accepted GitHub Classroom assignment
- [ ] Environment set up (Codespaces or local venv)
- [ ] `pip install -r requirements.txt` completed successfully
- [ ] Starter notebook runs without errors
- [ ] Test commit pushed to GitHub
- [ ] VS Code extensions installed (if local)

## Common Issues

**Import errors even after pip install:**
- VS Code: Check you selected the right Python interpreter (Command Palette → "Python: Select Interpreter")
- Terminal: Make sure venv is activated (you should see `(.venv)` in prompt)

**Jupyter kernel not found:**
- Install ipykernel: `pip install ipykernel`
- Register kernel: `python -m ipykernel install --user --name=.venv`

**Git push rejected:**
- Check you're on the right branch: `git branch`
- Make sure you committed: `git status`
- If first push: `git push -u origin main`

## Next Steps

Once your environment is verified:
1. Clear test files (`git rm test.md && git commit -m "chore: remove test file"`)
2. Keep this setup throughout the course—don't switch environments mid-semester
3. Proceed to Demo 2 for defensive programming exercises

## Resources

- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [GitHub Codespaces Docs](https://docs.github.com/en/codespaces)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [DataSci 217 Lecture 01](https://www.notion.so/1-Python-the-Command-Line-and-VS-Code-271d9fdd1a1a805784e1fe68dc985696) (deep dive on tooling)
