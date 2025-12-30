# Assignment 01: Reliable Notebooks and Debugging

**Due:** Before Lecture 02
**Points:** Pass/Fail (autograded)
**Skills:** Environment setup, defensive programming, VS Code debugging

---

## Overview

This assignment has **three parts** that build on Lecture 01 concepts:

1. **Email Verification** - Verify environment setup (1 script)
2. **Defensive Programming** - Add logging, validation, config to notebooks (3 notebooks)
3. **Debugging** - Fix bugs in script and notebook (1 script + 1 notebook)

**All work is autograded** via GitHub Actions - push frequently to see test results!

---

## Assignment Structure

```
assignment/
├── 01_process_email.py              # Part 1: Email verification
├── 02a_logging.ipynb                # Part 2a: Add logging
├── 02b_validation.ipynb             # Part 2b: Add validation
├── 02c_config_driven.ipynb          # Part 2c: Use config file
├── 03a_debug_script.py              # Part 3a: Debug Python script
├── 03b_debug_notebook.ipynb         # Part 3b: Debug notebook
├── config.yaml                      # Config for 02c
└── data/                            # Test data files
```

---

## Part 1: Email Verification (10%)

**File:** `01_process_email.py`

Verify your UCSF email and that your environment works.

### Tasks

1. Run the script with your UCSF email:
   ```bash
   python 01_process_email.py your.email@ucsf.edu
   ```

2. Verify output:
   ```bash
   cat processed_email.txt  # Should show a 64-character hash
   ```

3. Commit and push:
   ```bash
   git add processed_email.txt
   git commit -m "feat: add email verification"
   git push
   ```

### Success Criteria
- [ ] `processed_email.txt` exists with valid SHA256 hash
- [ ] Hash matches course roster

---

## Part 2: Defensive Programming (60%)

Convert brittle notebooks into robust, production-ready code through three progressive steps.

### Part 2a: Add Logging (15%)

**File:** `02a_logging.ipynb`

**Current state:** Notebook runs but provides no visibility into execution.

**Your task:** Add logging to trace execution and help debug issues.

**Requirements:**
1. Import and configure logging at the top:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
   ```

2. Add logging statements at key points:
   - After loading data: `logging.info("Loaded %d patients", len(df))`
   - Before calculations: `logging.info("Calculating BMI...")`
   - After filtering: `logging.info("Found %d high-risk patients", count)`

3. Run notebook - output should show execution trace

**Test:** All logging tests pass in pytest

---

### Part 2b: Add Validation (25%)

**File:** `02b_validation.ipynb`

**Current state:** Notebook crashes cryptically with bad data or produces wrong results silently.

**Your task:** Add schema and bounds validation with helpful error messages.

**Requirements:**

1. **Schema validation** - Check required columns exist:
   ```python
   required_cols = ["patient_id", "weight_kg", "height_cm", "age", "sex"]
   missing = [c for c in required_cols if c not in df.columns]
   if missing:
       raise ValueError(f"Missing columns: {missing}")
   ```

2. **Bounds validation** - Check realistic value ranges:
   ```python
   bounds = {
       "weight_kg": (30, 250),
       "height_cm": (120, 230),
       "age": (0, 110)
   }

   for col, (min_val, max_val) in bounds.items():
       out_of_bounds = ~df[col].between(min_val, max_val)
       if out_of_bounds.any():
           bad_rows = df.loc[out_of_bounds, ["patient_id", col]]
           raise ValueError(f"{col} out of bounds:\n{bad_rows}")
   ```

3. Test with bad data files (provided) - should fail with clear errors

**Test:** Run notebook with `data/patient_intake_bad_values.csv` - should raise helpful ValueError

---

### Part 2c: Config-Driven Development (20%)

**File:** `02c_config_driven.ipynb`

**Current state:** File paths and thresholds hardcoded throughout notebook.

**Your task:** Move all configuration to `config.yaml` file.

**Requirements:**

1. Create/update `config.yaml`:
   ```yaml
   data:
     input_file: "data/patient_intake.csv"

   bounds:
     weight_kg: {min: 30, max: 250}
     height_cm: {min: 120, max: 230}
     age: {min: 0, max: 110}

   bmi_thresholds:
     overweight: 25
     obese: 30
   ```

2. Load config in notebook:
   ```python
   import yaml
   from pathlib import Path

   with open("config.yaml") as f:
       config = yaml.safe_load(f)

   # Use config values
   data_path = Path(config["data"]["input_file"])
   bounds = config["bounds"]
   ```

3. Replace ALL hardcoded values with config lookups

**Test:** Change config values, rerun notebook - should use new values without code changes

---

## Part 3: Debugging with VS Code (30%)

Practice systematic debugging for both scripts and notebooks.

### Part 3a: Debug Python Script (15%)

**File:** `03a_debug_script.py`

**Current state:** Script has THREE intentional bugs.

**Your task:** Use VS Code debugger to find and fix all bugs, documenting each fix.

**Known bugs:**
1. **Formula error** - BMI calculation missing exponent
2. **NameError** - Variable typo (`catgory` vs `category`)
3. **IndexError** - Off-by-one loop error

**Debugging workflow:**
1. Set breakpoint on line with bug
2. Run debugger (Debug icon → Python File)
3. When paused:
   - Check **Variables panel** for unexpected values
   - Add **Watch expression** to compare correct calculation
   - Use **Debug Console** to test fixes
4. Fix bug and add comment:
   ```python
   # BUG 1: Used weight/height instead of weight/(height**2)
   # FIX: Added exponent for correct BMI formula
   bmi = weight / (height ** 2)
   ```

**Test:** Script runs without errors and produces correct BMI values

---

### Part 3b: Debug Notebook (15%)

**File:** `03b_debug_notebook.ipynb`

**Current state:** Notebook has THREE subtle bugs.

**Your task:** Use VS Code notebook debugger to find and fix bugs.

**Known bugs:**
1. **Type mismatch** - Age stored as string, comparison fails
2. **Off-by-one** - Loop skips first patient and crashes on last
3. **Logic error** - Risk categories inverted (high BMI = "low risk")

**Debugging workflow:**
1. Click **debug icon** beside cell (bug/play button)
2. Set breakpoint inside cell code
3. When paused:
   - **Variables panel:** Expand DataFrames, check dtypes
   - **Watch panel:** Add `df["age"].dtype` or `len(patients)`
   - **Debug Console:** Test `df.head()`, `df.info()`
4. Fix bug and add comment explaining the issue

**Test:** Restart kernel + Run All completes without errors

---

## Testing Your Work

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Test individual parts
python 01_process_email.py your.email@ucsf.edu
jupyter nbconvert --execute 02a_logging.ipynb
jupyter nbconvert --execute 02b_validation.ipynb
jupyter nbconvert --execute 02c_config_driven.ipynb
python 03a_debug_script.py
jupyter nbconvert --execute 03b_debug_notebook.ipynb

# Run all tests
pytest .github/tests/test_assignment.py -v
```

### GitHub Actions (Automatic)

Every push triggers autograding:
1. Push your changes
2. Go to **Actions** tab in GitHub
3. Click latest workflow run
4. Check which tests passed/failed
5. Fix failures, push again

---

## Submission Checklist

- [ ] `processed_email.txt` committed (Part 1)
- [ ] `02a_logging.ipynb` has logging statements
- [ ] `02b_validation.ipynb` has schema + bounds checks
- [ ] `02c_config_driven.ipynb` loads from `config.yaml`
- [ ] `03a_debug_script.py` all bugs fixed with comments
- [ ] `03b_debug_notebook.ipynb` all bugs fixed with comments
- [ ] All files pushed to GitHub
- [ ] GitHub Actions tests pass

---

## Grading Rubric

**Pass/Fail** based on autograding:

| Part | Weight | Criteria |
|------|--------|----------|
| **1. Email** | 10% | Valid hash file exists |
| **2a. Logging** | 15% | Logging configured and used throughout |
| **2b. Validation** | 25% | Schema + bounds validation with clear errors |
| **2c. Config** | 20% | All hardcoded values moved to config |
| **3a. Debug Script** | 15% | All bugs fixed with explanation comments |
| **3b. Debug Notebook** | 15% | All bugs fixed, notebook runs cleanly |

**To pass:** All GitHub Actions tests must pass (green checkmark)

---

## Getting Help

**Stuck?**
1. Review demos in `lectures/01/demo/` - they show these patterns
2. Read test output in GitHub Actions - tells you what's wrong
3. Ask on Discord - share your approach (not full code)

**Tips:**
- **Part 2:** Look at `02b_hardened_cleaning.ipynb` demo for patterns
- **Part 3a:** Similar bugs to `03a_buggy_bmi.py` demo
- **Part 3b:** Review `03b_buggy_analysis.ipynb` demo for notebook debugging
- **Git:** Commit after each part, push frequently to see test results

---

## Advanced (Optional)

Want more practice?
- Add type hints to all functions
- Create custom validation functions with unit tests
- Add data visualization showing BMI distributions
- Create a logging config file for different log levels

Won't affect your grade but great for learning!
