# Demo 1: Parallel Computing Practice

**Objective:** Experience the power of parallel computing by analyzing patient data with `multiprocessing.Pool`, observe real speedup, and create your first SLURM script.

## Step 1: Local Parallel Analysis (10 minutes)

### Create Sample Patient Data Files

```python
import pandas as pd
import numpy as np
import os

# DEBUG MODE: Set to True for faster testing
DEBUG_MODE = True  # Change to False for full analysis

# Configure based on mode
if DEBUG_MODE:
    n_files = 4          # Only 4 files for quick testing
    min_patients = 100   # Smaller cohorts
    max_patients = 500
    print("🐣 DEBUG MODE: Creating baby-sized dataset for quick testing...")
else:
    n_files = 20         # Full 20 files
    min_patients = 1000  # Full-size cohorts
    max_patients = 5000
    print("🚀 FULL MODE: Creating complete dataset...")

# Create directory for patient data
os.makedirs('patient_cohorts', exist_ok=True)

# Generate sample patient cohort files
for i in range(n_files):
    n_patients = np.random.randint(min_patients, max_patients)
    
    # Simulate patient data
    patients = pd.DataFrame({
        'patient_id': range(n_patients),
        'age': np.random.normal(65, 15, n_patients),
        'risk_score': np.random.beta(2, 5, n_patients),  # Most patients low risk
        'hospital_days': np.random.poisson(3, n_patients),
        'comorbidities': np.random.poisson(2, n_patients)
    })
    
    patients.to_csv(f'patient_cohorts/cohort_{i:02d}.csv', index=False)
    
print(f"Created {n_files} patient cohort files!")
```

### Run Parallel Analysis (Notebook-Friendly)

This cell uses `concurrent.futures.ProcessPoolExecutor` or `ThreadPoolExecutor`, which are more robust for use within Jupyter notebooks.

```python
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import sys # For checking if running in a notebook

# DEBUG MODE: Global variable accessible to worker processes
DEBUG_MODE = True  # Change to False for full analysis

def analyze_patient_cohort(patient_file_path): # Renamed argument to avoid conflict
    """Analyze a single patient cohort file"""
    df = pd.read_csv(patient_file_path)
    sleep_time = 0.01 if DEBUG_MODE else 0.1
    time.sleep(sleep_time)
    return {
        'file': patient_file_path,
        'n_patients': len(df),
        'avg_age': df['age'].mean(),
        'high_risk_count': (df['risk_score'] > 0.8).sum(),
        'completion_time': time.time()
    }

# Determine if running in a Jupyter-like environment for nbconvert robustness
# 'get_ipython' is a reliable check for IPython/Jupyter environments.
is_notebook_environment = 'get_ipython' in globals() or hasattr(sys, 'ps1')

n_files_to_process = 4 if DEBUG_MODE else 20
current_patient_files = [f'patient_cohorts/cohort_{i:02d}.csv' for i in range(n_files_to_process)]

print(f"📊 Analyzing {len(current_patient_files)} cohort files...")

# Sequential processing
print("⏳ Running sequential analysis...")
start_time_seq = time.time()
sequential_results = [analyze_patient_cohort(f) for f in current_patient_files]
sequential_time = time.time() - start_time_seq

# Parallel processing
print("⚡ Running parallel analysis...")
start_time_par = time.time()
parallel_results = []
executor_used = "None"

if is_notebook_environment:
    print("🔧 Detected notebook environment, preferring ThreadPoolExecutor for robustness.")
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(analyze_patient_cohort, current_patient_files))
        executor_used = "ThreadPoolExecutor (Notebook Safe)"
    except Exception as e_thread:
        print(f"⚠️ ThreadPoolExecutor failed ({e_thread}), falling back to sequential...")
        parallel_results = sequential_results
        executor_used = "Sequential Fallback (Thread)"
else:
    print("🚀 Detected script environment, attempting ProcessPoolExecutor for true parallelism.")
    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(analyze_patient_cohort, current_patient_files))
        executor_used = "ProcessPoolExecutor (Script)"
    except Exception as e_proc:
        print(f"⚠️ ProcessPoolExecutor failed ({e_proc}), falling back to sequential...")
        parallel_results = sequential_results
        executor_used = "Sequential Fallback (Process)"
    
parallel_time = time.time() - start_time_par

print(f"\n📈 RESULTS (using {executor_used}):")
print(f"Sequential: {sequential_time:.2f}s")
print(f"Parallel:   {parallel_time:.2f}s")

if parallel_time > 0 and sequential_time > 0:
    print(f"Speedup:    {sequential_time/parallel_time:.2f}x")
else:
    print("Speedup: N/A (Parallel processing might have failed or run too quickly)")
```

---

### Alternative: `multiprocessing.Pool` (Primarily for Standalone `.py` Scripts)

The following cell demonstrates using the original `multiprocessing.Pool`. **This method often causes issues in Jupyter Notebooks (especially on Windows/macOS without careful handling) due to how functions are pickled and how `__main__` is handled.** It's included for completeness and is best run by saving the notebook content as a `.py` file and executing it from the terminal.

**Do NOT run this cell directly in the notebook if you encounter pickling errors.**

```python
# THIS CELL IS INTENDED FOR .py SCRIPT EXECUTION AND EXPLANATION.
# It will NOT run correctly if executed directly in a Jupyter Notebook cell by nbconvert
# due to pickling issues with functions defined in __main__ of a notebook.
# To make this runnable by nbconvert, it would need to be in its own .py file and called.

if False: # This 'if False' ensures nbconvert --execute skips this block
    import multiprocessing as mp
    import pandas as pd # Assuming pandas is needed for analyze_patient_cohort
    import time         # Assuming time is needed
    import os           # Assuming os is needed (e.g. for file paths)
    
    # Ensure DEBUG_MODE and analyze_patient_cohort are defined here if running as a standalone script.
    # For this example, we'll assume they would be copied from above.
    # Example:
    # DEBUG_MODE = True
    # def analyze_patient_cohort(patient_file_path):
    #     df = pd.read_csv(patient_file_path)
    #     sleep_time = 0.01 if DEBUG_MODE else 0.1
    #     time.sleep(sleep_time)
    #     return { 'file': patient_file_path, 'n_patients': len(df) }

    if __name__ == '__main__':  # CRITICAL for multiprocessing.Pool on Windows/macOS
        
        # This code would be part of your run_mp_pool.py
        # Define n_files_mp and patient_files_mp appropriately
        n_files_mp = 4 if DEBUG_MODE else 20
        patient_files_mp = [f'patient_cohorts/cohort_{i:02d}.csv' for i in range(n_files_mp)]

        print(f"📊 (mp.Pool Example) Analyzing {len(patient_files_mp)} cohort files...")

        start_time_par_mp = time.time()
        try:
            # analyze_patient_cohort must be picklable (e.g. top-level in a .py file)
            with mp.Pool(processes=4) as pool:
                parallel_results_mp = pool.map(analyze_patient_cohort, patient_files_mp)
            parallel_time_mp = time.time() - start_time_par_mp
            print(f"Parallel (mp.Pool): {parallel_time_mp:.2f}s")
        except Exception as e:
            print(f"Error with mp.Pool: {e}")
            print("This often happens in notebooks. Ensure this code is in a .py file and analyze_patient_cohort is defined at the top level.")

```

### Test Different Process Counts

```python
# Test different numbers of workers (notebook-friendly version)
# Ensure DEBUG_MODE, analyze_patient_cohort, and current_patient_files are in scope from the main analysis cell.
# is_notebook_environment should also be in scope.

max_workers_to_test = 4 if DEBUG_MODE else 8
worker_counts_to_test = [1, 2, 4] if DEBUG_MODE else [1, 2, 4, 8]
worker_counts_to_test = [p for p in worker_counts_to_test if p <= max_workers_to_test]

print("\n🧪 Testing different worker counts...")
for n_workers in worker_counts_to_test:
    start_time_test = time.time()
    results_test = []
    executor_type_test = "None"

    if is_notebook_environment:
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results_test = list(executor.map(analyze_patient_cohort, current_patient_files))
            executor_type_test = "ThreadPoolExecutor"
        except Exception as e_thread_test:
            print(f"⚠️ ThreadPoolExecutor failed for {n_workers} workers ({e_thread_test}), falling back to sequential...")
            results_test = [analyze_patient_cohort(f) for f in current_patient_files]
            executor_type_test = "Sequential Fallback (Thread)"
    else: # Script environment
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results_test = list(executor.map(analyze_patient_cohort, current_patient_files))
            executor_type_test = "ProcessPoolExecutor"
        except Exception as e_proc_test:
            print(f"⚠️ ProcessPoolExecutor failed for {n_workers} workers ({e_proc_test}), falling back to sequential...")
            results_test = [analyze_patient_cohort(f) for f in current_patient_files]
            executor_type_test = "Sequential Fallback (Process)"
            
    elapsed_test = time.time() - start_time_test
    print(f"{n_workers} workers ({executor_type_test}): {elapsed_test:.2f}s")
```

## Step 2: SLURM Script Creation (5 minutes)

Create `my_health_analysis.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=my_first_health_analysis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/health_analysis_%j.out

echo "Starting health data analysis on $(hostname)"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"

# Load modules (if on Wynton)
# module load python/3.9

# Run your parallel analysis
python parallel_patient_analysis.py
echo "Analysis complete!"
```

## Success Criteria

✅ **Speedup Observed:** Parallel version runs 2-4x faster than sequential
✅ **Understanding Gained:** Can explain when to use processes vs threads
✅ **SLURM Ready:** Created a working SLURM script for cluster submission
✅ **Resource Awareness:** Understand relationship between data size and memory requirements

> 💡 **Debug Tip:** Start with `DEBUG_MODE = True` to quickly test your setup, then switch to `DEBUG_MODE = False` to see real performance differences with larger datasets.

## Common Issues & Solutions

- **"No speedup observed":** Check if analysis function is CPU-bound. Add more computation if needed.
- **"Runs slower with more processes":** Overhead domination. Try smaller datasets or more computation per task.
- **"Memory errors":** Reduce sample data size or increase memory allocation in SLURM script.
- **"Can't get attribute 'analyze_patient_cohort'":** Jupyter notebook limitation! Use the `concurrent.futures` version above or save as `.py` file.
- **"RuntimeError about bootstrapping":** You forgot the `if __name__ == '__main__':` guard! This is required on macOS/Windows.
- **"NameError: name 'DEBUG_MODE' is not defined":** Make sure DEBUG_MODE is defined at the global level, not inside the main block.

## 💡 Pro Tips

- **In Jupyter:** Use `ThreadPoolExecutor` or `ProcessPoolExecutor` from `concurrent.futures`
- **In Scripts:** Use `multiprocessing.Pool` for maximum performance
- **For I/O tasks:** Threading often works just as well as multiprocessing
- **For CPU tasks:** True multiprocessing gives the best speedup