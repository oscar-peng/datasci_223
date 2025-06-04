# Demo 1: Parallel Computing Practice

**Objective:** Experience the power of parallel computing by analyzing patient data with `multiprocessing.Pool`, observe real speedup, and create your first SLURM script.

## Step 1: Local Parallel Analysis (10 minutes)

### Create Sample Patient Data Files

```python
import pandas as pd
import numpy as np
import os

# Create directory for patient data
os.makedirs('patient_cohorts', exist_ok=True)

# Generate 20 sample patient cohort files
for i in range(20):
    n_patients = np.random.randint(1000, 5000)  # 1K-5K patients per cohort
    
    # Simulate patient data
    patients = pd.DataFrame({
        'patient_id': range(n_patients),
        'age': np.random.normal(65, 15, n_patients),
        'risk_score': np.random.beta(2, 5, n_patients),  # Most patients low risk
        'hospital_days': np.random.poisson(3, n_patients),
        'comorbidities': np.random.poisson(2, n_patients)
    })
    
    patients.to_csv(f'patient_cohorts/cohort_{i:02d}.csv', index=False)
    
print("Created 20 patient cohort files!")
```

### Run Parallel Analysis

```python
import multiprocessing as mp
import pandas as pd
import time

def analyze_patient_cohort(patient_file):
    """Analyze a single patient cohort file"""
    # Load patient data
    df = pd.read_csv(patient_file)
    
    # Simulate complex analysis
    time.sleep(0.1)  # Represents actual computation time
    
    # Return summary statistics
    return {
        'file': patient_file,
        'n_patients': len(df),
        'avg_age': df['age'].mean(),
        'high_risk_count': (df['risk_score'] > 0.8).sum(),
        'completion_time': time.time()
    }

# Sequential vs Parallel comparison
patient_files = [f'patient_cohorts/cohort_{i:02d}.csv' for i in range(20)]

# Sequential processing
start_time = time.time()
sequential_results = [analyze_patient_cohort(f) for f in patient_files]
sequential_time = time.time() - start_time

# Parallel processing
start_time = time.time()
with mp.Pool(processes=4) as pool:
    parallel_results = pool.map(analyze_patient_cohort, patient_files)
parallel_time = time.time() - start_time

print(f"Sequential: {sequential_time:.2f}s")
print(f"Parallel: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")
```

### Test Different Process Counts

```python
# Test different numbers of processes
for n_processes in [1, 2, 4, 8]:
    start_time = time.time()
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(analyze_patient_cohort, patient_files)
    elapsed = time.time() - start_time
    print(f"{n_processes} processes: {elapsed:.2f}s")
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

## Common Issues & Solutions

- **"No speedup observed":** Check if analysis function is CPU-bound. Add more computation if needed.
- **"Runs slower with more processes":** Overhead domination. Try smaller datasets or more computation per task.
- **"Memory errors":** Reduce sample data size or increase memory allocation in SLURM script.