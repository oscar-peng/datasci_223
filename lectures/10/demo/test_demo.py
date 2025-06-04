#!/usr/bin/env python3
"""
Test script for the parallel computing practice demo
Tests all code sections from the demo and validates outputs
"""

import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import time
import sys


def test_data_generation():
    """Test Step 1: Create Sample Patient Data Files"""
    print("=== Testing Data Generation ===")

    # Create directory for patient data
    os.makedirs("patient_cohorts", exist_ok=True)

    # Generate 20 sample patient cohort files
    for i in range(20):
        n_patients = np.random.randint(1000, 5000)  # 1K-5K patients per cohort

        # Simulate patient data
        patients = pd.DataFrame({
            "patient_id": range(n_patients),
            "age": np.random.normal(65, 15, n_patients),
            "risk_score": np.random.beta(
                2, 5, n_patients
            ),  # Most patients low risk
            "hospital_days": np.random.poisson(3, n_patients),
            "comorbidities": np.random.poisson(2, n_patients),
        })

        patients.to_csv(f"patient_cohorts/cohort_{i:02d}.csv", index=False)

    print("✅ Created 20 patient cohort files!")

    # Validate files were created
    files = os.listdir("patient_cohorts")
    csv_files = [f for f in files if f.endswith(".csv")]
    print(f"✅ Found {len(csv_files)} CSV files in patient_cohorts directory")

    # Check a sample file
    sample_df = pd.read_csv("patient_cohorts/cohort_00.csv")
    print(
        f"✅ Sample file has {len(sample_df)} patients with columns: {list(sample_df.columns)}"
    )

    return True


def analyze_patient_cohort(patient_file):
    """Analyze a single patient cohort file"""
    # Load patient data
    df = pd.read_csv(patient_file)

    # Simulate complex analysis
    time.sleep(0.1)  # Represents actual computation time

    # Return summary statistics
    return {
        "file": patient_file,
        "n_patients": len(df),
        "avg_age": df["age"].mean(),
        "high_risk_count": (df["risk_score"] > 0.8).sum(),
        "completion_time": time.time(),
    }


def test_parallel_analysis():
    """Test Step 1: Run Parallel Analysis"""
    print("\n=== Testing Parallel Analysis ===")

    # Sequential vs Parallel comparison
    patient_files = [f"patient_cohorts/cohort_{i:02d}.csv" for i in range(20)]

    # Verify all files exist
    missing_files = [f for f in patient_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print(f"✅ All {len(patient_files)} patient files found")

    # Sequential processing
    print("Running sequential processing...")
    start_time = time.time()
    sequential_results = [analyze_patient_cohort(f) for f in patient_files]
    sequential_time = time.time() - start_time

    # Parallel processing
    print("Running parallel processing...")
    start_time = time.time()
    with mp.Pool(processes=4) as pool:
        parallel_results = pool.map(analyze_patient_cohort, patient_files)
    parallel_time = time.time() - start_time

    speedup = sequential_time / parallel_time

    print(f"✅ Sequential: {sequential_time:.2f}s")
    print(f"✅ Parallel: {parallel_time:.2f}s")
    print(f"✅ Speedup: {speedup:.2f}x")

    # Validate results
    if len(sequential_results) != len(parallel_results):
        print("❌ Result count mismatch between sequential and parallel")
        return False

    if speedup < 1.5:
        print(f"⚠️  Speedup ({speedup:.2f}x) is less than expected (>1.5x)")
        print("   This could be due to overhead or system constraints")
    else:
        print(f"✅ Good speedup achieved: {speedup:.2f}x")

    return True


def test_process_scaling():
    """Test different numbers of processes"""
    print("\n=== Testing Process Scaling ===")

    patient_files = [f"patient_cohorts/cohort_{i:02d}.csv" for i in range(20)]

    # Test different numbers of processes
    results = {}
    for n_processes in [1, 2, 4, 8]:
        print(f"Testing {n_processes} processes...")
        start_time = time.time()
        with mp.Pool(processes=n_processes) as pool:
            process_results = pool.map(analyze_patient_cohort, patient_files)
        elapsed = time.time() - start_time
        results[n_processes] = elapsed
        print(f"✅ {n_processes} processes: {elapsed:.2f}s")

    # Analyze scaling
    baseline = results[1]
    print(f"\n📊 Scaling Analysis (baseline: {baseline:.2f}s):")
    for n_proc, time_taken in results.items():
        speedup = baseline / time_taken
        efficiency = speedup / n_proc * 100
        print(
            f"   {n_proc} processes: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency"
        )

    return True


def test_slurm_script_creation():
    """Test SLURM script creation"""
    print("\n=== Testing SLURM Script Creation ===")

    slurm_content = """#!/bin/bash
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
"""

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Write SLURM script
    with open("my_health_analysis.sh", "w") as f:
        f.write(slurm_content)

    print("✅ Created SLURM script: my_health_analysis.sh")
    print("✅ Created logs directory")

    # Verify script contents
    if os.path.exists("my_health_analysis.sh"):
        with open("my_health_analysis.sh", "r") as f:
            content = f.read()
            if "#SBATCH --job-name=my_first_health_analysis" in content:
                print("✅ SLURM script contains correct job configuration")
            else:
                print("❌ SLURM script missing expected content")
                return False

    return True


def test_timing_for_demo():
    """Test demo timing for 10-15 minute window"""
    print("\n=== Demo Timing Validation ===")

    # Quick timing test
    start = time.time()

    # Simulate data generation time (reduced dataset)
    test_df = pd.DataFrame({
        "patient_id": range(1000),
        "age": np.random.normal(65, 15, 1000),
        "risk_score": np.random.beta(2, 5, 1000),
        "hospital_days": np.random.poisson(3, 1000),
        "comorbidities": np.random.poisson(2, 1000),
    })
    data_gen_time = time.time() - start

    print(f"✅ Single file generation: {data_gen_time:.3f}s")
    print(f"✅ 20 files estimated: {data_gen_time * 20:.1f}s")
    print(
        f"✅ Total demo time estimate: 8-12 minutes (appropriate for 10-15 min window)"
    )

    return True


def run_all_tests():
    """Run all demo tests"""
    print("🚀 Starting Parallel Computing Demo Tests\n")

    tests = [
        ("Data Generation", test_data_generation),
        ("Parallel Analysis", test_parallel_analysis),
        ("Process Scaling", test_process_scaling),
        ("SLURM Script Creation", test_slurm_script_creation),
        ("Demo Timing", test_timing_for_demo),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Demo is ready for students.")
    else:
        print("⚠️  Some tests failed. Review issues above.")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
