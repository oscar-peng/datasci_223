#!/usr/bin/env python3
"""Test script for the parallel computing demo with debug mode - FIXED VERSION"""

import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import time


def analyze_patient_cohort(patient_file):
    """Analyze a single patient cohort file"""
    # Load patient data
    df = pd.read_csv(patient_file)

    # Simulate complex analysis - shorter sleep in debug mode
    sleep_time = 0.01 if DEBUG_MODE else 0.1
    time.sleep(sleep_time)  # Represents actual computation time

    # Return summary statistics
    return {
        "file": patient_file,
        "n_patients": len(df),
        "avg_age": df["age"].mean(),
        "high_risk_count": (df["risk_score"] > 0.8).sum(),
        "completion_time": time.time(),
    }


if __name__ == "__main__":
    print("🧪 Testing Parallel Computing Demo with Debug Mode")
    print("=" * 50)

    # DEBUG MODE: Set to True for faster testing
    DEBUG_MODE = True  # Change to False for full analysis

    # Configure based on mode
    if DEBUG_MODE:
        n_files = 4  # Only 4 files for quick testing
        min_patients = 100  # Smaller cohorts
        max_patients = 500
        print("🐣 DEBUG MODE: Creating baby-sized dataset for quick testing...")
    else:
        n_files = 20  # Full 20 files
        min_patients = 1000  # Full-size cohorts
        max_patients = 5000
        print("🚀 FULL MODE: Creating complete dataset...")

    # Create directory for patient data
    os.makedirs("lectures/10/demo/patient_cohorts", exist_ok=True)

    # Generate sample patient cohort files
    print(f"📁 Creating {n_files} patient cohort files...")
    for i in range(n_files):
        n_patients = np.random.randint(min_patients, max_patients)

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

        patients.to_csv(
            f"lectures/10/demo/patient_cohorts/cohort_{i:02d}.csv", index=False
        )

    print(f"✅ Created {n_files} patient cohort files!")

    # Sequential vs Parallel comparison
    patient_files = [
        f"lectures/10/demo/patient_cohorts/cohort_{i:02d}.csv"
        for i in range(n_files)
    ]

    print(f"📊 Analyzing {len(patient_files)} cohort files...")

    # Sequential processing
    print("⏳ Running sequential analysis...")
    start_time = time.time()
    sequential_results = [analyze_patient_cohort(f) for f in patient_files]
    sequential_time = time.time() - start_time

    # Parallel processing
    print("⚡ Running parallel analysis...")
    start_time = time.time()
    with mp.Pool(processes=4) as pool:
        parallel_results = pool.map(analyze_patient_cohort, patient_files)
    parallel_time = time.time() - start_time

    print("\n📈 RESULTS:")
    print(f"Sequential: {sequential_time:.3f}s")
    print(f"Parallel:   {parallel_time:.3f}s")
    print(f"Speedup:    {sequential_time / parallel_time:.2f}x")

    # Test different process counts
    print("\n🧪 Testing different process counts...")
    max_processes = 4 if DEBUG_MODE else 8  # Fewer processes in debug mode
    process_counts = [1, 2, 4] if DEBUG_MODE else [1, 2, 4, 8]

    for n_processes in process_counts:
        start_time = time.time()
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(analyze_patient_cohort, patient_files)
        elapsed = time.time() - start_time
        print(f"{n_processes} processes: {elapsed:.3f}s")

    print("\n✅ Demo test completed successfully!")
    print("💡 To test full performance, change DEBUG_MODE to False")
