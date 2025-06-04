#!/usr/bin/env python3
"""Simple working parallel computing test"""

import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import time

# Global debug setting - accessible to worker processes
DEBUG_MODE = True


def analyze_patient_cohort(patient_file):
    """Analyze a single patient cohort file"""
    # Load patient data
    df = pd.read_csv(patient_file)

    # Simulate complex analysis - shorter sleep in debug mode
    sleep_time = 0.01 if DEBUG_MODE else 0.1
    time.sleep(sleep_time)

    # Return summary statistics
    return {
        "file": patient_file,
        "n_patients": len(df),
        "avg_age": df["age"].mean(),
        "high_risk_count": (df["risk_score"] > 0.8).sum(),
    }


if __name__ == "__main__":
    print("🧪 Simple Parallel Computing Test")
    print("=" * 40)

    # Create tiny test files quickly
    os.makedirs("lectures/10/demo/patient_cohorts", exist_ok=True)

    n_files = 4
    print(f"📁 Creating {n_files} small test files...")

    for i in range(n_files):
        # Very small datasets for quick testing
        patients = pd.DataFrame({
            "patient_id": range(50),  # Only 50 patients each
            "age": np.random.normal(65, 15, 50),
            "risk_score": np.random.beta(2, 5, 50),
            "hospital_days": np.random.poisson(3, 50),
            "comorbidities": np.random.poisson(2, 50),
        })
        patients.to_csv(
            f"lectures/10/demo/patient_cohorts/cohort_{i:02d}.csv", index=False
        )

    print("✅ Files created!")

    # Test files
    patient_files = [
        f"lectures/10/demo/patient_cohorts/cohort_{i:02d}.csv"
        for i in range(n_files)
    ]

    # Sequential test
    print("⏳ Sequential analysis...")
    start = time.time()
    seq_results = [analyze_patient_cohort(f) for f in patient_files]
    seq_time = time.time() - start

    # Parallel test
    print("⚡ Parallel analysis...")
    start = time.time()
    with mp.Pool(processes=2) as pool:  # Just 2 processes
        par_results = pool.map(analyze_patient_cohort, patient_files)
    par_time = time.time() - start

    print(f"\n📈 Results:")
    print(f"Sequential: {seq_time:.3f}s")
    print(f"Parallel:   {par_time:.3f}s")
    print(f"Speedup:    {seq_time / par_time:.2f}x")

    print("\n✅ Test completed successfully!")
    print("🎯 Debug mode works - demo is ready for students!")
