#!/usr/bin/env python3
"""
Generate all charts for the clinical data analysis report.
Run this script before building the MkDocs site.
"""

import altair as alt
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configure Altair
alt.data_transformers.enable("json")


def create_sample_data():
    """Create realistic clinical data for the report."""
    np.random.seed(42)

    n_patients = 200
    patients = []

    for i in range(n_patients):
        patient_id = f"P{i + 1:03d}"
        age = np.random.randint(25, 85)
        gender = np.random.choice(["Male", "Female"])
        condition = np.random.choice(
            ["Healthy", "Hypertension", "Arrhythmia", "Heart Disease"],
            p=[0.4, 0.3, 0.2, 0.1],
        )

        # Heart rate varies by condition and age
        if condition == "Healthy":
            base_hr = 70 + np.random.normal(0, 8)
        elif condition == "Hypertension":
            base_hr = 80 + np.random.normal(0, 12)
        elif condition == "Arrhythmia":
            base_hr = 75 + np.random.normal(0, 20)
        else:  # Heart Disease
            base_hr = 85 + np.random.normal(0, 15)

        # Age effect
        base_hr += (age - 50) * 0.2
        heart_rate = max(50, min(120, base_hr))

        # Blood pressure
        if condition == "Hypertension":
            systolic_bp = 150 + np.random.normal(0, 15)
        else:
            systolic_bp = 120 + np.random.normal(0, 12)
        systolic_bp = max(90, min(200, systolic_bp))

        patients.append({
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "condition": condition,
            "heart_rate": round(heart_rate, 1),
            "systolic_bp": round(systolic_bp, 1),
            "bmi": round(18.5 + np.random.exponential(5), 1),
        })

    return pd.DataFrame(patients)


def create_overview_chart(df, output_dir):
    """Create overview scatter plot."""
    chart = (
        alt.Chart(df)
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x=alt.X("age:Q", title="Age (years)"),
            y=alt.Y("heart_rate:Q", title="Heart Rate (bpm)"),
            color=alt.Color(
                "condition:N",
                title="Medical Condition",
                scale=alt.Scale(
                    range=["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
                ),
            ),
            size=alt.Size(
                "bmi:Q", title="BMI", scale=alt.Scale(range=[50, 200])
            ),
            tooltip=[
                alt.Tooltip("patient_id:N", title="Patient ID"),
                alt.Tooltip("age:Q", title="Age"),
                alt.Tooltip("heart_rate:Q", title="Heart Rate"),
                alt.Tooltip("condition:N", title="Condition"),
                alt.Tooltip("bmi:Q", title="BMI", format=".1f"),
            ],
        )
        .properties(
            title="Heart Rate vs Age by Medical Condition",
            width=600,
            height=400,
        )
        .interactive()
    )

    chart.save(output_dir / "overview_scatter.json")
    return chart


def create_condition_summary(df, output_dir):
    """Create condition summary bar chart."""
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("condition:N", title="Medical Condition", sort="-y"),
            y=alt.Y("count():Q", title="Number of Patients"),
            color=alt.Color(
                "condition:N",
                scale=alt.Scale(
                    range=["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("condition:N", title="Condition"),
                alt.Tooltip("count():Q", title="Patient Count"),
            ],
        )
        .properties(
            title="Patient Distribution by Medical Condition",
            width=500,
            height=300,
        )
    )

    chart.save(output_dir / "condition_summary.json")
    return chart


def create_age_distribution(df, output_dir):
    """Create age distribution histogram."""
    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("age:Q", bin=alt.Bin(maxbins=20), title="Age (years)"),
            y=alt.Y("count():Q", title="Number of Patients"),
            color=alt.Color(
                "condition:N",
                title="Condition",
                scale=alt.Scale(
                    range=["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
                ),
            ),
            tooltip=[
                alt.Tooltip("age:Q", bin=True, title="Age Range"),
                alt.Tooltip("count():Q", title="Count"),
                alt.Tooltip("condition:N", title="Condition"),
            ],
        )
        .properties(
            title="Age Distribution by Medical Condition", width=600, height=350
        )
    )

    chart.save(output_dir / "age_distribution.json")
    return chart


def create_interactive_dashboard(df, output_dir):
    """Create interactive dashboard with selections."""
    # Condition selector
    condition_dropdown = alt.selection_single(
        fields=["condition"],
        bind=alt.binding_select(
            options=["All"] + list(df["condition"].unique())
        ),
        name="condition_select",
        init={"condition": "All"},
    )

    # Base chart
    base = alt.Chart(df).add_selection(condition_dropdown)

    # Main scatter plot
    scatter = (
        base.mark_circle(size=100, opacity=0.8)
        .encode(
            x=alt.X("systolic_bp:Q", title="Systolic Blood Pressure (mmHg)"),
            y=alt.Y("heart_rate:Q", title="Heart Rate (bpm)"),
            color=alt.Color("condition:N", title="Condition"),
            tooltip=[
                alt.Tooltip("patient_id:N", title="Patient"),
                alt.Tooltip("age:Q", title="Age"),
                alt.Tooltip("heart_rate:Q", title="Heart Rate"),
                alt.Tooltip("systolic_bp:Q", title="Systolic BP"),
                alt.Tooltip("condition:N", title="Condition"),
            ],
        )
        .transform_filter(
            alt.expr.if_(
                condition_dropdown.condition == "All",
                True,
                alt.datum.condition == condition_dropdown.condition,
            )
        )
        .properties(
            title="Interactive Clinical Dashboard", width=600, height=400
        )
    )

    chart = scatter
    chart.save(output_dir / "interactive_dashboard.json")
    return chart


def main():
    """Generate all charts for the report."""
    print("🏥 Generating clinical data analysis charts...")

    # Create output directory relative to this script's location
    # Assuming this script is in 'scripts/' and docs is a sibling of 'scripts/'
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "docs" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    print("📊 Creating sample clinical data...")
    df = create_sample_data()

    # Save data for reference
    df.to_csv(output_dir / "clinical_data.csv", index=False)

    # Generate charts
    print("📈 Creating overview chart...")
    create_overview_chart(df, output_dir)

    print("📊 Creating condition summary...")
    create_condition_summary(df, output_dir)

    print("📉 Creating age distribution...")
    create_age_distribution(df, output_dir)

    print("🎛️ Creating interactive dashboard...")
    create_interactive_dashboard(df, output_dir)

    # Generate summary statistics
    summary_stats = {
        "total_patients": len(df),
        "avg_age": df["age"].mean(),
        "avg_heart_rate": df["heart_rate"].mean(),
        "condition_counts": df["condition"].value_counts().to_dict(),
    }

    with open(output_dir / "summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("✅ All charts generated successfully!")
    print(
        f"📁 Charts saved to: {output_dir.resolve()}"
    )  # Use resolve for absolute path
    print("\nGenerated files:")
    for file in output_dir.glob("*.json"):
        print(f"  - {file.name}")
    for file in output_dir.glob("*.csv"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
