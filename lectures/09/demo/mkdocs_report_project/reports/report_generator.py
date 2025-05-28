#!/usr/bin/env python3
"""
Generates Altair chart JSON specifications and supporting data files
for the clinical data analysis report.
"""

import altair as alt
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add the parent directory of 'utils' to sys.path to allow direct import
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils import reporting  # For save_altair_chart_json

# Configure Altair
alt.data_transformers.enable("json")

# Define base paths
# Assuming this script is in 'reports/', and 'docs/' is a sibling of 'reports/' parent
DOCS_DIR = project_root / "docs"
CHARTS_DIR = DOCS_DIR / "charts"
DATA_DIR = DOCS_DIR / "data"  # For CSVs or other data outputs


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
        base_hr = 0
        if condition == "Healthy":
            base_hr = 70 + np.random.normal(0, 8)
        elif condition == "Hypertension":
            base_hr = 80 + np.random.normal(0, 12)
        elif condition == "Arrhythmia":
            base_hr = 75 + np.random.normal(0, 20)
        else:
            base_hr = 85 + np.random.normal(0, 15)  # Heart Disease
        base_hr += (age - 50) * 0.2
        heart_rate = max(50, min(120, base_hr))
        systolic_bp = 0
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


def generate_overview_chart_spec(df):
    """Generates the Altair spec for the overview scatter plot."""
    return (
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


def generate_condition_summary_chart_spec(df):
    """Generates the Altair spec for the condition summary bar chart."""
    return (
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


def generate_age_distribution_chart_spec(df):
    """Generates the Altair spec for the age distribution histogram."""
    return (
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


def generate_interactive_dashboard_chart_spec(df):
    """Generates the Altair spec for the interactive dashboard."""
    condition_dropdown = alt.selection_single(
        fields=["condition"],
        bind=alt.binding_select(
            options=["All"] + list(df["condition"].unique())
        ),
        name="condition_select",
        init={"condition": "All"},
    )
    base = alt.Chart(df).add_selection(condition_dropdown)
    return (
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


def generate_report_elements():
    """
    Generates all chart JSONs and supporting data files for the report.
    Markdown files are expected to be static and reference these outputs.
    """
    print("🏥 Generating report elements (charts and data)...")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure data directory exists

    df = create_sample_data()

    # Save the main dataframe for potential reference or other uses
    df.to_csv(DATA_DIR / "clinical_data.csv", index=False)
    print(f"Clinical data saved to: {DATA_DIR / 'clinical_data.csv'}")

    # Generate and save chart JSONs
    print("📈 Generating and saving chart specifications...")
    overview_chart = generate_overview_chart_spec(df)
    reporting.save_altair_chart_json(
        overview_chart, CHARTS_DIR / "overview_scatter.json"
    )

    condition_summary_chart = generate_condition_summary_chart_spec(df)
    reporting.save_altair_chart_json(
        condition_summary_chart, CHARTS_DIR / "condition_summary.json"
    )

    age_dist_chart = generate_age_distribution_chart_spec(df)
    reporting.save_altair_chart_json(
        age_dist_chart, CHARTS_DIR / "age_distribution.json"
    )

    interactive_dash_chart = generate_interactive_dashboard_chart_spec(df)
    reporting.save_altair_chart_json(
        interactive_dash_chart, CHARTS_DIR / "interactive_dashboard.json"
    )

    # Generate and save summary statistics (e.g., for manual inclusion in Markdown)
    summary_stats_data = {
        "total_patients": len(df),
        "avg_age": df["age"].mean(),
        "avg_heart_rate": df["heart_rate"].mean(),
        "condition_counts": df["condition"].value_counts().to_dict(),
        "condition_summary_table": df.groupby("condition")["heart_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_dict(orient="records"),
    }
    summary_stats_path = DATA_DIR / "summary_stats.json"
    with open(summary_stats_path, "w") as f:
        json.dump(summary_stats_data, f, indent=2)
    print(f"Summary statistics saved to: {summary_stats_path}")

    print("✅ All report elements generated successfully!")


if __name__ == "__main__":
    generate_report_elements()
