import altair as alt
import pandas as pd
import numpy as np
from pathlib import Path

# Create output directory
charts_dir = Path("media")
charts_dir.mkdir(parents=True, exist_ok=True)

# Generate synthetic health data
np.random.seed(42)
n_samples = 100

df = pd.DataFrame({
    "patient_id": range(1, n_samples + 1),
    "age": np.random.normal(45, 15, n_samples).astype(int),
    "blood_pressure": np.random.normal(120, 10, n_samples).astype(int),
    "heart_rate": np.random.normal(75, 8, n_samples).astype(int),
    "condition": np.random.choice(
        ["Healthy", "Hypertension", "Arrhythmia"], n_samples
    ),
    "visit_date": pd.date_range(
        start="2024-01-01", periods=n_samples, freq="D"
    ),
    "medication": np.random.choice(
        ["None", "Beta Blocker", "ACE Inhibitor", "Diuretic"], n_samples
    ),
    "dosage": np.random.uniform(0, 100, n_samples),
})

# 1. Basic Scatter Plot
scatter = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x="age:Q",
        y="blood_pressure:Q",
        color="condition:N",
        tooltip=["patient_id:N", "age:Q", "blood_pressure:Q", "condition:N"],
    )
    .properties(
        title="Age vs Blood Pressure by Condition", width=400, height=300
    )
)
scatter.save(charts_dir / "chart_basic_scatter.json")

# 2. Time Series with Multiple Metrics
time_series = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x="visit_date:T",
        y="blood_pressure:Q",
        color="condition:N",
        tooltip=["visit_date:T", "blood_pressure:Q", "condition:N"],
    )
    .properties(title="Blood Pressure Trends Over Time", width=600, height=300)
)
time_series.save(charts_dir / "chart_time_series.json")

# 3. Box Plot
box_plot = (
    alt.Chart(df)
    .mark_boxplot()
    .encode(
        x="condition:N",
        y="heart_rate:Q",
        color="condition:N",
        tooltip=["condition:N", "heart_rate:Q"],
    )
    .properties(
        title="Heart Rate Distribution by Condition", width=400, height=300
    )
)
box_plot.save(charts_dir / "chart_box_plot.json")

# 4. Heatmap
heatmap = (
    alt.Chart(df)
    .mark_rect()
    .encode(
        x=alt.X("condition:N", title="Condition"),
        y=alt.Y("medication:N", title="Medication"),
        color=alt.Color("mean(dosage):Q", title="Average Dosage"),
        tooltip=["condition:N", "medication:N", "mean(dosage):Q"],
    )
    .properties(
        title="Average Medication Dosage by Condition", width=400, height=300
    )
)
heatmap.save(charts_dir / "chart_heatmap.json")

# 5. Interactive Selection
selection = alt.selection_point(
    name="select", fields=["condition"], bind="legend"
)

interactive = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x="age:Q",
        y="blood_pressure:Q",
        color=alt.condition(selection, "condition:N", alt.value("lightgray")),
        tooltip=["patient_id:N", "age:Q", "blood_pressure:Q", "condition:N"],
    )
    .add_params(selection)
    .properties(title="Interactive Patient Data", width=400, height=300)
)
interactive.save(charts_dir / "chart_interactive.json")

# 6. Faceted Plot
faceted = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x="medication:N",
        y="count():Q",
        color="condition:N",
        tooltip=["medication:N", "count():Q", "condition:N"],
    )
    .properties(
        title="Medication Distribution by Condition", width=100, height=300
    )
    .facet(column="condition:N")
)
faceted.save(charts_dir / "chart_faceted.json")

print(f"Generated charts in {charts_dir}")
