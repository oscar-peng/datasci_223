#!/usr/bin/env python3
"""
Visualization module for neonatal feeding study.
Generates interactive Altair charts for the report.
"""

import altair as alt
import pandas as pd
from pathlib import Path
import json


def generate_charts(data: pd.DataFrame) -> None:
    """
    Generate interactive visualizations for the neonatal feeding study.

    Args:
        data (pd.DataFrame): Patient data to visualize
    """
    # Create output directory
    output_dir = Path(__file__).parent.parent / "docs" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save each chart
    charts = {
        "demographics_overview": create_demographics_chart(data),
        "feeding_progression": create_feeding_progression_chart(data),
        "clinical_factors": create_clinical_factors_chart(data),
        "intervention_timing": create_intervention_timing_chart(data),
        "primary_analysis": create_primary_analysis_chart(data),
    }

    # Save each chart
    for name, chart in charts.items():
        chart.save(str(output_dir / f"{name}.json"))


def create_demographics_chart(data: pd.DataFrame) -> alt.Chart:
    """Create demographics overview chart."""
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "gestational_age:Q", bin=True, title="Gestational Age (weeks)"
            ),
            y="count()",
            color="ventilation_status:N",
            tooltip=["count()", "gestational_age:Q", "ventilation_status:N"],
        )
        .properties(
            title="Patient Demographics by Gestational Age and Ventilation Status",
            width=600,
            height=400,
        )
    )


def create_feeding_progression_chart(data: pd.DataFrame) -> alt.Chart:
    """Create feeding progression chart."""
    return (
        alt.Chart(data)
        .mark_point()
        .encode(
            x="gestational_age:Q",
            y="time_to_fof:Q",
            color="ventilation_status:N",
            size="birth_weight:Q",
            tooltip=[
                "gestational_age:Q",
                "time_to_fof:Q",
                "ventilation_status:N",
                "birth_weight:Q",
            ],
        )
        .properties(
            title="Time to Full Oral Feeding by Gestational Age",
            width=600,
            height=400,
        )
    )


def create_clinical_factors_chart(data: pd.DataFrame) -> alt.Chart:
    """Create clinical factors comparison chart."""
    # Calculate mean time to FOF by ventilation and diabetes status
    summary = (
        data.groupby(["ventilation_status", "maternal_diabetes"])["time_to_fof"]
        .mean()
        .reset_index()
    )

    return (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x="ventilation_status:N",
            y="time_to_fof:Q",
            color="maternal_diabetes:N",
            column="maternal_diabetes:N",
            tooltip=[
                "ventilation_status:N",
                "maternal_diabetes:N",
                "time_to_fof:Q",
            ],
        )
        .properties(
            title="Impact of Clinical Factors on Time to Full Oral Feeding",
            width=300,
            height=400,
        )
    )


def create_intervention_timing_chart(data: pd.DataFrame) -> alt.Chart:
    """Create intervention timing chart."""
    # Create a base chart
    base = alt.Chart(data).encode(x="gestational_age:Q", y="time_to_fof:Q")

    # Add regression line
    regression = base.transform_regression(
        "gestational_age", "time_to_fof"
    ).mark_line(color="red")

    # Add points
    points = base.mark_point().encode(
        color="ventilation_status:N",
        tooltip=["gestational_age:Q", "time_to_fof:Q", "ventilation_status:N"],
    )

    return (points + regression).properties(
        title="Relationship Between Gestational Age and Time to Full Oral Feeding",
        width=600,
        height=400,
    )


def create_primary_analysis_chart(data: pd.DataFrame) -> alt.Chart:
    """Create primary analysis chart with multiple factors."""
    # Create a layered chart
    base = alt.Chart(data).encode(x="gestational_age:Q", y="time_to_fof:Q")

    # Add points colored by ventilation status
    points = base.mark_point().encode(
        color="ventilation_status:N",
        size="apgar_5min:Q",
        tooltip=[
            "gestational_age:Q",
            "time_to_fof:Q",
            "ventilation_status:N",
            "apgar_5min:Q",
            "birth_weight:Q",
        ],
    )

    # Add regression line
    regression = base.transform_regression(
        "gestational_age", "time_to_fof"
    ).mark_line(color="red")

    return (points + regression).properties(
        title="Primary Analysis: Multiple Factors Affecting Time to Full Oral Feeding",
        width=600,
        height=400,
    )
