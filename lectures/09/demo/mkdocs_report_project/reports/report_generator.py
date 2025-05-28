#!/usr/bin/env python3
"""
Generates Altair chart JSON specifications and supporting data files
for the neonatal feeding study analysis report.

This module creates realistic clinical data based on neonatal feeding research
and generates interactive visualizations for automated report generation.
"""

import altair as alt
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from scipy import stats

# Add the parent directory of 'utils' to sys.path to allow direct import
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils import reporting  # For save_altair_chart_json

# Configure Altair
alt.data_transformers.enable("json")

# Define base paths
DOCS_DIR = project_root / "docs"
CHARTS_DIR = DOCS_DIR / "charts"
DATA_DIR = DOCS_DIR / "data"


def create_neonatal_feeding_data():
    """
    Create realistic neonatal feeding study data.

    This simulates data from a study of optimal feeding times for premature babies,
    based on real clinical research patterns but simplified for education.

    Research Question: What factors predict time to full oral feeding in premature infants?
    """
    np.random.seed(42)  # Reproducible results for teaching
    n_patients = 180

    print("   📊 Generating realistic neonatal feeding data...")

    # Core patient characteristics
    gestational_age = np.random.normal(28.5, 2.5, n_patients)
    gestational_age = np.clip(gestational_age, 24, 32)  # Inclusion: <32 weeks

    # Birth weight correlates with gestational age (clinical reality)
    # Typical relationship: ~100g per week of gestation
    expected_weight = (gestational_age - 24) * 100 + 600
    birth_weight = np.random.normal(expected_weight, 200)
    birth_weight = np.clip(birth_weight, 400, 2000)

    # Other patient characteristics
    baby_sex = np.random.choice(["Male", "Female"], n_patients)
    multiple_births = np.random.choice(["N", "Y"], n_patients, p=[0.75, 0.25])
    c_section = np.random.choice(
        [0, 1], n_patients, p=[0.4, 0.6]
    )  # Higher rate in preemies

    # Medical complexity based on gestational age (earlier = more complex)
    complexity_prob = (
        32 - gestational_age
    ) / 8  # Earlier babies more likely to have complications
    mechanical_ventilation = np.random.binomial(1, complexity_prob)
    o2_at_36_weeks = np.random.binomial(
        1, complexity_prob * 0.7
    )  # Related to ventilation

    # Create base dataframe
    df = pd.DataFrame({
        "patient_id": [f"P{i + 1:03d}" for i in range(n_patients)],
        "gestational_age_weeks": gestational_age,
        "birth_weight_grams": birth_weight,
        "baby_sex": baby_sex,
        "multiple_births": multiple_births,
        "c_section": c_section,
        "mechanical_ventilation": mechanical_ventilation,
        "o2_at_36_weeks": o2_at_36_weeks,
    })

    # Calculate feeding outcomes with realistic clinical relationships
    df = _calculate_feeding_outcomes(df)

    print(
        f"   ✅ Generated {len(df)} patients with realistic clinical relationships"
    )
    return df


def _calculate_feeding_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate realistic feeding outcomes based on patient characteristics.

    Key clinical relationships modeled:
    - Earlier gestational age → later feeding start, longer time to full feeding
    - Lower birth weight → longer time to full feeding
    - Mechanical ventilation → delayed feeding progression
    - Multiple births → slightly delayed progression
    """
    n = len(df)

    # PMA at first oral feeding (usually starts around 32-34 weeks PMA)
    # Earlier babies start later relative to their corrected age
    base_start_pma = 33.0
    ga_delay = (
        30 - df["gestational_age_weeks"]
    ) * 0.3  # Earlier babies delayed
    ventilation_delay = (
        df["mechanical_ventilation"] * 0.5
    )  # Ventilation delays feeding
    multiple_delay = (df["multiple_births"] == "Y").astype(
        int
    ) * 0.2  # Multiples slightly delayed
    random_variation = np.random.normal(0, 0.8, n)

    pma_first_feeding = (
        base_start_pma
        + ga_delay
        + ventilation_delay
        + multiple_delay
        + random_variation
    )
    pma_first_feeding = np.clip(pma_first_feeding, 30, 37)

    # Time to full oral feeding (PRIMARY OUTCOME - days from first feeding to full feeding)
    base_time = 10  # Base 10 days

    # Clinical factors that increase time to full feeding
    ga_effect = (
        30 - df["gestational_age_weeks"]
    ) * 1.5  # Earlier GA = longer time
    weight_effect = (
        1200 - df["birth_weight_grams"]
    ) / 200  # Lower weight = longer time
    ventilation_effect = (
        df["mechanical_ventilation"] * 3
    )  # Ventilation adds ~3 days
    o2_effect = df["o2_at_36_weeks"] * 2  # O2 dependency adds ~2 days
    sex_effect = (df["baby_sex"] == "Male").astype(
        int
    ) * 1  # Males slightly slower
    multiple_effect = (df["multiple_births"] == "Y").astype(
        int
    ) * 1.5  # Multiples slower

    random_noise = np.random.normal(0, 4, n)

    time_to_fof = (
        base_time
        + ga_effect
        + weight_effect
        + ventilation_effect
        + o2_effect
        + sex_effect
        + multiple_effect
        + random_noise
    )
    time_to_fof = np.clip(time_to_fof, 1, 45)  # Realistic clinical bounds

    # PMA at full oral feeding
    pma_full_feeding = pma_first_feeding + (
        time_to_fof / 7
    )  # Convert days to weeks

    # Add calculated variables to dataframe
    df["PMA_at_first_oral_feeding"] = pma_first_feeding
    df["time_to_FOF"] = time_to_fof
    df["PMA_at_full_oral_feeding"] = pma_full_feeding

    return df


def generate_demographics_chart_spec(df):
    """Generate patient demographics overview chart."""

    # Gestational age distribution
    ga_hist = (
        alt.Chart(df)
        .mark_bar(opacity=0.7, color="steelblue")
        .encode(
            alt.X(
                "gestational_age_weeks:Q",
                bin=alt.Bin(maxbins=12),
                title="Gestational Age (weeks)",
            ),
            alt.Y("count()", title="Number of Patients"),
            tooltip=["count()"],
        )
        .properties(width=280, height=200, title="Gestational Age Distribution")
    )

    # Birth weight distribution
    weight_hist = (
        alt.Chart(df)
        .mark_bar(opacity=0.7, color="orange")
        .encode(
            alt.X(
                "birth_weight_grams:Q",
                bin=alt.Bin(maxbins=12),
                title="Birth Weight (grams)",
            ),
            alt.Y("count()", title="Number of Patients"),
            tooltip=["count()"],
        )
        .properties(width=280, height=200, title="Birth Weight Distribution")
    )

    return (
        alt.hconcat(ga_hist, weight_hist)
        .resolve_scale(x="independent", y="independent")
        .properties(title="Patient Demographics Overview")
        .configure_view(
            continuousWidth=300, continuousHeight=300, strokeWidth=0
        )
    )


def generate_primary_analysis_chart_spec(df):
    """
    Generate the primary analysis chart: Gestational Age vs Time to Full Oral Feeding.
    This addresses our main research question.
    """

    # Calculate correlation for subtitle
    correlation = df["gestational_age_weeks"].corr(df["time_to_FOF"])

    # Create interactive scatter plot with regression line
    base = alt.Chart(df).add_selection(
        alt.selection_interval(
            bind="scales", zoom=False
        )  # Enable pan but disable scroll zoom
    )

    # Scatter plot
    scatter = base.mark_circle(size=80, opacity=0.7).encode(
        alt.X(
            "gestational_age_weeks:Q",
            title="Gestational Age at Birth (weeks)",
            scale=alt.Scale(domain=[24, 32]),
        ),
        alt.Y(
            "time_to_FOF:Q",
            title="Time to Full Oral Feeding (days)",
            scale=alt.Scale(domain=[0, 45]),
        ),
        alt.Color(
            "mechanical_ventilation:N",
            title="Mechanical Ventilation",
            scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]),
            legend=alt.Legend(
                values=[0, 1], labelExpr="datum.value == 1 ? 'Yes' : 'No'"
            ),
        ),
        alt.Size(
            "birth_weight_grams:Q",
            title="Birth Weight (g)",
            scale=alt.Scale(range=[50, 200]),
        ),
        tooltip=[
            "patient_id:O",
            "gestational_age_weeks:Q",
            "time_to_FOF:Q",
            "birth_weight_grams:Q",
            "baby_sex:N",
            alt.Tooltip(
                "mechanical_ventilation:N",
                format=".0f",
                title="Mechanical Ventilation",
            ),
        ],
    )

    # Add regression line
    regression = (
        base.mark_line(color="red", strokeWidth=2)
        .transform_regression("gestational_age_weeks", "time_to_FOF")
        .encode(alt.X("gestational_age_weeks:Q"), alt.Y("time_to_FOF:Q"))
    )

    return (scatter + regression).properties(
        width=600,
        height=400,
        title=f"Gestational Age vs Time to Full Oral Feeding (r = {correlation:.3f})",
    )


def generate_feeding_progression_chart_spec(df):
    """Analyze feeding progression patterns by gestational age categories."""

    # Create GA categories for analysis
    df_analysis = df.copy()
    df_analysis["ga_category"] = pd.cut(
        df_analysis["gestational_age_weeks"],
        bins=[24, 27, 29, 32],
        labels=["24-26 weeks", "27-28 weeks", "29-31 weeks"],
        include_lowest=True,
    )

    # Box plot of time to FOF by GA category
    return (
        alt.Chart(df_analysis)
        .mark_boxplot(size=60, opacity=0.8)
        .encode(
            alt.X(
                "ga_category:O",
                title="Gestational Age Category",
                sort=["24-26 weeks", "27-28 weeks", "29-31 weeks"],
            ),
            alt.Y("time_to_FOF:Q", title="Time to Full Oral Feeding (days)"),
            alt.Color(
                "ga_category:O",
                legend=None,
                scale=alt.Scale(scheme="category10"),
            ),
            tooltip=[
                alt.Tooltip("ga_category:O", title="GA Category"),
                alt.Tooltip("time_to_FOF:Q", title="Time to FOF (days)"),
            ],
        )
        .properties(
            width=450,
            height=300,
            title="Time to Full Oral Feeding by Gestational Age Category",
        )
    )


def generate_intervention_timing_chart_spec(df):
    """Analyze the relationship between intervention timing and outcomes."""

    base = alt.Chart(df)

    # Scatter plot: PMA at first feeding vs time to full feeding
    scatter = base.mark_circle(size=60, opacity=0.7).encode(
        alt.X(
            "PMA_at_first_oral_feeding:Q",
            title="PMA at First Oral Feeding (weeks)",
            scale=alt.Scale(domain=[30, 37]),
        ),
        alt.Y("time_to_FOF:Q", title="Time to Full Oral Feeding (days)"),
        alt.Color(
            "baby_sex:N",
            title="Sex",
            scale=alt.Scale(range=["#ff7f0e", "#1f77b4"]),
        ),
        tooltip=[
            "patient_id:O",
            "PMA_at_first_oral_feeding:Q",
            "time_to_FOF:Q",
            "gestational_age_weeks:Q",
            "baby_sex:N",
        ],
    )

    # Add trend line
    trend = (
        base.mark_line(color="gray", strokeDash=[5, 5])
        .transform_regression("PMA_at_first_oral_feeding", "time_to_FOF")
        .encode(alt.X("PMA_at_first_oral_feeding:Q"), alt.Y("time_to_FOF:Q"))
    )

    return (scatter + trend).properties(
        width=500,
        height=350,
        title="Intervention Timing vs Feeding Progression",
    )


def generate_clinical_factors_chart_spec(df):
    """Analyze impact of clinical factors on feeding outcomes."""

    # Create a summary of mean time to FOF by clinical factors
    clinical_summary = []

    # Mechanical ventilation
    vent_summary = (
        df.groupby("mechanical_ventilation")["time_to_FOF"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    for _, row in vent_summary.iterrows():
        clinical_summary.append({
            "factor": "Mechanical Ventilation",
            "category": "Yes" if row["mechanical_ventilation"] == 1 else "No",
            "mean_time_to_fof": row["mean"],
            "std_time_to_fof": row["std"],
            "count": row["count"],
        })

    # Sex
    sex_summary = (
        df.groupby("baby_sex")["time_to_FOF"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    for _, row in sex_summary.iterrows():
        clinical_summary.append({
            "factor": "Sex",
            "category": row["baby_sex"],
            "mean_time_to_fof": row["mean"],
            "std_time_to_fof": row["std"],
            "count": row["count"],
        })

    # Multiple births
    multiple_summary = (
        df.groupby("multiple_births")["time_to_FOF"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    for _, row in multiple_summary.iterrows():
        clinical_summary.append({
            "factor": "Multiple Births",
            "category": "Yes" if row["multiple_births"] == "Y" else "No",
            "mean_time_to_fof": row["mean"],
            "std_time_to_fof": row["std"],
            "count": row["count"],
        })

    summary_df = pd.DataFrame(clinical_summary)

    # Create grouped bar chart
    return (
        alt.Chart(summary_df)
        .mark_bar(opacity=0.8)
        .encode(
            alt.X("factor:N", title="Clinical Factor"),
            alt.Y(
                "mean_time_to_fof:Q",
                title="Mean Time to Full Oral Feeding (days)",
            ),
            alt.Color("category:N", title="Category"),
            alt.Column("factor:N", title=""),
            tooltip=[
                "factor:N",
                "category:N",
                alt.Tooltip(
                    "mean_time_to_fof:Q", format=".1f", title="Mean Days"
                ),
                "count:Q",
            ],
        )
        .properties(
            width=120,
            height=250,
            title="Clinical Factors Impact on Feeding Progression",
        )
        .resolve_scale(x="independent")
    )


def perform_statistical_analysis(df):
    """Perform comprehensive statistical analysis for the report."""

    print("   📈 Performing statistical analysis...")

    # Correlation analysis
    correlations = {
        "ga_vs_time_to_fof": df["gestational_age_weeks"].corr(
            df["time_to_FOF"]
        ),
        "weight_vs_time_to_fof": df["birth_weight_grams"].corr(
            df["time_to_FOF"]
        ),
        "pma_first_vs_time_to_fof": df["PMA_at_first_oral_feeding"].corr(
            df["time_to_FOF"]
        ),
    }

    # T-tests for clinical factors
    vent_yes = df[df["mechanical_ventilation"] == 1]["time_to_FOF"]
    vent_no = df[df["mechanical_ventilation"] == 0]["time_to_FOF"]
    vent_ttest = stats.ttest_ind(vent_yes, vent_no)

    male = df[df["baby_sex"] == "Male"]["time_to_FOF"]
    female = df[df["baby_sex"] == "Female"]["time_to_FOF"]
    sex_ttest = stats.ttest_ind(male, female)

    multiple_yes = df[df["multiple_births"] == "Y"]["time_to_FOF"]
    multiple_no = df[df["multiple_births"] == "N"]["time_to_FOF"]
    multiple_ttest = stats.ttest_ind(multiple_yes, multiple_no)

    # Linear regression: GA predicting time to FOF
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["gestational_age_weeks"], df["time_to_FOF"]
    )

    return {
        "correlations": correlations,
        "mechanical_ventilation_effect": {
            "mean_with_vent": float(vent_yes.mean()),
            "mean_without_vent": float(vent_no.mean()),
            "p_value": float(vent_ttest.pvalue),
            "significant": vent_ttest.pvalue < 0.05,
        },
        "sex_differences": {
            "mean_male": float(male.mean()),
            "mean_female": float(female.mean()),
            "p_value": float(sex_ttest.pvalue),
            "significant": sex_ttest.pvalue < 0.05,
        },
        "multiple_birth_effect": {
            "mean_multiple": float(multiple_yes.mean()),
            "mean_singleton": float(multiple_no.mean()),
            "p_value": float(multiple_ttest.pvalue),
            "significant": multiple_ttest.pvalue < 0.05,
        },
        "ga_regression": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "interpretation": f"Each additional week of gestational age reduces time to FOF by {abs(slope):.1f} days",
        },
    }


def generate_report_elements():
    """
    Generates all chart JSONs and supporting data files for the neonatal feeding report.
    This is the main function called by the orchestrator.
    """
    print("🏥 Generating neonatal feeding study report elements...")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate realistic neonatal data
    df = create_neonatal_feeding_data()

    # Save the main dataframe
    df.to_csv(DATA_DIR / "neonatal_feeding_data.csv", index=False)
    print(
        f"   📁 Clinical data saved to: {DATA_DIR / 'neonatal_feeding_data.csv'}"
    )

    # Perform statistical analysis
    stats_results = perform_statistical_analysis(df)

    # Generate and save chart JSONs
    print("   📊 Generating and saving interactive chart specifications...")

    chart_files = {}

    # Demographics overview
    demographics_chart = generate_demographics_chart_spec(df)
    reporting.save_altair_chart_json(
        demographics_chart, CHARTS_DIR / "demographics_overview.json"
    )
    chart_files["demographics_overview"] = "charts/demographics_overview.json"

    # Primary analysis chart
    primary_chart = generate_primary_analysis_chart_spec(df)
    reporting.save_altair_chart_json(
        primary_chart, CHARTS_DIR / "primary_analysis.json"
    )
    chart_files["primary_analysis"] = "charts/primary_analysis.json"

    # Feeding progression by GA categories
    progression_chart = generate_feeding_progression_chart_spec(df)
    reporting.save_altair_chart_json(
        progression_chart, CHARTS_DIR / "feeding_progression.json"
    )
    chart_files["feeding_progression"] = "charts/feeding_progression.json"

    # Intervention timing analysis
    timing_chart = generate_intervention_timing_chart_spec(df)
    reporting.save_altair_chart_json(
        timing_chart, CHARTS_DIR / "intervention_timing.json"
    )
    chart_files["intervention_timing"] = "charts/intervention_timing.json"

    # Clinical factors impact
    factors_chart = generate_clinical_factors_chart_spec(df)
    reporting.save_altair_chart_json(
        factors_chart, CHARTS_DIR / "clinical_factors.json"
    )
    chart_files["clinical_factors"] = "charts/clinical_factors.json"

    # Generate comprehensive summary statistics
    summary_stats = {
        "study_overview": {
            "total_patients": len(df),
            "mean_gestational_age": float(df["gestational_age_weeks"].mean()),
            "std_gestational_age": float(df["gestational_age_weeks"].std()),
            "mean_birth_weight": float(df["birth_weight_grams"].mean()),
            "mean_time_to_fof": float(df["time_to_FOF"].mean()),
            "median_time_to_fof": float(df["time_to_FOF"].median()),
            "mean_pma_first_feeding": float(
                df["PMA_at_first_oral_feeding"].mean()
            ),
        },
        "patient_characteristics": {
            "sex_distribution": df["baby_sex"].value_counts().to_dict(),
            "multiple_birth_rate": float((df["multiple_births"] == "Y").mean()),
            "c_section_rate": float(df["c_section"].mean()),
            "mechanical_ventilation_rate": float(
                df["mechanical_ventilation"].mean()
            ),
            "o2_at_36_weeks_rate": float(df["o2_at_36_weeks"].mean()),
        },
        "statistical_analysis": stats_results,
        "clinical_insights": {
            "primary_finding": stats_results["ga_regression"]["interpretation"],
            "correlation_strength": "Strong"
            if abs(stats_results["correlations"]["ga_vs_time_to_fof"]) > 0.7
            else "Moderate",
            "ventilation_impact": f"Mechanical ventilation increases time to FOF by {stats_results['mechanical_ventilation_effect']['mean_with_vent'] - stats_results['mechanical_ventilation_effect']['mean_without_vent']:.1f} days on average",
        },
    }

    # Save comprehensive summary using the reporting utility
    reporting.save_json_data(
        summary_stats, DATA_DIR / "comprehensive_summary.json"
    )

    # Generate automated analysis report using the reporting structure
    print("   📝 Generating automated analysis report...")
    report_sections = reporting.create_neonatal_study_sections(
        summary_stats, stats_results, chart_files
    )

    # Save the automated report
    report_path = DOCS_DIR / "analysis" / "automated_analysis.md"
    reporting.generate_report(
        report_sections,
        output_path=report_path,
        title="Neonatal Feeding Study: Automated Analysis Report",
    )

    print(
        "   ✅ All neonatal feeding study report elements generated successfully!"
    )
    print(
        f"   📈 Key Finding: {stats_results['ga_regression']['interpretation']}"
    )
    print(
        f"   🔗 Correlation (GA vs Time to FOF): {stats_results['correlations']['ga_vs_time_to_fof']:.3f}"
    )

    return summary_stats, stats_results, chart_files


if __name__ == "__main__":
    generate_report_elements()
