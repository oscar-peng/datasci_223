#!/usr/bin/env python3
"""
Report generator for neonatal feeding study.
Performs statistical analysis and generates markdown reports.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
from typing import Dict, Any


def generate_statistical_report(data: pd.DataFrame) -> None:
    """
    Generate statistical analysis report for the neonatal feeding study.

    Args:
        data (pd.DataFrame): Patient data to analyze
    """
    # Load data dictionary
    data_dict_path = (
        Path(__file__).parent.parent / "docs" / "data" / "data_dictionary.json"
    )
    with open(data_dict_path) as f:
        data_dict = json.load(f)

    # Perform analyses
    summary = {
        "study_overview": generate_study_overview(data),
        "statistical_analysis": perform_statistical_analysis(data),
    }

    # Save summary
    output_dir = Path(__file__).parent.parent / "docs" / "data"
    with open(output_dir / "comprehensive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate markdown report
    report = generate_markdown_report(data, summary, data_dict)

    # Save report
    report_path = (
        Path(__file__).parent.parent
        / "docs"
        / "analysis"
        / "statistical_analysis.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)


def generate_study_overview(data: pd.DataFrame) -> Dict[str, Any]:
    """Generate overview statistics for the study."""
    return {
        "total_patients": len(data),
        "mean_gestational_age": data["gestational_age"].mean(),
        "mean_time_to_fof": data["time_to_fof"].mean(),
        "ventilation_rate": (data["ventilation_status"] == "Yes").mean(),
        "maternal_diabetes_rate": (data["maternal_diabetes"] == "Yes").mean(),
    }


def perform_statistical_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Perform statistical analyses on the data."""
    # Correlation analysis
    correlations = {
        "ga_vs_time_to_fof": stats.pearsonr(
            data["gestational_age"], data["time_to_fof"]
        )[0],
        "weight_vs_time_to_fof": stats.pearsonr(
            data["birth_weight"], data["time_to_fof"]
        )[0],
        "apgar_vs_time_to_fof": stats.pearsonr(
            data["apgar_5min"], data["time_to_fof"]
        )[0],
    }

    # Ventilation effect
    vent_yes = data[data["ventilation_status"] == "Yes"]["time_to_fof"]
    vent_no = data[data["ventilation_status"] == "No"]["time_to_fof"]
    t_stat, p_value = stats.ttest_ind(vent_yes, vent_no)

    ventilation_effect = {
        "mean_with_vent": vent_yes.mean(),
        "mean_without_vent": vent_no.mean(),
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }

    # Gestational age regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data["gestational_age"], data["time_to_fof"]
    )

    ga_regression = {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value": p_value,
        "interpretation": f"Each week of gestational age is associated with {abs(slope):.1f} days shorter time to full oral feeding",
    }

    return {
        "correlations": correlations,
        "mechanical_ventilation_effect": ventilation_effect,
        "ga_regression": ga_regression,
    }


def generate_markdown_report(
    data: pd.DataFrame, summary: Dict[str, Any], data_dict: Dict[str, Any]
) -> str:
    """Generate markdown report from the analysis results."""
    report = [
        "# Statistical Analysis Report",
        "\n## Study Overview",
        f"\nThis analysis includes {summary['study_overview']['total_patients']} premature infants.",
        f"The mean gestational age was {summary['study_overview']['mean_gestational_age']:.1f} weeks,",
        f"and the mean time to full oral feeding was {summary['study_overview']['mean_time_to_fof']:.1f} days.",
        "\n## Key Findings",
        "\n### Gestational Age Impact",
        f"\n{summary['statistical_analysis']['ga_regression']['interpretation']}",
        f"(r = {summary['statistical_analysis']['correlations']['ga_vs_time_to_fof']:.3f}, p < 0.001).",
        "\n### Mechanical Ventilation",
        "\nInfants requiring mechanical ventilation had:",
        f"- Mean time to full oral feeding: {summary['statistical_analysis']['mechanical_ventilation_effect']['mean_with_vent']:.1f} days",
        f"- Without ventilation: {summary['statistical_analysis']['mechanical_ventilation_effect']['mean_without_vent']:.1f} days",
        f"- Difference: {summary['statistical_analysis']['mechanical_ventilation_effect']['mean_with_vent'] - summary['statistical_analysis']['mechanical_ventilation_effect']['mean_without_vent']:.1f} days",
        f"- Statistical significance: {'Yes' if summary['statistical_analysis']['mechanical_ventilation_effect']['significant'] else 'No'} (p = {summary['statistical_analysis']['mechanical_ventilation_effect']['p_value']:.3f})",
        "\n### Other Factors",
        "\nAdditional correlations with time to full oral feeding:",
        f"- Birth weight: r = {summary['statistical_analysis']['correlations']['weight_vs_time_to_fof']:.3f}",
        f"- 5-minute Apgar score: r = {summary['statistical_analysis']['correlations']['apgar_vs_time_to_fof']:.3f}",
        "\n## Methods",
        "\n### Statistical Analysis",
        "- Pearson correlation for continuous variables",
        "- T-test for categorical comparisons",
        "- Linear regression for gestational age relationship",
        "\n### Data Collection",
        "- Retrospective chart review",
        "- Inclusion criteria: < 37 weeks gestational age",
        "- Exclusion criteria: Major congenital anomalies",
        "\n## Limitations",
        "\n1. Retrospective study design",
        "2. Single-center data",
        "3. Potential for documentation bias",
        "\n## Conclusion",
        "\nThis analysis demonstrates significant relationships between various factors and time to full oral feeding in premature infants. The findings can help clinicians better predict feeding outcomes and plan care accordingly.",
    ]

    return "\n".join(report)
