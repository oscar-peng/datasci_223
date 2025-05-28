#!/usr/bin/env python3
"""
Reporting utility functions for the neonatal feeding study analysis.
Based on the oof reporting structure but adapted for MkDocs integration.
"""

import os
from pathlib import Path
import pandas as pd
import json


def create_markdown_section(
    heading,
    content=None,
    table_data=None,
    chart_reference=None,
    image_location=None,
):
    """
    Create a markdown section with heading and various content types.

    Args:
        heading (str): The heading text, can include markdown heading levels (e.g., "# Main", "## Sub")
        content (str, optional): Text content for the section
        table_data (pandas.DataFrame or dict, optional): Data to display as a table
        chart_reference (str, optional): Path to chart JSON for vegalite embedding
        image_location (str, optional): Path to an image to include

    Returns:
        str: Markdown formatted section
    """
    # Extract heading level and text
    heading_level = 0
    heading_text = heading

    # Count leading # characters to determine heading level
    while heading_text.startswith("#"):
        heading_level += 1
        heading_text = heading_text[1:]

    # If no explicit heading level was provided, use default level 2
    if heading_level == 0:
        heading_level = 2

    # Create the heading
    section = f"{'#' * heading_level} {heading_text.strip()}\n\n"

    # Add text content if provided
    if content is not None:
        section += f"{content}\n\n"

    # Add table if provided
    if table_data is not None:
        if isinstance(table_data, pd.DataFrame):
            section += table_data.to_markdown(index=False)
        elif isinstance(table_data, dict):
            # Convert dict to DataFrame if it's a simple key-value structure
            if all(
                isinstance(v, (int, float, str, bool))
                for v in table_data.values()
            ):
                df = pd.DataFrame({
                    "Variable": list(table_data.keys()),
                    "Value": list(table_data.values()),
                })
                section += df.to_markdown(index=False)
            else:
                # For more complex dictionaries, show as formatted text
                for key, value in table_data.items():
                    if isinstance(value, (int, float)):
                        section += f"**{key}**: {value:.2f}\n\n"
                    else:
                        section += f"**{key}**: {value}\n\n"
        else:
            # For other types, convert to string
            section += str(table_data)
        section += "\n\n"

    # Add chart reference if provided (for MkDocs charts plugin)
    if chart_reference is not None:
        section += "```vegalite\n"
        section += "{\n"
        section += f'  "schema-url": "{chart_reference}"\n'
        section += "}\n"
        section += "```\n\n"

    # Add image if provided
    if image_location is not None:
        # Get just the filename for the markdown reference
        image_name = os.path.basename(image_location)
        section += f"![{image_name}](media/{image_name})\n\n"

    return section


def create_statistical_summary_section(stats_results):
    """
    Create a formatted section for statistical analysis results.

    Args:
        stats_results (dict): Statistical analysis results

    Returns:
        str: Formatted markdown section
    """
    content = "### Statistical Analysis Results\n\n"

    # Correlations
    if "correlations" in stats_results:
        content += "**Correlation Analysis:**\n\n"
        for var_pair, correlation in stats_results["correlations"].items():
            var_display = var_pair.replace("_", " ").title()
            content += f"- {var_display}: r = {correlation:.3f}\n"
        content += "\n"

    # Regression results
    if "ga_regression" in stats_results:
        reg = stats_results["ga_regression"]
        content += "**Linear Regression (Gestational Age → Time to FOF):**\n\n"
        content += f"- **R²**: {reg['r_squared']:.3f}\n"
        content += f"- **Slope**: {reg['slope']:.2f} days per week\n"
        content += f"- **P-value**: {reg['p_value']:.2e}\n"
        content += f"- **Interpretation**: {reg['interpretation']}\n\n"

    # Group comparisons
    if "mechanical_ventilation_effect" in stats_results:
        vent = stats_results["mechanical_ventilation_effect"]
        content += "**Mechanical Ventilation Effect:**\n\n"
        content += f"- **With ventilation**: {vent['mean_with_vent']:.1f} days (mean)\n"
        content += f"- **Without ventilation**: {vent['mean_without_vent']:.1f} days (mean)\n"
        content += f"- **Difference**: {vent['mean_with_vent'] - vent['mean_without_vent']:.1f} days\n"
        content += f"- **Significant**: {'Yes' if vent['significant'] else 'No'} (p = {vent['p_value']:.3f})\n\n"

    return content


def generate_report(report_sections, output_path=None, title="Analysis Report"):
    """
    Generate a complete markdown report from a list of sections.

    Args:
        report_sections (list): List of tuples (heading, content, table_data, chart_reference, image_location)
                               or (heading, table_data, image_location) for backward compatibility
        output_path (str or Path): The full path to save the report
        title (str): Title for the report

    Returns:
        Path: Path to the saved report
    """
    # Create report content with title
    content = f"# {title}\n\n"

    # Add each section
    for section_data in report_sections:
        if len(section_data) == 3:
            # Backward compatibility: (heading, table_data, image_location)
            heading, table_data, image_location = section_data
            content += create_markdown_section(
                heading, table_data=table_data, image_location=image_location
            )
        elif len(section_data) == 5:
            # New format: (heading, content, table_data, chart_reference, image_location)
            (
                heading,
                text_content,
                table_data,
                chart_reference,
                image_location,
            ) = section_data
            content += create_markdown_section(
                heading,
                content=text_content,
                table_data=table_data,
                chart_reference=chart_reference,
                image_location=image_location,
            )
        else:
            # Flexible format - assume it's a dict with keys
            section_dict = (
                section_data if isinstance(section_data, dict) else {}
            )
            content += create_markdown_section(**section_dict)

    # Save the report
    if output_path is None:
        output_path = Path("docs") / "generated_report.md"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"📝 Report saved to: {output_path}")
    return output_path


def save_altair_chart_json(chart, output_path_str):
    """
    Saves an Altair chart to a JSON file.
    output_path_str should be the full path to the .json file.
    """
    output_file = Path(output_path_str)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_file))
    print(f"📊 Chart JSON saved to: {output_file}")


def save_json_data(data, output_path_str):
    """
    Save data as JSON with proper formatting.

    Args:
        data (dict): Data to save
        output_path_str (str or Path): Path to save the JSON file
    """
    output_file = Path(output_path_str)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"💾 Data saved to: {output_file}")


def create_neonatal_study_sections(summary_data, stats_results, chart_files):
    """
    Create standardized report sections for the neonatal feeding study.

    Args:
        summary_data (dict): Study summary statistics
        stats_results (dict): Statistical analysis results
        chart_files (dict): Mapping of chart names to file paths

    Returns:
        list: List of report sections ready for generate_report()
    """
    sections = []

    # Executive Summary
    study = summary_data["study_overview"]
    exec_content = f"""This study analyzes feeding progression patterns in **{study["total_patients"]} premature infants** to identify optimal timing for oral feeding initiation.

**Key Findings:**
- Average time to full oral feeding: **{study["mean_time_to_fof"]:.1f} days**
- Mean gestational age: **{study["mean_gestational_age"]:.1f} weeks**
- Primary finding: {stats_results["ga_regression"]["interpretation"]}
- Statistical significance: R² = {stats_results["ga_regression"]["r_squared"]:.3f}, p < 0.001"""

    sections.append(("## Executive Summary", exec_content, None, None, None))

    # Demographics with chart
    demo_content = f"""Our study cohort represents a typical NICU population:

**Population Characteristics:**
- **Gestational Age**: {study["mean_gestational_age"]:.1f} ± {study["std_gestational_age"]:.1f} weeks
- **Birth Weight**: {study["mean_birth_weight"]:.0f} grams (average)
- **Sex Distribution**: {summary_data["patient_characteristics"]["sex_distribution"]}
- **Mechanical Ventilation**: {summary_data["patient_characteristics"]["mechanical_ventilation_rate"]:.1%}"""

    sections.append((
        "## Patient Demographics",
        demo_content,
        None,
        chart_files.get("demographics_overview"),
        None,
    ))

    # Primary Analysis
    primary_content = f"""Our central research question examines whether gestational age predicts feeding progression:

**Statistical Results:**
- **Correlation**: r = {stats_results["correlations"]["ga_vs_time_to_fof"]:.3f}
- **Clinical Impact**: {stats_results["ga_regression"]["interpretation"]}
- **Variance Explained**: R² = {stats_results["ga_regression"]["r_squared"]:.3f}

**Interactive Features:**
- Hover over points for patient details
- Zoom and pan to explore specific ranges
- Color coding shows mechanical ventilation history
- Point size represents birth weight"""

    sections.append((
        "## Primary Analysis: Gestational Age and Feeding Outcomes",
        primary_content,
        None,
        chart_files.get("primary_analysis"),
        None,
    ))

    # Clinical Factors
    factors_content = f"""Analysis of clinical factors affecting feeding outcomes:

**Significant Factors:**"""

    if stats_results["mechanical_ventilation_effect"]["significant"]:
        vent_diff = (
            stats_results["mechanical_ventilation_effect"]["mean_with_vent"]
            - stats_results["mechanical_ventilation_effect"][
                "mean_without_vent"
            ]
        )
        factors_content += f"\n- **Mechanical Ventilation**: Adds {vent_diff:.1f} days (p < 0.001)"

    if (
        "sex_differences" in stats_results
        and stats_results["sex_differences"]["significant"]
    ):
        sex_diff = (
            stats_results["sex_differences"]["mean_male"]
            - stats_results["sex_differences"]["mean_female"]
        )
        factors_content += f"\n- **Sex Differences**: Males take {sex_diff:.1f} days longer (p < 0.05)"

    sections.append((
        "## Clinical Factors Impact",
        factors_content,
        None,
        chart_files.get("clinical_factors"),
        None,
    ))

    # Statistical Summary
    stats_content = create_statistical_summary_section(stats_results)
    sections.append((
        "## Statistical Analysis",
        stats_content,
        None,
        None,
        None,
    ))

    return sections


# Backward compatibility functions
def create_markdown_heading(text, level=2):
    """Creates a Markdown heading string."""
    return f"{'#' * level} {text}\n\n"


def create_markdown_paragraph(text):
    """Creates a Markdown paragraph string."""
    return f"{text}\n\n"


def create_markdown_altair_chart_reference(
    chart_json_path_relative_to_docs, title=""
):
    """Creates a Markdown vegalite block to embed an Altair chart."""
    content = ""
    if title:
        content += f"### {title}\n"
    content += "```vegalite\n"
    content += "{\n"
    content += f'  "schema-url": "{chart_json_path_relative_to_docs}"\n'
    content += "}\n"
    content += "```\n\n"
    return content
