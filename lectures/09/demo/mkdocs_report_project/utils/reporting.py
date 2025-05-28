#!/usr/bin/env python3
"""
Reporting utility functions for the clinical data analysis report.
"""

import os
from pathlib import Path
import pandas as pd


def create_markdown_heading(text, level=2):
    """Creates a Markdown heading string."""
    return f"{'#' * level} {text}\n\n"


def create_markdown_paragraph(text):
    """Creates a Markdown paragraph string."""
    return f"{text}\n\n"


def create_markdown_altair_chart_reference(
    chart_json_path_relative_to_docs, title=""
):
    """
    Creates a Markdown vegalite block to embed an Altair chart
    by referencing its JSON specification.
    Assumes chart_json_path is relative to the 'docs' directory.
    e.g., "charts/my_chart.json"
    """
    content = ""
    if title:
        content += f"### {title}\n"  # Add a sub-heading for the chart
    content += "```vegalite\n"
    content += "{\n"
    content += f'  "schema-url": "{chart_json_path_relative_to_docs}"\n'
    content += "}\n"
    content += "```\n\n"
    return content


def create_markdown_mermaid_diagram(mermaid_code, title=""):
    """Creates a Markdown mermaid block."""
    content = ""
    if title:
        content += f"### {title}\n"
    content += "```mermaid\n"
    content += f"{mermaid_code.strip()}\n"
    content += "```\n\n"
    return content


def create_markdown_table(dataframe):
    """Creates a Markdown table from a Pandas DataFrame."""
    if dataframe is not None and not dataframe.empty:
        return dataframe.to_markdown(index=False) + "\n\n"
    return ""


def save_markdown_page(content, output_path_str):
    """
    Saves the generated markdown content to the specified output path.
    output_path_str should be the full path to the .md file.
    """
    output_file = Path(output_path_str)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Markdown page saved to: {output_file}")


def save_altair_chart_json(chart, output_path_str):
    """
    Saves an Altair chart to a JSON file.
    output_path_str should be the full path to the .json file.
    """
    output_file = Path(output_path_str)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_file))
    print(f"Altair chart JSON saved to: {output_file}")
