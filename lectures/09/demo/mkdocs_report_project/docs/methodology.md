# Methodology

This section details the methodology used for data collection, processing, analysis, and visualization.

## Data Source

The data used in this report is synthetically generated to mimic real-world clinical data, focusing on cardiovascular metrics. The generation script (`scripts/generate_charts.py`) includes parameters for age, gender, medical conditions (Healthy, Hypertension, Arrhythmia, Heart Disease), heart rate, systolic blood pressure, and BMI.

## Data Processing and Analysis

1.  **Data Generation:** A Python script using `pandas` and `numpy` generates a dataset of 200 patients.
2.  **Chart Generation:** `Altair` is used to create all visualizations. These charts are saved as JSON specifications.
3.  **Statistical Summaries:** Basic descriptive statistics (mean, counts) are calculated using `pandas`.

## Reporting Tool

This report is built using **MkDocs** with the **Material for MkDocs** theme. Interactive charts are embedded using the **mkdocs-charts-plugin**.

## Analysis Workflow Diagram

The overall workflow is visualized on the [Home](/) page.

*(Students should expand this section with more details about specific statistical methods or data cleaning steps if they were to use real data.)*