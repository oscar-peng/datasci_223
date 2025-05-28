# Heart Rate Pattern Analysis

## Age Distribution Analysis

```vegalite
{
  "schema-url": "charts/age_distribution.json"
}
```

The age distribution reveals interesting patterns across medical conditions:

!!! note "Key Observations"
    - **Healthy patients** are more evenly distributed across age groups
    - **Heart disease** patients skew toward older age groups (60+ years)
    - **Hypertension** shows a bimodal distribution with peaks at 45-55 and 65-75 years

## Interactive Clinical Dashboard

```vegalite
{
  "schema-url": "charts/interactive_dashboard.json"
}
```

This interactive dashboard allows you to:

=== "Filter by Condition"
    Use the dropdown menu to focus on specific medical conditions

=== "Explore Relationships"
    Examine the relationship between blood pressure and heart rate

=== "Identify Outliers"
    Spot patients with unusual cardiovascular profiles

## Statistical Summary

Based on our analysis of the clinical data:

| Condition | Avg Heart Rate | Std Dev | Sample Size |
|-----------|----------------|---------|-------------|
| Healthy | 70.2 bpm | 8.1 | 80 patients |
| Hypertension | 80.5 bpm | 12.3 | 60 patients |
| Arrhythmia | 75.8 bpm | 20.1 | 40 patients |
| Heart Disease | 85.1 bpm | 15.2 | 20 patients |

!!! warning "Clinical Significance"
    The elevated heart rate variability in arrhythmia patients (σ = 20.1) 
    suggests the need for continuous monitoring protocols.