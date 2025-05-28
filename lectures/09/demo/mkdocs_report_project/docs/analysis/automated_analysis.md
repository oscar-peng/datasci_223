# Neonatal Feeding Study: Automated Analysis Report

## Executive Summary

This study analyzes feeding progression patterns in **180 premature infants** to identify optimal timing for oral feeding initiation.

**Key Findings:**
- Average time to full oral feeding: **16.1 days**
- Mean gestational age: **28.4 weeks**
- Primary finding: Each additional week of gestational age reduces time to FOF by 2.5 days
- Statistical significance: R² = 0.603, p < 0.001

## Patient Demographics

Our study cohort represents a typical NICU population:

**Population Characteristics:**
- **Gestational Age**: 28.4 ± 2.2 weeks
- **Birth Weight**: 1044 grams (average)
- **Sex Distribution**: {'Male': 97, 'Female': 83}
- **Mechanical Ventilation**: 42.2%

```vegalite
{
  "schema-url": "charts/demographics_overview.json"
}
```

## Primary Analysis: Gestational Age and Feeding Outcomes

Our central research question examines whether gestational age predicts feeding progression:

**Statistical Results:**
- **Correlation**: r = -0.776
- **Clinical Impact**: Each additional week of gestational age reduces time to FOF by 2.5 days
- **Variance Explained**: R² = 0.603

**Interactive Features:**
- Hover over points for patient details
- Zoom and pan to explore specific ranges
- Color coding shows mechanical ventilation history
- Point size represents birth weight

```vegalite
{
  "schema-url": "charts/primary_analysis.json"
}
```

## Clinical Factors Impact

Analysis of clinical factors affecting feeding outcomes:

**Significant Factors:**
- **Mechanical Ventilation**: Adds 7.8 days (p < 0.001)

```vegalite
{
  "schema-url": "charts/clinical_factors.json"
}
```

## Statistical Analysis

### Statistical Analysis Results

**Correlation Analysis:**

- Ga Vs Time To Fof: r = -0.776
- Weight Vs Time To Fof: r = -0.670
- Pma First Vs Time To Fof: r = 0.584

**Linear Regression (Gestational Age → Time to FOF):**

- **R²**: 0.603
- **Slope**: -2.45 days per week
- **P-value**: 1.60e-37
- **Interpretation**: Each additional week of gestational age reduces time to FOF by 2.5 days

**Mechanical Ventilation Effect:**

- **With ventilation**: 20.6 days (mean)
- **Without ventilation**: 12.7 days (mean)
- **Difference**: 7.8 days
- **Significant**: Yes (p = 0.000)



