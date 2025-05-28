# Demo 2: Interactive Altair Chart - PhysioNet Heart Rate Analysis

**Goal:** Create an interactive Altair chart with selection mechanisms using real physiological data, then save it for embedding in reports.

## Step 1: Set Up Your Environment

Create a new Python file or Jupyter notebook and install required packages:

```bash
pip install altair pandas numpy vl-convert-python
```

## Step 2: Create Sample PhysioNet-Style Data

Since we're simulating PhysioNet data, let's create a realistic dataset:

```python
import altair as alt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Enable Altair to render in notebooks
alt.data_transformers.enable('json')

# Create synthetic PhysioNet-style heart rate data
np.random.seed(42)  # For reproducibility

# Generate patient data
n_patients = 150
patients = []

for i in range(n_patients):
    patient_id = f"P{i+1:03d}"
    age = np.random.randint(25, 85)
    gender = np.random.choice(['Male', 'Female'])
    condition = np.random.choice(['Healthy', 'Hypertension', 'Arrhythmia', 'Heart Disease'])
    
    # Heart rate varies by condition and age
    if condition == 'Healthy':
        base_hr = 70 + np.random.normal(0, 8)
    elif condition == 'Hypertension':
        base_hr = 80 + np.random.normal(0, 12)
    elif condition == 'Arrhythmia':
        base_hr = 75 + np.random.normal(0, 20)  # More variable
    else:  # Heart Disease
        base_hr = 85 + np.random.normal(0, 15)
    
    # Age effect
    base_hr += (age - 50) * 0.2
    
    # Ensure reasonable bounds
    heart_rate = max(50, min(120, base_hr))
    
    # Blood pressure (systolic)
    if condition == 'Hypertension':
        systolic_bp = 150 + np.random.normal(0, 15)
    else:
        systolic_bp = 120 + np.random.normal(0, 12)
    
    systolic_bp = max(90, min(200, systolic_bp))
    
    patients.append({
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'condition': condition,
        'heart_rate': round(heart_rate, 1),
        'systolic_bp': round(systolic_bp, 1),
        'bmi': round(18.5 + np.random.exponential(5), 1)
    })

physio_df = pd.DataFrame(patients)
print(f"Created dataset with {len(physio_df)} patients")
print(physio_df.head())
```

## Step 3: Basic Interactive Scatter Plot

Let's start with a simple interactive scatter plot:

```python
# Basic scatter plot with tooltips
basic_chart = alt.Chart(physio_df).mark_circle(size=100).encode(
    x=alt.X('age:Q', title='Age (years)'),
    y=alt.Y('heart_rate:Q', title='Heart Rate (bpm)'),
    color=alt.Color('condition:N', title='Medical Condition'),
    tooltip=[
        alt.Tooltip('patient_id:N', title='Patient ID'),
        alt.Tooltip('age:Q', title='Age'),
        alt.Tooltip('heart_rate:Q', title='Heart Rate'),
        alt.Tooltip('condition:N', title='Condition'),
        alt.Tooltip('gender:N', title='Gender')
    ]
).properties(
    title='Heart Rate vs Age by Medical Condition',
    width=500,
    height=350
).interactive()  # Enable pan and zoom

# Display the chart
basic_chart
```

## Step 4: Add Brush Selection

Now let's add a brush selection that highlights selected points:

```python
# Create a brush selection
brush = alt.selection_interval(name='brush')

# Chart with brush selection
brush_chart = alt.Chart(physio_df).mark_circle(size=100).encode(
    x=alt.X('age:Q', title='Age (years)'),
    y=alt.Y('heart_rate:Q', title='Heart Rate (bpm)'),
    color=alt.condition(
        brush,
        alt.Color('condition:N', title='Medical Condition'),
        alt.value('lightgray')  # Non-selected points are gray
    ),
    opacity=alt.condition(brush, alt.value(0.8), alt.value(0.3)),
    tooltip=[
        alt.Tooltip('patient_id:N', title='Patient ID'),
        alt.Tooltip('age:Q', title='Age'),
        alt.Tooltip('heart_rate:Q', title='Heart Rate'),
        alt.Tooltip('condition:N', title='Condition'),
        alt.Tooltip('systolic_bp:Q', title='Systolic BP')
    ]
).add_selection(
    brush
).properties(
    title='Heart Rate vs Age - Brush to Select Patients',
    width=500,
    height=350
)

brush_chart
```

## Step 5: Linked Charts with Selection

Create a more sophisticated visualization with linked charts:

```python
# Create a dropdown selection for medical condition
condition_dropdown = alt.selection_single(
    fields=['condition'],
    bind=alt.binding_select(options=['All'] + list(physio_df['condition'].unique())),
    name='condition_select',
    init={'condition': 'All'}
)

# Base chart
base = alt.Chart(physio_df).add_selection(condition_dropdown)

# Main scatter plot
scatter = base.mark_circle(size=80).encode(
    x=alt.X('age:Q', title='Age (years)', scale=alt.Scale(domain=[20, 90])),
    y=alt.Y('heart_rate:Q', title='Heart Rate (bpm)', scale=alt.Scale(domain=[45, 125])),
    color=alt.Color('condition:N', 
                   title='Medical Condition',
                   scale=alt.Scale(range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])),
    size=alt.Size('bmi:Q', title='BMI', scale=alt.Scale(range=[50, 200])),
    opacity=alt.condition(condition_dropdown, alt.value(0.8), alt.value(0.1)),
    tooltip=[
        alt.Tooltip('patient_id:N', title='Patient ID'),
        alt.Tooltip('age:Q', title='Age'),
        alt.Tooltip('heart_rate:Q', title='Heart Rate'),
        alt.Tooltip('systolic_bp:Q', title='Systolic BP'),
        alt.Tooltip('condition:N', title='Condition'),
        alt.Tooltip('bmi:Q', title='BMI', format='.1f')
    ]
).transform_filter(
    alt.expr.if_(condition_dropdown.condition == 'All', 
                 True, 
                 alt.datum.condition == condition_dropdown.condition)
).properties(
    title='Heart Rate vs Age (Size = BMI, Filter by Condition)',
    width=450,
    height=300
)

# Summary statistics bar chart
summary_bars = base.mark_bar().encode(
    x=alt.X('condition:N', title='Medical Condition'),
    y=alt.Y('mean(heart_rate):Q', title='Average Heart Rate'),
    color=alt.Color('condition:N', legend=None),
    opacity=alt.condition(condition_dropdown, alt.value(0.8), alt.value(0.3))
).transform_filter(
    alt.expr.if_(condition_dropdown.condition == 'All', 
                 True, 
                 alt.datum.condition == condition_dropdown.condition)
).properties(
    title='Average Heart Rate by Condition',
    width=450,
    height=200
)

# Combine charts vertically
linked_chart = alt.vconcat(scatter, summary_bars).resolve_scale(color='independent')

linked_chart
```

## Step 6: Save Charts for Embedding

Save your charts in different formats for use in reports:

```python
# Create output directory
import os
os.makedirs('charts', exist_ok=True)

# Save the linked chart as JSON for MkDocs
linked_chart.save('charts/heart_rate_analysis.json')

# Save as HTML for standalone viewing
linked_chart.save('charts/heart_rate_analysis.html')

# Save as PNG for presentations (requires vl-convert-python)
try:
    linked_chart.save('charts/heart_rate_analysis.png', scale_factor=2)
    print("✅ Saved PNG successfully")
except Exception as e:
    print(f"⚠️  PNG save failed: {e}")
    print("Install vl-convert-python for PNG export: pip install vl-convert-python")

print("✅ Charts saved successfully!")
print("Files created:")
print("- charts/heart_rate_analysis.json (for MkDocs)")
print("- charts/heart_rate_analysis.html (standalone)")
```

## Step 7: Create a "Clinical Dashboard" Style Chart

Let's create something that looks more like a clinical monitoring dashboard:

```python
# Clinical dashboard style
dashboard_chart = alt.Chart(physio_df).mark_circle(
    stroke='white',
    strokeWidth=1
).encode(
    x=alt.X('systolic_bp:Q', 
           title='Systolic Blood Pressure (mmHg)',
           scale=alt.Scale(domain=[80, 200])),
    y=alt.Y('heart_rate:Q', 
           title='Heart Rate (bpm)',
           scale=alt.Scale(domain=[45, 125])),
    size=alt.Size('age:Q', 
                 title='Age',
                 scale=alt.Scale(range=[100, 400])),
    color=alt.Color('condition:N',
                   title='Condition',
                   scale=alt.Scale(
                       domain=['Healthy', 'Hypertension', 'Arrhythmia', 'Heart Disease'],
                       range=['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
                   )),
    tooltip=[
        alt.Tooltip('patient_id:N', title='Patient'),
        alt.Tooltip('age:Q', title='Age'),
        alt.Tooltip('heart_rate:Q', title='HR (bpm)'),
        alt.Tooltip('systolic_bp:Q', title='SBP (mmHg)'),
        alt.Tooltip('condition:N', title='Condition')
    ]
).properties(
    title={
        "text": "Clinical Monitoring Dashboard",
        "subtitle": "Heart Rate vs Blood Pressure (bubble size = age)"
    },
    width=600,
    height=400,
    background='#fafafa'
).interactive()

# Add reference lines for normal ranges
hr_normal = alt.Chart(pd.DataFrame({'hr': [60, 100]})).mark_rule(
    color='green', strokeDash=[5, 5], opacity=0.7
).encode(y='hr:Q')

bp_normal = alt.Chart(pd.DataFrame({'bp': [120]})).mark_rule(
    color='green', strokeDash=[5, 5], opacity=0.7
).encode(x='bp:Q')

# Combine with reference lines
clinical_dashboard = dashboard_chart + hr_normal + bp_normal

clinical_dashboard.save('charts/clinical_dashboard.json')
clinical_dashboard
```

## Success Validation

Your interactive charts should demonstrate:

- ✅ **Basic interactivity**: Tooltips showing detailed patient information
- ✅ **Selection mechanisms**: Brush selection or dropdown filters
- ✅ **Linked visualizations**: Multiple charts that respond to the same selection
- ✅ **Professional appearance**: Appropriate colors, titles, and formatting
- ✅ **Saved outputs**: JSON files ready for embedding in MkDocs
- ✅ **Health data context**: Realistic physiological parameters and medical conditions

## Bonus: Quick Data Quality Check

Add this code to validate your data makes clinical sense:

```python
# Quick sanity check on our synthetic data
print("Data Quality Check:")
print(f"Heart rate range: {physio_df['heart_rate'].min():.1f} - {physio_df['heart_rate'].max():.1f} bpm")
print(f"Age range: {physio_df['age'].min()} - {physio_df['age'].max()} years")
print(f"Blood pressure range: {physio_df['systolic_bp'].min():.1f} - {physio_df['systolic_bp'].max():.1f} mmHg")
print(f"Conditions: {physio_df['condition'].value_counts().to_dict()}")

# Check for any impossible values
issues = []
if physio_df['heart_rate'].min() < 30 or physio_df['heart_rate'].max() > 200:
    issues.append("Heart rate out of physiological range")
if physio_df['age'].min() < 0 or physio_df['age'].max() > 120:
    issues.append("Age out of reasonable range")

if issues:
    print(f"⚠️  Issues found: {issues}")
else:
    print("✅ All values within reasonable physiological ranges")
```

This demo gives students hands-on experience with Altair's key interactive features while working with realistic health data! 🏥📊