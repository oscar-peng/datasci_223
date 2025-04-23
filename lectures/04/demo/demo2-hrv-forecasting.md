# Demo 2: ARIMA Forecasting with Heart Rate Variability Data

This notebook guides you through time series forecasting with ARIMA models using real RR interval data. Each section is annotated with visible context and teaching notes for health data science beginners.

---

## 1. Setup: Import Libraries and Configure Environment

**Teaching Note:**  
Importing libraries is the first step in any data science workflow. Here, we use pandas and numpy for data manipulation, matplotlib and seaborn for plotting, and statsmodels for time series analysis. Setting a random seed ensures reproducibility.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
import os
import glob
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
```

---

## 2. Load and Explore Patient Metadata

**Teaching Note:**  
Patient metadata provides context for the physiological data. Age groupings are useful for stratified analysis.

```python
data_dir = 'lectures/04/demo/rr-interval-time-series-from-healthy-subjects-1.0.0'
patient_info = pd.read_csv(f'{data_dir}/patient-info.csv')
print(f"Loaded information for {len(patient_info)} patients")
print(patient_info.head())
```

---

### Explore Patient Age Distribution

**Teaching Note:**  
Visualizing age distribution helps understand the diversity of the dataset and informs later analysis.

```python
def categorize_age(age):
    if age < 1:
        return "Infant (<1 year)"
    elif age < 6:
        return "Young Child (1-5 years)"
    elif age < 13:
        return "Child (6-12 years)"
    elif age < 18:
        return "Adolescent (13-17 years)"
    else:
        return "Adult (18+ years)"
patient_info['Age Group'] = patient_info['Age (years)'].apply(categorize_age)
age_group_counts = patient_info['Age Group'].value_counts()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(patient_info['Age (years)'], bins=20, alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.grid(True)
plt.subplot(1, 2, 2)
sns.countplot(y='Age Group', data=patient_info, order=age_group_counts.index)
plt.title('Age Group Distribution')
plt.xlabel('Count')
plt.ylabel('Age Group')
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()
```

---

## 3. Load and Process RR Interval Data

**Teaching Note:**  
RR intervals (the time between heartbeats) are often irregularly sampled. Converting to heart rate (bpm) and visualizing helps spot artifacts and physiological patterns.

```python
def load_rr_intervals(file_path):
    with open(file_path, 'r') as f:
        rr_intervals = [float(line.strip()) for line in f if line.strip()]
    return np.array(rr_intervals)

selected_patients = []
for age_group in age_group_counts.index:
    group_patients = patient_info[patient_info['Age Group'] == age_group].head(2)
    selected_patients.extend(group_patients['File'].tolist())

rr_data = {}
for patient_id in selected_patients:
    file_path = f'{data_dir}/{patient_id}.txt'
    if os.path.exists(file_path):
        rr_data[patient_id] = load_rr_intervals(file_path)
        print(f"Loaded {len(rr_data[patient_id])} RR intervals for patient {patient_id}")
```

---

### Convert RR Intervals to Heart Rate and Plot

**Teaching Note:**  
Heart rate (bpm) is easier to interpret than RR intervals (ms). Plotting the first 1000 beats for each patient gives a sense of variability and possible outliers.

```python
def rr_to_hr(rr_intervals):
    return 60000 / rr_intervals
hr_data = {pid: rr_to_hr(rr) for pid, rr in rr_data.items()}

plt.figure(figsize=(12, 8))
for i, (patient_id, hr) in enumerate(list(hr_data.items())[:4]):
    plt.subplot(2, 2, i+1)
    plt.plot(hr[:1000])
    patient_age = patient_info[patient_info['File'] == patient_id]['Age (years)'].values[0]
    plt.title(f'Patient {patient_id} (Age: {patient_age} years)')
    plt.xlabel('Beat Number')
    plt.ylabel('Heart Rate (bpm)')
    plt.grid(True)
plt.tight_layout()
plt.show()