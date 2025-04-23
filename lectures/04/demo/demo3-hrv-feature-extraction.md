# Demo 3: Advanced Feature Extraction from Heart Rate Variability Data

This notebook guides you through advanced feature extraction from RR interval data. Each section is annotated with visible context and teaching notes for health data science beginners.

---

## 1. Setup: Import Libraries and Configure Environment

**Teaching Note:**  
Importing libraries is the first step in any data science workflow. Here, we use pandas and numpy for data manipulation, matplotlib and seaborn for plotting, and statsmodels/scipy for time series and signal processing. Setting a random seed ensures reproducibility.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
import os
import glob
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
data_dir = 'rr-interval-time-series-from-healthy-subjects-1.0.0'
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
RR intervals (the time between heartbeats) are often irregularly sampled. Preprocessing and visualizing helps spot artifacts and physiological patterns.

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
    file_path = f'{data_dir}/{patient_id:03d}.txt'
    if os.path.exists(file_path):
        rr_data[patient_id] = load_rr_intervals(file_path)
        print(f"Loaded {len(rr_data[patient_id])} RR intervals for patient {patient_id}")
```

---

### Preprocess and Visualize RR Intervals

**Teaching Note:**  
Preprocessing removes outliers and ectopic beats. Visualizing before and after helps students see the impact of cleaning.

```python
def preprocess_rr(rr_intervals, remove_outliers=True, remove_ectopic=True):
    rr = np.array(rr_intervals)
    if remove_outliers:
        mean_rr = np.mean(rr)
        std_rr = np.std(rr)
        rr = rr[(rr > mean_rr - 3*std_rr) & (rr < mean_rr + 3*std_rr)]
    if remove_ectopic:
        rr_diff = np.abs(np.diff(rr) / rr[:-1])
        good_indices = np.where(rr_diff < 0.2)[0]
        rr = rr[good_indices]
    return rr

preprocessed_rr = {}
for patient_id, rr in rr_data.items():
    preprocessed_rr[patient_id] = preprocess_rr(rr)
    print(f"Preprocessed RR intervals for patient {patient_id}: {len(preprocessed_rr[patient_id])} intervals")

sample_patient = list(preprocessed_rr.keys())[0]
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(rr_data[sample_patient][:500])
plt.title(f'Original RR Intervals - Patient {sample_patient}')
plt.xlabel('Beat Number')
plt.ylabel('RR Interval (ms)')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(preprocessed_rr[sample_patient][:500])
plt.title(f'Preprocessed RR Intervals - Patient {sample_patient}')
plt.xlabel('Beat Number')
plt.ylabel('RR Interval (ms)')
plt.grid(True)
plt.tight_layout()
plt.show()