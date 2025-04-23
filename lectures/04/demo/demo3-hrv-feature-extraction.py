# Demo 3: Advanced Feature Extraction from Heart Rate Variability Data

# ## 1. Setting Up and Loading the Data
# 
# First, let's import the necessary libraries and load the RR interval time series dataset.

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Define the path to the dataset
data_dir = 'lectures/04/demo/rr-interval-time-series-from-healthy-subjects-1.0.0'

# Load patient information
patient_info = pd.read_csv(f'{data_dir}/patient-info.csv')
print(f"Loaded information for {len(patient_info)} patients")
print(patient_info.head())

# ## 2. Exploring the Dataset
# 
# Let's explore the dataset to understand its structure and characteristics.

# Create age groups for analysis
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

# Display the distribution of age groups
age_group_counts = patient_info['Age Group'].value_counts()
print("\nAge group distribution:")
print(age_group_counts)

# Plot the age distribution
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

# ## 3. Loading and Processing RR Interval Data
# 
# Now let's load the RR interval data for patients from different age groups.

# Function to load RR interval data from a file
def load_rr_intervals(file_path):
    """Load RR intervals from a text file."""
    with open(file_path, 'r') as f:
        rr_intervals = [float(line.strip()) for line in f if line.strip()]
    return np.array(rr_intervals)

# Select patients from different age groups
selected_patients = []
for age_group in age_group_counts.index:
    # Get up to 5 patients from each age group
    group_patients = patient_info[patient_info['Age Group'] == age_group].head(5)
    selected_patients.extend(group_patients['File'].tolist())

# Load RR interval data for selected patients
rr_data = {}
for patient_id in selected_patients:
    # Format the file name with leading zeros (e.g., 000.txt, 002.txt)
    # This matches the actual file names in the dataset
    file_path = f'{data_dir}/{patient_id:03d}.txt'
    if os.path.exists(file_path):
        rr_data[patient_id] = load_rr_intervals(file_path)
        print(f"Loaded {len(rr_data[patient_id])} RR intervals for patient {patient_id}")

# ## 4. Advanced Signal Processing
# 
# Let's apply advanced signal processing techniques to the RR interval data.

# Function to preprocess RR intervals
def preprocess_rr(rr_intervals, remove_outliers=True, remove_ectopic=True):
    """Preprocess RR intervals by removing outliers and ectopic beats."""
    # Convert to numpy array if not already
    rr = np.array(rr_intervals)
    
    # Remove outliers (values outside 3 standard deviations)
    if remove_outliers:
        mean_rr = np.mean(rr)
        std_rr = np.std(rr)
        rr = rr[(rr > mean_rr - 3*std_rr) & (rr < mean_rr + 3*std_rr)]
    
    # Remove ectopic beats (beats that differ by more than 20% from the previous beat)
    if remove_ectopic:
        rr_diff = np.abs(np.diff(rr) / rr[:-1])
        good_indices = np.where(rr_diff < 0.2)[0]
        rr = rr[good_indices]
    
    return rr

# Preprocess RR intervals for each patient
preprocessed_rr = {}
for patient_id, rr in rr_data.items():
    preprocessed_rr[patient_id] = preprocess_rr(rr)
    print(f"Preprocessed RR intervals for patient {patient_id}: {len(preprocessed_rr[patient_id])} intervals")

# Plot original vs. preprocessed RR intervals for one patient
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

# ## 5. Time-Domain Feature Extraction
#
# Let's extract time-domain features from the RR interval data.
#
# Time-domain analysis is the simplest approach to HRV analysis, working directly
# with the series of RR intervals and their statistical properties.
#
# Key time-domain measures include:
#
# 1. Basic statistics:
#    - Mean RR interval: Average duration between heartbeats
#    - SDNN (Standard Deviation of NN intervals): Overall variability
#    - Min/Max RR: Range of RR intervals
#
# 2. Heart rate statistics:
#    - Mean HR: Average heart rate in beats per minute
#    - Standard deviation of HR: Variability in heart rate
#
# 3. Beat-to-beat variability measures:
#    - RMSSD (Root Mean Square of Successive Differences): Short-term variability,
#      reflects parasympathetic activity
#    - pNN50 (Percentage of successive RR intervals differing by >50ms):
#      Another measure of parasympathetic activity
#
# These measures are clinically relevant and have been associated with various
# health outcomes, including cardiovascular health and autonomic nervous system function.

# Function to calculate time-domain HRV features
def calculate_time_domain_features(rr_intervals):
    """Calculate time-domain HRV features from RR intervals."""
    # Convert to seconds for standard HRV calculations
    rr_sec = rr_intervals / 1000.0
    
    # Basic statistics
    mean_rr = np.mean(rr_sec)
    std_rr = np.std(rr_sec)
    min_rr = np.min(rr_sec)
    max_rr = np.max(rr_sec)
    
    # Heart rate statistics
    heart_rates = 60 / rr_sec
    mean_hr = np.mean(heart_rates)
    std_hr = np.std(heart_rates)
    min_hr = np.min(heart_rates)
    max_hr = np.max(heart_rates)
    
    # HRV time-domain metrics
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_sec))))  # Root mean square of successive differences
    sdnn = np.std(rr_sec)  # Standard deviation of NN intervals
    nn50 = sum(abs(np.diff(rr_sec)) > 0.05)  # Number of pairs of successive NN intervals differing by more than 50 ms
    pnn50 = (nn50 / len(rr_sec)) * 100  # Percentage of NN50
    
    # Return features as a dictionary
    return {
        'mean_rr': mean_rr,
        'std_rr': std_rr,
        'min_rr': min_rr,
        'max_rr': max_rr,
        'mean_hr': mean_hr,
        'std_hr': std_hr,
        'min_hr': min_hr,
        'max_hr': max_hr,
        'rmssd': rmssd,
        'sdnn': sdnn,
        'nn50': nn50,
        'pnn50': pnn50
    }

# Calculate time-domain features for each patient
time_domain_features = {}
for patient_id, rr in preprocessed_rr.items():
    time_domain_features[patient_id] = calculate_time_domain_features(rr)

# Convert to DataFrame for easier analysis
time_domain_df = pd.DataFrame.from_dict(time_domain_features, orient='index')

# Add patient information
time_domain_df['patient_id'] = time_domain_df.index
time_domain_df = time_domain_df.merge(
    patient_info[['File', 'Age (years)', 'Gender', 'Age Group']], 
    left_on='patient_id', 
    right_on='File'
)

# Display the time-domain features
print("\nTime-Domain Features:")
print(time_domain_df.head())

# Plot some time-domain features by age group
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='Age Group', y='mean_hr', data=time_domain_df, order=age_group_counts.index)
plt.title('Mean Heart Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mean Heart Rate (bpm)')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.subplot(2, 2, 2)
sns.boxplot(x='Age Group', y='sdnn', data=time_domain_df, order=age_group_counts.index)
plt.title('SDNN by Age Group')
plt.xlabel('Age Group')
plt.ylabel('SDNN (s)')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.subplot(2, 2, 3)
sns.boxplot(x='Age Group', y='rmssd', data=time_domain_df, order=age_group_counts.index)
plt.title('RMSSD by Age Group')
plt.xlabel('Age Group')
plt.ylabel('RMSSD (s)')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.subplot(2, 2, 4)
sns.boxplot(x='Age Group', y='pnn50', data=time_domain_df, order=age_group_counts.index)
plt.title('pNN50 by Age Group')
plt.xlabel('Age Group')
plt.ylabel('pNN50 (%)')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# ## 6. Frequency-Domain Feature Extraction
#
# Let's extract frequency-domain features from the RR interval data.
#
# Frequency domain analysis transforms the time series data into the frequency domain,
# revealing oscillatory patterns that aren't visible in the time domain.
#
# In HRV analysis, the frequency domain reveals:
# 1. Very Low Frequency (VLF, 0.0033-0.04 Hz): Reflects long-term regulatory mechanisms
#    like thermoregulation and hormonal systems
#
# 2. Low Frequency (LF, 0.04-0.15 Hz): Reflects both sympathetic and parasympathetic
#    activity, including the baroreflex system (blood pressure regulation)
#
# 3. High Frequency (HF, 0.15-0.4 Hz): Primarily reflects parasympathetic (vagal) activity,
#    often associated with respiratory sinus arrhythmia
#
# The LF/HF ratio is commonly used as an index of sympathovagal balance, though
# this interpretation has some limitations and is still debated in research.

# Function to calculate frequency-domain HRV features
def calculate_frequency_domain_features(rr_intervals, fs=4.0):
    """Calculate frequency-domain HRV features from RR intervals.
    
    Frequency domain analysis of HRV reveals how power (variance) is distributed
    across different frequency bands:
    
    - VLF (0.0033-0.04 Hz): Very low frequency, influenced by long-term regulation
    - LF (0.04-0.15 Hz): Low frequency, reflects both sympathetic and parasympathetic activity
    - HF (0.15-0.4 Hz): High frequency, primarily reflects parasympathetic (vagal) activity
    
    The LF/HF ratio is often used as an index of sympathovagal balance.
    """
    # Convert to seconds
    rr_sec = rr_intervals / 1000.0
    
    # Interpolate RR intervals to get evenly sampled signal
    # First, create time array based on cumulative sum of RR intervals
    t = np.cumsum(rr_sec)
    t = np.insert(t, 0, 0)  # Insert 0 at the beginning
    
    # Create evenly sampled time array
    t_interp = np.arange(0, t[-1], 1/fs)
    
    # Interpolate RR intervals
    rr_interp = np.interp(t_interp, t[1:], rr_sec)
    
    # Remove mean
    rr_interp = rr_interp - np.mean(rr_interp)
    
    # Calculate power spectral density using Welch's method for better estimates
    # This is more robust than a simple FFT for real physiological data
    # Welch's method:
    # 1. Divides the signal into overlapping segments
    # 2. Computes periodogram for each segment
    # 3. Averages the periodograms to reduce noise
    # This produces a smoother, more reliable PSD estimate
    freqs, psd = signal.welch(rr_interp, fs=fs, nperseg=min(2048, len(rr_interp)),
                             scaling='density')
    
    # Define frequency bands
    vlf_band = (0.0033, 0.04)  # Very low frequency
    lf_band = (0.04, 0.15)     # Low frequency
    hf_band = (0.15, 0.4)      # High frequency
    
    # Calculate power in each band
    vlf_power = np.sum(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
    lf_power = np.sum(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
    hf_power = np.sum(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
    total_power = vlf_power + lf_power + hf_power
    
    # Calculate normalized powers
    lf_norm = lf_power / (lf_power + hf_power)
    hf_norm = hf_power / (lf_power + hf_power)
    
    # Calculate LF/HF ratio
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    # Return features and PSD for plotting
    return {
        'vlf_power': vlf_power,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'total_power': total_power,
        'lf_norm': lf_norm,
        'hf_norm': hf_norm,
        'lf_hf_ratio': lf_hf_ratio,
        'freqs': freqs,
        'psd': psd
    }

# Calculate frequency-domain features for each patient
freq_domain_features = {}
for patient_id, rr in preprocessed_rr.items():
    try:
        freq_domain_features[patient_id] = calculate_frequency_domain_features(rr)
        print(f"Calculated frequency-domain features for patient {patient_id}")
    except Exception as e:
        print(f"Error calculating frequency-domain features for patient {patient_id}: {e}")

# Extract features for DataFrame (excluding freqs and psd)
freq_features_for_df = {}
for patient_id, features in freq_domain_features.items():
    freq_features_for_df[patient_id] = {k: v for k, v in features.items() if k not in ['freqs', 'psd']}

# Convert to DataFrame
freq_domain_df = pd.DataFrame.from_dict(freq_features_for_df, orient='index')

# Add patient information
freq_domain_df['patient_id'] = freq_domain_df.index
freq_domain_df = freq_domain_df.merge(
    patient_info[['File', 'Age (years)', 'Gender', 'Age Group']], 
    left_on='patient_id', 
    right_on='File'
)

# Display the frequency-domain features
print("\nFrequency-Domain Features:")
print(freq_domain_df.head())

# Plot power spectral density for a few patients
plt.figure(figsize=(12, 8))
for i, patient_id in enumerate(list(freq_domain_features.keys())[:4]):
    plt.subplot(2, 2, i+1)
    features = freq_domain_features[patient_id]
    
    # Scale PSD for better visualization
    psd_scaled = features['psd'] * 1e6  # Scale up for visibility
    
    plt.plot(features['freqs'], psd_scaled)
    
    # Highlight frequency bands
    vlf_mask = (features['freqs'] >= 0.0033) & (features['freqs'] < 0.04)
    lf_mask = (features['freqs'] >= 0.04) & (features['freqs'] < 0.15)
    hf_mask = (features['freqs'] >= 0.15) & (features['freqs'] < 0.4)
    
    plt.fill_between(features['freqs'][vlf_mask], psd_scaled[vlf_mask], alpha=0.3, color='gray', label='VLF')
    plt.fill_between(features['freqs'][lf_mask], psd_scaled[lf_mask], alpha=0.3, color='blue', label='LF')
    plt.fill_between(features['freqs'][hf_mask], psd_scaled[hf_mask], alpha=0.3, color='red', label='HF')
    
    patient_age = patient_info[patient_info['File'] == patient_id]['Age (years)'].values[0]
    plt.title(f'Patient {patient_id} (Age: {patient_age} years)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.xlim(0, 0.4)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot frequency-domain features by age group
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='Age Group', y='lf_power', data=freq_domain_df, order=age_group_counts.index)
plt.title('LF Power by Age Group')
plt.xlabel('Age Group')
plt.ylabel('LF Power')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.subplot(2, 2, 2)
sns.boxplot(x='Age Group', y='hf_power', data=freq_domain_df, order=age_group_counts.index)
plt.title('HF Power by Age Group')
plt.xlabel('Age Group')
plt.ylabel('HF Power')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.subplot(2, 2, 3)
sns.boxplot(x='Age Group', y='lf_norm', data=freq_domain_df, order=age_group_counts.index)
plt.title('Normalized LF Power by Age Group')
plt.xlabel('Age Group')
plt.ylabel('LF Norm')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.subplot(2, 2, 4)
sns.boxplot(x='Age Group', y='lf_hf_ratio', data=freq_domain_df, order=age_group_counts.index)
plt.title('LF/HF Ratio by Age Group')
plt.xlabel('Age Group')
plt.ylabel('LF/HF Ratio')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# ## 7. Nonlinear Feature Extraction
#
# Let's extract nonlinear features from the RR interval data.
#
# Nonlinear analysis captures complex dynamics in heart rate that aren't
# detected by traditional time and frequency domain methods.
#
# Poincaré plots are a type of recurrence plot that visualize the correlation
# between successive RR intervals:
# - Each point represents an RR interval (x-axis) plotted against the next RR interval (y-axis)
# - The resulting cloud of points forms an ellipse-like shape
# - The width and length of this ellipse provide information about short and long-term variability
#
# Key Poincaré plot features:
# - SD1: Standard deviation perpendicular to the line of identity, representing
#        short-term variability (beat-to-beat)
# - SD2: Standard deviation along the line of identity, representing
#        long-term variability
# - SD1/SD2 ratio: Balance between short and long-term variability
# - Area: Total area of the ellipse, representing total variability
#
# These nonlinear measures can detect subtle patterns in heart rate dynamics
# that may indicate physiological or pathological conditions.

# Function to calculate Poincaré plot features
def calculate_poincare_features(rr_intervals):
    """Calculate Poincaré plot features from RR intervals."""
    # Convert to seconds
    rr_sec = rr_intervals / 1000.0
    
    # Create Poincaré plot data
    x = rr_sec[:-1]  # RR(n)
    y = rr_sec[1:]   # RR(n+1)
    
    # Calculate SD1 and SD2
    diff_rr = np.diff(rr_sec)
    sd1 = np.std(diff_rr) / np.sqrt(2)  # Standard deviation perpendicular to the line of identity
    sd2 = np.sqrt(2 * np.var(rr_sec) - sd1**2)  # Standard deviation along the line of identity
    
    # Calculate area of the ellipse
    area = np.pi * sd1 * sd2
    
    # Calculate SD1/SD2 ratio
    sd_ratio = sd1 / sd2 if sd2 > 0 else 0
    
    return {
        'sd1': sd1,
        'sd2': sd2,
        'area': area,
        'sd_ratio': sd_ratio,
        'x': x,
        'y': y
    }

# Calculate Poincaré plot features for each patient
poincare_features = {}
for patient_id, rr in preprocessed_rr.items():
    try:
        poincare_features[patient_id] = calculate_poincare_features(rr)
        print(f"Calculated Poincaré plot features for patient {patient_id}")
    except Exception as e:
        print(f"Error calculating Poincaré plot features for patient {patient_id}: {e}")

# Extract features for DataFrame (excluding x and y)
poincare_features_for_df = {}
for patient_id, features in poincare_features.items():
    poincare_features_for_df[patient_id] = {k: v for k, v in features.items() if k not in ['x', 'y']}

# Convert to DataFrame
poincare_df = pd.DataFrame.from_dict(poincare_features_for_df, orient='index')

# Add patient information
poincare_df['patient_id'] = poincare_df.index
poincare_df = poincare_df.merge(
    patient_info[['File', 'Age (years)', 'Gender', 'Age Group']], 
    left_on='patient_id', 
    right_on='File'
)

# Display the Poincaré plot features
print("\nPoincaré Plot Features:")
print(poincare_df.head())

# Plot Poincaré plots for a few patients
plt.figure(figsize=(12, 10))
for i, patient_id in enumerate(list(poincare_features.keys())[:4]):
    plt.subplot(2, 2, i+1)
    features = poincare_features[patient_id]
    
    # Plot points
    plt.scatter(features['x'], features['y'], alpha=0.3, s=1)
    
    # Plot identity line
    min_val = min(np.min(features['x']), np.min(features['y']))
    max_val = max(np.max(features['x']), np.max(features['y']))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add ellipse (simplified)
    center_x = np.mean(features['x'])
    center_y = np.mean(features['y'])
    plt.plot(center_x, center_y, 'ro')
    
    patient_age = patient_info[patient_info['File'] == patient_id]['Age (years)'].values[0]
    plt.title(f'Patient {patient_id} (Age: {patient_age} years)\nSD1: {features["sd1"]:.4f}, SD2: {features["sd2"]:.4f}')
    plt.xlabel('RR(n) (s)')
    plt.ylabel('RR(n+1) (s)')
    plt.axis('equal')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot nonlinear features by age group
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='Age Group', y='sd1', data=poincare_df, order=age_group_counts.index)
plt.title('SD1 by Age Group')
plt.xlabel('Age Group')
plt.ylabel('SD1 (s)')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.subplot(2, 2, 2)
sns.boxplot(x='Age Group', y='sd2', data=poincare_df, order=age_group_counts.index)
plt.title('SD2 by Age Group')
plt.xlabel('Age Group')
plt.ylabel('SD2 (s)')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.subplot(2, 2, 3)
sns.boxplot(x='Age Group', y='area', data=poincare_df, order=age_group_counts.index)
plt.title('Ellipse Area by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Area')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.subplot(2, 2, 4)
sns.boxplot(x='Age Group', y='sd_ratio', data=poincare_df, order=age_group_counts.index)
plt.title('SD1/SD2 Ratio by Age Group')
plt.xlabel('Age Group')
plt.ylabel('SD1/SD2 Ratio')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# ## 8. Summary and Conclusions
# 
# In this demo, we've explored advanced feature extraction from heart rate variability data:
# 
# 1. **Advanced Signal Processing**:
#    - Preprocessed RR intervals by removing outliers and ectopic beats
#    - Applied interpolation and windowing for frequency analysis
# 
# 2. **Feature Extraction**:
#    - Time-domain features (SDNN, RMSSD, pNN50)
#    - Frequency-domain features (VLF, LF, HF powers, LF/HF ratio)
#    - Nonlinear features (Poincaré plot metrics: SD1, SD2, area, SD1/SD2 ratio)
# 
# 3. **Age Group Analysis**:
#    - Visualized how HRV features vary across different age groups
#    - Observed patterns in heart rate dynamics across the lifespan
# 
# These features provide a comprehensive characterization of heart rate variability and can be used for various applications:
# 
# - Monitoring autonomic nervous system function
# - Assessing cardiovascular health
# - Detecting stress and recovery patterns
# - Tracking development across the lifespan
# 
# The techniques demonstrated here can be applied to other physiological time series data to extract meaningful features for analysis and interpretation.
