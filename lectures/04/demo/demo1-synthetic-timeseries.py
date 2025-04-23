# Demo 1: Exploring Temporal Patterns with Synthetic Healthcare Data

# ## 1. Setting Up and Generating Synthetic Data
# 
# First, let's import the necessary libraries and generate synthetic healthcare time series data.

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from datetime import datetime, timedelta
from lifelines import KaplanMeierFitter, CoxPHFitter

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Create a directory for data if it doesn't exist
os.makedirs('data', exist_ok=True)
os.makedirs('data/synthetic', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# ## 2. Generate Regular Time Series: Daily Blood Pressure Measurements
# 
# Let's create a synthetic dataset of daily blood pressure measurements over 1 year.

# Generate dates for one year
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate synthetic systolic blood pressure data with:
# - Baseline trend (slight increase over time)
# - Seasonal component (higher in winter, lower in summer)
# - Weekly pattern (higher on weekends)
# - Random noise

# Parameters
baseline = 120  # baseline blood pressure
trend_slope = 0.01  # slight upward trend
seasonal_amplitude = 5  # seasonal variation amplitude
weekly_amplitude = 3  # weekly variation amplitude
noise_level = 3  # random noise level

# Generate components
n_days = len(dates)
time_index = np.arange(n_days)

# Trend component
trend = baseline + trend_slope * time_index

# Seasonal component (yearly cycle)
yearly_cycle = seasonal_amplitude * np.sin(2 * np.pi * time_index / 365)

# Weekly component (higher on weekends)
weekly_cycle = weekly_amplitude * np.array([0, 0, 0, 0, 0.5, 1, 1])[pd.DatetimeIndex(dates).dayofweek]

# Random noise
noise = np.random.normal(0, noise_level, n_days)

# Combine components
systolic_bp = trend + yearly_cycle + weekly_cycle + noise

# Create a DataFrame
bp_data = pd.DataFrame({
    'date': dates,
    'systolic_bp': systolic_bp,
    'diastolic_bp': systolic_bp * 0.65 + np.random.normal(0, 2, n_days)  # diastolic is roughly 65% of systolic
})

# Save to CSV
bp_data.to_csv('data/synthetic/daily_blood_pressure.csv', index=False)

print("Generated daily blood pressure data with shape:", bp_data.shape)

# ## 3. Generate Irregular Time Series: Patient Visits
# 
# Now let's create an irregular time series of patient visits with various measurements.

# Parameters for patient visits
n_visits = 30  # number of visits over 1 year
visit_dates = sorted(np.random.choice(dates, size=n_visits, replace=False))

# Generate patient data with various measurements
patient_data = pd.DataFrame({
    'visit_date': visit_dates,
    'weight_kg': np.random.normal(70, 3, n_visits),  # weight with some variation
    'heart_rate': np.random.normal(75, 8, n_visits),  # heart rate with variation
    'temperature': np.random.normal(36.8, 0.3, n_visits),  # body temperature
    'glucose': np.random.normal(100, 15, n_visits)  # blood glucose
})

# Add a trend to weight (gradual weight loss)
time_index = np.arange(n_visits)
patient_data['weight_kg'] = patient_data['weight_kg'] - 0.1 * time_index

# Save to CSV
patient_data.to_csv('data/synthetic/patient_visits.csv', index=False)

print("Generated irregular patient visit data with shape:", patient_data.shape)

# ## 4. Generate Multivariate Time Series: Continuous Monitoring
# 
# Let's create a multivariate time series dataset simulating continuous monitoring
# for a shorter period (1 day) with measurements every minute.

# Generate timestamps for one day with minute frequency
day_start = datetime(2023, 6, 15, 0, 0, 0)
day_end = datetime(2023, 6, 15, 23, 59, 0)
minute_timestamps = pd.date_range(start=day_start, end=day_end, freq='min')

# Number of minutes
n_minutes = len(minute_timestamps)

# Generate synthetic heart rate data with:
# - Baseline pattern (lower during sleep, higher during day)
# - Activity spikes (exercise periods)
# - Random variation

# Time of day in hours (0-24)
hours = np.array([ts.hour + ts.minute/60 for ts in minute_timestamps])

# Baseline heart rate pattern (lower at night, higher during day)
baseline_hr = 60 + 20 * np.sin(np.pi * (hours - 4) / 12)  # lowest at 4am, highest at 4pm

# Add activity spikes (morning exercise 7-8am, evening exercise 6-7pm)
activity_mask_morning = (hours >= 7) & (hours <= 8)
activity_mask_evening = (hours >= 18) & (hours <= 19)
activity_hr = np.zeros(n_minutes)
activity_hr[activity_mask_morning] = 40  # morning exercise increases HR by 40
activity_hr[activity_mask_evening] = 35  # evening exercise increases HR by 35

# Random variation
hr_noise = np.random.normal(0, 3, n_minutes)

# Combine components for heart rate
heart_rate = baseline_hr + activity_hr + hr_noise

# Generate correlated oxygen saturation (normally high, drops slightly when HR is very high)
oxygen_saturation = 98 - 0.02 * (heart_rate - 60) + np.random.normal(0, 0.5, n_minutes)
oxygen_saturation = np.clip(oxygen_saturation, 94, 100)  # clip to realistic range

# Generate respiratory rate (correlated with heart rate)
respiratory_rate = 12 + 0.1 * (heart_rate - 60) + np.random.normal(0, 1, n_minutes)

# Create DataFrame
monitoring_data = pd.DataFrame({
    'timestamp': minute_timestamps,
    'heart_rate': heart_rate,
    'oxygen_saturation': oxygen_saturation,
    'respiratory_rate': respiratory_rate
})

# Save to CSV
monitoring_data.to_csv('data/synthetic/continuous_monitoring.csv', index=False)

print("Generated continuous monitoring data with shape:", monitoring_data.shape)

# ## 5. Generate Survival Data for Kaplan-Meier and Cox Proportional Hazards Analysis
#
# Let's create synthetic survival data to demonstrate survival analysis techniques.

# Number of patients
n_patients = 200

# Generate patient IDs
patient_ids = np.arange(1, n_patients + 1)

# Generate covariates
age = np.random.normal(65, 10, n_patients)  # Age, centered around 65
age = np.clip(age, 40, 90)  # Clip to realistic range

# Treatment group (0 = control, 1 = treatment)
treatment = np.random.binomial(1, 0.5, n_patients)

# Biomarker value (higher values indicate worse prognosis)
biomarker = np.random.gamma(2, 2, n_patients)

# Gender (0 = female, 1 = male)
gender = np.random.binomial(1, 0.5, n_patients)

# Generate true survival times based on covariates
# Higher age, higher biomarker, and male gender increase risk
# Treatment reduces risk
baseline_hazard = 0.1
age_effect = 0.02
biomarker_effect = 0.3
gender_effect = 0.2
treatment_effect = -0.5

# Linear predictor
linear_pred = (age_effect * (age - 65) + 
               biomarker_effect * biomarker + 
               gender_effect * gender + 
               treatment_effect * treatment)

# Generate survival times from exponential distribution
true_survival_times = np.random.exponential(
    scale=1.0 / (baseline_hazard * np.exp(linear_pred)), 
    size=n_patients
)

# Maximum follow-up time (5 years)
max_followup = 5.0

# Generate censoring times (some patients are lost to follow-up)
censoring_times = np.random.uniform(0.5, max_followup, n_patients)

# Observed time is the minimum of true survival time and censoring time
observed_times = np.minimum(true_survival_times, censoring_times)

# Event indicator (1 if the event was observed, 0 if censored)
event = (true_survival_times <= censoring_times).astype(int)

# Create DataFrame
survival_data = pd.DataFrame({
    'patient_id': patient_ids,
    'age': age,
    'treatment': treatment,
    'biomarker': biomarker,
    'gender': gender,
    'time': observed_times,
    'event': event
})

# Add treatment and gender labels for better visualization
survival_data['treatment_label'] = survival_data['treatment'].map({0: 'Control', 1: 'Treatment'})
survival_data['gender_label'] = survival_data['gender'].map({0: 'Female', 1: 'Male'})

# Save to CSV
survival_data.to_csv('data/synthetic/survival_data.csv', index=False)

print("Generated survival data with shape:", survival_data.shape)
print(f"Event rate: {survival_data['event'].mean():.2f}")

# ## 6. Exploring the Regular Time Series (Blood Pressure Data)
# 
# Let's visualize and analyze the daily blood pressure data.

# Plot the blood pressure time series
plt.figure(figsize=(12, 6))
plt.plot(bp_data['date'], bp_data['systolic_bp'], 'b-', label='Systolic')
plt.plot(bp_data['date'], bp_data['diastolic_bp'], 'r-', label='Diastolic')
plt.title('Daily Blood Pressure Measurements (1 Year)')
plt.xlabel('Date')
plt.ylabel('Blood Pressure (mmHg)')
plt.legend()
plt.grid(True)
plt.show()

# Decompose the time series to extract trend, seasonality, and residuals
systolic_series = bp_data.set_index('date')['systolic_bp']

# Perform decomposition
# Use a period of 30 days (monthly seasonality) instead of 365 days
decomposition = seasonal_decompose(systolic_series, model='additive', period=30)

# Plot the decomposition
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(decomposition.observed)
plt.title('Observed')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(decomposition.trend)
plt.title('Trend')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal)
plt.title('Seasonality')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(decomposition.resid)
plt.title('Residuals')
plt.grid(True)

plt.tight_layout()
plt.show()

# Weekly pattern analysis
bp_data['day_of_week'] = bp_data['date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(10, 6))
sns.boxplot(x='day_of_week', y='systolic_bp', data=bp_data, order=day_order)
plt.title('Blood Pressure by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Systolic Blood Pressure (mmHg)')
plt.grid(True, axis='y')
plt.show()

# Monthly pattern analysis
bp_data['month'] = bp_data['date'].dt.month_name()
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='systolic_bp', data=bp_data, order=month_order)
plt.title('Blood Pressure by Month')
plt.xlabel('Month')
plt.ylabel('Systolic Blood Pressure (mmHg)')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# ## 7. Analyzing the Irregular Time Series (Patient Visits)
# 
# Now let's analyze the irregular patient visit data.

# Plot the weight over time
plt.figure(figsize=(12, 6))
plt.plot(patient_data['visit_date'], patient_data['weight_kg'], 'bo-')
plt.title('Patient Weight Over Time')
plt.xlabel('Visit Date')
plt.ylabel('Weight (kg)')
plt.grid(True)
plt.show()

# Calculate time between visits
patient_data['visit_date'] = pd.to_datetime(patient_data['visit_date'])
patient_data['days_since_last_visit'] = (patient_data['visit_date'] - 
                                         patient_data['visit_date'].shift(1)).dt.days

# Plot the distribution of days between visits
plt.figure(figsize=(10, 6))
plt.hist(patient_data['days_since_last_visit'].dropna(), bins=15, alpha=0.7)
plt.title('Distribution of Days Between Patient Visits')
plt.xlabel('Days Since Last Visit')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Resampling irregular data to regular monthly intervals
# First, set the visit date as the index
patient_data_indexed = patient_data.set_index('visit_date')

# Resample to monthly frequency, using mean for aggregation
monthly_data = patient_data_indexed.resample('ME').mean()

# Plot the original irregular data vs. resampled regular data
plt.figure(figsize=(12, 6))
plt.plot(patient_data['visit_date'], patient_data['weight_kg'], 'bo-', alpha=0.7, label='Irregular Visits')
plt.plot(monthly_data.index, monthly_data['weight_kg'], 'ro-', label='Monthly Resampled')
plt.title('Original vs. Resampled Weight Measurements')
plt.xlabel('Date')
plt.ylabel('Weight (kg)')
plt.legend()
plt.grid(True)
plt.show()

# Correlation analysis between different measurements
plt.figure(figsize=(10, 8))
sns.heatmap(patient_data.drop(['visit_date', 'days_since_last_visit'], axis=1).corr(), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between Patient Measurements')
plt.show()

# Scatter plot of heart rate vs. glucose
plt.figure(figsize=(10, 6))
sns.scatterplot(x='glucose', y='heart_rate', data=patient_data)
plt.title('Heart Rate vs. Glucose Levels')
plt.xlabel('Glucose (mg/dL)')
plt.ylabel('Heart Rate (bpm)')
plt.grid(True)
plt.show()

# ## 8. Exploring the Multivariate Time Series (Continuous Monitoring)
# 
# Let's analyze the continuous monitoring data.

# Plot all vital signs
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Heart rate
axes[0].plot(monitoring_data['timestamp'], monitoring_data['heart_rate'], 'r-')
axes[0].set_title('Heart Rate')
axes[0].set_ylabel('BPM')
axes[0].grid(True)

# Oxygen saturation
axes[1].plot(monitoring_data['timestamp'], monitoring_data['oxygen_saturation'], 'b-')
axes[1].set_title('Oxygen Saturation')
axes[1].set_ylabel('SpO2 (%)')
axes[1].grid(True)

# Respiratory rate
axes[2].plot(monitoring_data['timestamp'], monitoring_data['respiratory_rate'], 'g-')
axes[2].set_title('Respiratory Rate')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Breaths/min')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Zoom in on exercise periods
morning_exercise = monitoring_data[(monitoring_data['timestamp'].dt.hour >= 7) & 
                                  (monitoring_data['timestamp'].dt.hour < 8)]

plt.figure(figsize=(12, 6))
plt.plot(morning_exercise['timestamp'], morning_exercise['heart_rate'], 'r-')
plt.title('Heart Rate During Morning Exercise (7-8 AM)')
plt.xlabel('Time')
plt.ylabel('Heart Rate (BPM)')
plt.grid(True)
plt.show()

# Calculate rolling statistics (moving average and standard deviation)
window_size = 30  # 30-minute window
monitoring_data['heart_rate_ma'] = monitoring_data['heart_rate'].rolling(window=window_size).mean()
monitoring_data['heart_rate_std'] = monitoring_data['heart_rate'].rolling(window=window_size).std()

# Plot heart rate with rolling statistics
plt.figure(figsize=(12, 6))
plt.plot(monitoring_data['timestamp'], monitoring_data['heart_rate'], 'b-', alpha=0.3, label='Raw')
plt.plot(monitoring_data['timestamp'], monitoring_data['heart_rate_ma'], 'r-', label='30-min Moving Avg')
plt.fill_between(monitoring_data['timestamp'], 
                 monitoring_data['heart_rate_ma'] - monitoring_data['heart_rate_std'],
                 monitoring_data['heart_rate_ma'] + monitoring_data['heart_rate_std'],
                 color='r', alpha=0.2)
plt.title('Heart Rate with 30-minute Moving Average and Standard Deviation')
plt.xlabel('Time')
plt.ylabel('Heart Rate (BPM)')
plt.legend()
plt.grid(True)
plt.show()

# Correlation between vital signs
plt.figure(figsize=(10, 8))
sns.heatmap(monitoring_data[['heart_rate', 'oxygen_saturation', 'respiratory_rate']].corr(), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between Vital Signs')
plt.show()

# Scatter plot of heart rate vs. respiratory rate
plt.figure(figsize=(10, 6))
plt.scatter(monitoring_data['heart_rate'], monitoring_data['respiratory_rate'], alpha=0.5)
plt.title('Respiratory Rate vs. Heart Rate')
plt.xlabel('Heart Rate (BPM)')
plt.ylabel('Respiratory Rate (breaths/min)')
plt.grid(True)
plt.show()

# ## 9. Survival Analysis with Kaplan-Meier and Cox Proportional Hazards
#
# Now let's demonstrate survival analysis techniques using our synthetic survival data.

# ### Kaplan-Meier Survival Analysis
#
# First, let's use the Kaplan-Meier estimator to visualize survival curves.

# Initialize the Kaplan-Meier fitter
kmf = KaplanMeierFitter()

# Fit the model to all data
kmf.fit(survival_data['time'], survival_data['event'], label='Overall')

# Plot the survival curve
plt.figure(figsize=(12, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# Compare survival curves by treatment group
plt.figure(figsize=(12, 6))

# Treatment group
kmf_treatment = KaplanMeierFitter()
kmf_treatment.fit(
    survival_data[survival_data['treatment'] == 1]['time'],
    survival_data[survival_data['treatment'] == 1]['event'],
    label='Treatment'
)
kmf_treatment.plot_survival_function()

# Control group
kmf_control = KaplanMeierFitter()
kmf_control.fit(
    survival_data[survival_data['treatment'] == 0]['time'],
    survival_data[survival_data['treatment'] == 0]['event'],
    label='Control'
)
kmf_control.plot_survival_function()

plt.title('Kaplan-Meier Survival Curves by Treatment Group')
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# Compare survival curves by gender
plt.figure(figsize=(12, 6))

# Female
kmf_female = KaplanMeierFitter()
kmf_female.fit(
    survival_data[survival_data['gender'] == 0]['time'],
    survival_data[survival_data['gender'] == 0]['event'],
    label='Female'
)
kmf_female.plot_survival_function()

# Male
kmf_male = KaplanMeierFitter()
kmf_male.fit(
    survival_data[survival_data['gender'] == 1]['time'],
    survival_data[survival_data['gender'] == 1]['event'],
    label='Male'
)
kmf_male.plot_survival_function()

plt.title('Kaplan-Meier Survival Curves by Gender')
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# ### Cox Proportional Hazards Model
#
# Now let's fit a Cox Proportional Hazards model to assess the effect of covariates on survival.

# Initialize the Cox model
cph = CoxPHFitter()

# Fit the model
cph.fit(
    survival_data[['time', 'event', 'age', 'treatment', 'biomarker', 'gender']],
    duration_col='time',
    event_col='event'
)

# Print the model summary
print("\nCox Proportional Hazards Model Summary:")
print(cph.summary)

# Plot the hazard ratios with confidence intervals
plt.figure(figsize=(10, 6))
cph.plot()
plt.title('Hazard Ratios with 95% Confidence Intervals')
plt.grid(True)
plt.show()

# Plot the partial effects of each covariate
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age effect
cph.plot_partial_effects_on_outcome('age', values=[50, 60, 70, 80], ax=axes[0, 0])
axes[0, 0].set_title('Effect of Age on Survival')
axes[0, 0].grid(True)

# Treatment effect
cph.plot_partial_effects_on_outcome('treatment', values=[0, 1], ax=axes[0, 1])
axes[0, 1].set_title('Effect of Treatment on Survival')
axes[0, 1].grid(True)

# Biomarker effect
cph.plot_partial_effects_on_outcome('biomarker', values=[1, 3, 5, 7], ax=axes[1, 0])
axes[1, 0].set_title('Effect of Biomarker on Survival')
axes[1, 0].grid(True)

# Gender effect
cph.plot_partial_effects_on_outcome('gender', values=[0, 1], ax=axes[1, 1])
axes[1, 1].set_title('Effect of Gender on Survival')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# ## 10. Summary and Conclusions
# 
# In this demo, we've explored different types of time series data and demonstrated various analysis techniques:
# 
# 1. **Regular Time Series**: Daily blood pressure measurements over a year
#    - Demonstrated trend, seasonality, and weekly patterns
#    - Used time series decomposition to separate components
# 
# 2. **Irregular Time Series**: Patient visits with various measurements
#    - Analyzed the challenges of irregular sampling
#    - Applied resampling to convert to regular intervals
#    - Explored correlations between different measurements
# 
# 3. **Multivariate Time Series**: Continuous monitoring of vital signs
#    - Visualized multiple related time series
#    - Identified patterns during specific activities
#    - Applied rolling statistics for smoothing and variability analysis
#
# 4. **Survival Analysis**: Time-to-event data with censoring
#    - Applied Kaplan-Meier method to estimate survival curves
#    - Used Cox Proportional Hazards model to assess covariate effects
#    - Interpreted hazard ratios and their confidence intervals
# 
# These techniques form the foundation of time series analysis in healthcare and can be applied to various real-world datasets.