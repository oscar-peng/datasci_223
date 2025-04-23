# Demo 2: ARIMA Forecasting with Heart Rate Variability Data

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
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
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

# Display basic statistics about patient ages
print("\nAge statistics:")
print(f"Age range: {patient_info['Age (years)'].min()} to {patient_info['Age (years)'].max()} years")
print(f"Mean age: {patient_info['Age (years)'].mean():.2f} years")
print(f"Median age: {patient_info['Age (years)'].median():.2f} years")

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
# Now let's load the RR interval data for a few patients from different age groups.

# Function to load RR interval data from a file
def load_rr_intervals(file_path):
    """Load RR intervals from a text file."""
    with open(file_path, 'r') as f:
        rr_intervals = [float(line.strip()) for line in f if line.strip()]
    return np.array(rr_intervals)

# Select a few patients from different age groups
selected_patients = []
for age_group in age_group_counts.index:
    # Get up to 2 patients from each age group
    group_patients = patient_info[patient_info['Age Group'] == age_group].head(2)
    selected_patients.extend(group_patients['File'].tolist())

# Load RR interval data for selected patients
rr_data = {}
for patient_id in selected_patients:
    file_path = f'{data_dir}/{patient_id}.txt'
    if os.path.exists(file_path):
        rr_data[patient_id] = load_rr_intervals(file_path)
        print(f"Loaded {len(rr_data[patient_id])} RR intervals for patient {patient_id}")

# ## 4. Converting RR Intervals to Heart Rate Time Series
# 
# Let's convert the RR intervals to heart rate time series for analysis.

# Function to convert RR intervals to heart rate
def rr_to_hr(rr_intervals):
    """Convert RR intervals (ms) to heart rate (bpm)."""
    return 60000 / rr_intervals

# Convert RR intervals to heart rate for each patient
hr_data = {}
for patient_id, rr_intervals in rr_data.items():
    hr_data[patient_id] = rr_to_hr(rr_intervals)

# Plot heart rate time series for a few patients
plt.figure(figsize=(12, 8))
for i, (patient_id, hr) in enumerate(list(hr_data.items())[:4]):
    plt.subplot(2, 2, i+1)
    plt.plot(hr[:1000])  # Plot first 1000 points for clarity
    patient_age = patient_info[patient_info['File'] == patient_id]['Age (years)'].values[0]
    plt.title(f'Patient {patient_id} (Age: {patient_age} years)')
    plt.xlabel('Beat Number')
    plt.ylabel('Heart Rate (bpm)')
    plt.grid(True)
plt.tight_layout()
plt.show()

# ## 5. Creating Regular Time Series for ARIMA Modeling
# 
# ARIMA models require regular time series. Let's convert our heart rate data to regular time series.

# Select one patient for detailed analysis
selected_patient = list(hr_data.keys())[0]
hr = hr_data[selected_patient]
patient_age = patient_info[patient_info['File'] == selected_patient]['Age (years)'].values[0]
print(f"\nSelected patient {selected_patient} (Age: {patient_age} years) for detailed analysis")

# Create a time index based on cumulative RR intervals
rr_intervals = rr_data[selected_patient]
time_seconds = np.cumsum(rr_intervals) / 1000  # Convert to seconds

# Create a DataFrame with time and heart rate
hr_df = pd.DataFrame({
    'time': time_seconds,
    'heart_rate': hr
})

# Resample to regular intervals (e.g., 1 second)
# First, set the time as index
hr_df.set_index('time', inplace=True)

# Create a regular time index
regular_time_index = np.arange(0, time_seconds[-1], 1)  # 1-second intervals

# Resample using interpolation
hr_regular = np.interp(regular_time_index, time_seconds, hr)

# Create a DataFrame with regular time series
hr_regular_df = pd.DataFrame({
    'time': regular_time_index,
    'heart_rate': hr_regular
})

# Plot original vs. resampled heart rate
plt.figure(figsize=(12, 6))
plt.plot(time_seconds[:500], hr[:500], 'b-', alpha=0.5, label='Original (Irregular)')
plt.plot(regular_time_index[:500], hr_regular[:500], 'r-', label='Resampled (Regular)')
plt.title(f'Original vs. Resampled Heart Rate - Patient {selected_patient}')
plt.xlabel('Time (seconds)')
plt.ylabel('Heart Rate (bpm)')
plt.legend()
plt.grid(True)
plt.show()

# ## 6. Testing for Stationarity
# 
# Before applying ARIMA models, we need to check if the time series is stationary.

# Perform Augmented Dickey-Fuller test
def test_stationarity(timeseries, window=30):
    """Test stationarity using the Augmented Dickey-Fuller test."""
    # Calculate rolling statistics
    rolling_mean = pd.Series(timeseries).rolling(window=window).mean()
    rolling_std = pd.Series(timeseries).rolling(window=window).std()
    
    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, label='Original')
    plt.plot(rolling_mean, label=f'{window}-period Rolling Mean')
    plt.plot(rolling_std, label=f'{window}-period Rolling Std')
    plt.legend()
    plt.title('Rolling Mean & Standard Deviation')
    plt.grid(True)
    plt.show()
    
    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    
    # Interpret the results
    if dftest[1] <= 0.05:
        print("\nConclusion: The time series is stationary (p-value <= 0.05)")
        return True
    else:
        print("\nConclusion: The time series is not stationary (p-value > 0.05)")
        return False

# Test stationarity of the regular heart rate time series
is_stationary = test_stationarity(hr_regular)

# If not stationary, try differencing
if not is_stationary:
    print("\nApplying differencing to make the series stationary...")
    hr_diff = np.diff(hr_regular)
    is_diff_stationary = test_stationarity(hr_diff)
    
    # If still not stationary, try second-order differencing
    if not is_diff_stationary:
        print("\nApplying second-order differencing...")
        hr_diff2 = np.diff(hr_diff)
        is_diff2_stationary = test_stationarity(hr_diff2)

# ## 7. Determining ARIMA Parameters
# 
# Let's analyze ACF and PACF plots to determine appropriate ARIMA parameters.

# Use the appropriate time series (original or differenced)
if is_stationary:
    ts_for_arima = hr_regular
    d_order = 0
elif 'is_diff_stationary' in locals() and is_diff_stationary:
    ts_for_arima = hr_diff
    d_order = 1
elif 'is_diff2_stationary' in locals() and is_diff2_stationary:
    ts_for_arima = hr_diff2
    d_order = 2
else:
    # If still not stationary, use first difference anyway
    ts_for_arima = hr_diff
    d_order = 1
    print("\nWarning: Series may not be fully stationary even after differencing.")

# Apply differencing for ACF/PACF analysis even if the series is stationary
# This can help reveal patterns more clearly
ts_diff = np.diff(ts_for_arima)

# Note on interpreting ACF/PACF plots for ARIMA modeling:
#
# ACF (Autocorrelation Function):
# - Shows correlation between a time series and its lagged values
# - Significant spikes at certain lags indicate correlation with those past values
# - Gradual decay suggests AR process
# - Sharp cutoff after q lags suggests MA(q) process
# - For differenced data, negative spike at lag 1 often indicates over-differencing
#
# PACF (Partial Autocorrelation Function):
# - Shows correlation between a time series and its lagged values after removing
#   the effects of intermediate lags
# - Sharp cutoff after p lags suggests AR(p) process
# - Gradual decay suggests MA process
#
# Common patterns:
# - AR(p): ACF gradually decays, PACF cuts off after lag p
# - MA(q): ACF cuts off after lag q, PACF gradually decays
# - ARMA(p,q): Both ACF and PACF decay gradually
#
# In this case, differencing helps reveal the underlying autocorrelation structure
# that might be obscured in the original data due to high persistence.

# Plot ACF and PACF of differenced data
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plot_acf(ts_diff, lags=40, ax=plt.gca())
plt.title('Autocorrelation Function (ACF) of Differenced Data')
plt.grid(True)

plt.subplot(2, 1, 2)
plot_pacf(ts_diff, lags=40, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF) of Differenced Data')
plt.grid(True)

plt.tight_layout()
plt.show()

# Based on ACF and PACF, suggest ARIMA parameters
print("\nBased on ACF and PACF plots:")
print(f"Suggested differencing (d): {d_order}")
print("Suggested AR order (p): Look for significant spikes in PACF")
print("Suggested MA order (q): Look for significant spikes in ACF")

# ## 8. Fitting ARIMA Models
# 
# Let's fit ARIMA models with different parameters and compare their performance.

# Define a function to fit ARIMA model and make predictions
def fit_arima_and_predict(timeseries, order, train_size=0.8):
    """Fit ARIMA model and make predictions."""
    # Split into training and testing sets
    n = len(timeseries)
    train_size = int(n * train_size)
    train, test = timeseries[:train_size], timeseries[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test))
    
    # Calculate error metrics
    mae = np.mean(np.abs(test - predictions))
    rmse = np.sqrt(np.mean((test - predictions)**2))
    
    return {
        'train': train,
        'test': test,
        'predictions': predictions,
        'model': model_fit,
        'mae': mae,
        'rmse': rmse
    }

# Try different ARIMA models
arima_orders = [
    (1, d_order, 1),
    (2, d_order, 2),
    (1, d_order, 0),
    (0, d_order, 1)
]

arima_results = {}
for order in arima_orders:
    print(f"\nFitting ARIMA{order} model...")
    try:
        result = fit_arima_and_predict(hr_regular, order)
        arima_results[order] = result
        print(f"MAE: {result['mae']:.4f}, RMSE: {result['rmse']:.4f}")
    except Exception as e:
        print(f"Error fitting ARIMA{order}: {e}")

# Find the best model based on MAE
if arima_results:
    best_order = min(arima_results.keys(), key=lambda x: arima_results[x]['mae'])
    best_result = arima_results[best_order]
    print(f"\nBest ARIMA model: {best_order}")
    print(f"MAE: {best_result['mae']:.4f}, RMSE: {best_result['rmse']:.4f}")
else:
    print("\nNo successful ARIMA models were fitted.")

# ## 9. Comparing Different Forecasting Methods
# 
# Let's compare ARIMA with simpler forecasting methods.

# Prepare data for comparison
train_size = int(0.8 * len(hr_regular))
train = hr_regular[:train_size]
test = hr_regular[train_size:]

# 1. Naive forecast (last value)
naive_forecast = np.repeat(train[-1], len(test))

# 2. Mean forecast
mean_forecast = np.repeat(np.mean(train), len(test))

# 3. Simple exponential smoothing
try:
    ses_model = SimpleExpSmoothing(train).fit()
    ses_forecast = ses_model.forecast(len(test))
except Exception as e:
    print(f"Error with Simple Exponential Smoothing: {e}")
    ses_forecast = np.repeat(np.nan, len(test))

# 4. Holt-Winters exponential smoothing
try:
    # Use a more conservative approach for Holt-Winters
    hw_model = ExponentialSmoothing(
        train,
        trend='add',
        seasonal=None,
        initialization_method="estimated"
    ).fit(smoothing_level=0.2, smoothing_trend=0.1)
    hw_forecast = hw_model.forecast(len(test))
    
    # Check if the forecast is reasonable (within 3 std of the mean)
    mean_train = np.mean(train)
    std_train = np.std(train)
    hw_forecast = np.clip(
        hw_forecast,
        mean_train - 3*std_train,
        mean_train + 3*std_train
    )
except Exception as e:
    print(f"Error with Holt-Winters: {e}")
    hw_forecast = np.repeat(np.nan, len(test))

# 5. Best ARIMA model
if 'best_result' in locals():
    arima_forecast = best_result['predictions']
else:
    arima_forecast = np.repeat(np.nan, len(test))

# Calculate error metrics for each method
methods = {
    'Naive': naive_forecast,
    'Mean': mean_forecast,
    'Simple Exponential Smoothing': ses_forecast,
    'Holt-Winters': hw_forecast,
    'ARIMA': arima_forecast
}

comparison = {}
for name, forecast in methods.items():
    if not np.isnan(forecast).any():
        mae = np.mean(np.abs(test - forecast))
        rmse = np.sqrt(np.mean((test - forecast)**2))
        comparison[name] = {'MAE': mae, 'RMSE': rmse}

# Display comparison results
comparison_df = pd.DataFrame(comparison).T
print("\nForecasting Method Comparison:")
print(comparison_df)

# Plot forecasts
plt.figure(figsize=(12, 8))
plt.plot(range(len(train)), train, 'b-', label='Training Data')
plt.plot(range(len(train), len(train) + len(test)), test, 'g-', label='Test Data')

for name, forecast in methods.items():
    if not np.isnan(forecast).any():
        plt.plot(range(len(train), len(train) + len(test)), forecast, '--', label=f'{name} Forecast')

plt.title('Comparison of Forecasting Methods')
plt.xlabel('Time Point')
plt.ylabel('Heart Rate (bpm)')
plt.legend()
plt.grid(True)
plt.show()

# ## 10. Forecasting Future Values
# 
# Let's use the best model to forecast future heart rate values.

# Use the full dataset to fit the final model
if 'best_order' in locals():
    print(f"\nFitting final ARIMA{best_order} model on full dataset...")
    final_model = ARIMA(hr_regular, order=best_order).fit()
    
    # Forecast future values
    forecast_steps = 1000  # Number of steps to forecast (increased for better visualization)
    forecast = final_model.forecast(steps=forecast_steps)
    
    # Plot the forecast
    plt.figure(figsize=(12, 6))
    # Show only the last 5000 points of historical data for better visualization
    history_points = 5000
    start_idx = max(0, len(hr_regular) - history_points)
    
    plt.plot(range(start_idx, len(hr_regular)), hr_regular[start_idx:], 'b-', label='Historical Data')
    plt.plot(range(len(hr_regular), len(hr_regular) + forecast_steps), forecast, 'r-', label='Forecast')
    plt.fill_between(
        range(len(hr_regular), len(hr_regular) + forecast_steps),
        forecast - 1.96 * np.std(hr_regular),
        forecast + 1.96 * np.std(hr_regular),
        color='r', alpha=0.2, label='95% Confidence Interval'
    )
    
    # Add a vertical line to separate historical data from forecast
    plt.axvline(x=len(hr_regular), color='k', linestyle='--', alpha=0.5)
    plt.title(f'ARIMA{best_order} Forecast - Patient {selected_patient}')
    plt.xlabel('Time Point')
    plt.ylabel('Heart Rate (bpm)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print forecast summary
    print("\nForecast Summary:")
    print(f"Mean forecast: {np.mean(forecast):.2f} bpm")
    print(f"Min forecast: {np.min(forecast):.2f} bpm")
    print(f"Max forecast: {np.max(forecast):.2f} bpm")
    print(f"Forecast standard deviation: {np.std(forecast):.2f} bpm")
else:
    print("\nNo best ARIMA model available for final forecasting.")

# ## 11. Summary and Conclusions
# 
# In this demo, we've explored time series forecasting with ARIMA models using heart rate variability data:
# 
# 1. **Data Preparation**:
#    - Loaded RR interval data from healthy subjects
#    - Converted RR intervals to heart rate
#    - Resampled irregular time series to regular intervals
# 
# 2. **Stationarity Analysis**:
#    - Tested for stationarity using the Augmented Dickey-Fuller test
#    - Applied differencing when necessary
# 
# 3. **ARIMA Modeling**:
#    - Determined appropriate parameters using ACF and PACF plots
#    - Fitted multiple ARIMA models with different parameters
#    - Evaluated model performance using MAE and RMSE
# 
# 4. **Forecasting Comparison**:
#    - Compared ARIMA with simpler forecasting methods
#    - Demonstrated the advantages of more sophisticated models
# 
# 5. **Future Forecasting**:
#    - Used the best model to forecast future heart rate values
#    - Visualized forecasts with confidence intervals
# 
# These techniques can be applied to various healthcare time series to predict future values and identify trends.