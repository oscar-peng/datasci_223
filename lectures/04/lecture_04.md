---
lecture_number: 04
pdf: true
---

# Time Series & Regression: Predicting the Future 📈⌛ [\[pdf\]](lecture_04.pdf)

<!---
This lecture covers regression and time series analysis in the context of health data science. Key points to emphasize:
- Time series data is everywhere in healthcare (vital signs, lab results, disease progression)
- Understanding patterns over time is crucial for patient care and resource planning
- We'll build from simple linear regression to advanced forecasting methods
- Focus on practical applications and common pitfalls
--->

> "The best thing about the future is that it comes one data point at a time."
> — Abraham Lincoln (if he was a data scientist)

![XKCD Extrapolating](media/extrapolating.png)
*Source: [XKCD 605](https://xkcd.com/605/) - A cautionary tale about extrapolation*

<!---
This comic perfectly illustrates the dangers of naive extrapolation. Just because you can fit a line to data doesn't mean you should extend it indefinitely. In healthcare, this might mean:
- Not assuming a patient's improvement will continue linearly
- Being cautious about extending seasonal patterns too far
- Considering biological and physical limits
--->

## Introduction: Why Time Series Matter in Healthcare 🏥

<!---
Time series data in healthcare is like a Netflix series - it's all about the patterns and plot twists:
- Each vital sign tells a story (heart rate is the action sequence, temperature is the dramatic tension)
- Missing data is like missing episodes - you need to figure out what happened
- Seasonality is the show's recurring themes
- Anomalies are the plot twists you need to catch
--->

> **Discussion**: What kinds of time series data have you encountered in healthcare or your studies?

## Types of Time Series in Health Data

Time series in health data science come in three main flavors:

| Type                | Example                        | Key Challenge         | Typical Method         |
|---------------------|-------------------------------|-----------------------|-----------------------|
| **Regular-interval (Panel)**    | Hourly vitals, daily labs for multiple patients | Missing data          | ARIMA, regression, panel models |
| **Irregular-interval**  | Lab results, medication events, symptom diaries | Resampling needed     | Interpolation, imputation |
| **Dense/Continuous**    | ECG, accelerometer, heart rate from wearables | Storage, noise        | Signal processing, feature extraction |

```mermaid
flowchart TD
    A[Regular-Interval (Panel)] -->|e.g. hourly vitals| B[Evenly spaced time points]
    C[Irregular-Interval] -->|e.g. lab results| D[Uneven time points]
    E[Dense Data] -->|e.g. ECG, accelerometer, heart rate| F[High-frequency, continuous]
```

<!---
This table and diagram help students distinguish between the main types of time series they will encounter in health data. Each type has unique challenges and requires different analysis strategies.
--->

### The Three Laws of Time Series 🤖

1. **A time series may not harm a patient, or through inaction, allow a patient to come to harm**
   - Monitoring systems must be reliable
   - Alerts should minimize false alarms
   - Models must be interpretable

2. **A time series must be clean, except where such cleanliness conflicts with the First Law**
   - Data quality is crucial
   - But don't discard "messy" data that might be clinically relevant
   - Document all preprocessing steps

3. **A time series must protect its integrity as long as such protection does not conflict with the First or Second Law**
   - No data leakage from the future
   - Respect temporal ordering
   - Handle missing values appropriately

> **Check**: What's your favorite example of "garbage in, garbage out" with time series data?

## Panel Data (Regular-Interval Repeated Measures)

Panel data consists of repeated measurements for multiple subjects at regular time intervals (e.g., daily heart rate for several patients). This structure is common in clinical trials, cohort studies, and EHRs.

**Visual:**
![Panel Data Example](media/panel_data_example.png)
*(If not present, generate with Altair/Matplotlib: grid with patients as rows, time as columns, colored cells for measurements.)*

**Example: Working with Real Panel Data**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

# Download a sample of the MMASH dataset
url = "https://physionet.org/files/mmash/1.0.0/sample_data.csv"
local_file = "mmash_sample.csv"
urlretrieve(url, local_file)

# Load the data
df = pd.read_csv(local_file, parse_dates=['timestamp'])

# Extract data for 3 subjects
subjects = df['subject_id'].unique()[:3]
df_subset = df[df['subject_id'].isin(subjects)]

# Pivot for visualization
pivot = df_subset.pivot(index='subject_id', columns='timestamp', values='heart_rate')

# Visualize
plt.figure(figsize=(10, 4))
plt.imshow(pivot, aspect='auto', cmap='viridis')
plt.colorbar(label='Heart Rate')
plt.xticks(ticks=range(0, len(pivot.columns), 5), 
           labels=pivot.columns[::5].strftime('%H:%M'), rotation=45)
plt.yticks(ticks=range(len(subjects)), labels=subjects)
plt.title('Panel Data: Heart Rate for 3 Subjects')
plt.tight_layout()
plt.show()
```

<!---
Panel data allows for both within-subject and between-subject analysis. In health, this is useful for tracking patient progress and comparing treatment effects. Common methods include mixed-effects models and repeated measures ANOVA.
--->

## Irregular-Interval Data

Irregular-interval data occurs when measurements are taken at uneven time points, such as lab results, medication events, or symptom diaries. This is common in real-world health data, where not all events are scheduled.

**Visual:**
![Irregular Data Example](media/irregular_data_example.png)
*(If not present, generate with Altair/Matplotlib: scatter plot with time on x-axis, dots at irregular intervals.)*

**Example: Handling Irregular Lab Test Data**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create sample irregular lab test data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=20)
# Make the sampling irregular by randomly selecting dates
irregular_dates = np.sort(np.random.choice(dates, size=10, replace=False))
glucose = 100 + 15 * np.random.randn(10)  # Blood glucose values

# Create DataFrame
df = pd.DataFrame({
    'date': irregular_dates,
    'glucose': glucose
})

# Plot original irregular data
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.scatter(df['date'], df['glucose'], color='blue', label='Original Measurements')
plt.title('Irregular Blood Glucose Measurements')
plt.ylabel('Glucose (mg/dL)')
plt.legend()

# Resample to regular daily intervals using interpolation
df_regular = df.set_index('date').resample('D').mean().interpolate(method='time')

# Plot resampled data
plt.subplot(2, 1, 2)
plt.plot(df_regular.index, df_regular['glucose'], 'r-', label='Daily Interpolated')
plt.scatter(df['date'], df['glucose'], color='blue', label='Original Measurements')
plt.title('Resampled to Regular Daily Intervals')
plt.ylabel('Glucose (mg/dL)')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Original data points: {len(df)}")
print(f"After resampling: {len(df_regular)}")
```

<!---
Irregular-interval data requires careful handling. Common strategies include resampling to regular intervals (with interpolation or imputation) or using models that can handle irregularity directly. Always visualize before and after resampling to check for artifacts.
--->

## Dense Data: When Time Gets Intense 🏃‍♀️

Dense (high-frequency) data is collected at very short intervals—sometimes hundreds or thousands of times per second. Examples include ECG, accelerometer, and heart rate data from wearables. These datasets are rich but require special handling.

**Visual:**
![Dense Data Example](media/dense_data_example.png)
*(If not present, generate with Altair/Matplotlib: continuous line plot with many points.)*

### The Dense Data Pipeline

```mermaid
flowchart LR
    A[Raw Sensor Data] --> B[Preprocessing (Filtering, Cleaning)]
    B --> C[Feature Extraction (Rolling Mean, Peaks, Variability)]
    C --> D[Modeling (Prediction, Classification)]
```

<!---
Dense data requires a pipeline approach: raw data is first cleaned and filtered, then features are extracted (e.g., rolling mean, heart rate variability), and finally models are applied. This is especially important for physiological signals like heart rate or ECG.
--->

**Example: Processing Real Heart Rate Data from Meditation**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

# Download heart rate data from meditation dataset
url = "https://physionet.org/files/meditation/1.0.0/data/s001.txt"
local_file = "meditation_s001.txt"
urlretrieve(url, local_file)

# Load the data (adjust column names based on actual format)
# Note: This is a simplified example - check actual data format
df = pd.read_csv(local_file, sep='\t', header=None, 
                names=['time', 'heart_rate'])

# Convert time to datetime if needed
df['time'] = pd.to_datetime(df['time'], unit='s')

# Preprocessing: rolling mean (30-second window)
window_size = 30  # Adjust based on sampling rate
df['hr_rolling'] = df['heart_rate'].rolling(window=window_size).mean()

# Feature extraction: heart rate variability (std over 1-min window)
hrv_window = 60  # Adjust based on sampling rate
df['hrv'] = df['heart_rate'].rolling(window=hrv_window).std()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['heart_rate'], alpha=0.3, label='Raw HR')
plt.plot(df['time'], df['hr_rolling'], color='red', label='Rolling Mean')
plt.xlabel('Time')
plt.ylabel('Heart Rate')
plt.title('Heart Rate During Meditation: Raw and Smoothed')
plt.legend()
plt.tight_layout()
plt.show()

# Display basic statistics
print(f"Data points: {len(df)}")
print(f"Time range: {df['time'].min()} to {df['time'].max()}")
print(f"Heart rate range: {df['heart_rate'].min():.1f} to {df['heart_rate'].max():.1f}")
print(f"Mean heart rate: {df['heart_rate'].mean():.1f}")
```

<!---
This example shows how to work with real meditation heart rate data from PhysioNet. We download the data, preprocess it with a rolling mean to smooth out noise, and extract heart rate variability as a feature. This approach is common in health monitoring applications.
--->

## Table of Contents

1. **Time Series Fundamentals**
    - Key concepts and components
    - Challenges in time series analysis
    - Applications in health data science
    - **Demo Break**: Exploring heart rate patterns during meditation

2. **Regression for Time Series**
    - Linear regression refresher
    - Feature engineering for temporal data
    - Model selection and evaluation
    - **Demo Break**: Building predictive models with sleep monitoring data

3. **Practical Applications**
    - Dense physiological data handling
    - Case studies in healthcare
    - Implementation considerations
    - **Demo Break**: Advanced analysis of sleep and activity data

> ### Time is an illusion. Lunchtime doubly so
>
> — Douglas Adams

## 1. Time Series Fundamentals 🕰️

### 1.1 What Makes Time Series Special?

Time series data is characterized by sequential measurements over intervals. Understanding its components is crucial for effective analysis:

- **Trend**: The underlying pattern in the data over time
- **Seasonality**: Regular variations tied to time intervals
- **Noise**: Random fluctuations that obscure patterns

**Example: Decomposing Heart Rate Data**

```python
# Example: Decomposing a real heart rate time series
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from urllib.request import urlretrieve

# Download sample data from MMASH dataset
url = "https://physionet.org/files/mmash/1.0.0/sample_hr.csv"
local_file = "sample_hr.csv"
urlretrieve(url, local_file)

# Load heart rate data
hr_data = pd.read_csv(local_file, parse_dates=['timestamp'])
hr_data = hr_data.set_index('timestamp')

# Ensure regular intervals (resample if needed)
hr_hourly = hr_data['heart_rate'].resample('H').mean().interpolate()

# Decompose the series
decomposition = seasonal_decompose(hr_hourly, period=24)  # 24 hours for daily pattern

# Plot components
decomposition.plot()
plt.tight_layout()
plt.show()

# Explain components
print("Trend: The long-term progression of heart rate")
print("Seasonal: The daily pattern of heart rate (higher during day, lower at night)")
print("Residual: Unexplained variations after accounting for trend and seasonality")
```

### 1.2 Challenges in Time Series Analysis

Working with time series data presents unique challenges:

1. **Missing Values**: Gaps in temporal data need careful handling
2. **Irregular Sampling**: Measurements may not be evenly spaced
3. **Seasonality**: Multiple seasonal patterns may exist
4. **Noise**: Signal-to-noise ratio can vary significantly

#### Common Pitfalls

- **Data Leakage**: Using future data to predict the past
- **Inappropriate Validation**: Not respecting temporal order in validation
- **Overlooking Context**: Ignoring domain knowledge about cycles

### 1.3 Applications in Health Data Science

Time series analysis is fundamental in healthcare:

- **Patient Monitoring**: Tracking vital signs and detecting anomalies
- **Disease Progression**: Modeling how conditions evolve over time
- **Resource Planning**: Predicting hospital admissions and resource needs
- **Treatment Response**: Analyzing how patients respond to interventions

**Example: Feature Engineering for Vital Signs**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create sample vital signs data
dates = pd.date_range('2024-01-01', periods=48, freq='H')
np.random.seed(42)
vitals = pd.DataFrame({
    'timestamp': dates,
    'heart_rate': 70 + 10*np.sin(2*np.pi*dates.hour/24) + np.random.normal(0, 3, len(dates)),
    'temperature': 37 + 0.3*np.sin(2*np.pi*dates.hour/24) + np.random.normal(0, 0.1, len(dates)),
    'blood_pressure': 120 + 5*np.sin(2*np.pi*dates.hour/24) + np.random.normal(0, 2, len(dates))
})

# Create features from time components
vitals['hour'] = vitals['timestamp'].dt.hour
vitals['day_of_week'] = vitals['timestamp'].dt.dayofweek
vitals['is_weekend'] = vitals['day_of_week'].isin([5, 6]).astype(int)

# Create lagged features
vitals['heart_rate_lag1'] = vitals['heart_rate'].shift(1)
vitals['heart_rate_lag2'] = vitals['heart_rate'].shift(2)

# Create rolling features
vitals['hr_rolling_mean_3h'] = vitals['heart_rate'].rolling(window=3).mean()
vitals['hr_rolling_std_3h'] = vitals['heart_rate'].rolling(window=3).std()

# Scale numerical features
scaler = StandardScaler()
vitals[['heart_rate', 'temperature', 'blood_pressure']] = scaler.fit_transform(
    vitals[['heart_rate', 'temperature', 'blood_pressure']]
)

# Display the engineered features
print(vitals[['timestamp', 'heart_rate', 'hour', 'is_weekend', 
             'heart_rate_lag1', 'hr_rolling_mean_3h']].head(10))
```

## DEMO BREAK: Exploring Heart Rate Patterns During Meditation

In this demo, we'll explore real heart rate data from the Heart Rate Oscillations during Meditation dataset on PhysioNet. You'll learn how to:

- Load and visualize real physiological time series data
- Identify patterns in heart rate during different meditation techniques
- Perform basic statistical tests on time series data
- Handle missing values and irregularities in real-world data

See: [`demo1-heart-rate-meditation`](./demo/demo1-heart-rate-meditation.ipynb)

## 2. Regression for Time Series 📊

### 2.1 Linear Regression Refresher

Linear regression models the relationship between predictors (X) and a target (y) as a linear combination:

y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε

Where:

- β₀ is the intercept
- βᵢ are coefficients
- Xᵢ are features
- ε is the error term

**Example: Predicting Sleep Quality from Activity Data**

```python
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

# Download sample data from MMASH dataset
url = "https://physionet.org/files/mmash/1.0.0/sample_sleep.csv"
local_file = "sample_sleep.csv"
urlretrieve(url, local_file)

# Load sleep and activity data
sleep_data = pd.read_csv(local_file)

# Prepare data for modeling
X = sleep_data[['daily_activity', 'evening_hr']]  # Features
y = sleep_data['sleep_quality']  # Target

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Print coefficients
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Visualize relationship
plt.figure(figsize=(10, 6))
plt.scatter(sleep_data['daily_activity'], sleep_data['sleep_quality'], 
            alpha=0.7, label='Actual data')

# Create prediction line for visualization
x_range = np.linspace(sleep_data['daily_activity'].min(), 
                      sleep_data['daily_activity'].max(), 100)
# Hold evening_hr constant at its mean for visualization
evening_hr_mean = sleep_data['evening_hr'].mean()
X_pred = np.column_stack([x_range, np.ones(100) * evening_hr_mean])
y_pred = model.predict(X_pred)

plt.plot(x_range, y_pred, 'r-', label='Regression line')
plt.xlabel('Daily Activity Level')
plt.ylabel('Sleep Quality')
plt.title('Relationship Between Activity and Sleep Quality')
plt.legend()
plt.show()
```

### 2.2 Feature Engineering for Temporal Data

Feature engineering is crucial for capturing temporal patterns:

#### 1. Time-Based Features

```python
def create_time_features(df, timestamp_col):
    """Create features from datetime components."""
    df = df.copy()
    
    # Basic time components
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['month'] = df[timestamp_col].dt.month
    
    # Cyclical encoding for periodic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Is weekend/holiday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df
```

#### 2. Lagged Features

```python
def create_lag_features(df, target_col, lags=[1, 2, 3, 24]):
    """Create lagged versions of target variable."""
    df = df.copy()
    
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
        
    return df
```

#### 3. Rolling Window Features

```python
def create_rolling_features(df, target_col, windows=[3, 6, 12, 24]):
    """Create rolling statistics."""
    df = df.copy()
    
    for window in windows:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    
    return df
```

### 2.3 Model Selection and Evaluation

Time series models require special consideration for evaluation:

#### Cross-Validation Strategies

1. **Traditional CV (Wrong!)**

```python
# ❌ Don't do this - mixes future and past
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True)
```

2. **Time Series CV (Correct!)**

```python
# ✅ Do this - respects temporal order
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, test_size=24*7)  # One week test size
```

![Time Series Cross Validation](media/time_series_cv.png)

#### Evaluation Metrics

1. **Scale-Dependent Metrics**
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)

   ```python
   from sklearn.metrics import mean_absolute_error, mean_squared_error
   
   mae = mean_absolute_error(y_true, y_pred)
   rmse = np.sqrt(mean_squared_error(y_true, y_pred))
   ```

2. **Percentage Errors**
   - Mean Absolute Percentage Error (MAPE)
   - Symmetric MAPE (sMAPE)

   ```python
   def mape(y_true, y_pred):
       return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
   
   def smape(y_true, y_pred):
       return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
   ```

3. **Scale-Independent Metrics**
   - R-squared (R²)
   - Adjusted R-squared

   ```python
   from sklearn.metrics import r2_score
   
   r2 = r2_score(y_true, y_pred)
   ```

## DEMO BREAK: Building Predictive Models with Sleep Monitoring Data

In this demo, we'll use the Multilevel Monitoring of Activity and Sleep dataset to build predictive models for sleep quality. You'll learn how to:

- Work with multivariate health time series data
- Engineer features from temporal health data
- Build and evaluate regression models
- Compare different cohorts and conditions

See: [`demo2-sleep-monitoring`](./demo/demo2-sleep-monitoring.ipynb)

## 3. Practical Applications 🚀

### 3.1 Time Series Forecasting Methods

There are several approaches to forecasting time series data in healthcare:

1. **Simple Methods**: Moving averages and exponential smoothing
2. **Statistical Models**: ARIMA (AutoRegressive Integrated Moving Average)
3. **Machine Learning**: Random Forests, XGBoost, and neural networks

For this course, we'll focus on simple methods and briefly introduce ARIMA.

#### ARIMA in a Nutshell

ARIMA models combine three components:

- **AR**: Using past values to predict future ones
- **I**: Making the data stationary through differencing
- **MA**: Using past prediction errors

```python
# Example: Simple ARIMA for body temperature forecasting
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

# Download sample temperature data
url = "https://physionet.org/files/mmash/1.0.0/sample_temp.csv"
local_file = "sample_temp.csv"
urlretrieve(url, local_file)

# Load temperature data
temp_data = pd.read_csv(local_file, parse_dates=['timestamp'])
temp_data = temp_data.set_index('timestamp')

# Fit a simple ARIMA model
model = ARIMA(temp_data['temperature'], order=(1,0,0))  # Simple AR(1) model
results = model.fit()

# Forecast next 6 hours
forecast = results.forecast(steps=6)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(temp_data.index, temp_data['temperature'], label='Historical Data')
forecast_index = pd.date_range(start=temp_data.index[-1], periods=7, freq=temp_data.index.freq)[1:]
plt.plot(forecast_index, forecast, 'r--', label='Forecast')
plt.title('Temperature Forecast with ARIMA')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

print("Next 6 hours temperature forecast:", forecast.values)
```

For more advanced ARIMA modeling, see the references at the end of this lecture.

### 3.2 Case Studies in Healthcare

Time series analysis has numerous applications in healthcare:

1. **Patient Monitoring**
   - Detecting anomalies in vital signs
   - Predicting clinical deterioration
   - Personalizing alarm thresholds

2. **Disease Progression**
   - Modeling chronic disease trajectories
   - Identifying patterns in symptom severity
   - Predicting disease flares

3. **Resource Planning**
   - Forecasting hospital admissions
   - Predicting equipment and medication needs
   - Staff scheduling optimization

### 3.3 Implementation Considerations

When implementing time series analysis in healthcare:

1. **Data Quality**
   - Handle missing values appropriately
   - Account for measurement errors
   - Document preprocessing steps

2. **Computational Efficiency**
   - Use appropriate data structures for time series
   - Consider downsampling for dense data
   - Implement incremental learning for streaming data

3. **Interpretability**
   - Ensure models can be explained to clinicians
   - Visualize predictions and uncertainty
   - Connect statistical results to clinical meaning

## DEMO BREAK: Advanced Analysis of Sleep and Activity Data

In this demo, we'll continue working with the Multilevel Monitoring of Activity and Sleep dataset to perform advanced analysis of dense physiological signals. You'll learn how to:

- Process and analyze dense accelerometer data
- Apply signal processing techniques to remove noise
- Extract meaningful features from raw sensor data
- Connect activity patterns to health outcomes

See: [`demo3-advanced-analysis`](./demo/demo3-advanced-analysis.ipynb)

## 4. Related Approaches: Survival Analysis

> **Q:** When would you use survival analysis instead of time series forecasting?

Survival analysis is a related approach that focuses on time-to-event data, such as:

- Time until disease recurrence
- Duration of hospital stay
- Time until treatment response

Unlike traditional time series analysis, survival analysis handles:

- Censored data (when the event hasn't occurred yet)
- Competing risks (multiple possible outcomes)
- Time-varying covariates

**Example: Kaplan-Meier Survival Curve**

```python
# Example: Kaplan-Meier curve
from lifelines import KaplanMeierFitter
import pandas as pd
import matplotlib.pyplot as plt

# Simulated survival data
np.random.seed(42)
n_patients = 50
data = pd.DataFrame({
    'time': np.random.exponential(scale=10, size=n_patients),  # Time to event
    'event': np.random.binomial(n=1, p=0.7, size=n_patients),  # 1=event occurred, 0=censored
    'group': np.random.choice(['A', 'B'], size=n_patients)     # Treatment group
})

# Fit Kaplan-Meier model
kmf = KaplanMeierFitter()
kmf.fit(data['time'], event_observed=data['event'])

# Plot survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

# Compare treatment groups
kmf_A = KaplanMeierFitter()
kmf_B = KaplanMeierFitter()
mask_A = data['group'] == 'A'
mask_B = data['group'] == 'B'
kmf_A.fit(data.loc[mask_A, 'time'], event_observed=data.loc[mask_A, 'event'], label='Group A')
kmf_B.fit(data.loc[mask_B, 'time'], event_observed=data.loc[mask_B, 'event'], label='Group B')
kmf_A.plot_survival_function()
kmf_B.plot_survival_function()

plt.title('Kaplan-Meier Survival Curves by Treatment Group')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()
```

## Homework Assignment

For this week's assignment, you'll apply the concepts from this lecture to real-world health data from the Wearable Device Dataset from Induced Stress and Exercise Sessions.

### Dataset

The dataset contains physiological signals (Electrodermal Activity, Blood Volume Pulse, Heart Rate, Temperature) from 36 healthy volunteers collected during structured acute stress induction and aerobic/anaerobic exercise sessions using the Empatica E4 wearable device.

Dataset link: <https://physionet.org/content/wearable-stress-affect/1.0.0/>

### Assignment Structure

1. **Part 1: Data Exploration and Preprocessing (30%)**
   - Load and explore the Wearable Stress/Exercise dataset
   - Visualize temporal patterns in physiological signals during stress vs. exercise
   - Handle missing values and irregularities in wearable device data
   - Perform basic statistical analysis comparing different conditions

2. **Part 2: Time Series Modeling (40%)**
   - Extract features from physiological time series data
   - Build regression models to predict stress levels from physiological signals
   - Compare different modeling approaches for stress detection
   - Evaluate model performance using appropriate time series metrics

3. **Part 3: Advanced Analysis (30%)**
   - Apply signal processing techniques to the wearable sensor data
   - Extract meaningful features from dense physiological signals
   - Identify patterns that differentiate stress from exercise responses
   - Create an interactive visualization comparing conditions

## References and Resources 📚

### Time Series Analysis

- [Python for Time Series Analysis](https://www.statsmodels.org/stable/tsa.html) - Statsmodels documentation
- [Time Series Forecasting](https://otexts.com/fpp3/) - Forecasting: Principles and Practice
- [Practical Time Series Analysis](https://www.oreilly.com/library/view/practical-time-series/9781492041641/) - O'Reilly book
- [Healthcare Time Series Analysis](https://www.nature.com/articles/s41746-020-00376-2) - Nature Digital Medicine

### Machine Learning for Time Series

- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - Linear regression
- [Feature Engineering for Time Series](https://www.featuretools.com/) - Automated feature engineering
- [Deep Learning for Time Series](https://www.tensorflow.org/tutorials/structured_data/time_series) - TensorFlow guide
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) - Scikit-learn

### Health Data Applications

- [Clinical Time Series Analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6823538/) - NCBI review
- [Vital Signs Monitoring](https://physionet.org/content/challenge-2019/1.0.0/) - PhysioNet Challenge
- [Disease Progression Modeling](https://www.nature.com/articles/s41598-020-78321-2) - Scientific Reports

### PhysioNet Datasets Used in This Lecture

- [Heart Rate Oscillations during Meditation](https://physionet.org/content/meditation/1.0.0/)
- [Multilevel Monitoring of Activity and Sleep in Healthy People](https://physionet.org/content/mmash/1.0.0/)
- [Wearable Device Dataset from Induced Stress and Exercise Sessions](https://physionet.org/content/wearable-stress-affect/1.0.0/)
