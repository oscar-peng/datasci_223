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

### Dense Data: When Time Gets Intense 🏃‍♀️

<!---
Dense time series are like trying to drink from a fire hose:
- ECG data comes in at 250+ samples per second
- Activity trackers generate thousands of readings per minute
- Audio recordings can be 44,100 samples per second
Key challenges:
- Storage and processing requirements
- Signal-to-noise ratio
- Feature extraction from raw signals
- Real-time analysis needs
--->

#### Example: Processing Dense ECG Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# Generate synthetic ECG-like data (250 Hz sampling rate)
t = np.linspace(0, 10, 2500)  # 10 seconds of data
ecg = np.zeros_like(t)

# Create QRS complexes
for i in range(10):  # 10 heartbeats
    t_beat = t - i
    qrs = 2 * signal.gaussian(len(t), std=25) * np.exp(-((t_beat - 1) ** 2) / 0.01)
    ecg += qrs

# Add noise
ecg += np.random.normal(0, 0.1, len(t))

# Plot raw vs filtered
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
ax1.plot(t, ecg)
ax1.set_title('Raw ECG Signal')

# Apply bandpass filter
fs = 250  # sampling frequency
w1 = 5 / (fs/2)  # 5 Hz high-pass
w2 = 50 / (fs/2)  # 50 Hz low-pass
b, a = signal.butter(4, [w1, w2], btype='band')
ecg_filtered = signal.filtfilt(b, a, ecg)

ax2.plot(t, ecg_filtered)
ax2.set_title('Filtered ECG Signal')
plt.tight_layout()
plt.show()
```

### Types of Healthcare Time Series

1. **Patient Monitoring Data**
   - Vital signs (heart rate, blood pressure, temperature)
   - Continuous glucose monitoring
   - ECG/EEG recordings
   
2. **Clinical Measurements**
   - Lab test results over time
   - Medication responses
   - Disease progression markers

3. **Healthcare Operations**
   - Hospital admissions
   - Resource utilization
   - Staff scheduling needs

### Common Questions in Healthcare Time Series

> **Check**: What kinds of time series data have you encountered in your work?

1. **Prediction Questions**
   - Will this patient develop complications?
   - When should we schedule follow-up tests?
   - How many beds will we need next month?

2. **Pattern Questions**
   - Is this vital sign pattern normal?
   - Has the treatment changed the trend?
   - Are there seasonal effects in disease occurrence?

3. **Relationship Questions**
   - Do medication changes affect symptoms?
   - How do different measurements relate over time?
   - What leads to increased hospital admissions?

### Example: Patient Monitoring System

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load continuous monitoring data
monitoring_data = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=1440, freq='T'),
    'heart_rate': np.random.normal(75, 5, 1440),  # One day of minute-by-minute data
    'blood_pressure_systolic': np.random.normal(120, 10, 1440),
    'temperature': np.random.normal(37, 0.3, 1440)
})

# Add circadian rhythm effect
time_of_day = pd.to_datetime(monitoring_data['timestamp']).dt.hour
monitoring_data['heart_rate'] += 10 * np.sin(2 * np.pi * time_of_day / 24)

# Plot the data
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
fig.suptitle('24-Hour Patient Monitoring Data')

monitoring_data.plot(x='timestamp', y='heart_rate', ax=axes[0], title='Heart Rate')
monitoring_data.plot(x='timestamp', y='blood_pressure_systolic', ax=axes[1], title='Blood Pressure')
monitoring_data.plot(x='timestamp', y='temperature', ax=axes[2], title='Temperature')

plt.tight_layout()
plt.show()
```

<!---
This example demonstrates several key concepts:
1. High-frequency healthcare data collection
2. Natural patterns in physiological measurements
3. Multiple related time series
4. Visualization of temporal patterns
--->

## Table of Contents

1. **Time Series Fundamentals**
    - Key concepts and components
    - Challenges in time series analysis
    - Applications in health data science
    - **Demo Break**: Exploring temporal patterns in health data

2. **Regression for Time Series**
    - Linear regression refresher
    - Feature engineering for temporal data
    - Model selection and evaluation
    - **Demo Break**: Building predictive models with health metrics

3. **Advanced Time Series Methods**
    - ARIMA and seasonal models
    - Machine learning approaches
    - Deep learning for time series
    - **Demo Break**: Comparing different forecasting approaches

> ### Time is an illusion. Lunchtime doubly so.
> — Douglas Adams

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

## 1. Time Series Fundamentals 🕰️

### 1.1 What Makes Time Series Special?

Time series data is characterized by sequential measurements over intervals. Understanding its components is crucial for effective analysis:

- **Trend**: The underlying pattern in the data over time
- **Seasonality**: Regular variations tied to time intervals
- **Noise**: Random fluctuations that obscure patterns

```python
# Example: Decomposing a time series
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

# Load patient temperature readings
temps = pd.read_csv("patient_temps.csv", parse_dates=["timestamp"])
temps = temps.set_index("timestamp")

# Decompose the series
decomposition = seasonal_decompose(temps["temperature"], period=24)  # 24 hours

# Plot components
decomposition.plot()
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

#### Example: Vital Signs Monitoring

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load vital signs data
vitals = pd.read_csv("patient_vitals.csv", parse_dates=["timestamp"])

# Create features from time components
vitals["hour"] = vitals["timestamp"].dt.hour
vitals["day_of_week"] = vitals["timestamp"].dt.dayofweek

# Scale numerical features
scaler = StandardScaler()
vitals[["heart_rate", "blood_pressure", "temperature"]] = scaler.fit_transform(
    vitals[["heart_rate", "blood_pressure", "temperature"]]
)
```

## DEMO BREAK: Exploring Temporal Patterns in Health Data

See: [`demo1-time-patterns`](./demo/demo1-time-patterns.ipynb)

## 2. Regression for Time Series 📊

<!---
Regression is a fundamental building block for time series analysis:
- Start with simple linear relationships before moving to complex models
- Important to understand assumptions and limitations
- Feature engineering often more important than model complexity
- Cross-validation must respect temporal order
--->

### 2.1 Linear Regression Refresher

Linear regression models the relationship between predictors (X) and a target (y) as a linear combination:

y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε

Where:
- β₀ is the intercept
- βᵢ are coefficients
- Xᵢ are features
- ε is the error term

#### Example: Predicting Patient Recovery Time

```python
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Sample patient data
np.random.seed(42)
n_patients = 100

patient_data = pd.DataFrame({
    'age': np.random.normal(60, 10, n_patients),
    'severity_score': np.random.uniform(1, 10, n_patients),
    'previous_conditions': np.random.randint(0, 5, n_patients)
})

# Generate recovery time with some noise
recovery_time = (
    0.5 * patient_data['age'] + 
    2.0 * patient_data['severity_score'] + 
    1.5 * patient_data['previous_conditions'] +
    np.random.normal(0, 5, n_patients)
)

# Fit linear regression
model = LinearRegression()
model.fit(patient_data, recovery_time)

print("Coefficients:")
for feature, coef in zip(patient_data.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
```

<!---
Key points about linear regression:
1. Interpretable coefficients show feature importance
2. Assumes linear relationship between features and target
3. Sensitive to outliers and scale of features
4. Good baseline model for comparison
--->

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

<!---
Feature engineering tips:
1. Consider domain knowledge when creating features
2. Watch out for data leakage in rolling/lagged features
3. Handle missing values created by lags/windows
4. Use cross-validation to validate feature importance
--->

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

<!---
Key points about evaluation:
1. Choose metrics based on business context
2. Consider multiple metrics for robust evaluation
3. Always validate on unseen future data
4. Be careful with percentage errors when values near zero
--->

## DEMO BREAK: Building Predictive Models with Health Metrics

See: [`demo2-predictive-models`](./demo/demo2-predictive-models.ipynb)

## 3. Advanced Time Series Methods 🚀

<!---
Advanced methods build on regression fundamentals:
- ARIMA models capture autoregressive and moving average patterns
- Machine learning models can handle non-linear relationships
- Deep learning excels with large, complex datasets
- Each approach has its strengths and ideal use cases
--->

### 3.1 ARIMA and Seasonal Models

ARIMA (AutoRegressive Integrated Moving Average) combines three components:

1. **AR (p)**: Autoregression - using past values
2. **I (d)**: Integration - differencing to make stationary
3. **MA (q)**: Moving Average - using past errors

#### Example: Modeling Patient Temperature

```python
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

# Generate hourly temperature data with daily pattern
np.random.seed(42)
hours = pd.date_range(start='2024-01-01', periods=24*30, freq='H')
temp_base = 37.0
daily_pattern = 0.3 * np.sin(2 * np.pi * hours.hour / 24)
noise = np.random.normal(0, 0.1, len(hours))
temperature = temp_base + daily_pattern + noise

# Fit ARIMA model
model = ARIMA(temperature, order=(24,1,1))  # p=24 for daily seasonality
results = model.fit()

# Make predictions
forecast = results.forecast(steps=24)  # Predict next 24 hours

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(hours, temperature, label='Actual')
plt.plot(pd.date_range(start=hours[-1], periods=25, freq='H')[1:],
         forecast, label='Forecast', linestyle='--')
plt.title('Patient Temperature: Actual vs Forecast')
plt.legend()
plt.show()

# Print model summary
print(results.summary())
```

<!---
ARIMA key points:
1. Good for regular patterns with clear seasonality
2. Requires stationary data (constant mean/variance)
3. Parameter selection (p,d,q) crucial for performance
4. Works best with evenly spaced observations
--->

#### Seasonal Decomposition

Understanding components of a time series:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the temperature series
decomposition = seasonal_decompose(
    temperature,
    period=24,  # 24 hours for daily pattern
    extrapolate_trend='freq'
)

# Plot components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=ax1)
ax1.set_title('Original Data')
decomposition.trend.plot(ax=ax2)
ax2.set_title('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
decomposition.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

### 3.2 Machine Learning Approaches

Modern ML methods offer flexible modeling approaches:

#### Random Forest for Time Series

```python
from sklearn.ensemble import RandomForestRegressor

def create_features_target(data, lookback=5):
    """Create features and target for ML model."""
    features, target = [], []
    for i in range(len(data) - lookback):
        features.append(data[i:i+lookback])
        target.append(data[i+lookback])
    return np.array(features), np.array(target)

# Prepare data
X, y = create_features_target(temperature, lookback=24)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Feature importance
importance = pd.DataFrame({
    'feature': [f'lag_{i+1}' for i in range(24)],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 5 most important lags:")
print(importance.head())
```

#### XGBoost with Time Features

```python
import xgboost as xgb

def create_advanced_features(data, timestamp_index):
    """Create advanced features for XGBoost."""
    features = pd.DataFrame(index=timestamp_index)
    
    # Time features
    features['hour'] = timestamp_index.hour
    features['day_of_week'] = timestamp_index.dayofweek
    features['month'] = timestamp_index.month
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        features[f'lag_{lag}'] = data.shift(lag)
    
    # Rolling features
    for window in [3, 6, 12, 24]:
        features[f'rolling_mean_{window}'] = data.rolling(window).mean()
        features[f'rolling_std_{window}'] = data.rolling(window).std()
    
    return features.dropna()

# Prepare features
features_df = create_advanced_features(pd.Series(temperature), hours)
X = features_df.values
y = temperature[len(temperature)-len(features_df):]

# Train XGBoost
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1
)
xgb_model.fit(X_train, y_train)
```

### 3.3 Deep Learning for Time Series

Neural networks excel at complex temporal patterns:

#### LSTM for Vital Signs

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def prepare_sequences(data, seq_length):
    """Prepare sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Scale data
scaler = MinMaxScaler()
scaled_temp = scaler.fit_transform(temperature.reshape(-1, 1))

# Create sequences
seq_length = 24
X, y = prepare_sequences(scaled_temp, seq_length)

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(30, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

<!---
Deep learning considerations:
1. Requires more data than traditional methods
2. GPU acceleration often beneficial
3. Hyperparameter tuning crucial
4. Risk of overfitting with small datasets
--->

### 3.4 Choosing the Right Approach

![XKCD Machine Learning](media/machine_learning.png)
*Source: [XKCD 1838](https://xkcd.com/1838/) - Machine Learning: Like regular learning, but with more arguing about algorithms*

<!---
This comic reminds us that:
- More complex isn't always better
- Different methods have different tradeoffs
- The goal is solving the problem, not using the fanciest algorithm
- Sometimes simple models work best
--->

Selection criteria for time series methods:

1. **Data Characteristics**
   - Amount of data available
   - Sampling frequency
   - Missing values
   - Multiple variables

2. **Problem Requirements**
   - Forecast horizon
   - Update frequency
   - Interpretability needs
   - Computational constraints

3. **Method Comparison**

| Method | Strengths | Weaknesses | Best For |
|--------|-----------|------------|-----------|
| ARIMA | Interpretable, handles seasonality | Requires stationarity, single variable | Regular patterns, clear seasonality |
| Random Forest | Handles non-linearity, feature importance | Memory intensive, black box | Multiple variables, complex patterns |
| XGBoost | High performance, handles missing values | Complex tuning, black box | Large datasets, competitions |
| LSTM | Learns long-term dependencies | Requires lots of data, slow training | Sequential data, complex patterns |

## DEMO BREAK: Comparing Different Forecasting Approaches

See: [`demo3-forecasting-comparison`](./demo/demo3-forecasting-comparison.ipynb)

## Exercise: Predicting Hospital Admissions 🏥

### Objective
Build a model to predict daily hospital admissions using historical data and relevant features.

### Data
- Daily admission counts
- Weather data
- Holiday information
- Special events

### Tasks
1. Load and prepare the data
2. Create temporal features
3. Train multiple models
4. Compare and evaluate results
5. Make future predictions

### Bonus
- Add confidence intervals to predictions
- Incorporate external factors
- Optimize model parameters

## It came from the Internet

> [!info] Forecasting at Scale  
> How we produce reliable forecasts for hundreds of millions of time series at Facebook.  
> [https://research.facebook.com/blog/2017/2/prophet-forecasting-at-scale/](https://research.facebook.com/blog/2017/2/prophet-forecasting-at-scale/)  

> [!info] Time Series Analysis in Python  
> A comprehensive guide to Time Series analysis in Python  
> [https://www.machinelearningplus.com/time-series/time-series-analysis-python/](https://www.machinelearningplus.com/time-series/time-series-analysis-python/)  

> [!info] Neural Prophet  
> Neural Network based Time-Series model, inspired by Facebook Prophet and AR-Net  
> [https://neuralprophet.com/](https://neuralprophet.com/) 

### Dense Data Deep Dive 🏊‍♂️

<!---
Dense time series are the Olympic swimmers of data - they generate a lot of waves:
- Sampling rates can be overwhelming (like trying to count individual raindrops)
- Signal processing is crucial (finding the melody in the noise)
- Storage and computation need careful planning (or your computer will have a bad time)
--->

#### 1. Motion Analysis

```python
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Decorator for timing function execution
def timer(func):
    """Time the execution of a function"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@timer
def process_motion_data(data, window_size=50):
    """Process accelerometer data with vectorized operations"""
    # Calculate magnitude using numpy operations
    magnitude = np.sqrt(np.sum(data**2, axis=1))
    
    # Efficient rolling calculations
    rolling_mean = pd.Series(magnitude).rolling(window_size).mean()
    rolling_std = pd.Series(magnitude).rolling(window_size).std()
    
    return magnitude, rolling_mean, rolling_std

# Generate synthetic accelerometer data (100 Hz for 60 seconds)
t = np.linspace(0, 60, 6000)
acc_data = np.column_stack([
    2 * np.sin(2 * np.pi * 0.5 * t),  # x-axis
    3 * np.cos(2 * np.pi * 0.3 * t),  # y-axis
    np.sin(2 * np.pi * 0.7 * t)       # z-axis
]) + np.random.normal(0, 0.1, (len(t), 3))

# Process data
magnitude, rolling_mean, rolling_std = process_motion_data(acc_data)

# Plot results using subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(t, acc_data)
axes[0].set_title('Raw Accelerometer Data')
axes[0].legend(['X', 'Y', 'Z'])

axes[1].plot(t, magnitude, 'k', alpha=0.3, label='Magnitude')
axes[1].plot(t, rolling_mean, 'r', label='Rolling Mean')
axes[1].fill_between(t, 
                    rolling_mean - 2*rolling_std,
                    rolling_mean + 2*rolling_std,
                    alpha=0.2, color='r')
axes[1].set_title('Processed Motion Data')
axes[1].legend()

plt.tight_layout()
plt.show()
```

#### 2. Audio Analysis

```python
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# List comprehension for efficient spectrogram calculation
def calculate_spectrograms(audio_data, fs, window_sizes=[256, 512, 1024]):
    return [signal.spectrogram(audio_data, fs, nperseg=size) 
            for size in window_sizes]

# Context manager for handling audio files
class AudioFile:
    def __init__(self, filename):
        self.filename = filename
        
    def __enter__(self):
        self.fs, self.data = wavfile.read(self.filename)
        return self.fs, self.data
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up if needed
        pass

# Example usage
with AudioFile('breath_sound.wav') as (fs, audio_data):
    # Calculate spectrograms at different resolutions
    specs = calculate_spectrograms(audio_data, fs)
    
    # Plot spectrograms
    fig, axes = plt.subplots(len(specs), 1, figsize=(12, 12))
    for (f, t, Sxx), ax, size in zip(specs, axes, [256, 512, 1024]):
        ax.pcolormesh(t, f, 10 * np.log10(Sxx))
        ax.set_title(f'Spectrogram (window size: {size})')
        ax.set_ylabel('Frequency [Hz]')
    
    axes[-1].set_xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()
```

#### 3. Continuous Glucose Monitoring

```python
# Type hints for better code documentation
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class GlucoseReading:
    """Data class for glucose readings"""
    timestamp: pd.Timestamp
    value: float
    is_valid: bool = True
    
    def __post_init__(self):
        """Validate reading after initialization"""
        if not (40 <= self.value <= 400):
            self.is_valid = False

class GlucoseMonitor:
    """Class for processing continuous glucose monitoring data"""
    def __init__(self, readings: List[GlucoseReading]):
        self.readings = readings
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert readings to DataFrame with error handling"""
        try:
            df = pd.DataFrame([
                {'timestamp': r.timestamp, 'value': r.value, 'is_valid': r.is_valid}
                for r in self.readings
            ])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return pd.DataFrame()
    
    def analyze_trends(self, window_size: str = '30min') -> Tuple[pd.Series, pd.Series]:
        """Calculate trends with proper error handling"""
        if self.df.empty:
            raise ValueError("No valid data available")
            
        try:
            valid_data = self.df[self.df['is_valid']]['value']
            trend = valid_data.rolling(window_size).mean()
            variability = valid_data.rolling(window_size).std()
            return trend, variability
        except Exception as e:
            print(f"Error calculating trends: {e}")
            return pd.Series(), pd.Series()

# Example usage
timestamps = pd.date_range(start='2024-01-01', periods=288, freq='5min')
values = 120 + 20 * np.sin(2 * np.pi * np.arange(288) / 288) + np.random.normal(0, 5, 288)
readings = [GlucoseReading(ts, val) for ts, val in zip(timestamps, values)]

monitor = GlucoseMonitor(readings)
trend, variability = monitor.analyze_trends()

# Plotting with error bars
plt.figure(figsize=(12, 6))
plt.plot(trend.index, trend, 'b-', label='Trend')
plt.fill_between(trend.index, 
                trend - 2*variability,
                trend + 2*variability,
                alpha=0.2, color='b')
plt.title('Continuous Glucose Monitoring')
plt.ylabel('Glucose Level (mg/dL)')
plt.legend()
plt.grid(True)
plt.show()
```

<!---
Key points about dense data:
1. Use appropriate data structures (NumPy for numerical, Pandas for time series)
2. Vectorize operations when possible
3. Consider memory usage with large datasets
4. Use proper error handling and validation
5. Document code with type hints and docstrings
6. Use object-oriented programming for complex analyses
---> 