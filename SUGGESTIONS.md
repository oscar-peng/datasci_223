# Suggestions for Lecture 4: Time Series & Regression

## General Lecture Content Adjustments

1. **Streamline Advanced Topics**:
   - Reduce depth on ARIMA, LSTM, and other advanced methods to avoid overwhelming beginners
   - Convert detailed explanations to brief mentions with references for further study
   - Focus more time on fundamental concepts and basic applications

2. **Enhance Beginner-Friendly Content**:
   - Add more visual explanations of key concepts (trend, seasonality, noise)
   - Include more step-by-step breakdowns of code examples
   - Add common pitfalls and misconceptions in speaking notes

3. **Improve Health Data Relevance**:
   - Connect each concept directly to health applications
   - Include more examples from clinical monitoring, epidemiology, and healthcare operations
   - Emphasize practical use cases students might encounter in their careers

4. **Adjust Time Allocation**:
   - Allocate ~30 minutes to time series fundamentals
   - ~30 minutes to regression and basic forecasting
   - ~30 minutes to practical applications and case studies

5. **Integrate Interactive Elements**:
   - Add discussion prompts at the start of each section
   - Include comprehension checkpoints throughout
   - Incorporate mini-exercises between major concepts

---

## Demo Improvements

### Demo 1: Exploring Temporal Patterns with Heart Rate Meditation Data
- **Dataset**: Heart Rate Oscillations during Meditation (https://physionet.org/content/meditation/1.0.0/)
- **Improvements**:
  - Replace synthetic data with real meditation heart rate data
  - Focus on visualization and basic pattern identification
  - Include examples of handling irregular time intervals
  - Add statistical tests relevant to time series (stationarity, autocorrelation)
  - Demonstrate proper handling of missing values

### Demo 2: Predictive Modeling with Sleep Monitoring Data
- **Dataset**: Multilevel Monitoring of Activity and Sleep (https://physionet.org/content/mmash/1.0.0/)
- **Improvements**:
  - Use real sleep monitoring data instead of synthetic examples
  - Focus on regression applications in health research
  - Demonstrate comparing relationships between different cohorts
  - Include simple feature engineering for time-based data
  - Simplify Random Forest explanation while retaining the concept

### Demo 3: Advanced Analysis of Sleep and Activity Data
- **Dataset**: Multilevel Monitoring of Activity and Sleep (continued)
- **Improvements**:
  - Focus on handling dense physiological data
  - Demonstrate preprocessing techniques for noisy signals
  - Include visualization of raw vs. processed data
  - Show practical applications for health monitoring
  - Connect to real-world clinical decision support

---

## Homework Assignment Structure

### Assignment Overview
- **Dataset**: Wearable Device Dataset from Induced Stress and Exercise Sessions (https://physionet.org/content/wearable-stress-affect/1.0.0/)
- **Structure**: Three progressive exercises building on lecture concepts
- **Learning Objectives**: Apply time series techniques to real health monitoring data in a new context
- **Implementation**: GitHub Classroom with automated testing via pytest

### Part 1: Data Exploration and Preprocessing (30%)
- Load and explore the Wearable Stress/Exercise dataset
- Visualize temporal patterns in physiological signals during stress vs. exercise
- Handle missing values and irregularities in wearable device data
- Perform basic statistical analysis comparing different conditions
- Create summary visualizations of physiological responses

### Part 2: Time Series Modeling (40%)
- Extract features from physiological time series data
- Build regression models to predict stress levels from physiological signals
- Compare different modeling approaches for stress detection
- Evaluate model performance using appropriate time series metrics
- Interpret results in a health monitoring context

### Part 3: Advanced Analysis (30%)
- Apply signal processing techniques to the wearable sensor data
- Extract meaningful features from dense physiological signals
- Identify patterns that differentiate stress from exercise responses
- Create an interactive visualization comparing conditions
- Discuss potential applications for stress monitoring and management

### Automated Testing Framework

The homework will include a pytest-based testing framework to enable automated grading through GitHub Classroom. The tests will verify:

1. **Data Loading and Preprocessing**:
   - Test that data is loaded correctly with expected dimensions
   - Verify handling of missing values (no NaN values in processed data)
   - Check that time series data is properly indexed
   - Validate statistical summaries against expected values

2. **Feature Engineering**:
   - Test that required features are created (e.g., time-based, rolling statistics)
   - Verify feature dimensions and data types
   - Check for expected value ranges in engineered features

3. **Model Implementation**:
   - Test model creation with expected parameters
   - Verify train/test split respects temporal ordering
   - Check prediction output format and dimensions
   - Validate evaluation metrics calculation

4. **Visualization and Analysis**:
   - Test that required plots are generated
   - Verify figure dimensions and components
   - Check for expected patterns in visualization output

5. **Code Quality**:
   - Verify function documentation
   - Check for proper error handling
   - Test for code efficiency (e.g., vectorized operations)

### Test Design Considerations

To make the tests robust against student variations:

1. **Flexible Output Checking**:
   - Use approximate equality for numerical comparisons
   - Test for patterns rather than exact values where appropriate
   - Allow multiple correct approaches for implementation

2. **Clear Error Messages**:
   - Provide informative feedback when tests fail
   - Include hints about potential issues
   - Reference specific requirements from the assignment

3. **Modular Test Structure**:
   - Separate tests for each component
   - Allow partial credit for passing some test sections
   - Include both basic and advanced test cases

4. **Edge Case Handling**:
   - Test with different subsets of the data
   - Include robustness checks for unusual inputs
   - Verify behavior with boundary conditions

### Example Test Structure

```python
# Example test for data loading
def test_data_loading():
    # Test that student function loads the correct number of subjects
    data = student_solution.load_data()
    assert len(data) >= 30, "Should load data from at least 30 subjects"
    
    # Test that required columns are present
    required_columns = ['EDA', 'BVP', 'HR', 'TEMP']
    for col in required_columns:
        assert col in data.columns, f"Missing required column: {col}"
    
    # Test that time index is properly formatted
    assert pd.api.types.is_datetime64_dtype(data.index), "Index should be datetime type"

# Example test for feature engineering
def test_feature_engineering():
    # Test that features are created with correct dimensions
    features = student_solution.extract_features(test_data)
    assert features.shape[1] >= 10, "Should extract at least 10 features"
    
    # Test that specific required features are present
    required_features = ['hr_mean', 'eda_std', 'temp_slope']
    for feature in required_features:
        assert feature in features.columns, f"Missing required feature: {feature}"
    
    # Test value ranges for specific features
    assert features['hr_mean'].between(50, 200).all(), "Heart rate mean outside expected range"

# Example test for model performance
def test_model_performance():
    # Test that model achieves minimum performance threshold
    predictions, metrics = student_solution.train_and_evaluate_model(test_data)
    assert metrics['r2_score'] > 0.3, "Model performance below minimum threshold"
    
    # Test prediction format
    assert len(predictions) == len(test_labels), "Predictions should match test set size"
```

---

## Revised Lecture Structure

### 1. Time Series Fundamentals (~30 minutes)
- **Introduction**: What makes time series special in healthcare?
- **Key Concepts**: Trend, seasonality, noise with health examples
- **Challenges**: Missing values, irregular sampling in clinical data
- **Feature Engineering**: Time-based, lagged, rolling features
- **Interactive Q&A**: Discuss challenges in health time series data
- **Demo**: Heart Rate Oscillations during Meditation dataset

### 2. Regression and Basic Forecasting (~30 minutes)
- **Linear Regression Review**: Application to temporal health data
- **Time Series Cross-Validation**: Respecting temporal order
- **Simple Forecasting**: Moving averages, exponential smoothing
- **Evaluation Metrics**: Choosing appropriate measures of success
- **Interactive Q&A**: Discuss forecasting applications in healthcare
- **Demo**: Sleep monitoring data analysis and prediction

### 3. Practical Applications (~30 minutes)
- **Dense Physiological Data**: Handling high-frequency health signals
- **Case Studies**: Patient monitoring, disease progression, resource planning
- **Implementation Considerations**: Computational efficiency, real-time analysis
- **Future Directions**: Brief overview of advanced methods
- **Interactive Q&A**: Discuss real-world applications
- **Demo**: Advanced analysis of sleep and activity data

---

## Detailed Demo Outlines

### Demo 1: Heart Rate Oscillations during Meditation
```python
# Demo 1: Exploring Temporal Patterns in Heart Rate during Meditation

# Learning objectives:
# 1. Load and explore real physiological time series data
# 2. Visualize temporal patterns in heart rate
# 3. Perform basic time series analysis
# 4. Handle missing values and irregularities

# Part 1: Data Loading and Exploration
# - Load the Heart Rate Oscillations during Meditation dataset
# - Examine data structure and characteristics
# - Visualize raw heart rate time series

# Part 2: Time Series Visualization
# - Create time plots of heart rate during different meditation techniques
# - Compare patterns between meditation types
# - Identify trends and cyclical patterns

# Part 3: Statistical Analysis
# - Test for stationarity using Augmented Dickey-Fuller test
# - Calculate and visualize autocorrelation
# - Identify significant lags in the data

# Part 4: Handling Data Challenges
# - Introduce artificial missing values
# - Demonstrate different imputation techniques
# - Compare results of different approaches
# - Discuss best practices for clinical time series

# Part 5: Summary and Discussion
# - Review key findings
# - Discuss implications for health monitoring
# - Suggest extensions and applications
```

### Demo 2: Sleep Monitoring Analysis
```python
# Demo 2: Predictive Modeling with Sleep Monitoring Data

# Learning objectives:
# 1. Work with multivariate health time series data
# 2. Engineer features from temporal health data
# 3. Build and evaluate regression models
# 4. Compare different cohorts and conditions

# Part 1: Data Loading and Exploration
# - Load the MMASH sleep monitoring dataset
# - Examine data structure and relationships
# - Visualize daily patterns in activity and heart rate

# Part 2: Feature Engineering
# - Create time-based features (hour of day, day of week)
# - Calculate rolling statistics (mean, std, min, max)
# - Extract circadian rhythm features
# - Create lag features for time-dependent effects

# Part 3: Regression Modeling
# - Split data into training and testing sets (respecting time order)
# - Build linear regression model to predict heart rate from activity
# - Evaluate model performance with appropriate metrics
# - Visualize predictions vs. actual values

# Part 4: Cohort Comparison
# - Group subjects by activity level or sleep quality
# - Compare regression models between groups
# - Test for significant differences in slopes/intercepts
# - Interpret findings in health context

# Part 5: Simple Random Forest Model
# - Introduce Random Forest as a non-linear alternative
# - Build a basic model with key features
# - Compare performance with linear regression
# - Discuss trade-offs between interpretability and accuracy
```

### Demo 3: Advanced Sleep and Activity Analysis
```python
# Demo 3: Advanced Analysis of Sleep and Activity Data

# Learning objectives:
# 1. Process and analyze dense physiological signals
# 2. Extract meaningful features from raw sensor data
# 3. Visualize complex temporal patterns
# 4. Connect analysis to health applications

# Part 1: Working with Dense Accelerometer Data
# - Load high-frequency accelerometer data from MMASH
# - Visualize raw signals and identify challenges
# - Discuss sampling rates and signal characteristics

# Part 2: Signal Processing Techniques
# - Apply filtering to remove noise (low-pass, high-pass filters)
# - Demonstrate signal smoothing techniques
# - Extract activity intensity from accelerometer magnitude
# - Visualize raw vs. processed signals

# Part 3: Feature Extraction
# - Calculate frequency domain features using FFT
# - Extract statistical features from windows of data
# - Identify activity patterns and transitions
# - Create derived metrics for health assessment

# Part 4: Connecting to Health Outcomes
# - Correlate activity patterns with sleep quality
# - Visualize relationships between daytime activity and nighttime rest
# - Demonstrate a simple classification of activity types
# - Discuss applications for personalized health monitoring

# Part 5: Interactive Visualization
# - Create an interactive time series dashboard
# - Enable exploration of patterns across time scales
# - Highlight key health insights from the data
# - Discuss effective visualization for clinical interpretation
```

---

## Implementation Plan

```mermaid
gantt
    title Time Series Lecture Improvement Plan
    dateFormat  YYYY-MM-DD
    section Lecture Content
    Review and streamline content           :2023-05-01, 3d
    Enhance beginner-friendly explanations  :2023-05-04, 2d
    Improve health data relevance           :2023-05-06, 2d
    section Demos
    Develop Demo 1 (Heart Rate)             :2023-05-08, 3d
    Develop Demo 2 (Sleep Monitoring)       :2023-05-11, 3d
    Develop Demo 3 (Advanced Analysis)      :2023-05-14, 3d
    section Homework
    Design assignment structure             :2023-05-17, 2d
    Create detailed instructions            :2023-05-19, 2d
    Develop pytest framework                :2023-05-21, 3d
    Setup GitHub Classroom                  :2023-05-24, 1d
    section Final Review
    Integrate all components                :2023-05-25, 2d
    Quality assurance                       :2023-05-27, 1d
