# Demo 2: Physiological Stress Classification Using Wearable Sensor Data


---- # KEEP THE OUTLINE FOR NOW AS CONTEXT

## 0. Project Overview

### Research Questions

1. Can we identify periods of high stress in students using physiological signals during exam sessions?
2. Do stress response patterns correlate with academic performance?

### Data Location

`wget -r -N -c -np https://physionet.org/files/wearable-exam-stress/1.0.0/`

### Classification Objectives

**Two-Stage Classification Task:**

1. **Stress Level Detection:**
   - Identify high-stress periods from physiological signals
   - Use sliding windows to detect stress state changes

2. **Performance Analysis:**
   - Group students by exam performance (above/below median)
   - Analyze stress response patterns between groups

### Analysis Thresholds

- **Stress Level:** Defined by significant deviations in physiological signals
- **Performance Groups:**
    - High: Students scoring above the median grade
    - Low: Students scoring below the median grade

## 1. Dataset Context

### Data Source

- Wearable Exam Stress Dataset from PhysioNet
- Empatica E4 wristband recordings
- 10 participants across three exam sessions
- All exams start at 9:00 AM (CT/CDT)
- Midterms: 1.5 hours, Finals: 3 hours

### Classification Features

1. Physiological Signals
   - Heart Rate
   - Electrodermal Activity (EDA)
   - Blood Volume Pulse
   - Skin Temperature
   - Inter-Beat Interval (IBI)
   - 3-axis Accelerometer

2. Derived Features
   - Heart Rate Variability (HRV) metrics
   - Frequency domain characteristics
   - Time-window statistical features
   - Movement/activity levels

## 2. Preprocessing Strategy

### Data Cleaning

- Handle missing values (<1%)
- Remove physiological outliers using rolling z-scores
- Normalize signal intensities
- Synchronize timestamps across sensors
- Filter motion artifacts using accelerometer data

### Feature Extraction Approach

1. **Time-Window Features (Stress Detection)**
   - Sliding windows (60-second, 30-second overlap)
   - Statistical features: mean, std, min, max
   - Physiological features: HRV, EDA peaks
   - Activity level from accelerometer

2. **Session-Level Features (Performance Analysis)**
   - Aggregate stress measures per exam
   - Stress pattern characteristics
   - Temporal stress distribution

## 3. Classification Methodology

### Stage 1: Stress Detection

- **Model:** XGBoost Classifier
- **Target:** Binary stress state (high/normal)
- **Validation:** Leave-one-subject-out cross-validation
- **Metrics:**
    - Precision-Recall curve
    - F1-Score
    - ROC-AUC

### Stage 2: Performance Analysis

- **Model:** Random Forest Classifier
- **Target:** Binary performance group
- **Features:** Aggregated stress patterns
- **Validation:** K-fold cross-validation (K=3)
- **Metrics:**
    - Accuracy
    - Balanced accuracy
    - Confusion matrix

### Cross-Validation Strategy

- Stage 1: Leave-one-subject-out
- Stage 2: Stratified K-Fold (K=3)
- Preserve exam session distribution
- Account for subject dependencies

## 4. Interpretability Techniques

### Stress Detection Analysis

- SHAP values for stress classification
- Feature importance in stress detection
- Temporal stress patterns visualization
- Physiological signal correlations

### Performance Analysis

- Aggregate stress patterns by group
- SHAP interaction values
- Partial dependence plots
- Individual stress response profiles

### Visualization Approaches

- Time series plots with stress highlights
- Group-wise stress pattern comparisons
- Feature importance rankings
- Individual subject trajectories

## 5. Ethical Considerations

### Data Privacy

- De-identified participant data
- Aggregate-level analysis
- No individual student identification
- Stress level confidentiality

### Methodological Limitations

- Small sample size
- Individual physiological variations
- Generalizability constraints
- Stress definition subjectivity

## Learning Objectives

1. Process and analyze multivariate physiological time series
2. Implement two-stage classification pipeline
3. Extract meaningful features from wearable sensor data
4. Apply cross-validation for time series data
5. Interpret stress patterns and their relationship to performance

## Comprehension Checkpoint

1. How can we identify stress periods from physiological signals?
2. What challenges exist in personalizing stress detection?
3. Why use a two-stage approach for this analysis?
4. How might different stress responses relate to performance?

## Recommended Workflow

1. Data exploration and cleaning
2. Signal preprocessing and feature extraction
3. Stress detection model development
4. Performance analysis model development
5. Model interpretation and visualization
6. Results analysis and validation