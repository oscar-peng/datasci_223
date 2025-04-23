# Time Series Analysis Homework Assignment

## Overview

In this assignment, you will apply the time series analysis techniques learned in class to real-world health data from the Wearable Device Dataset from Induced Stress and Exercise Sessions. This dataset contains physiological signals (Electrodermal Activity, Blood Volume Pulse, Heart Rate, Temperature) from healthy volunteers collected during structured acute stress induction and aerobic/anaerobic exercise sessions using the Empatica E4 wearable device.

Dataset link: https://physionet.org/content/wearable-stress-affect/1.0.0/

## Learning Objectives

By completing this assignment, you will:
- Gain experience working with real physiological time series data
- Apply preprocessing techniques to handle missing values and noise
- Extract meaningful features from time series data
- Build and evaluate predictive models for physiological responses
- Compare different conditions (stress vs. exercise) using time series analysis

## Setup

1. Clone this repository to your local machine
2. Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download the dataset from PhysioNet (instructions below)

## Dataset Download Instructions

The dataset is available on PhysioNet. You can download it using the following methods:

### Option 1: Manual Download
1. Visit https://physionet.org/content/wearable-stress-affect/1.0.0/
2. Click on "Download" and select your preferred format
3. Extract the downloaded files to the `data` directory in this repository

### Option 2: Automated Download (Python)
Run the provided script to download the dataset:
```bash
python download_data.py
```

## Assignment Structure

This assignment is divided into three parts, each building on the previous one:

### Part 1: Data Exploration and Preprocessing (30%)

In this part, you will:
- Load and explore the Wearable Stress/Exercise dataset
- Visualize temporal patterns in physiological signals during stress vs. exercise
- Handle missing values and irregularities in wearable device data
- Perform basic statistical analysis comparing different conditions
- Create summary visualizations of physiological responses

**Files to complete:**
- `part1_exploration.py` or `part1_exploration.ipynb`

### Part 2: Time Series Modeling (40%)

In this part, you will:
- Extract features from physiological time series data
- Build regression models to predict stress levels from physiological signals
- Compare different modeling approaches for stress detection
- Evaluate model performance using appropriate time series metrics
- Interpret results in a health monitoring context

**Files to complete:**
- `part2_modeling.py` or `part2_modeling.ipynb`

### Part 3: Advanced Analysis (30%)

In this part, you will:
- Apply signal processing techniques to the wearable sensor data
- Extract meaningful features from dense physiological signals
- Identify patterns that differentiate stress from exercise responses
- Create an interactive visualization comparing conditions
- Discuss potential applications for stress monitoring and management

**Files to complete:**
- `part3_advanced.py` or `part3_advanced.ipynb`

## Submission Guidelines

1. Complete all the required files
2. Make sure your code runs without errors
3. Include comments explaining your approach
4. Submit your work by pushing to your GitHub repository

## Grading Criteria

Your assignment will be graded based on:
1. **Correctness (40%)**: Does your code run without errors and produce the expected results?
2. **Methodology (30%)**: Did you apply appropriate techniques and justify your choices?
3. **Analysis (20%)**: How well did you interpret the results and draw meaningful conclusions?
4. **Code Quality (10%)**: Is your code well-organized, documented, and efficient?

## Automated Testing

This assignment includes automated tests to help you verify your solutions. To run the tests:

```bash
pytest test_assignment.py
```

The tests will check:
- Data loading and preprocessing
- Feature engineering
- Model implementation
- Visualization output

## Resources

- Lecture slides and demo notebooks
- [PhysioNet Wearable Dataset Documentation](https://physionet.org/content/wearable-stress-affect/1.0.0/)
- [Pandas Time Series Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/classes.html)
- [SciPy Signal Processing Documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)

## Due Date

This assignment is due on [DATE] at [TIME].

## Questions

If you have any questions about the assignment, please post them on the course forum or contact the instructor.