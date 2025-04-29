# Demo 2: Sensor Classification with Derived Features (Activity Recognition)

This demo explores classifying human activities (like sitting, walking, running) using synthetic accelerometer data. Wearable sensors are common in health monitoring, and classifying activities from sensor streams is a fundamental task. We will:
1. Generate synthetic 3-axis accelerometer data mimicking different activities.
2. Engineer features from time windows of this data (e.g., mean, standard deviation).
3. Train and compare two powerful tree-based classifiers: RandomForest and XGBoost.
4. Use SHAP (SHapley Additive exPlanations) to interpret the model and understand which features are most important for classification.
This demo highlights feature engineering for time series and introduces model interpretation techniques crucial for trusting ML models in health applications.

## 1. Setup: Import Libraries

We import our standard data science toolkit (`numpy`, `pandas`, `matplotlib`, `seaborn`) along with specific libraries for this task:
- `sklearn.model_selection`: For splitting data.
- `sklearn.ensemble`: For the RandomForestClassifier.
- `xgboost`: For the XGBClassifier (a popular gradient boosting library).
- `sklearn.metrics`: For evaluation metrics, including `confusion_matrix`.
- `shap`: For model interpretation (needs to be installed: `pip install shap`).
We set a random seed for reproducibility.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Added confusion_matrix
import shap # Make sure to install shap: pip install shap

# Configure plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis') # Using a different palette for variety
np.random.seed(42) # for reproducibility
```

## 2. Generate Synthetic Sensor Data

We'll simulate 3-axis accelerometer data for three activities: sitting, walking, and running. Each activity will have distinct statistical properties:
- **Sitting:** Low variance, mean close to (0, 0, 1) assuming Z is vertical axis against gravity.
- **Walking:** Moderate variance, periodic patterns.
- **Running:** High variance, more intense periodic patterns.

We generate data in segments (windows) for each activity. Each segment represents a short time interval (e.g., 2 seconds) with multiple sensor readings.

```python
def generate_activity_data(activity_type, n_segments=100, segment_len=50, noise_level=0.1):
    """Generates synthetic accelerometer data for a given activity."""
    data = []
    if activity_type == 'sitting':
        base_signal = np.array([0, 0, 1]) # Gravity on Z axis
        variance = 0.01
    elif activity_type == 'walking':
        base_signal = np.array([0, 0, 1])
        variance = 0.2
        # Add simple periodic component to X/Y
        time = np.linspace(0, 2*np.pi, segment_len)
        periodic_x = 0.5 * np.sin(time)
        periodic_y = 0.3 * np.cos(time)
    elif activity_type == 'running':
        base_signal = np.array([0, 0, 1])
        variance = 0.8
        # Add stronger periodic component
        time = np.linspace(0, 4*np.pi, segment_len) # Faster pace
        periodic_x = 1.5 * np.sin(time)
        periodic_y = 1.0 * np.cos(time)
    else:
        raise ValueError("Unknown activity type")

    for _ in range(n_segments):
        segment = np.random.normal(loc=0, scale=np.sqrt(variance), size=(segment_len, 3)) + base_signal
        if activity_type == 'walking':
            segment[:, 0] += periodic_x
            segment[:, 1] += periodic_y
        elif activity_type == 'running':
            segment[:, 0] += periodic_x
            segment[:, 1] += periodic_y
        segment += np.random.normal(0, noise_level, size=(segment_len, 3)) # Add noise
        data.append(segment)
    return np.array(data)

# Generate data for each activity
sitting_data = generate_activity_data('sitting', n_segments=200)
walking_data = generate_activity_data('walking', n_segments=200)
running_data = generate_activity_data('running', n_segments=200)

# Combine data and create labels
# Labels: 0=sitting, 1=walking, 2=running
all_data = np.concatenate([sitting_data, walking_data, running_data], axis=0)
labels = np.concatenate([
    np.zeros(sitting_data.shape[0]),
    np.ones(walking_data.shape[0]),
    np.full(running_data.shape[0], 2)
])

print("Total data shape (segments, segment_length, axes):", all_data.shape)
print("Labels shape:", labels.shape)

# Visualize a sample segment for each activity
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(sitting_data[0])
axes[0].set_title('Sample Sitting Segment')
axes[0].set_ylabel('Acceleration')
axes[0].legend(['X', 'Y', 'Z'])
axes[1].plot(walking_data[0])
axes[1].set_title('Sample Walking Segment')
axes[1].set_ylabel('Acceleration')
axes[2].plot(running_data[0])
axes[2].set_title('Sample Running Segment')
axes[2].set_xlabel('Time Step within Segment')
axes[2].set_ylabel('Acceleration')
plt.tight_layout()
plt.show()
```

## 3. Feature Engineering from Time Windows

Machine learning models often work better with summary statistics (features) derived from raw sensor data rather than the raw data itself. We'll calculate simple time-domain features for each axis (X, Y, Z) within each segment:
- Mean
- Standard Deviation (Std)
- Minimum (Min)
- Maximum (Max)
- Variance (Var)

This transforms our (segments, segment_length, axes) data into a (segments, num_features) structure suitable for standard classifiers.

```python
def extract_features(data_segments):
    """Extracts statistical features from each segment."""
    n_segments = data_segments.shape[0]
    n_axes = data_segments.shape[2]
    features = []

    for i in range(n_segments):
        segment = data_segments[i, :, :]
        segment_features = []
        for axis in range(n_axes):
            axis_data = segment[:, axis]
            segment_features.extend([
                np.mean(axis_data),
                np.std(axis_data),
                np.min(axis_data),
                np.max(axis_data),
                np.var(axis_data)
            ])
        features.append(segment_features)
    return np.array(features)

# Extract features
X_features = extract_features(all_data)

# Create feature names for clarity
axis_names = ['X', 'Y', 'Z']
stat_names = ['mean', 'std', 'min', 'max', 'var']
feature_names = [f'{ax}_{stat}' for ax in axis_names for stat in stat_names]

# Convert features to a DataFrame
df_features = pd.DataFrame(X_features, columns=feature_names)
df_features['Activity'] = labels.astype(int) # Add labels as integers

print("Feature matrix shape:", df_features.shape)
print("\nFirst 5 rows of features:")
print(df_features.head())
```

## 4. Split Data into Training and Testing Sets

As in the previous demo, we split our engineered features (`X_features`) and corresponding labels (`labels`) into training and testing sets. This allows us to evaluate how well our trained models generalize to unseen data segments. We use `stratify=labels` to maintain the proportion of each activity in both sets.

```python
# Separate features (X) and target (y)
X = df_features.drop('Activity', axis=1)
y = df_features['Activity']

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
```

## 5. Train and Compare Models: RandomForest vs. XGBoost

We'll train two popular and powerful tree-based ensemble models:

- **RandomForestClassifier:** Builds multiple decision trees on different subsets of data and features, and averages their predictions. Known for robustness and ease of use.
- **XGBClassifier:** Uses gradient boosting, where trees are built sequentially, each correcting the errors of the previous one. Often achieves state-of-the-art results on tabular data.

We train both models on the same training data.

```python
# --- RandomForest ---
print("Training RandomForest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Use 100 trees
rf_model.fit(X_train, y_train)
print("RandomForest training complete.")

# --- XGBoost ---
# Note: XGBoost expects labels to start from 0. Ours already do (0, 1, 2).
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax', # Specify multi-class classification
    num_class=3,              # Number of classes
    n_estimators=100,
    use_label_encoder=False,  # Recommended practice
    eval_metric='mlogloss',   # Evaluation metric for multi-class
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
print("XGBoost training complete.")
```

## 6. Evaluate Models

We evaluate both models on the unseen test set using:
- **Accuracy:** Overall correct prediction rate.
- **Classification Report:** Precision, Recall, F1-score for each activity class (sitting, walking, running). This gives a more detailed view, especially important if class performance differs.
- **Confusion Matrix:** Visualizes the counts of correct and incorrect predictions for each class.

We compare the results to see which model performed better on this task.

```python
# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# --- Evaluate RandomForest ---
print("\n--- RandomForest Evaluation ---")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['sitting', 'walking', 'running']))

# Confusion Matrix for RandomForest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['sitting', 'walking', 'running'],
            yticklabels=['sitting', 'walking', 'running'])
plt.title('RandomForest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# --- Evaluate XGBoost ---
print("\n--- XGBoost Evaluation ---")
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy: {accuracy_xgb:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['sitting', 'walking', 'running']))

# Confusion Matrix for XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', # Different color map
            xticklabels=['sitting', 'walking', 'running'],
            yticklabels=['sitting', 'walking', 'running'])
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

## 7. Model Interpretation with SHAP

While high accuracy is good, in health applications, understanding *why* a model makes certain predictions is crucial for trust and clinical relevance. SHAP (SHapley Additive exPlanations) helps us do this.

We'll use SHAP's `TreeExplainer` (optimized for tree models like RF and XGBoost) to calculate SHAP values. These values represent the contribution of each feature to the prediction for each individual data point (segment).

The `summary_plot` visualizes these contributions:
- **Features:** Listed vertically, ordered by overall importance.
- **Points:** Each point is a SHAP value for a feature for a single prediction.
- **Horizontal Location:** Shows the impact on the model's output (e.g., pushing towards a higher class prediction).
- **Color:** Represents the original feature value (high/low).

This helps us see which features (like 'Z_std', 'X_var') are most influential and how their values affect the activity classification. We'll interpret the XGBoost model as an example.

```python
print("\n--- SHAP Interpretation (XGBoost) ---")

# Initialize the SHAP explainer for the XGBoost model
# For tree models, TreeExplainer is efficient
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for the test set
# This can take a moment
shap_values = explainer.shap_values(X_test)

print("SHAP values calculated.")

# Create the summary plot
# For multi-class, shap_values is a list of arrays (one per class)
# We can plot for one class or use plot_type='bar' for overall importance
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=['sitting', 'walking', 'running'], feature_names=feature_names)

# Detailed summary plot for one class (e.g., class 1: walking)
# shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, plot_title="SHAP Values for Walking Class")

# Dependence plot for a specific feature (e.g., Z_std)
# Shows how the feature value affects prediction, potentially colored by another feature
# shap.dependence_plot("Z_std", shap_values[1], X_test, feature_names=feature_names, interaction_index="X_std")

```

## 8. Interpretation & Conclusion

Let's analyze the results:
- **Model Performance:** Both RandomForest and XGBoost likely performed well on this synthetic data, achieving high accuracy. XGBoost might have a slight edge, which is common. The classification reports and confusion matrices show how well each activity was identified and where errors occurred (e.g., confusing walking and running).
- **Feature Engineering:** Simple statistical features (mean, std, min, max, var) derived from sensor windows were effective for distinguishing these activities. Standard deviation and variance features are often key for activity recognition, as they capture movement intensity.
- **SHAP Interpretation:** The SHAP summary plot reveals the most important features globally. For instance, features related to the variance or standard deviation (like `Z_std`, `X_var`, `Y_var`) are probably high on the list, as they directly reflect the amount of movement. The plot shows *which* features the model relies on most heavily. Dependence plots (optional code included) could further reveal *how* feature values influence predictions (e.g., higher variance pushing towards 'running').

This demo illustrated a common workflow for sensor-based classification: segmenting data, extracting features, training models, and interpreting results. Understanding feature importance via SHAP builds confidence in the model's decisions, a critical step in applying ML in healthcare.

**🧠 Comprehension Checkpoint:**

1.  Why did we extract features (like mean, std) instead of feeding the raw sensor data directly into RandomForest/XGBoost? (Hint: These models work best with tabular feature matrices).
2.  What is the main difference in how RandomForest and XGBoost build their trees? (Hint: Independent vs. Sequential/Error-Correcting).
3.  What does the SHAP summary plot tell us about the model?
4.  Based on the likely SHAP results, which features would you expect to be most important for distinguishing 'sitting' from 'running'?

Answers:
1. Standard tree-based models like RandomForest and XGBoost expect a 2D array (samples x features). Feature engineering converts the time series segments into this format, summarizing the information in each window. (More advanced models like CNNs/LSTMs can handle raw sequences).
2. RandomForest builds trees independently on bootstrapped samples. XGBoost builds trees sequentially, with each new tree trying to correct the errors made by the previous ones (gradient boosting).
3. It shows the overall importance of each feature across all predictions and gives an idea of how feature values (high/low) tend to influence the model's output towards different classes.
4. Features related to variance or standard deviation (e.g., `X_std`, `Y_std`, `Z_std`, `X_var`, etc.) because running involves much more movement (higher variance) than sitting.