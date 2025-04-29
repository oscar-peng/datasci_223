# Demo 1: Binary Classification with Logistic Regression (Diabetes Prediction)

This demo introduces binary classification using one of the simplest yet effective models: Logistic Regression. We'll use synthetic data resembling a diabetes prediction task. The goal is to predict whether a patient has diabetes (1) or not (0) based on simulated health metrics. We'll cover data generation, model training, and essential evaluation metrics. This provides a foundational understanding before moving to more complex models.

## 1. Setup: Import Libraries

We start by importing the necessary Python libraries.
- `numpy` for numerical operations.
- `pandas` for data manipulation (DataFrames).
- `matplotlib.pyplot` and `seaborn` for plotting.
- `sklearn.model_selection` for splitting data.
- `sklearn.linear_model` for the Logistic Regression model.
- `sklearn.metrics` for evaluating the model (accuracy, confusion matrix, classification report, ROC curve, AUC).
- `sklearn.datasets` to generate synthetic data easily.
Setting a random seed (`np.random.seed`) ensures that our results are reproducible – if we run the code again, the synthetic data generated will be the same.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.datasets import make_classification

# Configure plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
np.random.seed(42) # for reproducibility
```

## 2. Generate Synthetic Diabetes Data

Real health data can be complex and requires significant cleaning. For this introductory demo, we'll use `make_classification` from scikit-learn to generate synthetic data that mimics a diabetes prediction scenario.
We create:
- `X`: Features (e.g., simulated glucose level, BMI, age). We'll create 1000 samples with 5 features.
- `y`: Target variable (0 for no diabetes, 1 for diabetes).
We specify `n_informative=3` meaning only 3 out of 5 features actually help predict the outcome, adding some realism. `flip_y=0.05` introduces a small amount of label noise (5% incorrect labels), also common in real datasets.
Finally, we convert the data into a pandas DataFrame for easier handling.

```python
# Generate synthetic data for a binary classification problem
X, y = make_classification(
    n_samples=1000,      # Number of patients
    n_features=5,        # Number of health indicators (e.g., glucose, BMI, age, bp, cholesterol)
    n_informative=3,     # Number of features that actually predict diabetes
    n_redundant=1,       # Number of features that are linear combinations of informative features
    n_clusters_per_class=1, # How features group for each class
    weights=[0.8, 0.2],  # 80% non-diabetic (0), 20% diabetic (1) -> Imbalanced data!
    flip_y=0.05,         # Introduce some noise in labels
    random_state=42
)

# Create a pandas DataFrame
feature_names = ['Glucose', 'BMI', 'Age', 'BP', 'Cholesterol']
df = pd.DataFrame(X, columns=feature_names)
df['Diabetes'] = y

print("Synthetic Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['Diabetes'].value_counts(normalize=True))
```

## 3. Explore the Data

Before modeling, it's crucial to understand the data. We'll use `pairplot` from seaborn to visualize the relationships between features and how they differ between the two classes (diabetes vs. no diabetes). This helps identify potentially useful predictors.
Look for features where the distributions (histograms on the diagonal) or scatter plots (off-diagonal) show separation between the blue (0) and orange (1) points.

```python
# Visualize relationships between features, colored by the target variable
sns.pairplot(df, hue='Diabetes', vars=feature_names[:3]) # Plot first 3 features for simplicity
plt.suptitle('Pair Plot of Features by Diabetes Status', y=1.02)
plt.show()
```

## 4. Split Data into Training and Testing Sets

To evaluate our model fairly, we need to test it on data it hasn't seen during training. We split the dataset into:
- Training set (e.g., 80%): Used to train the Logistic Regression model.
- Testing set (e.g., 20%): Used to evaluate the trained model's performance on unseen data.
`train_test_split` shuffles the data before splitting (`stratify=y` ensures the proportion of diabetic/non-diabetic patients is similar in both train and test sets, which is important for imbalanced data).

```python
# Separate features (X) and target (y)
X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42,  # Ensure reproducibility
    stratify=y        # Keep class proportions consistent in splits
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
```

## 5. Train a Logistic Regression Model

Now we train our classification model. Logistic Regression is a good starting point because it's relatively simple and interpretable.
1. Create an instance of the `LogisticRegression` model.
2. `fit` the model to the training data (`X_train`, `y_train`). The model learns the relationship between the features and the diabetes outcome from this data.

```python
# Initialize the Logistic Regression model
log_reg = LogisticRegression(random_state=42)

# Train the model on the training data
print("Training the Logistic Regression model...")
log_reg.fit(X_train, y_train)
print("Model training complete.")
```

## 6. Make Predictions on the Test Set

With the trained model, we can now predict the outcome for the unseen test data (`X_test`). The `predict` method outputs the predicted class label (0 or 1) for each sample in the test set.

```python
# Use the trained model to predict diabetes status on the test set
y_pred = log_reg.predict(X_test)

print("Predictions made on the test set.")
# print("First 10 predictions:", y_pred[:10])
# print("Actual first 10 labels:", y_test.values[:10])
```

## 7. Evaluate the Model

How well did our model do? We need metrics to quantify its performance.

**Accuracy:** The overall percentage of correct predictions. Simple, but can be misleading with imbalanced data.
**Confusion Matrix:** A table showing correct and incorrect predictions broken down by class:
  - True Positives (TP): Correctly predicted Diabetes (1)
  - True Negatives (TN): Correctly predicted No Diabetes (0)
  - False Positives (FP): Incorrectly predicted Diabetes (Type I Error)
  - False Negatives (FN): Incorrectly predicted No Diabetes (Type II Error - often critical in health!)
**Classification Report:** Provides key metrics per class:
  - **Precision:** TP / (TP + FP) - Of those predicted positive, how many actually are? (Minimizes false alarms)
  - **Recall (Sensitivity):** TP / (TP + FN) - Of those actually positive, how many did we catch? (Minimizes missed cases)
  - **F1-Score:** Harmonic mean of Precision and Recall - A balanced measure.
**ROC Curve & AUC:**
  - **ROC Curve:** Plots True Positive Rate (Recall) vs. False Positive Rate at different probability thresholds. A good model hugs the top-left corner.
  - **AUC (Area Under the Curve):** A single number summarizing the ROC curve. 1.0 is perfect, 0.5 is random guessing. Measures how well the model distinguishes between classes.

```python
# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plotting the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes (0)', 'Diabetes (1)'],
            yticklabels=['No Diabetes (0)', 'Diabetes (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 3. Classification Report
report = classification_report(y_test, y_pred, target_names=['No Diabetes (0)', 'Diabetes (1)'])
print("\nClassification Report:")
print(report)

# 4. ROC Curve and AUC
# Get prediction probabilities for the positive class (Diabetes=1)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC: {auc:.4f}")

# Plot ROC Curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing') # Diagonal line
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()
```

## 8. Interpretation & Conclusion

Let's interpret the results:
- **Accuracy:** Gives an overall sense, but look deeper due to class imbalance.
- **Confusion Matrix:** Shows *how* the model makes mistakes. Are we missing diabetic patients (FN) or falsely alarming healthy ones (FP)? In healthcare, minimizing FN is often critical.
- **Classification Report:** Precision tells us about the reliability of positive predictions. Recall tells us how well we find actual positive cases. F1 balances both. Notice the metrics might be lower for the minority class (Diabetes=1) due to imbalance.
- **AUC:** Provides a threshold-independent measure of how well the model separates the classes. An AUC significantly above 0.5 indicates the model has discriminative ability.

This demo covered the basics of binary classification: generating data, training a simple model, and evaluating its performance using standard metrics. Logistic Regression provides a solid baseline. In the next demos, we'll explore more complex models and techniques for handling different data challenges.

**🧠 Comprehension Checkpoint:**

1.  Why is splitting data into training and testing sets important?
2.  What is the difference between Precision and Recall? When might one be more important than the other in a health context?
3.  What does an AUC of 0.6 tell you compared to an AUC of 0.9?
4.  Look at the confusion matrix: How many diabetic patients did the model miss (False Negatives)?

Answers:
1. To evaluate the model's ability to generalize to new, unseen data and avoid overfitting.
2. Precision = TP / (TP + FP) - Accuracy of positive predictions. Recall = TP / (TP + FN) - Ability to find all positive cases. Recall is often more critical when missing a positive case (e.g., a disease) has severe consequences.
3. An AUC of 0.9 indicates a much better ability to distinguish between classes than an AUC of 0.6 (which is only slightly better than random guessing).
4. Find the value in the bottom-left cell of the confusion matrix.