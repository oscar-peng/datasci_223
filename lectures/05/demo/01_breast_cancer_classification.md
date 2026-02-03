# Demo 1: Binary Classification with Breast Cancer Data

This demo introduces binary classification using real medical data: the Wisconsin Breast Cancer dataset. The goal is to predict whether a tumor is **malignant** (cancerous) or **benign** (non-cancerous) based on measurements of cell nuclei from fine needle aspirate (FNA) images.

## Learning Objectives

By the end of this demo, you will be able to:

1. Load and explore a real medical dataset
2. Split data properly with stratification
3. Train a logistic regression classifier
4. Evaluate model performance with confusion matrix, classification report, and ROC curve

## 0. Setup

```python
%pip install -q scikit-learn matplotlib seaborn pandas numpy
```

## 1. Load the Breast Cancer Dataset

The Wisconsin Breast Cancer dataset is built into scikit-learn. It contains 569 samples with 30 features computed from digitized images of breast mass FNA.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()

# Create a DataFrame for easier exploration
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})

print(f"Dataset shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['diagnosis'].value_counts())
print(f"\nFeature names (first 10):")
print(list(data.feature_names[:10]))
```

## 2. Explore the Data

Understanding feature distributions helps identify which measurements differ between malignant and benign tumors.

```python
# Select a few key features to visualize
key_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, feature in zip(axes.flat, key_features):
    sns.histplot(data=df, x=feature, hue='diagnosis', kde=True, ax=ax)
    ax.set_title(f'{feature} by Diagnosis')
plt.tight_layout()
plt.show()

# Correlation of features with target
correlations = df.drop(['target', 'diagnosis'], axis=1).corrwith(df['target'])
top_features = correlations.abs().sort_values(ascending=False).head(10)
print("Top 10 features correlated with diagnosis:")
print(top_features)
```

## 3. Prepare Data for Modeling

Split the data into training and test sets. Use `stratify` to ensure both sets have similar proportions of malignant and benign cases.

```python
from sklearn.model_selection import train_test_split

# Separate features and target
X = data.data
y = data.target  # 0 = malignant, 1 = benign

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTraining class distribution:")
print(f"  Malignant (0): {sum(y_train == 0)}")
print(f"  Benign (1): {sum(y_train == 1)}")
```

## 4. Train a Logistic Regression Model

Logistic regression is a good starting point for binary classification - it's interpretable and often performs well on medical data.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")
print(f"Training accuracy: {model.score(X_train_scaled, y_train):.4f}")
```

## 5. Evaluate the Model

Use multiple metrics to understand model performance, especially important in medical contexts where false negatives (missing cancer) can be serious.

```python
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant', 'Benign'])
disp.plot(ax=ax, cmap='Blues')
ax.set_title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nAUC Score: {auc_score:.4f}")
```

## 6. Interpret the Model

Examine which features are most important for predicting malignancy.

```python
# Get feature importances (coefficients)
feature_importance = pd.DataFrame({
    'feature': data.feature_names,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

# Plot top 10 features
plt.figure(figsize=(10, 6))
top_10 = feature_importance.head(10)
colors = ['red' if c < 0 else 'green' for c in top_10['coefficient']]
plt.barh(top_10['feature'], top_10['coefficient'], color=colors)
plt.xlabel('Coefficient (negative = toward malignant)')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

## Summary

In this demo, we:

1. Loaded a real medical dataset with 30 tumor measurements
2. Explored feature distributions between malignant and benign cases
3. Split data with stratification to preserve class balance
4. Trained a logistic regression model with feature scaling
5. Evaluated using confusion matrix, classification report, and ROC/AUC
6. Interpreted which features most strongly predict malignancy

The model achieved high accuracy on this dataset, but in real clinical settings, we'd want to carefully consider the costs of false negatives (missing cancer) vs false positives (unnecessary follow-up tests).
