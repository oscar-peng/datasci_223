# Demo 1: Systematic Model Selection for Health Data

This demo shows how to implement systematic model selection using cross-validation techniques. We'll use a patient readmission prediction dataset to demonstrate different validation strategies and their importance in healthcare applications.

## Setup

First, we need to install and import the necessary libraries:

```python
# Install required packages
%pip install -q numpy pandas matplotlib seaborn scikit-learn

%reset -f

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
```

## Load and Prepare Data

We'll use a synthetic patient readmission dataset for this demo. In real healthcare applications, you would use actual patient data with proper privacy considerations.

```python
# Generate synthetic patient data
np.random.seed(42)
n_samples = 1000

# Create synthetic features
data = {
    'age': np.random.normal(65, 15, n_samples),
    'length_of_stay': np.random.exponential(5, n_samples),
    'num_medications': np.random.poisson(8, n_samples),
    'num_diagnoses': np.random.poisson(5, n_samples),
    'emergency_admission': np.random.binomial(1, 0.3, n_samples)
}

# Create target variable (readmission within 30 days)
# More complex relationship to make it interesting
readmission_prob = (
    0.1 +  # Base probability
    0.01 * (data['age'] - 65) +  # Age effect
    0.05 * data['length_of_stay'] +  # Length of stay effect
    0.02 * data['num_medications'] +  # Medication effect
    0.03 * data['num_diagnoses'] +  # Diagnosis effect
    0.1 * data['emergency_admission']  # Emergency admission effect
)
readmission_prob = np.clip(readmission_prob, 0, 1)  # Clip to valid probability range
data['readmission'] = np.random.binomial(1, readmission_prob)

# Convert to DataFrame
df = pd.DataFrame(data)

# Display basic statistics
print("Dataset Summary:")
print(df.describe())
print("\nClass Distribution:")
print(df['readmission'].value_counts(normalize=True))
```

## Visualize Data Distribution

Understanding the data distribution is crucial in healthcare applications:

```python
# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Patient Data Distribution', fontsize=16)

# Plot histograms for each feature
features = ['age', 'length_of_stay', 'num_medications', 'num_diagnoses', 'emergency_admission']
for idx, feature in enumerate(features):
    row = idx // 3
    col = idx % 3
    sns.histplot(data=df, x=feature, hue='readmission', ax=axes[row, col])
    axes[row, col].set_title(f'{feature} Distribution')

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.show()
```

## Simple K-Fold Cross Validation

Let's start with simple k-fold cross-validation:

```python
# Prepare features and target
X = df.drop('readmission', axis=1)
y = df['readmission']

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("5-Fold Cross Validation Scores:")
print(f"Mean Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Visualize CV scores
plt.figure(figsize=(8, 4))
plt.bar(range(1, 6), cv_scores)
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label='Mean')
plt.title('Cross-Validation Scores Across Folds')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Stratified K-Fold Cross Validation

In healthcare, we often deal with imbalanced datasets. Stratified k-fold helps maintain class distribution:

```python
# Perform stratified 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_strat = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("Stratified 5-Fold Cross Validation Scores:")
print(f"Mean Accuracy: {cv_scores_strat.mean():.3f} (+/- {cv_scores_strat.std() * 2:.3f})")

# Compare simple vs stratified
plt.figure(figsize=(10, 4))
plt.bar(range(1, 6), cv_scores, width=0.35, label='Simple K-Fold')
plt.bar(np.arange(1, 6) + 0.35, cv_scores_strat, width=0.35, label='Stratified K-Fold')
plt.title('Comparison of Cross-Validation Methods')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Nested Cross-Validation

For hyperparameter tuning, we use nested cross-validation to avoid data leakage:

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Outer cross-validation loop
outer_scores = []
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Inner cross-validation for hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=inner_cv,
        scoring='accuracy'
    )
    
    # Fit the model with best parameters
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    score = best_model.score(X_test, y_test)
    outer_scores.append(score)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test set score: {score:.3f}\n")

print("Nested Cross-Validation Results:")
print(f"Mean Accuracy: {np.mean(outer_scores):.3f} (+/- {np.std(outer_scores) * 2:.3f})")

# Visualize nested CV results
plt.figure(figsize=(8, 4))
plt.bar(range(1, 6), outer_scores)
plt.axhline(y=np.mean(outer_scores), color='r', linestyle='--', label='Mean')
plt.title('Nested Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Key Takeaways

1. **Cross-Validation Importance**
   - Helps assess model performance on unseen data
   - Reduces overfitting risk
   - Provides confidence intervals for performance metrics

2. **Stratification**
   - Maintains class distribution across folds
   - Crucial for imbalanced healthcare datasets
   - More reliable performance estimates

3. **Nested Cross-Validation**
   - Prevents data leakage in hyperparameter tuning
   - More reliable performance estimates
   - Important for model selection in healthcare

4. **Healthcare-Specific Considerations**
   - Need to consider class imbalance
   - Importance of interpretable models
   - Validation of clinical relevance 