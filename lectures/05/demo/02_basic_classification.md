# Demo 2: Basic Classification with Synthetic Health Data 🏥

## Learning Objectives 🎯

By the end of this demo, you will be able to:

1. Generate and visualize synthetic health data
2. Compare different classification algorithms
3. Evaluate model performance using various metrics
4. Interpret model decisions using feature importance

```python
%pip install -r requirements.txt --quiet
```

## Setup and Imports 🛠️

```python
# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

# Model evaluation
from sklearn.metrics import (confusion_matrix, classification_report,
                           roc_curve, auc)

# Set random seed for reproducibility
np.random.seed(42)
```

## 1. Generate Synthetic Health Data 🧬

Let's create a synthetic dataset that mimics health measurements for diabetes risk prediction:

```python
# Generate synthetic data
n_samples = 1000

# Generate features with more realistic distributions and correlations
age = np.random.normal(50, 15, n_samples)  # Age: mean=50, std=15
bmi = np.random.normal(28, 5, n_samples)   # BMI: mean=28, std=5
glucose = np.random.normal(100, 25, n_samples)  # Glucose: mean=100, std=25
bp = np.random.normal(130, 15, n_samples)  # Blood Pressure: mean=130, std=15

# Add some noise and interactions
noise = np.random.normal(0, 0.2, n_samples)  # Random noise
age_bmi_interaction = (age/50) * (bmi/25) * 0.1  # Interaction effect
glucose_bp_interaction = (glucose/100) * (bp/120) * 0.1  # Another interaction

# Create more complex, noisy risk score
risk_score = (
    0.2 * stats.norm.cdf((age - 50) / 15) +  # Smoother age effect
    0.2 * stats.norm.cdf((bmi - 25) / 5) +   # Smoother BMI effect
    0.3 * stats.norm.cdf((glucose - 110) / 25) +  # Smoother glucose effect
    0.1 * stats.norm.cdf((bp - 130) / 15) +      # Smoother BP effect
    0.1 * age_bmi_interaction +  # Add interaction effects
    0.1 * glucose_bp_interaction +
    noise  # Add random noise
)

# Convert to binary outcome with some randomness
probability = 1 / (1 + np.exp(-(risk_score - np.mean(risk_score)) * 3))  # Logistic function
y = (probability > np.random.uniform(0.3, 0.7, n_samples)).astype(int)  # Random threshold

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'BMI': bmi,
    'Glucose': glucose,
    'BloodPressure': bp,
    'DiabetesRisk': y
})

print("\nDataset Shape:", df.shape)
print("\nClass Distribution:")
print(df['DiabetesRisk'].value_counts(normalize=True))

# Visualize feature distributions by class
plt.figure(figsize=(12, 8))
for i, feature in enumerate(['Age', 'BMI', 'Glucose', 'BloodPressure']):
    plt.subplot(2, 2, i+1)
    sns.kdeplot(data=df, x=feature, hue='DiabetesRisk')
    plt.title(f'{feature} Distribution by Risk')
plt.tight_layout()
plt.show()

# Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0)
plt.title('Feature Correlations')
plt.show()
```

## 2. Prepare Data for Modeling 📊

Split first to avoid data leakage, then fit the scaler only on the training set. Use `stratify=y` so both splits keep similar class proportions.

```python
# Split features and target
X = df.drop('DiabetesRisk', axis=1)
y = df['DiabetesRisk']

# Split FIRST (before scaling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit scaler on training data only, then transform both train and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("Training set shape:", X_train_scaled_df.shape)
print("Test set shape:", X_test_scaled_df.shape)
```

## 3. Cross-Validation for Model Selection 🔄

Before picking a final model, use **cross-validation** to get more reliable performance estimates. A single train/test split can be misleading if the split happens to be particularly easy or hard.

```python
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
}

# Use the same CV splitter for fair comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validate each model on the TRAINING data only
print("Cross-Validation Results (5-fold on training set):\n")
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled_df, y_train, cv=cv, scoring='roc_auc')
    cv_results[name] = scores
    print(f"{name}:")
    print(f"  AUC per fold: {scores.round(3)}")
    print(f"  Mean AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})\n")
```

## 4. Train Final Models and Evaluate on Test Set 🤖

Now train on the full training set and evaluate on the held-out test set:

```python
# Train and evaluate each model on the test set
results = {}
plt.figure(figsize=(10, 8))

for name, model in models.items():
    # Train model on full training set
    model.fit(X_train_scaled_df, y_train)

    # Get predictions and probabilities
    y_pred = model.predict(X_test_scaled_df)
    y_prob = model.predict_proba(X_test_scaled_df)[:, 1]

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # Store results
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_prob,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.show()

# Print classification reports
for name, result in results.items():
    print(f"\n{name} Results:")
    print(result['classification_report'])
```

## 5. Feature Importance Analysis 🔍

Let's examine which features are most important for each model:

```python
# Get feature importance for each model
plt.figure(figsize=(12, 4))

for i, (name, result) in enumerate(results.items()):
    model = result['model']
    plt.subplot(1, 3, i+1)

    if name == 'Logistic Regression':
        importance = np.abs(model.coef_[0])
    else:
        importance = model.feature_importances_

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=True)

    # Plot horizontal bar chart
    sns.barplot(data=importance_df, y='feature', x='importance')
    plt.title(f'{name}\nFeature Importance')

plt.tight_layout()
plt.show()
```

## 6. SHAP Values for Model Interpretation 🔬

SHAP (SHapley Additive exPlanations) provides more detailed feature importance by showing how each feature contributes to individual predictions.

```python
# Use SHAP to explain the XGBoost model
xgb_model = results['XGBoost']['model']

# Create SHAP explainer for tree-based model
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled_df)

# Summary plot: shows feature importance and direction of effect
print("SHAP Summary Plot (XGBoost):")
shap.summary_plot(shap_values, X_test_scaled_df, plot_type="dot", show=False)
plt.tight_layout()
plt.show()

# Bar plot: global feature importance
print("\nSHAP Feature Importance (bar):")
shap.summary_plot(shap_values, X_test_scaled_df, plot_type="bar", show=False)
plt.tight_layout()
plt.show()
```

The SHAP summary plot shows:
- **Position on x-axis**: How much each feature pushed the prediction higher (right) or lower (left)
- **Color**: The feature value (red = high, blue = low)
- **Spread**: Features with wider spread have more variable impact across patients

## 7. Interactive Prediction Function 🎯

Let's create a function to make predictions for new patients:

```python
def predict_diabetes_risk(age, bmi, glucose, bp, model=models['Random Forest']):
    """Make diabetes risk prediction for a new patient."""
    # Create feature array
    X_new = np.array([[age, bmi, glucose, bp]])
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    # Get prediction and probability
    prediction = model.predict(X_new_scaled)[0]
    probability = model.predict_proba(X_new_scaled)[0][1]
    
    print(f"\nPatient Information:")
    print(f"Age: {age} years")
    print(f"BMI: {bmi:.1f}")
    print(f"Glucose: {glucose} mg/dL")
    print(f"Blood Pressure: {bp} mmHg")
    print(f"\nPrediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
    print(f"Risk Probability: {probability:.1%}")
    
    return prediction, probability

# Example prediction
predict_diabetes_risk(
    age=65,
    bmi=32,
    glucose=140,
    bp=150
)
```

Cross-validation gave us confidence in model performance before committing to a final evaluation; the ROC curves and classification reports show how each model trades off true positives and false positives; both feature importance plots and SHAP values reveal which inputs (e.g., glucose, BMI) the model relied on most. SHAP additionally shows the *direction* of effect—for example, high glucose pushing predictions toward "high risk."
