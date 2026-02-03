# Assignment 5: Classification Showdown

In this assignment, you'll build and compare classification models on the **Fashion-MNIST** dataset. Your task is to classify images as either **clothing** or **footwear**.

## Dataset

Fashion-MNIST contains 70,000 grayscale images (28x28) of clothing items across 10 categories:

| Label | Category | Your Class |
|:---:|:---|:---|
| 0 | T-shirt/top | Clothing |
| 1 | Trouser | Clothing |
| 2 | Pullover | Clothing |
| 3 | Dress | Clothing |
| 4 | Coat | Clothing |
| 5 | Sandal | **Footwear** |
| 6 | Shirt | Clothing |
| 7 | Sneaker | **Footwear** |
| 8 | Bag | *Exclude* |
| 9 | Ankle boot | **Footwear** |

Your binary classification task:
- **Class 0 (Clothing)**: T-shirt, Trouser, Pullover, Dress, Coat, Shirt
- **Class 1 (Footwear)**: Sandal, Sneaker, Ankle boot
- **Exclude**: Bag (label 8) - filter these out during data preparation

## Learning Objectives

- Load and prepare image data for classification
- Apply train/test splitting with stratification
- Train multiple classification models (LogisticRegression, RandomForest)
- Use k-fold cross-validation to compare models
- Evaluate models using confusion matrix, classification report, and ROC/AUC
- Generate output artifacts documenting your results

## Your Tasks

### 1. Complete the Scaffold Code

Open `classifier.py` and implement the TODO functions:

**`train_model(model, X_train, y_train)`**
- Fit the model on the training data
- Return the fitted model

**`evaluate_model(model, X_test, y_test)`**
- Generate predictions using the fitted model
- Calculate accuracy
- Return a dictionary with `'accuracy'` and `'predictions'` keys

### 2. Run the Classification Pipeline

After implementing the functions, run the main script:

```bash
python classifier.py
```

This will:
1. Load Fashion-MNIST and prepare binary labels (excluding bags)
2. Train and evaluate multiple models using cross-validation
3. Generate output files in the `output/` directory

### 3. Generated Artifacts

Your code should produce these files in `output/`:

| File | Description |
|:---|:---|
| `cv_results.csv` | Cross-validation scores for each model and fold |
| `test_results.csv` | Final test set performance metrics |
| `confusion_matrix.png` | Confusion matrix for the best model |
| `roc_curve.png` | ROC curve comparing all models |

## Grading

The autograder checks:

1. **Functions work correctly**: `train_model` and `evaluate_model` return expected types
2. **Models achieve reasonable accuracy**: At least 90% on the test set (this is an easier task than 10-class)
3. **Output artifacts exist**: All required files in `output/` directory
4. **Cross-validation completed**: `cv_results.csv` contains results for multiple folds

## Hints

- The `load_and_prepare_data()` function handles data loading, filtering, and binary label creation
- Use `model.fit(X, y)` to train a scikit-learn model
- Use `model.predict(X)` to get predictions
- Use `accuracy_score(y_true, y_pred)` to calculate accuracy
- The main block shows example usage

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```
