"""
Classification Assignment: Clothing vs Footwear on Fashion-MNIST

Complete the functions below to train and evaluate classification models
on the Fashion-MNIST dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


def load_and_prepare_data():
    """
    Load Fashion-MNIST data and prepare for binary classification.

    Creates a binary classification task:
    - Class 0 (Clothing): T-shirt, Trouser, Pullover, Dress, Coat, Shirt
    - Class 1 (Footwear): Sandal, Sneaker, Ankle boot
    - Excludes: Bag (label 8)

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test (scaled features)
    """
    from tensorflow.keras.datasets import fashion_mnist

    # Load Fashion-MNIST
    (X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

    # Combine for unified processing
    X_all = np.vstack([X_train_full, X_test_full])
    y_all = np.hstack([y_train_full, y_test_full])

    # Filter out bags (label 8)
    mask = y_all != 8
    X_filtered = X_all[mask]
    y_filtered = y_all[mask]

    # Create binary labels: footwear (5, 7, 9) = 1, clothing = 0
    footwear_labels = {5, 7, 9}  # Sandal, Sneaker, Ankle boot
    y_binary = np.array([1 if label in footwear_labels else 0 for label in y_filtered])

    # Flatten images from 28x28 to 784
    X_flat = X_filtered.reshape(X_filtered.shape[0], -1).astype(np.float32)

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    """
    Train a classification model.

    Parameters
    ----------
    model : sklearn estimator
        An unfitted scikit-learn compatible classifier
    X_train : array-like
        Training features
    y_train : array-like
        Training labels

    Returns
    -------
    model
        The fitted model
    """
    # TODO: Implement this function
    # Hint: Use model.fit(X_train, y_train) and return the fitted model
    pass


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model.

    Parameters
    ----------
    model : sklearn estimator
        A fitted scikit-learn compatible classifier
    X_test : array-like
        Test features
    y_test : array-like
        Test labels

    Returns
    -------
    dict
        Dictionary containing:
        - 'accuracy': float
        - 'predictions': array of predictions
    """
    # TODO: Implement this function
    # Hint: Use model.predict() and accuracy_score()
    pass


def run_cross_validation(models, X_train, y_train, cv=5):
    """
    Run cross-validation for multiple models.

    Parameters
    ----------
    models : dict
        Dictionary of model name -> model instance
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    cv : int
        Number of cross-validation folds

    Returns
    -------
    pd.DataFrame
        Cross-validation results with columns: model, fold, score
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='accuracy')
        for fold, score in enumerate(scores, 1):
            results.append({'model': name, 'fold': fold, 'score': score})

    return pd.DataFrame(results)


def plot_confusion_matrix(model, X_test, y_test, output_path):
    """Generate and save confusion matrix plot."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clothing', 'Footwear'])
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix: Best Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curves(models, X_test, y_test, output_path):
    """Generate and save ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Model Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading Fashion-MNIST data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Class distribution (train): Clothing={sum(y_train==0)}, Footwear={sum(y_train==1)}")

    # Define models to compare
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    # Step 1: Cross-validation on training data
    print("\nRunning cross-validation...")
    cv_results = run_cross_validation(models, X_train, y_train, cv=5)
    cv_results.to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
    print("Cross-validation results saved to output/cv_results.csv")

    # Print CV summary
    cv_summary = cv_results.groupby('model')['score'].agg(['mean', 'std'])
    print("\nCross-Validation Summary:")
    print(cv_summary)

    # Step 2: Train final models and evaluate on test set
    print("\nTraining final models...")
    test_results = []
    trained_models = {}

    for name, model in models.items():
        # Train on full training set
        fitted_model = train_model(model, X_train, y_train)

        if fitted_model is None:
            print(f"  {name}: train_model returned None - please implement the function!")
            continue

        trained_models[name] = fitted_model

        # Evaluate on test set
        result = evaluate_model(fitted_model, X_test, y_test)

        if result is None:
            print(f"  {name}: evaluate_model returned None - please implement the function!")
            continue

        test_results.append({
            'model': name,
            'accuracy': result['accuracy'],
            'auc': roc_auc_score(y_test, fitted_model.predict_proba(X_test)[:, 1])
        })
        print(f"  {name}: Accuracy = {result['accuracy']:.4f}")

    # Save test results
    if test_results:
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)
        print("\nTest results saved to output/test_results.csv")

        # Find best model
        best_model_name = test_df.loc[test_df['accuracy'].idxmax(), 'model']
        best_model = trained_models[best_model_name]

        # Generate plots
        print("\nGenerating plots...")
        plot_confusion_matrix(best_model, X_test, y_test,
                              os.path.join(output_dir, "confusion_matrix.png"))
        print("  Confusion matrix saved to output/confusion_matrix.png")

        plot_roc_curves(trained_models, X_test, y_test,
                        os.path.join(output_dir, "roc_curve.png"))
        print("  ROC curves saved to output/roc_curve.png")

        print("\nDone! Check the output/ directory for results.")
    else:
        print("\nNo results generated. Make sure to implement train_model and evaluate_model.")
