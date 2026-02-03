"""
Classification Assignment: Digits vs. Letters on EMNIST

Complete the functions below to train and evaluate classification models
on the EMNIST dataset.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def load_and_prepare_data(subset_size=10000):
    """
    Load EMNIST data and prepare for classification.

    Creates a binary classification task: digits (1) vs letters (0).

    Parameters
    ----------
    subset_size : int
        Number of samples to use (for faster experimentation)

    Returns
    -------
    tuple
        X_train, X_test, X_val, y_train, y_test, y_val
    """
    from emnist import extract_training_samples, extract_test_samples

    # Load EMNIST balanced dataset (contains digits and letters)
    X_train_full, y_train_full = extract_training_samples("balanced")
    X_test_full, y_test_full = extract_test_samples("balanced")

    # Combine train and test for re-splitting
    X = np.vstack([X_train_full, X_test_full])
    y = np.hstack([y_train_full, y_test_full])

    # Take a subset for faster processing
    if subset_size and subset_size < len(X):
        indices = np.random.choice(len(X), subset_size, replace=False)
        X = X[indices]
        y = y[indices]

    # Create binary labels: digits (0-9 map to labels 0-9) vs letters
    # In EMNIST balanced: 0-9 are digits, 10+ are letters
    y_binary = (y < 10).astype(int)  # 1 for digits, 0 for letters

    # Flatten images from 28x28 to 784
    X_flat = X.reshape(X.shape[0], -1)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # Split: 60% train, 20% test, 20% validation
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_binary, test_size=0.4, random_state=42, stratify=y_binary
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, X_test, X_val, y_train, y_test, y_val


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
    # Hint: model.fit(X_train, y_train)
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


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    X_train, X_test, X_val, y_train, y_test, y_val = load_and_prepare_data(
        subset_size=5000
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Validation set: {X_val.shape}")

    # TODO: Train and evaluate your models here
    # Example:
    # from sklearn.linear_model import LogisticRegression
    # lr = LogisticRegression(max_iter=1000)
    # lr = train_model(lr, X_train, y_train)
    # results = evaluate_model(lr, X_test, y_test)
    # print(f"Logistic Regression Accuracy: {results['accuracy']:.4f}")
