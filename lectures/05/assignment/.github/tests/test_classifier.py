"""
Tests for Classification Assignment (Fashion-MNIST)

These tests verify that:
1. Data loading and preparation works correctly
2. Model training function is implemented
3. Model evaluation function returns expected structure
4. At least one model achieves reasonable accuracy (90%+ for binary task)
5. Output artifacts are generated correctly
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Output directory path
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")


class TestDataPreparation:
    """Test data loading and preparation functions."""

    def test_load_and_prepare_data_returns_correct_splits(self):
        """Test that load_and_prepare_data returns 4 arrays."""
        from classifier import load_and_prepare_data

        result = load_and_prepare_data()

        assert len(result) == 4, (
            "Should return 4 arrays (X_train, X_test, y_train, y_test)"
        )

    def test_data_shapes_are_consistent(self):
        """Test that X and y shapes are consistent within each split."""
        from classifier import load_and_prepare_data

        X_train, X_test, y_train, y_test = load_and_prepare_data()

        assert X_train.shape[0] == len(y_train), (
            "X_train and y_train should have same number of samples"
        )
        assert X_test.shape[0] == len(y_test), (
            "X_test and y_test should have same number of samples"
        )

    def test_labels_are_binary(self):
        """Test that labels are binary (0 or 1)."""
        from classifier import load_and_prepare_data

        _, _, y_train, y_test = load_and_prepare_data()

        all_labels = np.concatenate([y_train, y_test])
        unique_labels = np.unique(all_labels)

        assert set(unique_labels).issubset({0, 1}), (
            "Labels should be binary (0 for Clothing, 1 for Footwear)"
        )

    def test_bags_excluded(self):
        """Test that bags (label 8) are excluded from the dataset."""
        from classifier import load_and_prepare_data

        X_train, X_test, y_train, y_test = load_and_prepare_data()

        # Fashion-MNIST has 7000 bags (6000 train + 1000 test)
        # After exclusion, we should have 63000 samples (70000 - 7000)
        total_samples = len(y_train) + len(y_test)

        # Allow some tolerance for random split variations
        assert 60000 < total_samples < 65000, (
            f"Total samples should be ~63000 (bags excluded), got {total_samples}"
        )


class TestModelTraining:
    """Test model training function."""

    def test_train_model_returns_fitted_model(self):
        """Test that train_model returns a fitted model."""
        from classifier import load_and_prepare_data, train_model
        from sklearn.linear_model import LogisticRegression

        X_train, _, y_train, _ = load_and_prepare_data()

        # Use small subset for faster testing
        X_small = X_train[:500]
        y_small = y_train[:500]

        model = LogisticRegression(max_iter=100)
        fitted_model = train_model(model, X_small, y_small)

        assert fitted_model is not None, "train_model should return a model"
        assert hasattr(fitted_model, "predict"), (
            "Returned model should have predict method"
        )


class TestModelEvaluation:
    """Test model evaluation function."""

    def test_evaluate_model_returns_dict(self):
        """Test that evaluate_model returns a dictionary."""
        from classifier import (
            load_and_prepare_data,
            train_model,
            evaluate_model,
        )
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, y_train, y_test = load_and_prepare_data()

        # Use small subset for faster testing
        X_train_small = X_train[:500]
        y_train_small = y_train[:500]
        X_test_small = X_test[:100]
        y_test_small = y_test[:100]

        model = LogisticRegression(max_iter=100)
        model = train_model(model, X_train_small, y_train_small)

        result = evaluate_model(model, X_test_small, y_test_small)

        assert isinstance(result, dict), "evaluate_model should return a dictionary"

    def test_evaluate_model_contains_accuracy(self):
        """Test that evaluation result contains accuracy."""
        from classifier import (
            load_and_prepare_data,
            train_model,
            evaluate_model,
        )
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, y_train, y_test = load_and_prepare_data()

        X_train_small = X_train[:500]
        y_train_small = y_train[:500]
        X_test_small = X_test[:100]
        y_test_small = y_test[:100]

        model = LogisticRegression(max_iter=100)
        model = train_model(model, X_train_small, y_train_small)

        result = evaluate_model(model, X_test_small, y_test_small)

        assert "accuracy" in result, "Result should contain 'accuracy' key"
        assert 0 <= result["accuracy"] <= 1, "Accuracy should be between 0 and 1"

    def test_evaluate_model_contains_predictions(self):
        """Test that evaluation result contains predictions."""
        from classifier import (
            load_and_prepare_data,
            train_model,
            evaluate_model,
        )
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, y_train, y_test = load_and_prepare_data()

        X_train_small = X_train[:500]
        y_train_small = y_train[:500]
        X_test_small = X_test[:100]
        y_test_small = y_test[:100]

        model = LogisticRegression(max_iter=100)
        model = train_model(model, X_train_small, y_train_small)

        result = evaluate_model(model, X_test_small, y_test_small)

        assert "predictions" in result, "Result should contain 'predictions' key"
        assert len(result["predictions"]) == len(y_test_small), (
            "Predictions should match test set size"
        )


class TestModelPerformance:
    """Test that models achieve reasonable performance."""

    def test_model_achieves_minimum_accuracy(self):
        """Test that model achieves >90% accuracy on binary clothing/footwear task."""
        from classifier import (
            load_and_prepare_data,
            train_model,
            evaluate_model,
        )
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, y_train, y_test = load_and_prepare_data()

        # Use larger subset for realistic accuracy assessment
        X_train_subset = X_train[:5000]
        y_train_subset = y_train[:5000]

        model = LogisticRegression(max_iter=500)
        model = train_model(model, X_train_subset, y_train_subset)

        result = evaluate_model(model, X_test, y_test)

        assert result["accuracy"] > 0.90, (
            f"Model should achieve >90% accuracy on binary task, got {result['accuracy']:.2%}"
        )


class TestOutputArtifacts:
    """Test that required output artifacts exist and have correct format."""

    def test_output_directory_exists(self):
        """Test that output directory exists."""
        assert os.path.isdir(OUTPUT_DIR), (
            f"Output directory should exist at {OUTPUT_DIR}. "
            "Run 'python classifier.py' to generate outputs."
        )

    def test_cv_results_file_exists(self):
        """Test that cv_results.csv exists."""
        cv_path = os.path.join(OUTPUT_DIR, "cv_results.csv")
        assert os.path.exists(cv_path), (
            "cv_results.csv should exist in output/ directory"
        )

    def test_cv_results_has_correct_columns(self):
        """Test that cv_results.csv has required columns."""
        cv_path = os.path.join(OUTPUT_DIR, "cv_results.csv")
        if not os.path.exists(cv_path):
            pytest.skip("cv_results.csv not found")

        df = pd.read_csv(cv_path)
        required_columns = {"model", "fold", "score"}

        assert required_columns.issubset(set(df.columns)), (
            f"cv_results.csv should have columns: {required_columns}"
        )

    def test_cv_results_has_multiple_folds(self):
        """Test that cv_results.csv contains multiple folds."""
        cv_path = os.path.join(OUTPUT_DIR, "cv_results.csv")
        if not os.path.exists(cv_path):
            pytest.skip("cv_results.csv not found")

        df = pd.read_csv(cv_path)

        assert df["fold"].nunique() >= 3, (
            "cv_results.csv should contain at least 3 folds"
        )

    def test_test_results_file_exists(self):
        """Test that test_results.csv exists."""
        test_path = os.path.join(OUTPUT_DIR, "test_results.csv")
        assert os.path.exists(test_path), (
            "test_results.csv should exist in output/ directory"
        )

    def test_test_results_has_correct_columns(self):
        """Test that test_results.csv has required columns."""
        test_path = os.path.join(OUTPUT_DIR, "test_results.csv")
        if not os.path.exists(test_path):
            pytest.skip("test_results.csv not found")

        df = pd.read_csv(test_path)
        required_columns = {"model", "accuracy"}

        assert required_columns.issubset(set(df.columns)), (
            f"test_results.csv should have columns: {required_columns}"
        )

    def test_confusion_matrix_exists(self):
        """Test that confusion_matrix.png exists."""
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        assert os.path.exists(cm_path), (
            "confusion_matrix.png should exist in output/ directory"
        )

    def test_roc_curve_exists(self):
        """Test that roc_curve.png exists."""
        roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
        assert os.path.exists(roc_path), (
            "roc_curve.png should exist in output/ directory"
        )
