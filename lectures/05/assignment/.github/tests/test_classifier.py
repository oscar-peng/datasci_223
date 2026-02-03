"""
Tests for Classification Assignment

These tests verify that:
1. Data loading and preparation works correctly
2. Model training function is implemented
3. Model evaluation function returns expected structure
4. At least one model achieves reasonable accuracy
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestDataPreparation:
    """Test data loading and preparation functions."""

    def test_load_and_prepare_data_returns_correct_splits(self):
        """Test that load_and_prepare_data returns 6 arrays."""
        from classifier import load_and_prepare_data

        result = load_and_prepare_data(subset_size=1000)

        assert len(result) == 6, (
            "Should return 6 arrays (X_train, X_test, X_val, y_train, y_test, y_val)"
        )

    def test_data_shapes_are_consistent(self):
        """Test that X and y shapes are consistent within each split."""
        from classifier import load_and_prepare_data

        X_train, X_test, X_val, y_train, y_test, y_val = load_and_prepare_data(
            subset_size=1000
        )

        assert X_train.shape[0] == len(y_train), (
            "X_train and y_train should have same number of samples"
        )
        assert X_test.shape[0] == len(y_test), (
            "X_test and y_test should have same number of samples"
        )
        assert X_val.shape[0] == len(y_val), (
            "X_val and y_val should have same number of samples"
        )

    def test_labels_are_binary(self):
        """Test that labels are binary (0 or 1)."""
        from classifier import load_and_prepare_data

        _, _, _, y_train, y_test, y_val = load_and_prepare_data(
            subset_size=1000
        )

        all_labels = np.concatenate([y_train, y_test, y_val])
        unique_labels = np.unique(all_labels)

        assert set(unique_labels).issubset({0, 1}), (
            "Labels should be binary (0 or 1)"
        )


class TestModelTraining:
    """Test model training function."""

    def test_train_model_returns_fitted_model(self):
        """Test that train_model returns a fitted model."""
        from classifier import load_and_prepare_data, train_model
        from sklearn.linear_model import LogisticRegression

        X_train, _, _, y_train, _, _ = load_and_prepare_data(subset_size=500)

        model = LogisticRegression(max_iter=100)
        fitted_model = train_model(model, X_train, y_train)

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

        X_train, X_test, _, y_train, y_test, _ = load_and_prepare_data(
            subset_size=500
        )

        model = LogisticRegression(max_iter=100)
        model = train_model(model, X_train, y_train)

        result = evaluate_model(model, X_test, y_test)

        assert isinstance(result, dict), (
            "evaluate_model should return a dictionary"
        )

    def test_evaluate_model_contains_accuracy(self):
        """Test that evaluation result contains accuracy."""
        from classifier import (
            load_and_prepare_data,
            train_model,
            evaluate_model,
        )
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, _, y_train, y_test, _ = load_and_prepare_data(
            subset_size=500
        )

        model = LogisticRegression(max_iter=100)
        model = train_model(model, X_train, y_train)

        result = evaluate_model(model, X_test, y_test)

        assert "accuracy" in result, "Result should contain 'accuracy' key"
        assert 0 <= result["accuracy"] <= 1, (
            "Accuracy should be between 0 and 1"
        )

    def test_evaluate_model_contains_predictions(self):
        """Test that evaluation result contains predictions."""
        from classifier import (
            load_and_prepare_data,
            train_model,
            evaluate_model,
        )
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, _, y_train, y_test, _ = load_and_prepare_data(
            subset_size=500
        )

        model = LogisticRegression(max_iter=100)
        model = train_model(model, X_train, y_train)

        result = evaluate_model(model, X_test, y_test)

        assert "predictions" in result, (
            "Result should contain 'predictions' key"
        )
        assert len(result["predictions"]) == len(y_test), (
            "Predictions should match test set size"
        )


class TestModelPerformance:
    """Test that models achieve reasonable performance."""

    def test_model_achieves_minimum_accuracy(self):
        """Test that at least one model achieves >70% accuracy (sanity check)."""
        from classifier import (
            load_and_prepare_data,
            train_model,
            evaluate_model,
        )
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, _, y_train, y_test, _ = load_and_prepare_data(
            subset_size=2000
        )

        model = LogisticRegression(max_iter=500)
        model = train_model(model, X_train, y_train)

        result = evaluate_model(model, X_test, y_test)

        assert result["accuracy"] > 0.70, (
            f"Model should achieve >70% accuracy, got {result['accuracy']:.2%}"
        )


class TestResultsDocumentation:
    """Test that results documentation exists."""

    def test_results_file_exists(self):
        """Test that RESULTS.md file exists."""
        results_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "RESULTS.md"
        )

        assert os.path.exists(results_path), "RESULTS.md file should exist"

    def test_results_file_not_empty(self):
        """Test that RESULTS.md has content."""
        results_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "RESULTS.md"
        )

        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                content = f.read().strip()
            assert len(content) > 100, (
                "RESULTS.md should have meaningful content"
            )
