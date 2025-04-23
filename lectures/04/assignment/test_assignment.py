import os
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import student solutions
# Note: These imports will fail until students create these files
try:
    import part1_exploration as student_part1
except ImportError:
    # Check for notebook instead
    if os.path.exists('part1_exploration.ipynb'):
        print("Found notebook for Part 1, but couldn't import module. Will test functions directly.")
    else:
        print("Warning: part1_exploration.py or part1_exploration.ipynb not found")

try:
    import part2_modeling as student_part2
except ImportError:
    # Check for notebook instead
    if os.path.exists('part2_modeling.ipynb'):
        print("Found notebook for Part 2, but couldn't import module. Will test functions directly.")
    else:
        print("Warning: part2_modeling.py or part2_modeling.ipynb not found")

try:
    import part3_advanced as student_part3
except ImportError:
    # Check for notebook instead
    if os.path.exists('part3_advanced.ipynb'):
        print("Found notebook for Part 3, but couldn't import module. Will test functions directly.")
    else:
        print("Warning: part3_advanced.py or part3_advanced.ipynb not found")


# Helper functions for testing
def get_function_from_notebook(notebook_path, function_name):
    """Extract a function from a Jupyter notebook."""
    import nbformat
    from nbformat import read
    import re
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = read(f, as_version=4)
    
    # Find the cell with the function definition
    function_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if re.search(f"def {function_name}\s*\(", cell.source):
                function_code = cell.source
                break
    
    if not function_code:
        raise ValueError(f"Function {function_name} not found in notebook {notebook_path}")
    
    # Execute the function definition
    namespace = {}
    exec(function_code, namespace)
    
    # Return the function
    return namespace[function_name]


# Test fixtures
@pytest.fixture
def data_dir():
    """Fixture for data directory path."""
    return Path("data")


@pytest.fixture
def sample_data():
    """Fixture for sample data for testing."""
    # Create a small sample dataset for testing
    # This is a simplified version of what the actual data might look like
    timestamps = pd.date_range(start='2023-01-01', periods=100, freq='1s')
    data = pd.DataFrame({
        'timestamp': timestamps,
        'EDA': np.random.normal(2.0, 0.5, 100),
        'BVP': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
        'HR': 70 + 10 * np.sin(np.linspace(0, 2, 100)) + np.random.normal(0, 2, 100),
        'TEMP': 37 + 0.2 * np.sin(np.linspace(0, 1, 100)) + np.random.normal(0, 0.05, 100),
        'condition': ['stress'] * 50 + ['exercise'] * 50,
        'subject_id': np.repeat(np.arange(1, 6), 20)
    })
    return data


# Part 1: Data Exploration and Preprocessing Tests
class TestPart1:
    def test_data_loading(self, data_dir, sample_data):
        """Test that data loading function works correctly."""
        # Try to get the load_data function
        load_data = None
        
        if 'student_part1' in globals():
            if hasattr(student_part1, 'load_data'):
                load_data = student_part1.load_data
        elif os.path.exists('part1_exploration.ipynb'):
            try:
                load_data = get_function_from_notebook('part1_exploration.ipynb', 'load_data')
            except:
                pass
        
        if load_data is None:
            pytest.skip("load_data function not found")
        
        # If data directory doesn't exist, use sample data
        if not data_dir.exists():
            # Mock the function to return sample data
            def mock_load_data(*args, **kwargs):
                return sample_data
            
            data = mock_load_data()
        else:
            # Use the actual function
            data = load_data()
        
        # Test that data is loaded correctly
        assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame"
        assert len(data) > 0, "Data should not be empty"
        
        # Check for required columns
        required_columns = ['timestamp', 'EDA', 'BVP', 'HR', 'TEMP']
        for col in required_columns:
            assert col in data.columns, f"Missing required column: {col}"
        
        # Check that timestamp is datetime
        assert pd.api.types.is_datetime64_dtype(data['timestamp']) or isinstance(data['timestamp'].iloc[0], pd.Timestamp), \
            "timestamp should be datetime type"

    def test_preprocessing(self, sample_data):
        """Test data preprocessing function."""
        # Try to get the preprocess_data function
        preprocess_data = None
        
        if 'student_part1' in globals():
            if hasattr(student_part1, 'preprocess_data'):
                preprocess_data = student_part1.preprocess_data
        elif os.path.exists('part1_exploration.ipynb'):
            try:
                preprocess_data = get_function_from_notebook('part1_exploration.ipynb', 'preprocess_data')
            except:
                pass
        
        if preprocess_data is None:
            pytest.skip("preprocess_data function not found")
        
        # Add some NaN values to sample data
        data_with_nans = sample_data.copy()
        data_with_nans.loc[10:15, 'EDA'] = np.nan
        data_with_nans.loc[20:25, 'HR'] = np.nan
        
        # Preprocess the data
        processed_data = preprocess_data(data_with_nans)
        
        # Check that NaNs are handled
        assert processed_data['EDA'].isna().sum() == 0, "NaN values should be handled in EDA column"
        assert processed_data['HR'].isna().sum() == 0, "NaN values should be handled in HR column"
        
        # Check that data is still a DataFrame with the same columns
        assert isinstance(processed_data, pd.DataFrame), "Processed data should be a pandas DataFrame"
        assert set(processed_data.columns) >= set(data_with_nans.columns), "Processed data should have at least the same columns"

    def test_visualization(self, sample_data):
        """Test visualization functions."""
        # Try to get the plot_time_series function
        plot_time_series = None
        
        if 'student_part1' in globals():
            if hasattr(student_part1, 'plot_time_series'):
                plot_time_series = student_part1.plot_time_series
        elif os.path.exists('part1_exploration.ipynb'):
            try:
                plot_time_series = get_function_from_notebook('part1_exploration.ipynb', 'plot_time_series')
            except:
                pass
        
        if plot_time_series is None:
            pytest.skip("plot_time_series function not found")
        
        # Create a figure to test plotting
        fig, ax = plt.subplots()
        
        # Call the plotting function
        plot_time_series(sample_data, 'HR', ax=ax)
        
        # Check that the plot has data
        assert len(ax.lines) > 0, "Plot should have at least one line"
        assert ax.get_xlabel() != "", "Plot should have an x-label"
        assert ax.get_ylabel() != "", "Plot should have a y-label"
        
        # Close the figure to avoid memory leaks
        plt.close(fig)


# Part 2: Time Series Modeling Tests
class TestPart2:
    def test_feature_engineering(self, sample_data):
        """Test feature engineering function."""
        # Try to get the extract_features function
        extract_features = None
        
        if 'student_part2' in globals():
            if hasattr(student_part2, 'extract_features'):
                extract_features = student_part2.extract_features
        elif os.path.exists('part2_modeling.ipynb'):
            try:
                extract_features = get_function_from_notebook('part2_modeling.ipynb', 'extract_features')
            except:
                pass
        
        if extract_features is None:
            pytest.skip("extract_features function not found")
        
        # Extract features
        features = extract_features(sample_data)
        
        # Check that features is a DataFrame
        assert isinstance(features, pd.DataFrame), "Features should be a pandas DataFrame"
        
        # Check that we have at least some features
        assert features.shape[1] >= 5, "Should extract at least 5 features"
        
        # Check that features don't have NaN values
        assert features.isna().sum().sum() == 0, "Features should not have NaN values"

    def test_model_training(self, sample_data):
        """Test model training function."""
        # Try to get the train_model function
        train_model = None
        
        if 'student_part2' in globals():
            if hasattr(student_part2, 'train_model'):
                train_model = student_part2.train_model
        elif os.path.exists('part2_modeling.ipynb'):
            try:
                train_model = get_function_from_notebook('part2_modeling.ipynb', 'train_model')
            except:
                pass
        
        if train_model is None:
            pytest.skip("train_model function not found")
        
        # Create a simple feature set
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        y = np.random.normal(0, 1, 100)
        
        # Train the model
        model = train_model(X, y)
        
        # Check that model has a predict method
        assert hasattr(model, 'predict'), "Model should have a predict method"
        
        # Check that model can make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y), "Predictions should have the same length as y"

    def test_model_evaluation(self, sample_data):
        """Test model evaluation function."""
        # Try to get the evaluate_model function
        evaluate_model = None
        
        if 'student_part2' in globals():
            if hasattr(student_part2, 'evaluate_model'):
                evaluate_model = student_part2.evaluate_model
        elif os.path.exists('part2_modeling.ipynb'):
            try:
                evaluate_model = get_function_from_notebook('part2_modeling.ipynb', 'evaluate_model')
            except:
                pass
        
        if evaluate_model is None:
            pytest.skip("evaluate_model function not found")
        
        # Create simple test data
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5.2])
        
        # Evaluate the model
        metrics = evaluate_model(y_true, y_pred)
        
        # Check that metrics is a dictionary
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        
        # Check that it has some common metrics
        common_metrics = ['mae', 'mse', 'rmse', 'r2']
        for metric in common_metrics:
            assert any(metric in key.lower() for key in metrics.keys()), f"Metrics should include {metric}"
        
        # Check that metrics have reasonable values
        for value in metrics.values():
            assert isinstance(value, (int, float)), "Metric values should be numbers"
            assert not np.isnan(value), "Metric values should not be NaN"


# Part 3: Advanced Analysis Tests
class TestPart3:
    def test_signal_processing(self, sample_data):
        """Test signal processing function."""
        # Try to get the apply_filter function
        apply_filter = None
        
        if 'student_part3' in globals():
            if hasattr(student_part3, 'apply_filter'):
                apply_filter = student_part3.apply_filter
        elif os.path.exists('part3_advanced.ipynb'):
            try:
                apply_filter = get_function_from_notebook('part3_advanced.ipynb', 'apply_filter')
            except:
                pass
        
        if apply_filter is None:
            pytest.skip("apply_filter function not found")
        
        # Apply filter to sample data
        filtered_data = apply_filter(sample_data, 'BVP')
        
        # Check that filtered data is returned
        assert isinstance(filtered_data, pd.DataFrame) or isinstance(filtered_data, np.ndarray), \
            "Filtered data should be a DataFrame or numpy array"
        
        if isinstance(filtered_data, pd.DataFrame):
            assert len(filtered_data) == len(sample_data), "Filtered data should have the same length as input data"
        else:
            assert len(filtered_data) == len(sample_data), "Filtered data should have the same length as input data"

    def test_advanced_features(self, sample_data):
        """Test advanced feature extraction function."""
        # Try to get the extract_advanced_features function
        extract_advanced_features = None
        
        if 'student_part3' in globals():
            if hasattr(student_part3, 'extract_advanced_features'):
                extract_advanced_features = student_part3.extract_advanced_features
        elif os.path.exists('part3_advanced.ipynb'):
            try:
                extract_advanced_features = get_function_from_notebook('part3_advanced.ipynb', 'extract_advanced_features')
            except:
                pass
        
        if extract_advanced_features is None:
            pytest.skip("extract_advanced_features function not found")
        
        # Extract advanced features
        advanced_features = extract_advanced_features(sample_data)
        
        # Check that advanced_features is a DataFrame
        assert isinstance(advanced_features, pd.DataFrame), "Advanced features should be a pandas DataFrame"
        
        # Check that we have at least some features
        assert advanced_features.shape[1] >= 3, "Should extract at least 3 advanced features"
        
        # Check that features don't have NaN values
        assert advanced_features.isna().sum().sum() == 0, "Advanced features should not have NaN values"

    def test_condition_comparison(self, sample_data):
        """Test condition comparison function."""
        # Try to get the compare_conditions function
        compare_conditions = None
        
        if 'student_part3' in globals():
            if hasattr(student_part3, 'compare_conditions'):
                compare_conditions = student_part3.compare_conditions
        elif os.path.exists('part3_advanced.ipynb'):
            try:
                compare_conditions = get_function_from_notebook('part3_advanced.ipynb', 'compare_conditions')
            except:
                pass
        
        if compare_conditions is None:
            pytest.skip("compare_conditions function not found")
        
        # Compare conditions
        comparison_results = compare_conditions(sample_data, 'stress', 'exercise')
        
        # Check that comparison_results is a dictionary or DataFrame
        assert isinstance(comparison_results, (dict, pd.DataFrame)), \
            "Comparison results should be a dictionary or DataFrame"
        
        # If it's a dictionary, check that it has some results
        if isinstance(comparison_results, dict):
            assert len(comparison_results) > 0, "Comparison results should not be empty"
        # If it's a DataFrame, check that it has some rows
        else:
            assert len(comparison_results) > 0, "Comparison results should not be empty"


if __name__ == "__main__":
    pytest.main(["-v"])