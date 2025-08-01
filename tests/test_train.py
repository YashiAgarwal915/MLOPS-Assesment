import pytest
from sklearn.linear_model import LinearRegression
from src.utils import load_data
from src.train import model as trained_model # A way to access the model after running train.py

def test_data_loading():
    """Tests that data loads correctly."""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]

def test_model_training():
    """Tests the model training process and output."""
    # This test relies on train.py having been run
    # In the CI/CD pipeline, this will be the case
    from src.train import model as trained_model, r2
    assert isinstance(trained_model, LinearRegression)
    assert hasattr(trained_model, 'coef_') # Check if model is fitted
    assert r2 > 0.5 # Ensure R2 score meets a minimum threshold
