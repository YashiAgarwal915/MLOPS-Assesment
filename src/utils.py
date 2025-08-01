import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import os

def load_data():
    """Loads and splits the California Housing dataset."""
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def save_model(model, filepath):
    """Saves the given model to the specified path."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")