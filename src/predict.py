# src/predict.py
import joblib
import numpy as np
import os

print("--- Container Verification Started ---")
model_path = 'model/sklearn_model.joblib'

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit(1)
    
model = joblib.load(model_path)
# The California Housing dataset has 8 features
dummy_sample = np.random.rand(1, 8) 
prediction = model.predict(dummy_sample)
print(f"Model loaded and test prediction successful: {prediction}")
print("--- Container Verification Finished ---")    