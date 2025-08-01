
import joblib
import numpy as np
from src.utils import load_data, save_model

print("--- Quantization Started ---")

# Load the trained model
sklearn_model = joblib.load('model/sklearn_model.joblib')
X_train, X_test, y_train, y_test = load_data()

# Extract, save, and print info for raw parameters
coef = sklearn_model.coef_
intercept = sklearn_model.intercept_
unquant_params = {'coef': coef, 'intercept': intercept}
save_model(unquant_params, 'model/unquant_params.joblib')

# Manual Quantization (with epsilon for stability)
def quantize(params):
    scale = (np.max(params) - np.min(params)) / 255.0
    if scale == 0:
        scale = 1e-8 # Add epsilon to prevent division by zero
    zero_point = -np.min(params) / scale
    quantized_params = np.round((params / scale) + zero_point).astype(np.uint8)
    return quantized_params, scale, zero_point

quantized_coef, scale_coef, zp_coef = quantize(coef)
quantized_intercept, scale_intercept, zp_intercept = quantize(np.array([intercept]))

quant_params = {
    'quantized_coef': quantized_coef, 'scale_coef': scale_coef, 'zp_coef': zp_coef,
    'quantized_intercept': quantized_intercept, 'scale_intercept': scale_intercept, 'zp_intercept': zp_intercept
}
save_model(quant_params, 'model/quant_params.joblib')

# De-quantization for inference
def dequantize(quantized_params, scale, zero_point):
    return (quantized_params.astype(np.float32) - zero_point) * scale

dequant_coef = dequantize(quantized_coef, scale_coef, zp_coef)
dequant_intercept = dequantize(quantized_intercept, scale_intercept, zp_intercept)

# Perform inference with de-quantized weights
y_pred_quant = X_test @ dequant_coef.T + dequant_intercept
print("Inference with de-quantized weights completed.")

print("--- Quantization Finished ---")