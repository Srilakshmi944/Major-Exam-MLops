import joblib
import numpy as np
import os
from utils import load_data, evaluate_model
from sklearn.metrics import r2_score

# Load model
model_path = "model.joblib"
model = joblib.load(model_path)

# Print model size
model_size = os.path.getsize(model_path) / 1024  # in KB
print(f"Model file size before quantization: {model_size:.2f} KB")

# Save raw parameters
params = {"coef": model.coef_, "intercept": model.intercept_}
joblib.dump(params, "unquant_params.joblib")

# Custom per-coefficient min-max quantization to uint8
coef = model.coef_
intercept = model.intercept_

coef_min = coef.copy()
coef_max = coef.copy()
scales = np.ones_like(coef)

# Compute scale for each coefficient independently
for i in range(len(coef)):
    min_val = coef[i].min()
    max_val = coef[i].max()
    range_val = max(abs(min_val), abs(max_val))
    if range_val == 0:
        scales[i] = 1.0
    else:
        scales[i] = range_val / 127  # mapping to uint8 centered at 128

# Quantize each coefficient independently
q_coef = np.clip((coef / scales + 128), 0, 255).astype(np.uint8)

# Store quantization parameters
quantized = {
    "coef": q_coef,
    "scales": scales,
    "intercept": intercept  # Store exact intercept value
}
joblib.dump(quantized, "quant_params.joblib")

# Print quantized model size
quant_model_size = os.path.getsize("quant_params.joblib") / 1024  # in KB
print(f"Model file size after quantization: {quant_model_size:.2f} KB")

# Dequantize
q_coef = quantized["coef"].astype(np.float32)
deq_coef = (q_coef - 128) * scales

deq_intercept = quantized["intercept"]  # Restore exact intercept value

# Inference
sample_input = np.random.rand(len(deq_coef))
prediction = np.dot(sample_input, deq_coef) + deq_intercept
print("Sample Inference:", prediction)

# Evaluate
X_train, X_test, y_train, y_test = load_data()
y_pred = np.dot(X_test, deq_coef) + deq_intercept
_, r_square_before_quantization = evaluate_model(model, X_test, y_test)
print(f"R² Score before quantization: {r_square_before_quantization}")
r_square_after_quantization = r2_score(y_test, y_pred)
print(f"R² Score after quantization: {r_square_after_quantization}")
