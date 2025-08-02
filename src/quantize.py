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

# Quantize using only uint8
max_val = np.max(np.abs(model.coef_))
scale_factor = 255 / max_val if max_val != 0 else 1
q_coef = np.clip((model.coef_ * scale_factor), 0, 255).astype(np.uint8)
q_intercept = np.clip((model.intercept_ * scale_factor), 0, 255).astype(np.uint8)
quantized = {"coef": q_coef, "intercept": q_intercept, "scale_factor": scale_factor}
joblib.dump(quantized, "quant_params.joblib")

# Print quantized model size
quant_model_size = os.path.getsize("quant_params.joblib") / 1024  # in KB
print(f"Model file size after quantization: {quant_model_size:.2f} KB")

# Dequantize and inference
deq_coef = q_coef.astype(np.float32) / scale_factor
deq_intercept = q_intercept.astype(np.float32) / scale_factor
sample_input = np.random.rand(len(deq_coef))
prediction = np.dot(sample_input, deq_coef) + deq_intercept
print("Sample Inference:", prediction)

# Calculate R² scores
X_train, X_test, y_train, y_test = load_data()
y_pred = np.dot(X_test, deq_coef) + deq_intercept
_, r_square_before_quantization = evaluate_model(model, X_test, y_test)

print(f"R² Score before quantization: {r_square_before_quantization}")
r_square_after_quantization = r2_score(y_test, y_pred)
print(f"R² Score after quantization: {r_square_after_quantization}")
