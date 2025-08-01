import joblib
from utils import load_data

X_train, X_test, y_train, y_test = load_data()
model = joblib.load("model.joblib")

preds = model.predict(X_test[:5])
print("Sample Predictions:", preds)