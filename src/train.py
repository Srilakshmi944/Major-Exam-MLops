from utils import load_data, train_model, evaluate_model
import joblib

X_train, X_test, y_train, y_test = load_data()
model = train_model(X_train, y_train)
mse, r2 = evaluate_model(model, X_test, y_test)

print(f"R² Score: {r2}")
print(f"Loss (MSE): {mse}")

joblib.dump(model, "model.joblib")