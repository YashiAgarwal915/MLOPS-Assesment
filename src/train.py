from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from .utils import load_data, save_model

print("--- Training Started ---")

# Load data
X_train, X_test, y_train, y_test = load_data()

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model and print metrics
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Model R2 Score: {r2:.4f}")
print(f"Model Mean Squared Error (Loss): {mse:.4f}")

# Save model
save_model(model, 'model/sklearn_model.joblib')

print("--- Training Finished ---")