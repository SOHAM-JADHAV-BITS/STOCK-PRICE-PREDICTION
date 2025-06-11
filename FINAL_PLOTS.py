import model1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# 1. Make Predictions
print("Making predictions...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 2. Calculate Performance Metrics
print("\n=== MODEL PERFORMANCE METRICS ===")

# Training metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Training Metrics:")
print(f"MSE: {train_mse:.6f}")
print(f"RMSE: {train_rmse:.6f}")
print(f"MAE: {train_mae:.6f}")
print(f"R² Score: {train_r2:.6f}")

# Testing metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTesting Metrics:")
print(f"MSE: {test_mse:.6f}")
print(f"RMSE: {test_rmse:.6f}")
print(f"MAE: {test_mae:.6f}")
print(f"R² Score: {test_r2:.6f}")

# 3. Plot Training History
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()



plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Test Set)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))

# Plot last 200 points for better visualization
plot_size = min(200, len(y_test))
plt.plot(range(plot_size), y_test[-plot_size:], label='Actual', color='blue', linewidth=2)
plt.plot(range(plot_size), y_test_pred[-plot_size:], label='Predicted', color='red', linewidth=2)
plt.title('LSTM Model: Actual vs Predicted Stock Prices (Last 200 Test Points)')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Stock Price')
plt.legend()
plt.grid(True)
plt.show()

