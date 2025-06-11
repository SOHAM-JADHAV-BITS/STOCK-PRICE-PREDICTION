from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

data=pd.read_csv(r"C:\Users\SOHAM\OneDrive\Desktop\AMAZON STOCK PREDICTION\normalized data from 02-02-1998.csv")
features = ['close',"RSI_30","EMA_30","EMA_200"]
target='close'

scaled_data = data[features].values
window_size=100
X = []
y = []

for i in range(len(data) - window_size):
    X.append(data[features].iloc[i:i+window_size].values)
    y.append(data['close'].iloc[i+window_size])

X = np.array(X)  # shape: (num_samples, 100, num_features)
y = np.array(y)  # shape: (num_samples,)

'''X[0] = [
    [day0_close, day0_RSI_30, day0_SMA_50, day0_SMA_100, day0_SMA_200, day0_EMA_30, day0_EMA_200],
    [day1_close, day1_RSI_30, day1_SMA_50, day1_SMA_100, day1_SMA_200, day1_EMA_30, day1_EMA_200],
    ......
    '''

#df[features.values creates a list of list and the inner list has values like volume RSI_30 appended one after the other


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


model = keras.models.Sequential()
model.add(keras.layers.LSTM(128, return_sequences=True, input_shape=(window_size, 4)))  # x features
model.add(keras.layers.LSTM(80, return_sequences=False))
model.add(keras.layers.Dense(128,activation=gelu))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1))  # Output layer for regression
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# Make Predictions
print("Making predictions...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#  Calculate Performance Metrics
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


plot_size = min(200, len(y_test))
plt.plot(range(plot_size), y_test[-plot_size:], label='Actual', color='blue', linewidth=2)
plt.plot(range(plot_size), y_test_pred[-plot_size:], label='Predicted', color='red', linewidth=2)
plt.title('LSTM Model: Actual vs Predicted Stock Prices (Last 200 Test Points)')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Stock Price')
plt.legend()
plt.grid(True)
plt.show()



