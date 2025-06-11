import math
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv(r"C:\Users\SOHAM\OneDrive\Desktop\AMAZON STOCK PREDICTION\2.csv")
df['date'] = pd.to_datetime(df['date'],dayfirst=False)
df = df.sort_values('date')

columns_to_scale = ['RSI_30', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_30', 'EMA_200', 'close', 'volume']


df_scaled = df.dropna(subset=columns_to_scale).copy()

# Apply MinMaxScaler
scaler = MinMaxScaler()
df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

# (Optional) Save to new CSV
df_scaled.to_csv("2_scaled.csv", index=False)

print("Normalization complete. Scaled data saved to '2_scaled.csv'.")
