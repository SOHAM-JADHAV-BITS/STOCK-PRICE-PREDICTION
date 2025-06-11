import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates


df = pd.read_csv(r"C:\Users\SOHAM\OneDrive\Desktop\AMAZON STOCK PREDICTION\1.csv")
df['date'] = pd.to_datetime(df['date'],dayfirst=False)
df = df.sort_values('date')

columns_to_check = ['SMA_50', 'SMA_100', 'SMA_200', 'EMA_30', 'EMA_200', 'RSI_30']
df_clean = df.dropna(subset=columns_to_check)


correlation_matrix = df_clean[columns_to_check].corr()

print(correlation_matrix)

df.to_csv(r"C:\Users\SOHAM\OneDrive\Desktop\AMAZON STOCK PREDICTION\1.csv", index=False)