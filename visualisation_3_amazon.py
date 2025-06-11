import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates


df = pd.read_csv(r"C:\Users\SOHAM\OneDrive\Desktop\AMAZON STOCK PREDICTION\1.csv")
df['date'] = pd.to_datetime(df['date'],dayfirst=False)
df = df.sort_values('date')

df['EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()
df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Set minor ticks at every month, and format as "Jan", "Feb", ...
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%b'))



plt.figure(figsize=(14, 7))
plt.plot(df.index, df['close'], label='Close Price')
plt.plot(df['EMA_30'], label='EMA 30-day', linestyle='--')
plt.plot(df['EMA_200'], label='EMA 200-day', linestyle='--')
plt.title('Stock Price with EMAs')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

df.to_csv(r"C:\Users\SOHAM\OneDrive\Desktop\AMAZON STOCK PREDICTION\1.csv", index=False)