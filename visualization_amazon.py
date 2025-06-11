import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

df = pd.read_csv(r"C:\Users\SOHAM\OneDrive\Desktop\portfolio management\1.csv")

print(df.columns)


df['date'] = pd.to_datetime(df['date'], dayfirst=False)
df = df.sort_values('date')
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['SMA_100'] = df['close'].rolling(window=100).mean()
df['SMA_200'] = df['close'].rolling(window=200).mean()

df['delta'] = df['close'].diff()
df['gains'] = df['delta'].where(df['delta'] > 0, 0)
df['loss'] = -df['delta'].where(df['delta'] < 0, 0)

df['avg_gain'] = df['gains'].rolling(window=30).mean()
df['avg_loss'] = df['loss'].rolling(window=30).mean()

df['rs'] = df['avg_gain'] / df['avg_loss']
df['RSI_30'] = 100 - (100 / (1 + df['rs']))


df.to_csv(r"C:\Users\SOHAM\OneDrive\Desktop\portfolio management\1.csv"
, index=False)









print(df)


plt.figure(figsize=(14, 7))


plt.plot(df['date'], df['SMA_50'], label='SMA 50')
plt.plot(df['date'], df['SMA_100'], label='SMA 100')
plt.plot(df['date'], df['SMA_200'], label='SMA 200')


plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


plt.gcf().autofmt_xdate()


plt.title('Amazon Stock SMAs Over 20 Years')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





plt.figure(figsize=(20,20))
plt.plot(df['date'],df['RSI_30'],label="RSI_30")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gcf().autofmt_xdate()


plt.title('Amazon Stock RSI Over 20 Years')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()







