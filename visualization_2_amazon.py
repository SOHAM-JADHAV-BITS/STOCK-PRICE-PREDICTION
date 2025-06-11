import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

df = pd.read_csv(r"C:\Users\SOHAM\OneDrive\Desktop\AMAZON STOCK PREDICTION\1.csv")


df['date'] = pd.to_datetime(df['date'],dayfirst=True)


df = df.sort_values('date')


df['monthly_mean'] = df['close'].rolling(window=30).mean()

df.to_csv(r"C:\Users\SOHAM\OneDrive\Desktop\AMAZON STOCK PREDICTION\1.csv", index=False)


plt.figure(figsize=(14, 8))
plt.plot(df['date'], df['monthly_mean'], label='MM')

# Format the x-axis
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gcf().autofmt_xdate(rotation=45)

# Labels and titles
plt.title('AMAZON MONTHLY_MEAN ')
plt.xlabel('Year')
plt.ylabel('PRICE')
plt.legend()
plt.grid(True)



df = pd.read_csv(r"C:\Users\SOHAM\OneDrive\Desktop\AMAZON STOCK PREDICTION\1.csv")
df['date'] = pd.to_datetime(df['date'])  # Convert to datetime

plt.figure(figsize=(14, 8))
plt.plot(df['date'], df['open'], label='CLOSE PRICE')

# Format the x-axis
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())  # One tick per year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gcf().autofmt_xdate(rotation=45)  # Rotate x-labels

# Labels and titles
plt.title('AMAZON OPENING PRICE ')
plt.xlabel('Year')
plt.ylabel('PRICE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




df['Daily Return %'] = df['close'].pct_change() * 100


df = df.dropna(subset=['Daily Return %'])

plt.figure(figsize=(12,6))
plt.plot(df['date'], df['Daily Return %'], label='Daily Return %')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.title('Daily Returns Over Time')
plt.legend()
plt.grid(True)
plt.show()

