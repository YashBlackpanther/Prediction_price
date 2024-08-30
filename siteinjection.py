import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data and handle date parsing
fields = ['Modal_Price', 'Price_Date']
df = pd.read_csv(r"C:\Users\INTEL\Downloads\SIH appp\FramMart-main\Future-crop-price-prediction-master\data\\raw_wheat.csv", usecols=fields)

# Convert 'Price_Date' to datetime and set as index
df['Price_Date'] = pd.to_datetime(df['Price_Date'], errors='coerce')
df.dropna(subset=['Price_Date'], inplace=True)
df.set_index('Price_Date', inplace=True)

# Resample data monthly and fill missing values
y = df['Modal_Price'].resample('MS').mean()
y.fillna(method='bfill', inplace=True)

# Plot the resampled data
plt.figure(figsize=(10, 6))
plt.plot(y, label='Resampled Modal Price')
plt.title('Monthly Resampled Modal Price of Wheat')
plt.xlabel('Date')
plt.ylabel('Modal Price')
plt.legend()
plt.show()

# Fit a basic SARIMA model
# Simple parameters for demonstration: (1, 1, 1) for ARIMA and (1, 1, 1, 12) for seasonal
model = sm.tsa.statespace.SARIMAX(y,
                                  order=(1, 1, 1),
                                  seasonal_order=(1, 1, 1, 12),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)

results = model.fit()

# Forecast next 12 months
forecast = results.get_forecast(steps=12)
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(10, 6))
plt.plot(y, label='Observed')
forecast.predicted_mean.plot(label='Forecast', alpha=.7)
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='k', alpha=.2)
plt.title('Forecast of Modal Price of Wheat')
plt.xlabel('Date')
plt.ylabel('Modal Price')
plt.legend()
plt.show()




# Save the model to a file
pickle.dump(model, open(r'C:\Users\INTEL\Desktop\Model save\modelsave.pkl', 'wb'))
