import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st

# Load data and handle date parsing
fields = ['Modal_Price', 'Price_Date']
df = pd.read_csv(r"C:\Users\INTEL\Downloads\SIH appp\FramMart-main\Future-crop-price-prediction-master\data\raw_wheat.csv", usecols=fields)

# Convert 'Price_Date' to datetime and set as index
df['Price_Date'] = pd.to_datetime(df['Price_Date'], errors='coerce')
df.dropna(subset=['Price_Date'], inplace=True)
df.set_index('Price_Date', inplace=True)

# Resample data monthly and fill missing values
y = df['Modal_Price'].resample('MS').mean()
y.fillna(method='bfill', inplace=True)

# Plot the resampled data
st.title("Monthly Resampled Modal Price of Wheat")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y, label='Resampled Modal Price')
ax.set_xlabel('Date')
ax.set_ylabel('Modal Price')
ax.legend()
st.pyplot(fig)

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
st.title("Forecast of Modal Price of Wheat")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y, label='Observed')
forecast.predicted_mean.plot(label='Forecast', alpha=.7, ax=ax)
ax.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Modal Price')
ax.legend()
st.pyplot(fig)

# Save the model to a file
pickle.dump(model, open("modelsave.pkl", "wb"))

# Download the saved model
@st.cache
def download_model():
    with open("modelsave.pkl", "rb") as f:
        return f.read()

st.download_button("Download Model", download_model(), file_name="modelsave.pkl")