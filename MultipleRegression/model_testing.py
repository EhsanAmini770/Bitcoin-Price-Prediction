import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Load the saved Multiple Regression model using joblib
model = joblib.load('C:\\Users\\hasib\\Desktop\\ai project\\MultipleRegression\\MR_model_lr.pkl')

# Load the test data
test_data = pd.read_csv('C:\\Users\\hasib\\Desktop\\ai project\\test_data.csv', index_col='Date', parse_dates=True)

# Calculate technical indicators
test_data["RSI"] = ta.rsi(test_data.Close, length=15)
test_data["EMAF"] = ta.ema(test_data.Close, length=20)
test_data["EMAM"] = ta.ema(test_data.Close, length=100)
test_data["EMAS"] = ta.ema(test_data.Close, length=150)

# Calculate targets
test_data["TARGET"] = test_data["Adj Close"] - test_data.Open
test_data["TARGET"] = test_data["TARGET"].shift(-1)
test_data["TargetClass"] = [1 if test_data["TARGET"].iloc[i] > 0 else 0 for i in range(len(test_data))]
test_data["TargetNextClose"] = test_data["Adj Close"].shift(-1)

# Clean data
test_data.dropna(inplace=True)

# Ensure we have enough samples after dropping NaNs
if len(test_data) < 30:
    raise ValueError("Not enough data after cleaning to create sequences.")

test_data.reset_index(inplace=True)
test_data.drop(["Volume", "Close", "Date"], axis=1, inplace=True)

# Prepare dataset
test_data_set = test_data.iloc[:, :-1]  # Ensure we include all the features used in training

print(test_data_set.head())
y_test_data_set = test_data['TargetNextClose']

# Scale data
sc = MinMaxScaler(feature_range=(0, 1))
test_data_set_scaled = sc.fit_transform(test_data_set)

# Create sequences (for consistency, although Linear Regression does not require sequences)
backcandles = 30
X_test_new = test_data_set_scaled[backcandles:]
y_test_new = y_test_data_set.iloc[backcandles:].values

# Predict
y_pred_new = model.predict(X_test_new)

# Inverse transform the predictions and actual values (if necessary)
y_test_new_unscaled = y_test_new  # The target is already in original scale
y_pred_new_unscaled = y_pred_new  # As Linear Regression does not alter scale

# Save the actual and predicted prices to a CSV file
results = pd.DataFrame({'Actual': y_test_new_unscaled.flatten(), 'Predicted': y_pred_new_unscaled.flatten()})
results.to_csv('C:\\Users\\hasib\\Desktop\\ai project\\MultipleRegression\\MR_actual_vs_predicted.csv', index=False)

# Plot results
plt.figure(figsize=(16, 8))
plt.plot(y_test_new_unscaled, color='black', label='Test Data')
plt.plot(y_pred_new_unscaled, color='green', label='Prediction')
plt.legend()
plt.show()

# Calculate MSE and MAE
mse_new = mean_squared_error(y_test_new_unscaled, y_pred_new_unscaled)
mae_new = mean_absolute_error(y_test_new_unscaled, y_pred_new_unscaled)

print('MSE for the model on new test data:', mse_new)
print('MAE for the model on new test data:', mae_new)
