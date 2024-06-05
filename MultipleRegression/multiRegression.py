import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import joblib

# Load data
data = pd.read_csv('data.csv', index_col='Date', parse_dates=True)

# Calculate technical indicators
data["RSI"] = ta.rsi(data.Close, length=15)
data["EMAF"] = ta.ema(data.Close, length=20)
data["EMAM"] = ta.ema(data.Close, length=100)
data["EMAS"] = ta.ema(data.Close, length=150)

# Calculate targets
data["TARGET"] = data["Adj Close"] - data.Open
data["TARGET"] = data["TARGET"].shift(-1)
data["TargetClass"] = [1 if data.TARGET.iloc[i] > 0 else 0 for i in range(len(data))]
data["TargetNextClose"] = data["Adj Close"].shift(-1)

# Clean data
data.dropna(inplace=True)
data.reset_index(inplace=True)
data.drop(["Volume", "Close", "Date"], axis=1, inplace=True)

# Prepare dataset
data_set = data.iloc[:, 0:10]
y_data_set = data.iloc[:, 10]

# Scale data
sc = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = sc.fit_transform(data_set)

# Scale y_data_set
y_sc = MinMaxScaler(feature_range=(0, 1))
y_data_set_scaled = y_sc.fit_transform(y_data_set.values.reshape(-1, 1))

# Prepare flat data for Linear Regression
X = data_set_scaled
y = y_data_set_scaled

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
mae_scores = []

# K-Fold Cross-Validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Build Linear Regression model
    model = LinearRegression()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict and calculate MSE and MAE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse_scores.append(mse)
    mae_scores.append(mae)

# Plot results for the last fold
plt.figure(figsize=(16, 8))
plt.plot(y_test, color='black', label='Test')
plt.plot(y_pred, color='green', label='Prediction')
plt.legend()
plt.show()

print('MSE scores for each fold:', mse_scores)
print('Mean MSE score:', np.mean(mse_scores))
print('MAE scores for each fold:', mae_scores)
print('Mean MAE score:', np.mean(mae_scores))

# Save the model (optional, if you want to save the last model)
joblib.dump(model, "MultipleRegression/MR_model_lr.pkl")

# Predict next 30 days using the last model (optional)
future_predictions = []
last_available_data = X[-1].reshape(1, -1)  # Use the last available data point from the original dataset

print('Last available data:', last_available_data)

for _ in range(30):
    next_day_prediction = model.predict(last_available_data)[0]
    future_predictions.append(next_day_prediction)
    last_available_data = np.roll(last_available_data, -1)  # Shift the data
    last_available_data[0, -1] = next_day_prediction  # Update the last feature with the new predicted value

# Inverse transform the predictions to get the original scale
future_predictions_unscaled = y_sc.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# # Save the actual and predicted prices to a file
# future_predictions_unscaled = pd.DataFrame(future_predictions_unscaled)
# future_predictions_unscaled.to_csv('MultipleRegression/future_predictions.csv', index=False)

# Plot future predictions (optional)
plt.figure(figsize=(16, 8))
plt.plot(np.arange(len(y)), y_sc.inverse_transform(y).flatten(), color='black', label='Test Data')
plt.plot(np.arange(len(y), len(y) + 30), future_predictions_unscaled, color='green', label='Future Predictions')
plt.legend()
plt.show()
