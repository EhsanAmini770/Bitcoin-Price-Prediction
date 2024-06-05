import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Activation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

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
data["TargetClass"] = [1 if data.TARGET[i] > 0 else 0 for i in range(len(data))]
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
y_sc = MinMaxScaler(feature_range=(0, 1))
y_data_set_scaled = y_sc.fit_transform(y_data_set.values.reshape(-1, 1))

# Create sequences
backcandles = 30
X = np.array([data_set_scaled[i-backcandles:i] for i in range(backcandles, len(data_set_scaled))])
y = np.reshape(y_data_set_scaled[backcandles:], (len(y_data_set_scaled[backcandles:]), 1))

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
mae_scores = []

# K-Fold Cross-Validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Build LSTM model
    input_shape = (backcandles, X_train.shape[2])
    lstm_input = Input(shape=input_shape, name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    
    # Compile model
    model.compile(optimizer=optimizers.Adam(), loss='mse')
    
    # Train model
    model.fit(X_train, y_train, batch_size=16, epochs=60, shuffle=True, validation_split=0.1, verbose=0)
    
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
model.save("LSTM/lstm_model.h5")

# Predict next 30 days using the last model (optional)
future_days = 30
future_predictions = []
last_available_data = X_test[-1]

for _ in range(future_days):
    prediction_input = np.reshape(last_available_data, (1, backcandles, input_shape[1]))
    next_day_prediction = model.predict(prediction_input)[0][0]
    future_predictions.append(next_day_prediction)
    last_available_data = np.roll(last_available_data, -1, axis=0)
    last_available_data[-1][-1] = next_day_prediction

# Plot future predictions (optional)
plt.figure(figsize=(16, 8))
plt.plot(y_test, color='black', label='Test Data')
plt.plot(range(len(y_test), len(y_test) + future_days), future_predictions, color='green', label='Future Predictions')
plt.legend()
plt.show()
