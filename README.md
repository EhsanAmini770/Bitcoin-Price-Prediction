# Bitcoin Price Prediction Using Machine Learning and Neural Network Models

## Overview
This project aims to predict Bitcoin prices using various machine learning and neural network models. The models compared in this project include RNN, GRU, LSTM, Random Forest, and Multiple Regression. The performance of each model is evaluated based on Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Project Structure
The project directory is organized as follows:

- `GRU/` - Contains code and results for the GRU model.
- `LSTM/` - Contains code and results for the LSTM model.
- `MultipleRegression/` - Contains code and results for the Multiple Regression model.
- `RandomForest/` - Contains code and results for the Random Forest model.
- `RNN/` - Contains code and results for the RNN model.
- `tables/` - Contains tables of results and performance metrics.
- `data.csv` - The main dataset used for training and testing the models.
- `test.py` - Python script used for testing the models.
- `test_data.csv` - Additional test dataset.
- `test_data_set.csv` - Another test dataset.

## Data
The data used in this project is sourced from Yahoo Finance and includes daily Bitcoin prices with the following attributes:
- Open
- High
- Low
- Close
- Volume
- Adjusted Close

Technical indicators such as RSI and EMA are also calculated and included in the dataset.

## Models
The following models are implemented and compared in this project:
1. **RNN**: Recurrent Neural Network
2. **GRU**: Gated Recurrent Unit
3. **LSTM**: Long Short-Term Memory
4. **Random Forest**: Ensemble learning method using multiple decision trees
5. **Multiple Regression**: Linear regression model with multiple predictors

## Installation
To run the code in this repository, you'll need to have Python installed along with the necessary libraries.
