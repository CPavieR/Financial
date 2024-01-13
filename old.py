import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to safely get financial data
def get_financial_data(ticker, data_type):
    try:
        data = getattr(ticker, data_type)()
        print(data)
        return data
    except Exception:
        return pd.DataFrame()
ticker_symbol = 'AAPL'
# Fetch historical stock price data
#check if data is already saved
if os.path.isfile('stock_data.csv'):
    stock_data = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)
else:
    
    stock_data = yf.download(ticker_symbol, start='2020-09-30', end='2023-01-01')
    #save to file
    stock_data.to_csv('stock_data.csv')

# Initialize ticker
ticker = yf.Ticker(ticker_symbol)
#print(ticker.info)
def add_financial_feature(stock_data, financial_data, feature_name):
    financial_data.index = pd.to_datetime(financial_data.index)
    print(financial_data.index)
#    if feature_name in financial_data.columns:
#        # Assuming that the financial data's index is DateTime
    feature_series = financial_data[feature_name].resample('D').ffill().reindex(stock_data.index, method='ffill')
    stock_data[feature_name] = feature_series
#    else:
#        stock_data[feature_name] = -1  # Fill with -1 if the feature does not exist

# Get financial data
#get_financial_data(ticker, 'balance_sheet')
#verify that the data is saved
if os.path.isfile('balance_sheet.csv'):
    balance_sheet = pd.read_csv('balance_sheet.csv',parse_dates=True, index_col=0)
else:
    balance_sheet = ticker.balance_sheet
    balance_sheet.to_csv('balance_sheet.csv')

#verufiy all three data sets are saved
if os.path.isfile('cash_flow.csv'):
    cash_flow = pd.read_csv('cash_flow.csv', parse_dates=True, index_col=0)
else:
    cash_flow = ticker.cashflow
    cash_flow.to_csv('cash_flow.csv')
"""
if os.path.isfile('earnings.csv'):
    earnings = pd.read_csv('earnings.csv', index_col='Date', parse_dates=True)
else:
    earnings = ticker.earnings
    earnings.to_csv('earnings.csv')
"""
if os.path.isfile('income_statement.csv'):
    #date is the first row
    income_statement = pd.read_csv('income_statement.csv', parse_dates=True, index_col=0)
else:
    income_statement = ticker.income_stmt
    income_statement.to_csv('income_statement.csv')
#dividends = ticker.dividends.resample('D').ffill().reindex(stock_data.index, method='ffill')
balance_sheet = balance_sheet.T
cash_flow = cash_flow.T
income_statement = income_statement.T
# Example: Incorporate selected financial data into stock_data
financial_features = ['Current Assets', 'Current Liabilities', 'Total Debt', 'Net PPE','Cash Financial','Investments And Advances']
for feature in financial_features:
    add_financial_feature(stock_data, balance_sheet, feature)

add_financial_feature(stock_data, cash_flow, 'Free Cash Flow')
add_financial_feature(stock_data, income_statement, 'Net Income')
add_financial_feature(stock_data, income_statement, 'Total Revenue')
add_financial_feature(stock_data, income_statement, 'Gross Profit')


#add_financial_feature(stock_data, earnings, 'Earnings')
#stock_data['Cash Flow'] = cash_flow.loc['Total Cash From Operating Activities'].resample('D').ffill().reindex(stock_data.index, method='ffill')
#stock_data['Earnings'] = earnings['Earnings'].resample('D').ffill().reindex(stock_data.index, method='ffill')
#stock_data['Dividends'] = dividends

# Fill missing values with -1
stock_data.fillna(-1, inplace=True)
"""
# Preprocess data
features = stock_data.drop('Close', axis=1).values
target = stock_data['Close'].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.reshape(-1, 1)).ravel()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)
"""
# Preprocess data for predicting next day's closing price
number_days = 30  # Number of days to consider for prediction

# Function to create a dataset for time series prediction
def create_time_series_dataset(data, n):
    X, y = [], []
    for i in range(len(data) - n):
        X.append(data.iloc[i:(i + n), :-1].values)
        y.append(data.iloc[i + n, -1])
    return np.array(X), np.array(y)

# Shift the 'Close' column to get next day's closing price as the target
stock_data['Next Close'] = stock_data['Close'].shift(-1)
stock_data.dropna(inplace=True)  # Drop the last row with NaN target
print(stock_data)
#save to a file
stock_data.to_csv('stock_data_final.csv')
# Create time series dataset
X, y = create_time_series_dataset(stock_data, number_days)

# Scale the features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Initialize models
models = {
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression(),
    'SVR': SVR()
}

# Train and evaluate models
# ... [earlier parts of the script remain the same] ...

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    
    # Predict on the test set
    predictions = model.predict(X_test.reshape(X_test.shape[0], -1))

    # Inverse transform predictions and true values to original scale
    predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).ravel()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Calculate MSE in original dollar scale
    mse_dollars = mean_squared_error(y_test_original, predictions_original)
    print(f"{name} - Mean Squared Error in dollars: ${mse_dollars:.2f}")
# Reshape data for LSTM model (required 3D shape)
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Build LSTM model
lstm_model = build_lstm_model(X_train_lstm.shape[1:])
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32)

# Predict on the test set
lstm_predictions = lstm_model.predict(X_test_lstm)

# Inverse transform predictions and true values to original scale
lstm_predictions_original = scaler_y.inverse_transform(lstm_predictions).ravel()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Calculate MSE in original dollar scale for LSTM
lstm_mse_dollars = mean_squared_error(y_test_original, lstm_predictions_original)
print(f"LSTM Neural Network - Mean Squared Error in dollars: ${lstm_mse_dollars:.2f}")
