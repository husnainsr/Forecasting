import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from prophet import Prophet
import sqlite3
import pickle


# Read data
data = pd.read_csv("data/sp500_stocks.csv")
data = data[data['Symbol'] == 'MMM'].drop(['Symbol'], axis=1)
date_column = 'Date'
target_column = 'Close'

# Preprocessing with normalization and differencing
data['Close_diff'] = data[target_column].diff()
data.dropna(inplace=True)
numerical_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data[numerical_columns])
normalized_df = pd.DataFrame(normalized_data, columns=numerical_columns)
normalized_df['Close_diff'] = normalized_df['Close'].diff()
normalized_df.dropna(inplace=True)
normalized_df['Date'] = data[date_column]

# ARIMA model
X = normalized_df.drop(columns=['Date', 'Close'])
y = normalized_df['Close_diff']
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

order = (22, 1, 21)
model = ARIMA(y_train, order=order)
fitted_model = model.fit()
arima_predictions_testing_diff = fitted_model.forecast(steps=len(X_test))
mse_arima_diff = mean_squared_error(y_test, arima_predictions_testing_diff)
r_squared_arima_diff = r2_score(y_test, arima_predictions_testing_diff)

# Save ARIMA model
with open('models/arima_model.pkl', 'wb') as f:
    pickle.dump(fitted_model, f)

# SARIMA model
sarima_model = SARIMAX(y_train, order=(21, 1, 1), seasonal_order=(0, 0, 0, 12))
fitted_sarima_model = sarima_model.fit()
sarima_predictions_diff = fitted_sarima_model.forecast(steps=len(X_test))
mse_sarima_diff = mean_squared_error(y_test, sarima_predictions_diff)
r_squared_sarima_diff = r2_score(y_test, sarima_predictions_diff)

# SVR model
svr = SVR()
param_grid = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_svr = SVR(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'])
best_svr.fit(X_train, y_train)
svr_prediction_testing_diff = best_svr.predict(X_test)
mse_svr_diff = mean_squared_error(y_test, svr_prediction_testing_diff)
r_squared_svr_diff = r2_score(y_test, svr_prediction_testing_diff)

# Prophet model
prophet_df = normalized_df.rename(columns={"Close_diff": "y", "Date": "ds"})
model = Prophet(seasonality_mode='additive', changepoint_prior_scale=0.5)
model.fit(prophet_df)
future = model.make_future_dataframe(periods=len(X_test))
forecast = model.predict(future)
mse_pro_diff = mean_squared_error(prophet_df['y'][-len(X_test):], forecast['yhat'])
r_squared_pro_diff = r2_score(prophet_df['y'][-len(X_test):], forecast['yhat'])

# LSTM model
X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=27, batch_size=32)
lstm_prediction_testing_diff = model.predict(X_test)
mse_lstm_diff = mean_squared_error(y_test, lstm_prediction_testing_diff)
r_squared_lstm_diff = r2_score(y_test, lstm_prediction_testing_diff)

# Save evaluation results to SQLite database
conn = sqlite3.connect('forecast_results.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS evaluation_diff (
                    model_name TEXT,
                    mse REAL,
                    r_squared REAL
                 )''')
evaluation_data_diff = [
    ('ARIMA', mse_arima_diff, r_squared_arima_diff),
    ('SARIMA', mse_sarima_diff, r_squared_sarima_diff),
    ('SVR', mse_svr_diff, r_squared_svr_diff),
    ('Prophet', mse_pro_diff, r_squared_pro_diff),
    ('LSTM', mse_lstm_diff, r_squared_lstm_diff)
]
cursor.executemany('''INSERT INTO evaluation_diff (model_name, mse, r_squared) VALUES (?, ?, ?)''', evaluation_data_diff)
conn.commit()

# Create DataFrame for testing predictions
df_testing = pd.DataFrame({
    "index_value": range(len(X_test)),
    "Test": y_test.values,
    "ARIMA": arima_predictions_testing_diff,
    "SARIMA": sarima_predictions_diff,
    "SVR": svr_prediction_testing_diff,
    "Prophet": forecast['yhat'][-len(X_test):].values,
    "LSTM": lstm_prediction_testing_diff.flatten()
})

# Save DataFrame to SQLite database
cursor.execute("DROP TABLE IF EXISTS testing")
cursor.execute('''CREATE TABLE testing (
                    index_value INTEGER PRIMARY KEY,
                    Test REAL,
                    ARIMA REAL,
                    SARIMA REAL,
                    SVR REAL,
                    Prophet REAL,
                    LSTM REAL
                 )''')
df_testing.to_sql('testing', conn, if_exists='append', index=False)

# Commit changes and close the connection
conn.commit()
conn.close()
