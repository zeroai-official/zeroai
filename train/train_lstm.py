from keras.models import Sequential
from keras.layers import LSTM, Dense
import os
import pandas as pd
import numpy as np
from preprocess_data import preprocess_data
import json

current_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_path))
config_path = os.path.join(root_path, 'config.json')
data_path = os.path.join(root_path, 'dataset/data')

with open(config_path) as f:
    config = json.load(f)

symbol = config['symbol']
modeltype = config['modeltype']
data_frequency = config['data_frequency']
test_size = config['test_size']
hidden_dim = config['hidden_dim']
batch_size = config['batch_size']
epochs = config['epochs']

if data_frequency == '1d':
    units = hidden_dim[0]
elif data_frequency == '1h':
    units = hidden_dim[1]
else:
    units = 256  # default value


X, y, price_scaler, volume_scaler, dates = preprocess_data(symbol, f'{data_path}/{symbol}/{symbol}_{data_frequency}.csv', 100, data_frequency,'lstm')

X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

model = Sequential()
model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units, activation='relu'))
model.add(Dense(7)) 
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

model.save(f'../models/{symbol}_{data_frequency}_lstm.h5')

y_pred = model.predict(X_test)

price_pred = y_pred[:, :6]
other_pred = y_pred[:, 6:]

price_pred_inv = price_scaler.inverse_transform(price_pred)
other_pred_inv = volume_scaler.inverse_transform(other_pred)

y_pred_inv = np.concatenate([price_pred_inv, other_pred_inv], axis=1)

price_test = y_test[:, :6]
other_test = y_test[:, 6:]
price_test_inv = price_scaler.inverse_transform(price_test)
other_test_inv = volume_scaler.inverse_transform(other_test)
y_test_inv = np.concatenate([price_test_inv, other_test_inv], axis=1)

feature_names = ['open', 'high', 'low', 'close', 'average', 'moving_average', 'volume']

df_pred = pd.DataFrame(y_pred_inv, columns=feature_names)
df_pred['date'] = dates[-len(y_pred_inv):]

df_test = pd.DataFrame(y_test_inv, columns=feature_names)
df_test['date'] = dates[-len(y_test_inv):]

print(df_pred.tail())
print(df_test.tail())