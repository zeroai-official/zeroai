from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.layers import Dropout
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from joblib import dump
import numpy as np
import pandas as pd
import os
import json
from preprocess_data import preprocess_data

current_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_path))
config_path = os.path.join(root_path, 'config.json')
data_path = os.path.join(root_path, 'dataset/stackingdata')

with open(config_path) as f:
    config = json.load(f)

symbol = config['symbol']
modeltype = config['modeltype']
data_frequency = config['data_frequency']
test_size = config['test_size']
hidden_dim = config['hidden_dim']
batch_size = config['batch_size']
epochs = config['epochs']

def custom_round(x):
    return round(x, 2) if x >= 1 else round(x, 8)

X, y = preprocess_data(symbol, f'{data_path}/{symbol}/{symbol}_pred_{data_frequency}.csv', 100, data_frequency,'stacking')

X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

model = Ridge(alpha=1.0)

model.fit(X_train,y_train)

dump(model, f'../models/{symbol}_{data_frequency}_stacking.joblib')

predictions = model.predict(X_test)

loss = mean_squared_error(y_test, predictions)
print('Test loss:', loss)

rmse = np.sqrt(loss)
print('Test RMSE:', rmse)

df = pd.DataFrame()

df['y_test'] = y_test

df['predictions'] = predictions

print(df.tail(10))
