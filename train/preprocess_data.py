import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import sys
sys.path.insert(0, '../')
from get_realtime_data import get_data

current_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_path))
config_path = os.path.join(root_path, 'config.json')

# Load configuration file
with open(config_path) as f:
    config = json.load(f)

symbol = config['symbol']
modeltype = config['modeltype']

# Function to create sequences of data for time series analysis
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# Preprocessing function for stacking model
def preprocess_data_stacking(df):
    X = df[['prophet_price', 'lstm_price', 'forest_price']]
    y = df['real_price']
    return X, y

# Preprocessing function for LSTM model
def preprocess_data_lstm(df, seq_length):
    df['date'] = pd.to_datetime(df['timestamp'])
    df.set_index('date', inplace=True)
    price_features = ['open', 'high', 'low', 'close']
    volume_features = ['volume']
    df_price = df[price_features].copy()
    df_volume = df[volume_features]

    df_price.loc[:, 'average'] = (df_price['high'] + df_price['low']) / 2
    df_price.loc[:, 'moving_average'] = df_price['close'].rolling(window=5).mean()

    df = pd.concat([df_price, df_volume], axis=1)

    feature_names = ['open', 'high', 'low', 'close', 'average', 'moving_average', 'volume']

    df.columns = feature_names
    dates = df.index

    df = df.dropna(subset=feature_names)

    price_scaler = MinMaxScaler(feature_range=(0, 1))
    df[['open', 'high', 'low', 'close', 'average', 'moving_average']] = price_scaler.fit_transform(df[['open', 'high', 'low', 'close', 'average', 'moving_average']])
    volume_scaler = MinMaxScaler(feature_range=(0, 1))
    df[['volume']] = volume_scaler.fit_transform(df[['volume']])

    X, y = create_sequences(df.values, seq_length)

    return X, y, price_scaler, volume_scaler, dates

# Preprocessing function for Prophet model
def preprocess_data_prophet(df):
    regressors = {'open', 'high', 'low', 'volume'}

    df = df[['timestamp', 'close'] + list(regressors)]
    df = df.dropna()

    for col in df.columns:
        if col not in ['timestamp']:
            df[col] = df[col].astype(float)

    df = df.reset_index().rename(columns={'timestamp': 'ds', 'close': 'y'})
    return df, regressors

# Preprocessing function for RandomForest model
def preprocess_data_randomforest(df):
    feature_names = ['open', 'high', 'low', 'volume', 'average', 'moving_average']

    df['average'] = (df['high'] + df['low']) / 2
    df['moving_average'] = df['close'].rolling(window=5).mean()

    df = df.dropna(subset=feature_names)

    features = df[feature_names]
    target = df['close']

    return features, target

# Main preprocessing function
def preprocess_data(symbol, data, seq_length, data_frequency, modeltype):
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, int):
        df = get_data(symbol, data, data_frequency)
    else:
        raise ValueError("data must be a file path or a pandas DataFrame")

    if modeltype == 'lstm':
        X, y, price_scaler, volume_scaler, dates = preprocess_data_lstm(df, seq_length)
        return X, y, price_scaler, volume_scaler, dates
    elif modeltype == 'prophet':
        df, regressors = preprocess_data_prophet(df)
        return df, regressors
    elif modeltype == 'randomforest':
        features, target = preprocess_data_randomforest(df)
        return features, target
    elif modeltype == 'stacking':
        X, y = preprocess_data_stacking(df)
        return X, y
    else:
        raise ValueError("Invalid model: {}".format(modeltype))