from joblib import load
from keras.models import load_model
import neuralprophet.utils
import pandas as pd
import os
from preprocess_data import preprocess_data
import json

# Load configuration file
current_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_path))
config_path = os.path.join(root_path, 'config.json')
data_path = os.path.join(root_path, 'dataset/stackingdata')

with open(config_path) as f:
    config = json.load(f)

symbol = config['symbol']

data_frequency = config['data_frequency']
modeltype = config['modeltype']


def custom_round(x):
    return round(x, 2) if x >= 1 else round(x, 8)


def predict_neuralprophet():

    model = neuralprophet.utils.load(
        f"{root_path}/models/{symbol}_{data_frequency}_prophet.np")

    df, regressors = preprocess_data(
        symbol, f'{data_path}/{symbol}/{symbol}_{data_frequency}.csv', 100,
        data_frequency, 'prophet')

    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    predictions = model.predict(df)
    close_pred = predictions['yhat1'][104:].apply(custom_round)
    close_pred = close_pred.reset_index(drop=True)
    timestamp = df['ds'][104:].reset_index(drop=True)
    real_price = df['y'][104:].reset_index(drop=True)
    print(close_pred)

    return close_pred, timestamp, real_price


def predict_lstm():
    model = load_model(
        f"{root_path}/models/{symbol}_{data_frequency}_lstm.h5")

    X, y, price_scaler, volume_scaler, dates = preprocess_data(
        symbol, f'{data_path}/{symbol}/{symbol}_{data_frequency}.csv', 100,
        data_frequency, 'lstm')

    predictions = model.predict(X)

    pred = predictions[:, :6]
    pred_inv = pd.DataFrame(price_scaler.inverse_transform(pred))
    pred_inv = pred_inv.applymap(custom_round)
    close_pred = pred_inv[3].astype(float)

    print(close_pred)

    return close_pred


def predict_randomforest():
    model = load(
        f"{root_path}/models/{symbol}_{data_frequency}_randomforest.joblib"
    )
    features, target = preprocess_data(
        symbol, f'{data_path}/{symbol}/{symbol}_{data_frequency}.csv', 100,
        data_frequency, 'randomforest')
    predictions = model.predict(features)
    close_pred_array = pd.Series(predictions)[100:]
    close_pred_array = close_pred_array.reset_index(drop=True)
    close_pred = close_pred_array.astype(float).apply(custom_round)

    print(close_pred)
    return close_pred

close_pred_1, timestamp, real_price = predict_neuralprophet()
close_pred_2 = predict_lstm()
close_pred_3 = predict_randomforest()

# Remove the first element from timestamp and real_price
timestamp = timestamp[1:].reset_index(drop=True)
real_price = real_price[1:].reset_index(drop=True)

# Remove the last element
close_pred_1 = close_pred_1[:-1]
close_pred_2 = close_pred_2[:-1]
close_pred_3 = close_pred_3[:-1]

df = pd.DataFrame({
    'timestamp': timestamp,
    'real_price': real_price,
    'prophet_price': close_pred_1,
    'lstm_price': close_pred_2,
    'forest_price': close_pred_3
})

filename = f"{data_path}/{symbol}/{symbol}_pred_{data_frequency}.csv"
df.to_csv(filename, index=False)
print(
    f"Saved {len(df)} rows of 1d data from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']} to {filename}"
)