from joblib import load
from keras.models import load_model
import neuralprophet.utils
import pandas as pd
import plotly.graph_objects as go
import os
from train.preprocess_data import preprocess_data
import json

# Load configuration file
current_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_path))
config_path = os.path.join(root_path, 'config.json')
data_path = os.path.join(root_path, 'dataset/stackingdata')

with open('config.json', 'r') as f:
    config = json.load(f)

symbol = config['symbol']

data_frequency = config['data_frequency']
modeltype = config['modeltype']


def custom_round(x):
    return round(x, 2) if x >= 1 else round(x, 8)


def predict_neuralprophet():
    model = neuralprophet.utils.load(
        f"models/{symbol}_{data_frequency}_prophet.np")

    df, regressors = preprocess_data(
        symbol, f'dataset/stackingdata/{symbol}/{symbol}_{data_frequency}.csv', 100,
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
        f"models/{symbol}_{data_frequency}_lstm.h5")

    X, y, price_scaler, volume_scaler, dates = preprocess_data(
        symbol, f'dataset/stackingdata/{symbol}/{symbol}_{data_frequency}.csv', 100,
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
        f"models/{symbol}_{data_frequency}_randomforest.joblib"
    )
    features, target = preprocess_data(
        symbol, f'dataset/stackingdata/{symbol}/{symbol}_{data_frequency}.csv', 100,
        data_frequency, 'randomforest')
    predictions = model.predict(features)
    close_pred_array = pd.Series(predictions)[100:]
    close_pred_array = close_pred_array.reset_index(drop=True)
    close_pred = close_pred_array.astype(float).apply(custom_round)

    print(close_pred)
    return close_pred

def predict_stacking():

    model = load(f'models/{symbol}_{data_frequency}_stacking.joblib')
    X, y = preprocess_data(
        symbol, f'dataset/stackingdata/{symbol}/{symbol}_pred_{data_frequency}.csv', 100,
        data_frequency, 'stacking')

    predictions = model.predict(X)

    close_pred = predictions
    return close_pred

close_pred_1, timestamp, real_price = predict_neuralprophet()
close_pred_2 = predict_lstm()
close_pred_3 = predict_randomforest()
close_pred_4 = predict_stacking()
print(len(close_pred_1))
print(len(close_pred_2))
print(len(close_pred_3))
print(len(close_pred_4))
print(len(timestamp))
print(len(real_price))

timestamp = timestamp[2:].reset_index(drop=True)
real_price = real_price[2:].reset_index(drop=True)

close_pred_1 = close_pred_1[:-2]
close_pred_2 = close_pred_2[:-2]
close_pred_3 = close_pred_3[:-2]
close_pred_4 = close_pred_4[:-1]


df = pd.DataFrame({
    'timestamp': timestamp,
    'real_price': real_price,
    'prophet_price': close_pred_1,
    'lstm_price': close_pred_2,
    'forest_price': close_pred_3,
    'stacking_price': close_pred_4
})


df = df.tail(720) 
print(df)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['real_price'], mode='lines', name='Real Price', line_shape='spline'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['prophet_price'], mode='lines', name='Prophet Price', line_shape='spline'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['lstm_price'], mode='lines', name='LSTM Price', line_shape='spline'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['forest_price'], mode='lines', name='Forest Price', line_shape='spline'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stacking_price'], mode='lines', name='Stacking Price', line_shape='spline'))

fig.update_xaxes(title_text="Timestamp")

fig.update_yaxes(title_text="Price")

fig.update_layout(title=f'{symbol} {data_frequency} Price Predictions')

fig.show()