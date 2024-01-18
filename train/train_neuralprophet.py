from neuralprophet import NeuralProphet
import os
from preprocess_data import preprocess_data
import json
from neuralprophet.utils import save

current_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_path))
config_path = os.path.join(root_path, 'config.json')
data_path = os.path.join(root_path, 'dataset/data')

with open(config_path) as f:
    config = json.load(f)

symbol = config['symbol']
data_frequency = config['data_frequency']
modeltype = config['modeltype']
test_size = config['test_size']

df, regressors = preprocess_data(symbol, f'{data_path}/{symbol}/{symbol}_{data_frequency}.csv', 100, data_frequency,'prophet')

duplicated_rows = df[df.duplicated('ds')]
if not duplicated_rows.empty:
    df = df.drop_duplicates('ds')

if 'index' in df.columns:
    df = df.drop(columns=['index'])

train_size = len(df) - test_size

train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

yearly_seasonality = False
weekly_seasonality = False
daily_seasonality = False

if data_frequency == '1d':
    n_lags = 64
    freq = 'D'
    yearly_seasonality = True
    weekly_seasonality = True
    daily_seasonality = False
elif data_frequency == '1h':
    n_lags = 72
    freq = 'H'
    yearly_seasonality = False
    weekly_seasonality = False
    daily_seasonality = True

model = NeuralProphet(
    yearly_seasonality=yearly_seasonality,
    weekly_seasonality=weekly_seasonality,
    daily_seasonality=daily_seasonality,
    n_lags=n_lags,
    learning_rate=0.003,
    n_forecasts=1,
    quantiles=[0.2, 0.8],
)

for feature in regressors:
    model.add_future_regressor(name=feature)

model.fit(train_df, freq=freq)

forecast = model.predict(test_df)

save(model, f'../models/{symbol}_{data_frequency}_prophet.np')

selected_columns = ['ds', 'y', 'yhat1', 'yhat1 20.0%', 'yhat1 80.0%']

forecast_selected = forecast[selected_columns]

print(forecast_selected.tail())