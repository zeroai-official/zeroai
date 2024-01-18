from joblib import load
from keras.models import load_model
import neuralprophet.utils
import pandas as pd
from train.preprocess_data import preprocess_data
import json
import logging
import plotly.express as px
import plotly.graph_objects as go

logging.disable(logging.CRITICAL+1)

# Load configuration file
with open('config.json', 'r') as f:
    config = json.load(f)

symbol = config['symbol']
modeltype = config['modeltype']
data_frequency = config['data_frequency']


def custom_round(x):
    return round(x, 2) if x >= 1 else round(x, 8)


def predict_lstm(symbol, modeltype, data_frequency):
    print(f"Starting prediction with {modeltype} model...")
    model = load_model(f'models/{symbol}_{data_frequency}_{modeltype}.h5')

    X, y, price_scaler, volume_scaler, dates = preprocess_data(
        symbol, 105, 100, data_frequency, modeltype)

    predictions = model.predict(X)
    actual = y[-1]

    pred = predictions[:, :6]
    actual = actual[:6]

    # Inverse normalization
    pred_inv = price_scaler.inverse_transform(pred)
    actual_inv = price_scaler.inverse_transform([actual])

    feature_names = [
        'open', 'high', 'low', 'close', 'average', 'moving_average'
    ]

    df_pred = pd.DataFrame(pred_inv, columns=feature_names)
    close_pred = df_pred['close']

    actual_inv = pd.DataFrame(actual_inv, columns=feature_names)
    actual_inv['date'] = dates[-len(actual_inv):]

    return close_pred, actual_inv


def predict_neuralprophet(symbol, modeltype, data_frequency):
    print(f"Starting prediction with {modeltype} model...")
    model = neuralprophet.utils.load(
        f'models/{symbol}_{data_frequency}_{modeltype}.np')
    df, regressors = preprocess_data(symbol, 100, 100, data_frequency, modeltype)

    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    if data_frequency == '1d':
        num_data_points = -65
    elif data_frequency == '1h':
        num_data_points = -73
    predictions = model.predict(df[num_data_points:])

    pred = predictions.iloc[-1]
    close_pred = pred['yhat1']

    return close_pred


def predict_randomforest(symbol, modeltype, data_frequency):
    print(f"Starting prediction with {modeltype} model...")
    model = load(f'models/{symbol}_{data_frequency}_{modeltype}.joblib')

    features, target = preprocess_data(symbol, 5, 100, data_frequency, modeltype)

    predictions = model.predict(features)

    close_pred = predictions

    return close_pred


def predict_stacking(symbol, modeltype, data_frequency, df):
    print(f"Starting prediction with {modeltype} model...")
    model = load(f'models/{symbol}_{data_frequency}_{modeltype}.joblib')
    X, y = preprocess_data(symbol, df, 100, data_frequency, modeltype)
    predictions = model.predict(X)

    close_pred = predictions
    return close_pred

# Make predictions


def make_predictions(symbol, data_frequency):
    predictions_randomforest = predict_randomforest(symbol, 'randomforest',
                                                    data_frequency)
    predictions_neuralprophet = predict_neuralprophet(symbol, 'prophet',
                                                      data_frequency)
    predictions_lstm = predict_lstm(symbol, 'lstm', data_frequency)

    results = pd.DataFrame()

    # Add actual values
    results = predictions_lstm[1]

    # Add predictions
    results['forest_price'] = predictions_randomforest[0]
    results['prophet_price'] = predictions_neuralprophet
    results['lstm_price'] = predictions_lstm[0]

    # Move 'date' column to the end
    date = results.pop('date')
    results.rename(columns={'close': 'real_price'}, inplace=True)

    predictions_stacking = predict_stacking(symbol, 'stacking', data_frequency,
                                            results)
    results['stacking_price'] = predictions_stacking
    results.insert(len(results.columns), 'date', date)
    columns_to_round = [
        'average', 'moving_average', 'forest_price', 'prophet_price',
        'lstm_price', 'stacking_price'
    ]
    results[columns_to_round] = results[columns_to_round].applymap(
        custom_round)
    # Add symbol and data_frequency
    symbol_df = pd.DataFrame([symbol] * len(results), columns=['symbol'])
    data_frequency_df = pd.DataFrame([data_frequency] * len(results),
                                     columns=['data_frequency'])
    results = pd.concat([symbol_df, data_frequency_df, results], axis=1)
    return results

if __name__ == "__main__":

    # List of data frequencies
    data_frequencies = ['1d', '1h']
    all_results = pd.DataFrame()

    # Loop over all combinations of data frequencies
    for data_frequency in data_frequencies:
        results = make_predictions(symbol, data_frequency)
        print(symbol)
        all_results = pd.concat([all_results, results])

    # Print all results
    print(all_results.to_string(index=False))

    # Create a bar plot for all data
    fig = go.Figure()

    # Melt the DataFrame into a format that Plotly can use
    melted_results = all_results.melt(id_vars=['date', 'data_frequency'], value_vars=['real_price', 'forest_price', 'prophet_price', 'lstm_price', 'stacking_price'])

    # Add traces for each prediction method
    for method in ['real_price', 'forest_price', 'prophet_price', 'lstm_price', 'stacking_price']:
        for freq in melted_results['data_frequency'].unique():
            fig.add_trace(go.Bar(x=[f'{freq} forecast'] * len(melted_results[(melted_results['variable'] == method) & (melted_results['data_frequency'] == freq)]),
                                y=melted_results['value'][(melted_results['variable'] == method) & (melted_results['data_frequency'] == freq)],
                                name=f'{method} {freq}',
                                text=['Current Price' if method == 'real_price' else 'Future Price' if method == 'stacking_price' else ''] * len(melted_results[(melted_results['variable'] == method) & (melted_results['data_frequency'] == freq)]),
                                textposition='auto'))

    # Set the y-axis title
    fig.update_yaxes(title_text="Price")

    # Set the x-axis title
    fig.update_xaxes(title_text="Forecast")

    # Set the overall layout
    fig.update_layout(barmode='group', title='Price Predictions')

    # Calculate the minimum and maximum values of your data
    min_value = melted_results['value'].min()
    max_value = melted_results['value'].max()

    # Add a small padding to the minimum and maximum values
    padding = (max_value - min_value) * 0.1
    min_value -= padding
    max_value += padding

    # Set the y-axis range to focus on the differences
    fig.update_yaxes(range=[min_value, max_value])

    fig.show()