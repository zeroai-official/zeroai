import requests
import json
import pandas as pd
import calendar
from datetime import datetime, timedelta
import pytz


def get_bitmex_data(symbol, binSize, startTime):
    url = f"https://www.bitmex.com/api/v1/trade/bucketed?binSize={binSize}&partial=false&symbol={symbol}&startTime={startTime}&count=750&reverse=false"
    response = requests.get(url)
    data = json.loads(response.text)
    return pd.DataFrame(data) 

def get_binance_data(symbol, binSize, startTime):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={binSize}&startTime={startTime}&limit=1000"
    response = requests.get(url)
    data = json.loads(response.text)
    return pd.DataFrame(data)

fieldnames = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "base asset volume",
    "close time",
    "volume",
    "number of trades",
    "taker buy base asset volume",
    "taker buy quote asset volume",
    "ignore",
]

def get_utc_now():
    return datetime.now(pytz.utc)


def get_data(symbol, sample, data_frequency):
    if symbol == "XBTUSD":
        symbol = "BTCUSDT"
    else:
        symbol = symbol + "T"

    startTime = get_utc_now() - timedelta(
            days=sample) if data_frequency == '1d' else get_utc_now() - timedelta(
                hours=sample)
    startTime = calendar.timegm(startTime.timetuple()) * 1000

    data = get_binance_data(symbol, data_frequency, startTime)

    data.columns = fieldnames

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    data['timestamp'] = data['timestamp'].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    columns_to_convert = [
        'open', 'high', 'low', 'close', 'base asset volume', 'volume',
        'number of trades', 'taker buy base asset volume',
        'taker buy quote asset volume'
    ]

    for column in columns_to_convert:
        data[column] = data[column].astype(float)

    return data
