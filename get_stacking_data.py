import requests
from datetime import datetime, timedelta
import pandas as pd
import time
import os
import json


with open("config.json") as f:
    config = json.load(f)

symbol = config["symbol"]
tsym = symbol[-3:]


if config["symbol"] == "XBTUSD":
    fsym = "BTC"
else:
    fsym = symbol[:-3]


# Convert startTime to datetime object
startTime = datetime.strptime(config["startTime"], "%Y-%m-%d")

os.makedirs(f"dataset/stackingdata/{symbol}", exist_ok=True)

def get_cryptocompare_data(fsym, tsym, binSize, startTime):
    url = f"https://min-api.cryptocompare.com/data/v2/{binSize}"
    parameters = {
        "fsym": fsym,
        "tsym": tsym,
        "toTs": startTime,
        "limit": 2000,
    }
    response = requests.get(url, params=parameters)
    data = json.loads(response.text)
    return data

# Fetching daily data
binSize = "histoday"
data = []
endtime = startTime
startTime = startTime + timedelta(days=1950)
while endtime < datetime.now():
    try:
        startTime_timestamp = int(startTime.timestamp())
        batch = get_cryptocompare_data(fsym, tsym, binSize,
                                       startTime_timestamp)
        batch = batch['Data']['Data']
        data.extend(batch)
        if batch:
            startTime = datetime.fromtimestamp(
                batch[-1]["time"]) + timedelta(days=2000)
            endtime = datetime.fromtimestamp(
                batch[-1]["time"]) + timedelta(days=1)
            print(f"Got data up to {endtime}, total {len(data)} rows.")
        time.sleep(2)
    except Exception as e:
        print(f"Error occurred: {e}")
        break


df = pd.DataFrame(data)

df['time'] = df['time'].astype(int)
# Check for duplicates
df.drop_duplicates(subset='time', keep='first', inplace=True)
# Check for continuity
time_diff = df['time'].diff()
discontinuities = df[time_diff.ne(86400)]

print(discontinuities)

df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')

startTime = pd.Timestamp(config["startTime"], tz='UTC')

df = df[df['time'] >= startTime]
df = df.rename(columns={'time': 'timestamp', 'volumeto': 'volume'})

filename = f"dataset/stackingdata/{symbol}/{symbol}_1d.csv"
df.to_csv(filename, index=False)
print(
    f"Saved {len(df)} rows of 1d data from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']} to {filename}"
)
# Fetching hourly data
# re-set startTime to start date
startTime = datetime.strptime(config["startTime"], "%Y-%m-%d")
binSize = "histohour" 
data = []
endtime = startTime
startTime = startTime + timedelta(hours=1950)
while endtime < datetime.now():
    try:
        startTime_timestamp = int(startTime.timestamp())
        batch = get_cryptocompare_data(fsym, tsym, binSize,
                                       startTime_timestamp)
        batch = batch['Data']['Data']
        data.extend(batch)
        if batch:
            startTime = datetime.fromtimestamp(
                batch[-1]["time"]) + timedelta(hours=2000)
            endtime = datetime.fromtimestamp(
                batch[-1]["time"]) + timedelta(hours=1)
            print(f"Got data up to {endtime}, total {len(data)} rows.")
        time.sleep(2)
    except Exception as e:
        print(f"Error occurred: {e}")
        break

df = pd.DataFrame(data)

df['time'] = df['time'].astype(int)
# Check for duplicates
df.drop_duplicates(subset='time', keep='first', inplace=True)
# Check for continuity
time_diff = df['time'].diff()
discontinuities = df[time_diff.ne(3600)]

print(discontinuities)

df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')

startTime = pd.Timestamp(config["startTime"], tz='UTC')

df = df[df['time'] >= startTime]
df = df.rename(columns={'time': 'timestamp', 'volumeto': 'volume'})

filename = f"dataset/stackingdata/{symbol}/{symbol}_1h.csv"
df.to_csv(filename, index=False)
print(
    f"Saved {len(df)} rows of 1d data from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']} to {filename}"
)
