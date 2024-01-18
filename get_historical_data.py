import requests
import json
import csv
import os
import time
from datetime import datetime, timedelta


def get_bitmex_data(symbol, binSize, startTime):
    url = f"https://www.bitmex.com/api/v1/trade/bucketed?binSize={binSize}&partial=false&symbol={symbol}&startTime={startTime}&count=750&reverse=false"
    response = requests.get(url)
    data = json.loads(response.text)
    return data


def save_data_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)

with open('config.json') as f:
    config = json.load(f)

symbol = config['symbol']
binSize = "1d"
startTime = datetime.strptime(
    config['startTime'], "%Y-%m-%d")

os.makedirs(f'dataset/data/{symbol}', exist_ok=True)


data = []
while startTime < datetime.now():
    try:
        batch = get_bitmex_data(
            symbol, binSize, startTime.strftime("%Y-%m-%d"))
        data.extend(batch)
        if batch:
            startTime = datetime.strptime(
                batch[-1]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(days=1)
            print(f"Got data up to {startTime}, total {len(data)} rows.")
        else:
            startTime += timedelta(days=1)
    except Exception as e:
        print(f"Error occurred: {e}")
        break

filename = f'dataset/data/{symbol}/{symbol}_1d.csv'
save_data_to_csv(data, filename)
print(f"Saved {len(data)} rows of 1d data from {data[0]['timestamp']} to {data[-1]['timestamp']} to {filename}")


startTime = datetime.strptime(config['startTime'], "%Y-%m-%d")

binSize = "1h"
data = []
while startTime < datetime.now():
    try:
        batch = get_bitmex_data(
            symbol, binSize, startTime.strftime("%Y-%m-%dT%H:%M:%S"))
        data.extend(batch)
        if batch:
            startTime = datetime.strptime(
                batch[-1]['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(hours=1)
            print(f"Got data up to {startTime}, total {len(data)} rows.")
        else:
            startTime += timedelta(hours=1)
        time.sleep(2)
    except Exception as e:
        print(f"Error occurred: {e}")
        break

if data:
    filename = f'dataset/data/{symbol}/{symbol}_1h.csv'
    save_data_to_csv(data, filename)
    print(f"Saved {len(data)} rows of 1h data from {data[0]['timestamp']} to {data[-1]['timestamp']} to {filename}")
else:
    print("No 1h data to save.")
