import requests

url = "https://api.binance.us/api/v3/klines"

params = {
    "symbol": "BTCUSDT",
    "interval": "15m",
    "limit": 5
}

response = requests.get(url, params=params, timeout=15)
response.raise_for_status()

data = response.json()

print(data)