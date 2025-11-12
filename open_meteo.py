import requests

lat, lon = 10.5, -60.2
url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={lat}&longitude={lon}"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation,cloudcover,windspeed_10m"
    "&forecast_days=1&timezone=auto"
)

resp = requests.get(url).json()
print(resp)
