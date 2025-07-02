import requests

url = "http://0.0.0.0:8000"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()  # Convert response to a Python dict
    print(data)
else:
    print(f"Request failed with status code {response.status_code}")
