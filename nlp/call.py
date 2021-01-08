import requests

response = requests.get("http://localhost:8080/api/sites")
data = response.json()
print(data[0])