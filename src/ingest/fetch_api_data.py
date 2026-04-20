import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

url = "https://v3.football.api-sports.io/fixtures"

headers = {
    "x-apisports-key": API_KEY
}

params = {
    "league": 39,
    "season": 2024
}

response = requests.get(url, headers=headers, params=params)
data = response.json()

matches = []

for fixture in data["response"]:
    matches.append({
        "date": fixture["fixture"]["date"],
        "home_team": fixture["teams"]["home"]["name"],
        "away_team": fixture["teams"]["away"]["name"],
        "home_goals": fixture["goals"]["home"],
        "away_goals": fixture["goals"]["away"],
        "status": fixture["fixture"]["status"]["short"]
    })

df = pd.DataFrame(matches)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/matches.csv", index=False)

print("Premier League data saved ✅")