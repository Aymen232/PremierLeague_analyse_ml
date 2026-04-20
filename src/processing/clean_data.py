import pandas as pd
import os

INPUT_PATH = "data/raw/matches.csv"
OUTPUT_PATH = "data/processed/matches_cleaned.csv"


def get_match_result(home_goals, away_goals):
    if home_goals > away_goals:
        return "Home Win"
    elif home_goals < away_goals:
        return "Away Win"
    else:
        return "Draw"


def main():
    print("Loading raw dataset...")

    df = pd.read_csv(INPUT_PATH)

    print("Initial shape:", df.shape)

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep only finished matches
    df = df[df["status"] == "FT"].copy()

    # Remove missing values
    df = df.dropna(subset=[
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals"
    ])

    # Convert goals to numeric
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")

    df = df.dropna(subset=["home_goals", "away_goals"])

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # Create ML target
    df["result"] = df.apply(
        lambda row: get_match_result(row["home_goals"], row["away_goals"]),
        axis=1
    )

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    print("Final shape:", df.shape)

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Cleaned dataset saved to:", OUTPUT_PATH)
    print(df.head())


if __name__ == "__main__":
    main()