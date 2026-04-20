import pandas as pd
import os

INPUT_PATH = "data/processed/matches_cleaned.csv"
OUTPUT_PATH = "data/features/matches_features.csv"

WINDOW = 5  # nombre de matchs passés


def compute_team_form_features(df):
    df = df.sort_values("date").reset_index(drop=True)

    # Lists to store features
    home_avg_goals = []
    away_avg_goals = []

    home_avg_conceded = []
    away_avg_conceded = []

    home_points_form = []
    away_points_form = []

    # Historique des équipes
    history = {}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Initialisation si équipe jamais vue
        if home not in history:
            history[home] = []

        if away not in history:
            history[away] = []

        # ===== FEATURES HOME =====
        last_home = history[home][-WINDOW:]

        if len(last_home) == 0:
            home_avg_goals.append(0.0)
            home_avg_conceded.append(0.0)
            home_points_form.append(0.0)
        else:
            home_avg_goals.append(
                sum(match["scored"] for match in last_home) / len(last_home)
            )
            home_avg_conceded.append(
                sum(match["conceded"] for match in last_home) / len(last_home)
            )
            home_points_form.append(
                sum(match["points"] for match in last_home) / len(last_home)
            )

        # ===== FEATURES AWAY =====
        last_away = history[away][-WINDOW:]

        if len(last_away) == 0:
            away_avg_goals.append(0.0)
            away_avg_conceded.append(0.0)
            away_points_form.append(0.0)
        else:
            away_avg_goals.append(
                sum(match["scored"] for match in last_away) / len(last_away)
            )
            away_avg_conceded.append(
                sum(match["conceded"] for match in last_away) / len(last_away)
            )
            away_points_form.append(
                sum(match["points"] for match in last_away) / len(last_away)
            )

        # ===== UPDATE HISTORIQUE APRES LE MATCH =====
        home_goals = row["home_goals"]
        away_goals = row["away_goals"]

        if home_goals > away_goals:
            home_pts = 3
            away_pts = 0
        elif home_goals < away_goals:
            home_pts = 0
            away_pts = 3
        else:
            home_pts = 1
            away_pts = 1

        history[home].append(
            {
                "scored": home_goals,
                "conceded": away_goals,
                "points": home_pts
            }
        )

        history[away].append(
            {
                "scored": away_goals,
                "conceded": home_goals,
                "points": away_pts
            }
        )

    # Ajout des nouvelles colonnes
    df["home_avg_goals_last5"] = home_avg_goals
    df["away_avg_goals_last5"] = away_avg_goals
    df["home_avg_conceded_last5"] = home_avg_conceded
    df["away_avg_conceded_last5"] = away_avg_conceded
    df["home_points_form_last5"] = home_points_form
    df["away_points_form_last5"] = away_points_form

    return df


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    df = compute_team_form_features(df)

    os.makedirs("data/features", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Feature dataset created ✅")
    print(df.head())
    print("Shape:", df.shape)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()