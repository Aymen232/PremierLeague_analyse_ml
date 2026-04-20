import pandas as pd
import joblib


MODEL_PATH = "models/random_forest.pkl"
HOME_ENCODER_PATH = "models/home_team_encoder.pkl"
AWAY_ENCODER_PATH = "models/away_team_encoder.pkl"
RESULT_ENCODER_PATH = "models/result_encoder.pkl"
DATA_PATH = "data/features/matches_features.csv"


def load_artifacts():
    print("Loading saved model and encoders...")

    model = joblib.load(MODEL_PATH)
    le_home = joblib.load(HOME_ENCODER_PATH)
    le_away = joblib.load(AWAY_ENCODER_PATH)
    le_result = joblib.load(RESULT_ENCODER_PATH)

    return model, le_home, le_away, le_result


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_team_last_stats(df, team):
    team_matches = df[
        (df["home_team"] == team) | (df["away_team"] == team)
    ].sort_values("date")

    if team_matches.empty:
        raise ValueError(f"Aucun match trouvé pour l'équipe : {team}")

    last_match = team_matches.iloc[-1]

    if last_match["home_team"] == team:
        return {
            "avg_goals": last_match["home_avg_goals_last5"],
            "avg_conceded": last_match["home_avg_conceded_last5"],
            "form": last_match["home_points_form_last5"],
        }

    return {
        "avg_goals": last_match["away_avg_goals_last5"],
        "avg_conceded": last_match["away_avg_conceded_last5"],
        "form": last_match["away_points_form_last5"],
    }


def main():
    model, le_home, le_away, le_result = load_artifacts()
    df = load_data()

    all_teams = sorted(set(df["home_team"]).union(set(df["away_team"])))

    print("\nAvailable teams:")
    for team in all_teams:
        print("-", team)

    home_team = input("\nEnter HOME team: ").strip()
    away_team = input("Enter AWAY team: ").strip()

    if home_team not in le_home.classes_:
        print(f"Unknown home team: {home_team}")
        return

    if away_team not in le_away.classes_:
        print(f"Unknown away team: {away_team}")
        return

    home_stats = get_team_last_stats(df, home_team)
    away_stats = get_team_last_stats(df, away_team)

    form_diff = home_stats["form"] - away_stats["form"]
    goals_diff_form = home_stats["avg_goals"] - away_stats["avg_goals"]
    conceded_diff_form = home_stats["avg_conceded"] - away_stats["avg_conceded"]

    X_pred = pd.DataFrame(
        [[
            le_home.transform([home_team])[0],
            le_away.transform([away_team])[0],
            home_stats["avg_goals"],
            away_stats["avg_goals"],
            home_stats["avg_conceded"],
            away_stats["avg_conceded"],
            home_stats["form"],
            away_stats["form"],
            form_diff,
            goals_diff_form,
            conceded_diff_form,
        ]],
        columns=[
            "home_team_enc",
            "away_team_enc",
            "home_avg_goals_last5",
            "away_avg_goals_last5",
            "home_avg_conceded_last5",
            "away_avg_conceded_last5",
            "home_points_form_last5",
            "away_points_form_last5",
            "form_diff",
            "goals_diff_form",
            "conceded_diff_form",
        ]
    )

    prediction_encoded = model.predict(X_pred)[0]
    prediction_label = le_result.inverse_transform([prediction_encoded])[0]

    probabilities = model.predict_proba(X_pred)[0]

    print("\n===== PREDICTION =====")
    print(f"{home_team} vs {away_team}")
    print(f"Predicted result: {prediction_label}")

    print("\nProbabilities:")
    for class_name, prob in zip(le_result.classes_, probabilities):
        print(f"{class_name}: {prob:.2%}")


if __name__ == "__main__":
    main()