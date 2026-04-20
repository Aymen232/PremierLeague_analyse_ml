import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_PATH = "data/features/matches_features.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
HOME_ENCODER_PATH = os.path.join(MODEL_DIR, "home_team_encoder.pkl")
AWAY_ENCODER_PATH = os.path.join(MODEL_DIR, "away_team_encoder.pkl")
RESULT_ENCODER_PATH = os.path.join(MODEL_DIR, "result_encoder.pkl")


def load_data():
    print("Loading feature dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    required_columns = [
        "date",
        "home_team",
        "away_team",
        "home_avg_goals_last5",
        "away_avg_goals_last5",
        "home_avg_conceded_last5",
        "away_avg_conceded_last5",
        "home_points_form_last5",
        "away_points_form_last5",
        "result",
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing_cols}")

    df = df.dropna(subset=required_columns).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    print("\nDataset shape after cleaning:", df.shape)
    print("\nClass distribution:")
    print(df["result"].value_counts())

    return df


def prepare_features(df):
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    le_result = LabelEncoder()

    df["home_team_enc"] = le_home.fit_transform(df["home_team"])
    df["away_team_enc"] = le_away.fit_transform(df["away_team"])

    # Features dérivées
    df["form_diff"] = df["home_points_form_last5"] - df["away_points_form_last5"]
    df["goals_diff_form"] = df["home_avg_goals_last5"] - df["away_avg_goals_last5"]
    df["conceded_diff_form"] = (
        df["home_avg_conceded_last5"] - df["away_avg_conceded_last5"]
    )

    X = df[
        [
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
    ]

    y = le_result.fit_transform(df["result"])

    return X, y, le_home, le_away, le_result, df


def chronological_split(df, X, y, train_ratio=0.8):
    split_index = int(len(df) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    print("\n===== CHRONOLOGICAL SPLIT =====")
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)
    print("Train period:", df.iloc[0]["date"], "->", df.iloc[split_index - 1]["date"])
    print("Test period :", df.iloc[split_index]["date"], "->", df.iloc[-1]["date"])

    return X_train, X_test, y_train, y_test


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, class_names):
    print(f"\n{'=' * 20} {model_name} {'=' * 20}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return acc


def save_artifacts(model, le_home, le_away, le_result):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le_home, HOME_ENCODER_PATH)
    joblib.dump(le_away, AWAY_ENCODER_PATH)
    joblib.dump(le_result, RESULT_ENCODER_PATH)

    print("\n===== SAVED ARTIFACTS =====")
    print(f"Model saved to         : {MODEL_PATH}")
    print(f"Home encoder saved to  : {HOME_ENCODER_PATH}")
    print(f"Away encoder saved to  : {AWAY_ENCODER_PATH}")
    print(f"Result encoder saved to: {RESULT_ENCODER_PATH}")


def main():
    df = load_data()
    X, y, le_home, le_away, le_result, df = prepare_features(df)
    X_train, X_test, y_train, y_test = chronological_split(df, X, y)

    class_names = list(le_result.classes_)
    print("\nTarget classes:", class_names)

    # Modèle 1 : Logistic Regression
    logistic_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    )

    logistic_acc = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_train,
        X_test,
        y_train,
        y_test,
        class_names
    )

    # Modèle 2 : Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    rf_acc = evaluate_model(
        "Random Forest",
        rf_model,
        X_train,
        X_test,
        y_train,
        y_test,
        class_names
    )

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(f"Logistic Regression Accuracy : {logistic_acc:.4f}")
    print(f"Random Forest Accuracy      : {rf_acc:.4f}")

    if rf_acc >= logistic_acc:
        print("\nBest model: Random Forest")
        best_model = rf_model
    else:
        print("\nBest model: Logistic Regression")
        best_model = logistic_model

    save_artifacts(best_model, le_home, le_away, le_result)


if __name__ == "__main__":
    main()