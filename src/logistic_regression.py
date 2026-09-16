import polars as pl
import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss, roc_auc_score)

INPUT_PATH = Path("data/processed/team_game_features.parquet")

MODEL_OUTPUT_PATH = Path("data/models/logistic_regression_model.joblib")
RESULTS_OUTPUT_PATH = Path("data/processed/logistic_regression_results.csv")
COEFFICIENTS_OUTPUT_PATH = Path("data/processed/logistic_regression_coefficients.csv")

TRAIN_END_SEASON = 2021
TEST_START_SEASON = 2022

FEATURES = [
    "off_epa_diff",
    "def_epa_diff",
    "pace_diff",
    "rest_diff"
]

def load_data():
    print(f"Loading team-game features from: {INPUT_PATH}")

    team_games = pl.read_parquet(INPUT_PATH)

    print(f"Team-game rows: {len(team_games):,}")

    return team_games

def build_game_dataset(team_games):
    print("\nBuilding game-level dataset")

    home = team_games.filter(pl.col("home") == 1).select([
        "game_id", "season", "week", "gameday",
        pl.col("team").alias("home_team"),
        pl.col("opponent").alias("away_team"),
        pl.col("win").alias("home_win"),
        pl.col("pregame_off_epa").alias("home_off_epa"),
        pl.col("pregame_def_epa").alias("home_def_epa"),
        pl.col("pregame_pace").alias("home_pace"),
        pl.col("rest_days").alias("home_rest_days")
    ])

    away = team_games.filter(pl.col("home") == 0).select([
        "game_id", "season", "week", "gameday",
        pl.col("team").alias("away_team"),
        pl.col("opponent").alias("home_team"),
        pl.col("win").alias("away_win"),
        pl.col("pregame_off_epa").alias("away_off_epa"),
        pl.col("pregame_def_epa").alias("away_def_epa"),
        pl.col("pregame_pace").alias("away_pace"),
        pl.col("rest_days").alias("away_rest_days")
    ])

    games = home.join(away, on = ["game_id", "home_team", "away_team"], how = "inner")

    games = games.with_columns([
        (pl.col("home_off_epa") - pl.col("away_off_epa")).alias("off_epa_diff"),
        (pl.col("home_def_epa") - pl.col("away_def_epa")).alias("def_epa_diff"),
        (pl.col("home_pace") - pl.col("away_pace")).alias("pace_diff"),
        (pl.col("home_rest_days") - pl.col("away_rest_days")).alias("rest_diff")
    ])

    return games

def check_games_dataset(games):
    print("\nChecking game-level dataset")

    print(f"Games: {len(games):,}")

    print("\nGames by season:")
    print(games.group_by("season").agg(pl.len().alias("games")).sort("season"))

    print("\nChecking duplicate games:")
    duplicates = (games.group_by("game_id").agg(pl.len().alias("rows")).filter(pl.col("rows") > 1))

    print(duplicates)

    print("\nChecking missing model features:")

    missing = games.filter(pl.any_horizontal([pl.col(feature).is_null() for feature in FEATURES]))

    print(f"Games with missing features: {len(missing):,}")

    if len(missing) > 0:
        print(missing.select([
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            *FEATURES
        ]).head(20))

def split_data(games):
    print("\nCreating chronological train/test split")

    train = games.filter(pl.col("season") <= TRAIN_END_SEASON)
    test = games.filter(pl.col("season") > TRAIN_END_SEASON)

    print(f"Training seasons: {train.get_column('season').min()} - {train.get_column('season').max()}")

    print(f"Testing seasons: {test.get_column('season').min()} - {test.get_column('season').max()}")

    print(f"Training games: {len(train):,}")
    print(f"Testing games: {len(test):,}")

    return train, test

def prepare_model_data(train, test):
    train = train.drop_nulls(FEATURES + ["home_win"])
    test = test.drop_nulls(FEATURES + ["home_win"])

    X_train = train.select(FEATURES).to_numpy()
    y_train = train.get_column("home_win").to_numpy()

    X_test = test.select(FEATURES).to_numpy()
    y_test = test.get_column("home_win").to_numpy()

    return train, test, X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    print("\nTraining logistic regression")

    model = LogisticRegression(max_iter = 1000, random_state = 42)

    model.fit(X_train, y_train)

    print("Model training complete")

    return model

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    loss = log_loss(y_test, probabilities)

    matrix = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Log Loss: {loss:.4f}")
    print("Confusion Matrix:")
    print(matrix)

    return predictions, probabilities, accuracy, auc, loss, matrix

def save_results(test, predictions, probabilities):
    print("\nSaving predictions")

    results = test.select([
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_win"
    ]).with_columns([
        pl.Series("predicted_home_win", predictions),
        pl.Series("predicted_home_win_prob", probabilities)
    ])

    RESULTS_OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    results.write_csv(RESULTS_OUTPUT_PATH)

    print(f"Predictions saved to: {RESULTS_OUTPUT_PATH}")
    return results

def save_coefficients(model):
    print("\nSaving model coefficients")

    coefficients = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_[0]
    })

    coefficients["odds_ratio"] = coefficients["coefficient"].apply(lambda x: float(__import__("math").exp(x)))

    coefficients.to_csv(COEFFICIENTS_OUTPUT_PATH, index = False)

    print("\nModel coefficients:")
    print(coefficients)

def save_model(model):
    print("\nSaving model")

    MODEL_OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

def main():
    team_games = load_data()

    games = build_game_dataset(team_games)

    check_games_dataset(games)

    train, test = split_data(games)

    (train, test, X_train, y_train, X_test, y_test) = prepare_model_data(train, test)

    print("\nFinal model data:")
    print(f"Training games: {len(X_train):,}")
    print(f"Testing games: {len(X_test):,}")

    print("\nFeatures:")
    print(FEATURES)

    model = train_model(X_train, y_train)

    (predictions, probabilities, accuracy, auc, loss, matrix) = evaluate_model(model, X_test, y_test)

    save_results(test, predictions, probabilities)

    save_coefficients(model)

    save_model(model)

if __name__ == "__main__":
    main()