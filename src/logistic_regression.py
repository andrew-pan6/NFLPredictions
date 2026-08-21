import polars as pl
import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss, roc_auc_score)

INPUT_PATH = Path("data/processed/team_game_features.parquet")

MODEL_OUTPUT_PATH = Path("data/models/logsitic_regression_model.joblib")
RESULTS_OUTPUT_PATH = Path("data/processed/logistic_regression_results.parquet")
COEFFICIENTS_OUTPUT_PATH = Path("data/processed/logsitic_regression_coefficients.csv")

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