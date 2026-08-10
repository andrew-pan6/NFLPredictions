import nflreadpy as nfl
import polars as pl
from pathlib import Path

START_SEASON = 1998
END_SEASON = 2025

OUTPUT_PATH = Path("data/processed/team_game_features.parquet")

def load_data():
    seasons = list(range(START_SEASON, END_SEASON + 1))

    print(f"Loading schedules from {START_SEASON} - {END_SEASON}...")
    schedules = nfl.load_schedules(seasons = seasons)

    print(f"Loading schedules from {START_SEASON} - {END_SEASON}...")
    pbp = nfl.load_pbp(seasons = seasons)

    print(f"Schedules: {len(schedules):,} rows")
    print(f"Play-by-play: {len(pbp):,} rows")

    return schedules, pbp

def build_games(schedules):
    games = (schedules.filter(pl.col("game_type") == "REG").select([
        "game_id", "season", "week", "gameday", "away_team", "home_team",
        "away_score", "home_score", "away_rest", "home_rest"
    ])).with_columns([pl.col("gameday").str.to_date(), (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_win")])

    return games

def build_play_stats(pbp):
    plays = (pbp.filter(
        (pl.col("season_type") == "REG") &
        pl.col("posteam").is_not_null() &
        pl.col("defteam").is_not_null() &
        pl.col("epa").is_not_null() &
        (pl.col("play_type").is_in(["pass", "run"])) &
        (pl.col("qb_kneel") != 1) &
        (pl.col("qb_spike") != 1)&
        (pl.col("aborted_play") != 1) &
        (pl.col("play_deleted") != 1)
    )).select([
        "game_id", "season", "week", "posteam", "defteam", "eta"
    ])

    offensive = plays.group_by([
        "game_id", "season", "week", "posteam"
    ]).agg([
        pl.col("epa").mean().alias("off_epa"),
        pl.col("epa").sum().alias("off_epa_total"),
        pl.len().alias("off_plays")
    ]).rename({"posteam": "team"})

    defensive = plays.group_by([
        "game_id", "season", "week", "defteam"
    ]).agg([
        pl.col("epa").mean().alias("def_epa_allowed"),
        pl.col("epa").sum().alias("off_epa_def_epa_allowed_total"),
    ]).rename({"defteam": "team"})

    team_stats = offensive.join(defensive, on =["game_id", "season", "week", "team"], how = "inner")

    return team_stats

def build_team_games(games, team_stats):
    