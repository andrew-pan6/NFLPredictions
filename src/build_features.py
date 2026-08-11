import nflreadpy as nfl
import polars as pl
from pathlib import Path

START_SEASON = 1999
END_SEASON = 2025

OUTPUT_PATH = Path("data/processed/team_game_features.parquet")


def load_data():
    seasons = list(range(START_SEASON, END_SEASON + 1))

    print(f"Loading schedules from {START_SEASON} - {END_SEASON}")
    schedules = nfl.load_schedules(seasons = seasons)

    print(f"Loading play-by-play from {START_SEASON} - {END_SEASON}")
    pbp = nfl.load_pbp(seasons = seasons)

    print(f"Schedules: {len(schedules):,} rows")
    print(f"Play-by-play: {len(pbp):,} rows")

    return schedules, pbp


def normalize_team_names(df):
    return df.with_columns(
        pl.col("team").replace({
            "STL": "LA",
            "SD": "LAC",
            "OAK": "LV",
            "WAS": "WAS"
        }).alias("team")
    )


def normalize_schedule_team_names(games):
    return games.with_columns([
        pl.col("away_team").replace({
            "STL": "LA",
            "SD": "LAC",
            "OAK": "LV",
            "WAS": "WAS"
        }).alias("away_team"),
        pl.col("home_team").replace({
            "STL": "LA",
            "SD": "LAC",
            "OAK": "LV",
            "WAS": "WAS"
        }).alias("home_team")
    ])


def build_games(schedules):
    games = schedules.filter(
        pl.col("game_type") == "REG"
    ).select([
        "game_id", "season", "week", "gameday", "away_team", "home_team",
        "away_score", "home_score", "away_rest", "home_rest"
    ]).with_columns([
        pl.col("gameday").str.to_date(),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_win"),
        (pl.col("home_score") - pl.col("away_score")).alias("point_margin")
    ])

    games = normalize_schedule_team_names(games)

    return games


def build_play_stats(pbp):
    plays = pbp.filter(
        (pl.col("season_type") == "REG") &
        pl.col("posteam").is_not_null() &
        pl.col("defteam").is_not_null() &
        pl.col("epa").is_not_null() &
        (pl.col("play_type").is_in(["pass", "run"])) &
        (pl.col("qb_kneel").fill_null(0) != 1) &
        (pl.col("qb_spike").fill_null(0) != 1) &
        (pl.col("aborted_play").fill_null(0) != 1) &
        (pl.col("play_deleted").fill_null(0) != 1)
    ).select([
        "game_id", "season", "week", "posteam", "defteam", "epa"
    ]).with_columns([
        pl.col("posteam").replace({
            "STL": "LA",
            "SD": "LAC",
            "OAK": "LV",
            "WAS": "WAS"
        }).alias("posteam"),
        pl.col("defteam").replace({
            "STL": "LA",
            "SD": "LAC",
            "OAK": "LV",
            "WAS": "WAS"
        }).alias("defteam")
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
        pl.col("epa").sum().alias("def_epa_allowed_total"),
        pl.len().alias("def_plays")
    ]).rename({"defteam": "team"})

    team_stats = offensive.join(
        defensive,
        on = ["game_id", "season", "week", "team"],
        how = "left"
    )

    return team_stats


def build_team_games(games, team_stats):
    home = games.select([
        "game_id", "season", "week", "gameday",
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(1).alias("home"),
        pl.col("home_win").alias("win"),
        pl.col("home_score").alias("points_for"),
        pl.col("away_score").alias("points_against"),
        pl.col("home_rest").alias("rest_days"),
        pl.col("away_rest").alias("opponent_rest_days")
    ])

    away = games.select([
        "game_id", "season", "week", "gameday",
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(0).alias("home"),
        (1 - pl.col("home_win")).alias("win"),
        pl.col("away_score").alias("points_for"),
        pl.col("home_score").alias("points_against"),
        pl.col("away_rest").alias("rest_days"),
        pl.col("home_rest").alias("opponent_rest_days")
    ])

    team_games = pl.concat([home, away])

    team_games = team_games.join(
        team_stats,
        on=["game_id", "season", "week", "team"],
        how="left"
    )

    team_games = team_games.with_columns(
        (pl.col("rest_days") - pl.col("opponent_rest_days")).alias("rest_advantage")
    )

    return team_games.sort(["team", "gameday"])


def check_team_game_joins(games, pbp, team_stats, team_games):
    print("\nChecking raw PBP for missing games:")

    missing_games = games.join(
        team_stats.select("game_id").unique(),
        on = "game_id",
        how = "anti"
    )

    print(missing_games.select([
        "game_id", "season", "week", "away_team", "home_team"
    ]))

    if len(missing_games) > 0:
        print("\nRaw PBP rows for missing games:")

        missing_game_ids = missing_games.get_column("game_id").implode()

        print(
            pbp.filter(
                pl.col("game_id").is_in(missing_game_ids)
            ).group_by("game_id").agg([
                pl.len().alias("pbp_rows"),
                pl.col("posteam").is_not_null().sum().alias("posteam_rows"),
                pl.col("defteam").is_not_null().sum().alias("defteam_rows"),
                pl.col("epa").is_not_null().sum().alias("epa_rows")
            ])
        )

    print("\nChecking historical team identifiers:")

    print(
        games.select([
            pl.col("away_team").alias("team"),
            "season"
        ]).vstack(
            games.select([
                pl.col("home_team").alias("team"),
                "season"
            ])
        ).filter(
            pl.col("team").is_in(["LA", "LAC", "LV", "WAS"])
        ).group_by([
            "season", "team"
        ]).agg(
            pl.len().alias("games")
        ).sort([
            "season", "team"
        ])
    )

    print("\nChecking team-game joins")

    missing_stats = team_games.filter(
        pl.col("off_epa").is_null()
    )

    print(f"Team-games missing play stats: {len(missing_stats):,}")

    print("\nMissing play stats by season:")
    print(
        missing_stats.group_by("season")
        .agg(pl.len().alias("missing"))
        .sort("season")
    )

    print("\nMissing play stats by team:")
    print(
        missing_stats.group_by("team")
        .agg(pl.len().alias("missing"))
        .sort("missing", descending=True)
    )

    print("\nExample missing play stats:")
    print(
        missing_stats.select([
            "game_id", "season", "week", "team", "opponent"
        ]).head(20)
    )

    print("\nChecking game IDs")

    game_ids = games.get_column("game_id").unique()
    team_stat_game_ids = team_stats.get_column("game_id").unique()

    print(f"Games: {len(game_ids):,}")
    print(f"Team-stat game IDs: {len(team_stat_game_ids):,}")

    # Changed from is_in() to an anti join to avoid the Polars warning.
    missing_game_stats = games.join(
        team_stats.select("game_id").unique(),
        on="game_id",
        how="anti"
    )

    print("\nGames missing from team stats:")
    print(
        missing_game_stats.select([
            "game_id", "season", "week"
        ])
    )

    print("\nChecking missing team keys")

    expected_team_keys = team_games.select([
        "game_id", "team"
    ]).unique()

    actual_team_keys = team_stats.select(["game_id", "team"]).unique()

    missing_team_keys = expected_team_keys.join(
        actual_team_keys,
        on = ["game_id", "team"],
        how = "anti"
    )

    print(f"Missing team-stat keys: {len(missing_team_keys):,}")
    print(missing_team_keys.head(30))


def add_pregame_features(team_games):
    team_games = team_games.sort(["team", "gameday"])

    team_games = team_games.with_columns([
        pl.col("off_epa_total").fill_null(0).shift(1).cum_sum().over("team").alias("previous_off_epa_total"),
        pl.col("def_epa_allowed_total").fill_null(0).shift(1).cum_sum().over("team").alias("previous_def_epa_total"),
        pl.col("off_plays").fill_null(0).shift(1).cum_sum().over("team").alias("previous_plays"),
        pl.col("def_plays").fill_null(0).shift(1).cum_sum().over("team").alias("previous_def_plays"),
        pl.col("off_plays").is_not_null().cast(pl.Int8).shift(1).cum_sum().over("team").alias("previous_games")
    ])

    team_games = team_games.with_columns([
        pl.when(pl.col("previous_plays") > 0)
        .then(
            pl.col("previous_off_epa_total") / pl.col("previous_plays")
        )
        .otherwise(None)
        .alias("pregame_off_epa"),

        pl.when(pl.col("previous_def_plays") > 0)
        .then(
            pl.col("previous_def_epa_total") / pl.col("previous_def_plays")
        )
        .otherwise(None)
        .alias("pregame_def_epa"),

        pl.when(pl.col("previous_games") > 0)
        .then(
            pl.col("previous_plays") / pl.col("previous_games")
        )
        .otherwise(None)
        .alias("pregame_pace")
    ])

    return team_games


def check_pregame_features(team_games):
    missing = team_games.filter(
        pl.col("pregame_off_epa").is_null()
    )

    print("\nMissing pregame features:")
    print(
        missing.select([
            "season", "week", "team", "opponent",
            "previous_games", "previous_plays",
            "pregame_off_epa", "pregame_def_epa", "pregame_pace"
        ]).head(20)
    )

    missing = team_games.filter(pl.col("pregame_off_epa").is_null())

    missing_with_reason = missing.with_columns(
        pl.when(pl.col("previous_games").fill_null(0) == 0)
        .then(pl.lit("no_previous_history"))
        .otherwise(pl.lit("missing_play_stats"))
        .alias("missing_reason")
    )

    print("\nMissing pregame features by cause:")
    print(
        missing_with_reason.group_by("missing_reason")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    print("\nMissing pregame features by team:")
    print(
        missing.group_by("team")
        .agg(pl.len().alias("missing"))
        .sort("missing", descending=True)
    )

    print(f"Team-games without pregame history: {len(missing):,}")

    print("\nMissing pregame features by season:")
    print(
        missing.group_by("season")
        .agg(pl.len().alias("missing"))
        .sort("season")
    )


def check_dataset(team_games, games):
    print("\nChecking duplicate team-game rows:")

    duplicates = (
        team_games.group_by(["game_id", "team"])
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
    )

    print(duplicates)

    print("\nChecking impossible wins:")

    impossible_wins = team_games.filter(
        ((pl.col("win") == 1) & (pl.col("points_for") <= pl.col("points_against"))) |
        ((pl.col("win") == 0) & (pl.col("points_for") > pl.col("points_against")))
    )

    print(impossible_wins)

    print("\nChecking expected row count:")

    expected_rows = len(games) * 2

    print(f"Expected rows: {expected_rows:,}")
    print(f"Actual rows: {len(team_games):,}")

    print(f"\nColumns: {len(team_games.columns)}")

    print("\nColumns:")
    print(team_games.columns)


def main():
    schedules, pbp = load_data()

    print("\nBuilding games")
    games = build_games(schedules)
    print(f"Regular-season games: {len(games):,}")

    print("\nCalculating play-level statistics")
    team_stats = build_play_stats(pbp)
    print(f"Team-game stat rows: {len(team_stats):,}")

    print("\nTeam-game stats by season:")
    print(
        team_stats.group_by("season")
        .agg(pl.len().alias("team_games"))
        .sort("season")
    )

    print("\nBuilding team-game dataset")
    team_games = build_team_games(games, team_stats)

    check_team_game_joins(games, pbp, team_stats, team_games)

    print("\nCalculating pregame features")
    team_games = team_games.filter(pl.col("off_epa").is_not_null())
    team_games = add_pregame_features(team_games)

    check_pregame_features(team_games)

    check_dataset(team_games, games)

    print("\nTeam-games missing play stats:")

    missing_stats = team_games.filter(
        pl.col("off_epa").is_null()
    )

    print(
        missing_stats.select(["season", "week", "game_id", "team", "opponent"]).head(20)
    )

    print(f"Team-games missing play stats: {len(missing_stats):,}")

    print("\nMissing play stats by season:")
    print(
        missing_stats.group_by("season")
        .agg(pl.len().alias("missing"))
        .sort("season")
    )

    print("\nSaving dataset")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    team_games.write_parquet(OUTPUT_PATH)

    print(f"Saved to: {OUTPUT_PATH}")

    print("\nSample:")
    print(
        team_games.select([
            "season", "week", "team", "opponent", "win",
            "pregame_off_epa", "pregame_def_epa", "pregame_pace",
            "rest_days", "rest_advantage", "home"
        ]).tail(10)
    )


if __name__ == "__main__":
    main()