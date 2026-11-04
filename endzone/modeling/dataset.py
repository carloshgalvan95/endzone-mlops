"""Load marts for training and check that a requested feature set is usable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from endzone.data.warehouse import DatabricksClient
from endzone.features.pregame import feature_columns_for
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

# A requested feature column at or above this null share is not a real signal.
DEFAULT_MAX_NULL_FRACTION: float = 0.5
FEATURE_COLUMN_REBUILD_HINT: str = "run make dbt-build-databricks"


def load_data(
    fixture_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``fct_games`` and ``fct_team_game_epa`` from Databricks or fixtures.

    Args:
        fixture_path: Optional directory of parquet fixtures. None reads the warehouse.

    Returns:
        ``(fct_games, fct_team_game_epa)``.
    """
    if fixture_path:
        fixture_dir = Path(fixture_path)
        games_file = fixture_dir / "fct_games.parquet"
        epa_file = fixture_dir / "fct_team_game_epa.parquet"
        if not games_file.exists() or not epa_file.exists():
            raise FileNotFoundError(
                f"Fixtures not found in {fixture_dir}. Expected: {games_file.name}, {epa_file.name}"
            )
        logger.info(f"Loading data from fixtures: {fixture_dir}")
        fct_games = pd.read_parquet(games_file)
        fct_team_game_epa = pd.read_parquet(epa_file)
    else:
        logger.info("Loading data from Databricks")
        with DatabricksClient() as client:
            fct_games = client.read_table("fct_games")
            fct_team_game_epa = client.read_table("fct_team_game_epa")
    logger.info(f"Loaded {len(fct_games)} games, {len(fct_team_game_epa)} team-game records")
    return fct_games, fct_team_game_epa


def load_upcoming_games(fixture_path: str | None = None) -> pd.DataFrame:
    """Load ``fct_upcoming_games`` from Databricks or a fixture directory.

    Args:
        fixture_path: Optional directory containing ``fct_upcoming_games.parquet``.
            None reads the warehouse.

    Returns:
        One row per scheduled game that has no final score.

    Raises:
        FileNotFoundError: The fixture directory has no upcoming-games file.
    """
    if fixture_path:
        fixture_dir = Path(fixture_path)
        upcoming_file = fixture_dir / "fct_upcoming_games.parquet"
        if not upcoming_file.exists():
            raise FileNotFoundError(
                f"Fixtures not found in {fixture_dir}. Expected: {upcoming_file.name}"
            )
        logger.info("Loading upcoming games from fixtures: %s", fixture_dir)
        return pd.read_parquet(upcoming_file)
    logger.info("Loading upcoming games from Databricks")
    with DatabricksClient() as client:
        return client.read_table("fct_upcoming_games")


def load_qb_inputs(
    fixture_path: str | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load the expected-starter and dropback marts.

    A fixture directory without both parquet files returns None so the column
    check can name the missing ratings. A warehouse read uses
    ``fct_expected_starter`` and ``fct_qb_dropbacks``.
    """
    if fixture_path:
        fixture_dir = Path(fixture_path)
        starter_file = fixture_dir / "fct_expected_starter.parquet"
        dropback_file = fixture_dir / "fct_qb_dropbacks.parquet"
        if not starter_file.exists() or not dropback_file.exists():
            logger.info("H5 fixture tables are absent in %s", fixture_dir)
            return None, None
        logger.info("Loading H5 tables from fixtures: %s", fixture_dir)
        return pd.read_parquet(starter_file), pd.read_parquet(dropback_file)
    logger.info("Loading H5 tables from Databricks")
    with DatabricksClient() as client:
        return client.read_table("fct_expected_starter"), client.read_table("fct_qb_dropbacks")


def cv_dataset_frame(
    games: pd.DataFrame,
    features: pd.DataFrame,
    cv_end_season: int,
) -> pd.DataFrame:
    """Rows that can enter rolling-origin CV, in a stable column order.

    Seasons after ``cv_end_season`` stay out. The frame is the full feature
    matrix plus the label, not one feature set.
    """
    labels: pd.DataFrame = games.loc[:, ["game_id", "gameday", "season", "home_won"]].copy()
    merged: pd.DataFrame = features.merge(labels, on="game_id", how="inner")
    kept: pd.DataFrame = merged.loc[merged["season"] <= cv_end_season].copy()
    ordered: pd.DataFrame = kept.sort_values(["gameday", "game_id"], kind="mergesort")
    id_columns: list[str] = ["game_id", "gameday", "season", "home_won"]
    feature_columns: list[str] = [column for column in features.columns if column != "game_id"]
    return ordered.loc[:, id_columns + feature_columns].reset_index(drop=True)


def validate_feature_columns(
    frame: pd.DataFrame,
    feature_sets: tuple[str, ...],
    max_null_fraction: float = DEFAULT_MAX_NULL_FRACTION,
) -> dict[str, dict[str, float]]:
    """Refuse a requested feature set whose columns are missing or mostly null.

    The check runs on the feature frame before any fold is fit. A column that
    is absent, or whose null share is at least ``max_null_fraction``, fails
    the set. The default share is one half.

    Raises:
        ValueError: The threshold is invalid, the frame has no rows, or a set
            has a missing or too-null column. The message names the feature
            set and tells the reader to run ``make dbt-build-databricks``.
    """
    threshold: float = float(max_null_fraction)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("max_null_fraction must be in (0, 1]")
    if len(frame) == 0:
        raise ValueError("feature frame has no rows to validate")

    problems: list[str] = []
    passed: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    row_count: int = len(frame)
    for feature_set in feature_sets:
        if feature_set in seen:
            continue
        seen.add(feature_set)
        columns: tuple[str, ...] = feature_columns_for(feature_set)
        missing: list[str] = [column for column in columns if column not in frame.columns]
        null_columns: list[str] = []
        fractions: dict[str, float] = {}
        for column in columns:
            if column not in frame.columns:
                continue
            null_fraction: float = float(frame[column].isna().sum()) / float(row_count)
            fractions[column] = null_fraction
            if null_fraction >= threshold:
                null_columns.append(f"{column} ({null_fraction:.0%} null)")
        if missing or null_columns:
            missing_text: str = ", ".join(missing) if missing else "none"
            null_text: str = ", ".join(null_columns) if null_columns else "none"
            problems.append(
                f"{feature_set}: missing columns: {missing_text}. "
                f"Columns at or above {threshold:.0%} null: {null_text}."
            )
            continue
        passed[feature_set] = fractions
    if problems:
        detail: str = " ".join(problems)
        raise ValueError(
            "Feature columns are missing or too null to score. "
            f"{detail} Hint: {FEATURE_COLUMN_REBUILD_HINT}."
        )
    return passed
