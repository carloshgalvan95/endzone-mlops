"""
NFLverseFeed: Historical feed implementation using nflverse Parquet files.

License: CC BY 4.0
Source: https://github.com/nflverse/nflverse-data
Coverage: 1999-present, updated nightly
"""

from urllib.parse import urljoin

import pandas as pd

from .feeds import HistoricalFeed

# Base URL for nflverse data releases
NFLVERSE_BASE_URL = "https://github.com/nflverse/nflverse-data/releases/latest/download/"


class NFLverseFeed(HistoricalFeed):
    """
    Historical feed implementation for nflverse data.

    Downloads and caches Parquet files from nflverse GitHub releases.
    Data is read-only and updated nightly by nflverse maintainers.

    Attribution: This implementation uses data from the nflverse project,
    licensed under CC BY 4.0. See docs/ATTRIBUTION.md for full credits.
    """

    def __init__(self, cache_dir: str = "./data/nflverse_cache"):
        """
        Initialize nflverse feed.

        Args:
            cache_dir: Local directory for caching downloaded Parquet files.
                      Avoids re-downloading on subsequent calls.
        """
        self.cache_dir = cache_dir
        self.base_url = NFLVERSE_BASE_URL

    def get_games(
        self,
        season: int,
        week: int | None = None,
        game_type: str = "REG",
    ) -> pd.DataFrame:
        """
        Fetch historical game data from nflverse.

        Args:
            season: NFL season year (1999-present)
            week: Optional week filter (1-18 regular, 19+ playoffs)
            game_type: Game type ('REG', 'POST', 'PRE')

        Returns:
            DataFrame with game-level data (schedule, scores, odds, conditions)

        Raises:
            ValueError: If season is before 1999 (nflverse coverage starts 1999)
            IOError: If download fails
        """
        if season < 1999:
            raise ValueError(
                f"nflverse data starts in 1999. Requested season: {season}"
            )

        # nflverse games.parquet contains all seasons in one file
        url = urljoin(self.base_url, "games.parquet")

        # Download (pandas caches if local copy exists)
        df = pd.read_parquet(url)

        # Filter by season
        df = df[df["season"] == season].copy()

        # Optional filters
        if week is not None:
            df = df[df["week"] == week]

        if game_type:
            df = df[df["game_type"] == game_type]

        return df

    def get_plays(self, game_id: str) -> pd.DataFrame:
        """
        Fetch play-by-play data for a specific game.

        Args:
            game_id: nflverse game_id (format: YYYY_WW_AWAY_HOME)

        Returns:
            DataFrame with ~370 columns of play-level data

        Raises:
            ValueError: If game_id format is invalid
            IOError: If download fails

        Note:
            Play-by-play files are split by season (~19.5 MB each).
            This method downloads the full season file and filters to the requested game.
        """
        # Parse season from game_id (format: YYYY_WW_AWAY_HOME)
        try:
            season = int(game_id.split("_")[0])
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid game_id format: {game_id}. Expected: YYYY_WW_AWAY_HOME"
            ) from e

        if season < 1999:
            raise ValueError(
                f"nflverse play-by-play starts in 1999. Game season: {season}"
            )

        # Download play-by-play for the season
        url = urljoin(self.base_url, f"play_by_play_{season}.parquet")
        df = pd.read_parquet(url)

        # Filter to the specific game
        df = df[df["game_id"] == game_id].copy()

        if df.empty:
            raise ValueError(
                f"Game {game_id} not found in season {season} play-by-play data"
            )

        return df

    def get_data_version(self) -> str:
        """
        Return nflverse data version.

        Returns:
            Version string indicating data source and freshness.

        Note:
            nflverse releases are tagged 'latest' on GitHub.
            For reproducibility, could pin to specific commit SHA or date.
        """
        return "nflverse-latest"


# Example usage (for documentation and testing)
if __name__ == "__main__":
    # This module can be run standalone for testing
    feed = NFLverseFeed()

    # Fetch 2023 Week 1 games
    games_2023_w1 = feed.get_games(season=2023, week=1, game_type="REG")
    print(f"2023 Week 1 games: {len(games_2023_w1)} found")

    # Fetch play-by-play for first game (if exists)
    if not games_2023_w1.empty:
        first_game_id = games_2023_w1.iloc[0]["game_id"]
        plays = feed.get_plays(first_game_id)
        print(f"Plays in {first_game_id}: {len(plays)}")

    print(f"Data version: {feed.get_data_version()}")
