"""
Feed interfaces for NFL data ingestion.

Implements ports-and-adapters pattern: HistoricalFeed vs LiveFeed behind same contract.
This enables training on historical data and scoring on live data with identical feature code.

v0.1: Only HistoricalFeed (nflverse) is implemented.
LiveFeed is a documented stub for v0.2+ when BALLDONTLIE API is activated.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol

import pandas as pd


class NFLFeedProtocol(Protocol):
    """
    Protocol defining the contract for NFL data feeds.

    Any feed (historical or live) must implement these methods
    to be compatible with feature engineering and scoring pipelines.
    """

    def get_games(
        self,
        season: int,
        week: int | None = None,
        game_type: str = "REG",
    ) -> pd.DataFrame:
        """
        Fetch game-level data (schedule, scores, betting lines).

        Args:
            season: NFL season year (e.g., 2023)
            week: Week number (1-18 for regular season, 19+ playoffs) or None
            game_type: Game type filter ('REG', 'POST', 'PRE')

        Returns:
            DataFrame with columns:
                - game_id: str
                - season: int
                - week: int
                - game_type: str
                - away_team: str
                - home_team: str
                - gameday: date
                - away_score: int | None
                - home_score: int | None
                - spread_line: float | None
                - total_line: float | None
                - roof: str | None
                - surface: str | None
                - temp: float | None
                - wind: float | None
                - away_rest: int | None
                - home_rest: int | None
        """
        ...

    def get_plays(
        self,
        game_id: str,
    ) -> pd.DataFrame:
        """
        Fetch play-by-play data for a specific game.

        Args:
            game_id: Unique game identifier

        Returns:
            DataFrame with columns (subset of ~370 nflverse columns):
                - play_id: str
                - game_id: str
                - season: int
                - week: int
                - down: int | None
                - ydstogo: int | None
                - yardline_100: int | None
                - quarter_seconds_remaining: int
                - score_differential: int
                - posteam: str | None
                - defteam: str | None
                - play_type: str
                - yards_gained: int
                - ... (many more columns)

        Note:
            Play-by-play is ~19.5 MB per season per nflverse.
            For live feeds, only in-progress game plays are returned.
        """
        ...


class HistoricalFeed(ABC):
    """
    Abstract base class for historical (batch) NFL data feeds.

    Implementations read from static sources like nflverse Parquet files.
    Data is immutable once finalized (though corrections may occur).

    v0.1 Implementation: NFLverseFeed (nflverse Parquet from GitHub releases)
    """

    @abstractmethod
    def get_games(
        self,
        season: int,
        week: int | None = None,
        game_type: str = "REG",
    ) -> pd.DataFrame:
        """Fetch historical game data. See NFLFeedProtocol for schema."""
        pass

    @abstractmethod
    def get_plays(self, game_id: str) -> pd.DataFrame:
        """Fetch historical play-by-play data. See NFLFeedProtocol for schema."""
        pass

    @abstractmethod
    def get_data_version(self) -> str:
        """
        Return version or timestamp of historical data snapshot.

        Returns:
            Version string (e.g., 'nflverse-2023-12-31' or commit SHA)
        """
        pass


class LiveFeed(ABC):
    """
    Abstract base class for live (near-real-time) NFL data feeds.

    Implementations poll or subscribe to live APIs like BALLDONTLIE.
    Data includes in-progress games and may be updated as plays occur.

    v0.1 Status: STUB - Not implemented. Raises NotImplementedError.
    v0.2+ Target: BALLDONTLIE NFL GOAT API (~$39.99/month)

    Decision gate: Activation requires explicit owner approval and is post-Day 14.
    """

    @abstractmethod
    def get_games(
        self,
        season: int,
        week: int | None = None,
        game_type: str = "REG",
        in_progress_only: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch live or upcoming game data.

        Args:
            season: NFL season year
            week: Optional week number
            game_type: Game type filter
            in_progress_only: If True, return only games currently in progress

        Returns:
            DataFrame with same schema as HistoricalFeed.get_games()
            For live games, scores update as game progresses.
        """
        pass

    @abstractmethod
    def get_plays(
        self,
        game_id: str,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch live play-by-play data.

        Args:
            game_id: Unique game identifier
            as_of: Optional timestamp for historical replay of live game

        Returns:
            DataFrame with same schema as HistoricalFeed.get_plays()
            For in-progress games, only plays that have occurred are returned.
        """
        pass

    @abstractmethod
    def get_latest_update(self, game_id: str) -> datetime:
        """
        Get timestamp of most recent data update for a game.

        Args:
            game_id: Game to check

        Returns:
            Timestamp of last update (UTC)

        Note:
            BALLDONTLIE webhooks can have up to 1 min delay.
            Polling-based feeds may have higher latency.
        """
        pass


class LiveFeedStub(LiveFeed):
    """
    Stub implementation of LiveFeed for v0.1.

    Raises NotImplementedError with helpful message pointing to roadmap.
    Serves as documentation of the intended interface for v0.2+.

    Target: BALLDONTLIE NFL GOAT API
    - Pricing: ~$39.99/month (600 req/min)
    - Coverage: Play-by-play, stats, injuries, odds (2002-current)
    - Latency: Near-real-time, no published SLA for in-play updates
    - License: Commercial API, ML use allowed, no raw feed redistribution
    """

    def get_games(
        self,
        season: int,
        week: int | None = None,
        game_type: str = "REG",
        in_progress_only: bool = False,
    ) -> pd.DataFrame:
        """Not implemented in v0.1. See roadmap for BALLDONTLIE activation."""
        raise NotImplementedError(
            "LiveFeed is not active in v0.1. "
            "For v0.2+ with BALLDONTLIE GOAT API, see docs/adr/ADR-001-databricks.md "
            "and docs/architecture.md for activation checklist. "
            "Requires: (1) explicit owner approval, (2) API subscription, "
            "(3) secrets configuration, (4) crosswalk mapping nflverse <-> BALLDONTLIE IDs."
        )

    def get_plays(
        self,
        game_id: str,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        """Not implemented in v0.1. See roadmap for BALLDONTLIE activation."""
        raise NotImplementedError(
            "LiveFeed play-by-play not active in v0.1. "
            "Target: BALLDONTLIE GOAT tier endpoint '/plays' (GOAT-only). "
            "See docs/architecture.md for v0.2+ implementation plan."
        )

    def get_latest_update(self, game_id: str) -> datetime:
        """Not implemented in v0.1. See roadmap for BALLDONTLIE activation."""
        raise NotImplementedError("LiveFeed not active in v0.1.")
