"""The clean EPA filter is fixed in the team-game mart SQL.

These checks read the model text. They do not need a warehouse.
"""

from pathlib import Path


def _model_text(relative_path: str) -> str:
    """Return one dbt model file with comments and whitespace kept."""
    path = Path(__file__).parent.parent / relative_path
    return path.read_text(encoding="utf-8")


class TestCleanEpaMartSql:
    """Clean columns are added beside the existing play = 1 averages."""

    def test_staging_exposes_kneel_spike_and_wp(self) -> None:
        """qb_kneel, qb_spike, and wp are selected from the nflverse source."""
        sql = _model_text("dbt/models/staging/stg_nflverse__pbp.sql")
        assert "qb_kneel," in sql
        assert "qb_spike," in sql
        assert "wp as win_probability" in sql

    def test_clean_filter_is_fixed_and_existing_epa_still_uses_play(self) -> None:
        """Pass and run only, no kneel or spike, and wp outside 0.05 to 0.95 is out."""
        sql = _model_text("dbt/models/marts/fct_team_game_epa.sql")
        assert "where play = 1" in sql
        assert "play_type in ('pass', 'run')" in sql
        assert "coalesce(qb_kneel, 0) = 0" in sql
        assert "coalesce(qb_spike, 0) = 0" in sql
        assert "win_probability is null" in sql
        assert "win_probability >= 0.05" in sql
        assert "win_probability <= 0.95" in sql
        assert "from pbp" in sql
        assert "from clean_pbp" in sql
        assert "avg(epa) as avg_offensive_epa" in sql
        assert "avg(epa) as avg_clean_offensive_epa" in sql
        assert "avg(epa) as avg_defensive_epa_allowed" in sql
        assert "avg(epa) as avg_clean_defensive_epa_allowed" in sql
        assert "as clean_success_rate" in sql
        assert "as clean_defensive_success_rate" in sql
        offensive_block = sql.split("clean_offensive_stats as", maxsplit=1)[0]
        assert "from pbp" in offensive_block
        assert "from clean_pbp" not in offensive_block
