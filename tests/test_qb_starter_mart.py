"""Expected-starter SQL does not read who played in the game.

These checks read the model text. They do not need a warehouse.
"""

from pathlib import Path


def _model_text(relative_path: str) -> str:
    """Return one dbt model file with comments and whitespace kept."""
    path = Path(__file__).parent.parent / relative_path
    return path.read_text(encoding="utf-8")


class TestExpectedStarterSql:
    """The starter mart is the depth chart, not the play-by-play passer."""

    def test_starter_model_does_not_read_plays(self) -> None:
        """No play-by-play source, passer id, or dropback flag."""
        sql = _model_text("dbt/models/marts/fct_expected_starter.sql")
        lowered = sql.lower()
        assert "stg_nflverse__pbp" not in lowered
        assert "raw_play_by_play" not in lowered
        assert "passer_player_id" not in lowered
        assert "qb_dropback" not in lowered
        assert "play_by_play" not in lowered
        assert "cast(position as string)) = 'qb'" in lowered
        assert "cast(formation as string)) = 'offense'" in lowered
        assert "depth_rank = 1" in lowered
        assert "try_cast(week as double)" in lowered
        assert "try_cast(depth_team as double)" in lowered
        assert "chr(10)" in lowered
        assert "depth_position_norm" in lowered
        assert "row_number()" in lowered
        assert "depth_chart_club_map" in lowered
        assert "week is not null" in lowered
        assert "stg_nflverse__depth_charts" in lowered
        assert "depth_team = '1'" not in lowered
        assert "min(gsis_id)" not in lowered
        assert "depth_rank = 2" not in lowered

    def test_dropback_model_is_who_played_and_keeps_cpoe(self) -> None:
        """Dropbacks are aggregated from staging. The comment says this is not the starter."""
        sql = _model_text("dbt/models/marts/fct_qb_dropbacks.sql")
        lowered = sql.lower()
        assert "who played" in lowered
        assert "not the starter" in lowered
        assert "qb_dropback = 1" in lowered
        assert "passer_player_id" in lowered
        assert "cpoe" in lowered
        assert "epa" in lowered
        assert "stg_nflverse__pbp" in lowered
        assert "coalesce(epa, 0)" in lowered

    def test_staging_selects_passer_dropback_and_cpoe(self) -> None:
        """The play-by-play view exposes the columns the dropback mart needs."""
        sql = _model_text("dbt/models/staging/stg_nflverse__pbp.sql")
        assert "passer_player_id," in sql
        assert "qb_dropback," in sql
        assert "cpoe," in sql
