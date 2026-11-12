"""Pre-game starter absences use only the last report before kickoff."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from endzone.features.injuries import (
    INJURY_DIFF_COLUMNS,
    attach_injury_features,
    starter_absence_diffs,
)
from endzone.features.pregame import (
    FEATURE_SET_PKG_CORE_INJ,
    FEATURE_SET_PKG_CORE_INJ_L2,
    PREGAME_FEATURE_COLUMNS_PKG_CORE,
    PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ,
    build_pregame_features,
)
from endzone.modeling.models import feature_column_groups
from tests.test_h6 import _multiweek_slate


def _game(
    game_id: str = "g1",
    gameday: str = "2016-09-11",
    gametime: str = "13:00",
    home: str = "A",
    away: str = "B",
) -> pd.DataFrame:
    """One scheduled game with the columns the absence count requires."""
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": 2016,
                "week": 1,
                "game_type": "REG",
                "gameday": gameday,
                "gametime": gametime,
                "home_team": home,
                "away_team": away,
            }
        ]
    )


def _starter(team: str, gsis_id: str, position: str, depth: int = 1) -> dict[str, object]:
    """One depth-chart row on the synthetic week-1 slate."""
    return {
        "season": 2016,
        "week": 1,
        "game_type": "REG",
        "team": team,
        "gsis_id": gsis_id,
        "position": position,
        "depth_team": depth,
    }


def _report(
    team: str,
    gsis_id: str,
    status: str | None,
    modified: str | None,
) -> dict[str, object]:
    """One injury row. ``modified`` is UTC, or None when the release has no clock."""
    return {
        "season": 2016,
        "week": 1,
        "game_type": "REG",
        "team": team,
        "gsis_id": gsis_id,
        "report_status": status,
        "date_modified": modified,
    }


def _diffs(
    games: pd.DataFrame,
    starters: list[dict[str, object]],
    reports: list[dict[str, object]],
) -> pd.Series:
    """Return the four differences for the only game."""
    frame: pd.DataFrame = starter_absence_diffs(
        games,
        pd.DataFrame(starters),
        pd.DataFrame(reports),
    )
    assert list(frame.columns) == ["game_id", *INJURY_DIFF_COLUMNS]
    return frame.iloc[0]


def test_post_kickoff_and_null_timestamps_are_excluded() -> None:
    """A row at or after kickoff, or with no clock, does not count."""
    games: pd.DataFrame = _game()
    starters: list[dict[str, object]] = [
        _starter("A", "early", "WR"),
        _starter("A", "late", "WR"),
        _starter("A", "blank", "WR"),
    ]
    before: pd.Series = _diffs(
        games,
        starters,
        [_report("A", "early", "Out", "2016-09-11 16:30:00+00:00")],
    )
    after: pd.Series = _diffs(
        games,
        starters,
        [
            _report("A", "early", "Out", "2016-09-11 16:30:00+00:00"),
            _report("A", "late", "Out", "2016-09-11 17:00:00+00:00"),
            _report("A", "blank", "Out", None),
        ],
    )
    assert before["diff_skill_starters_out_or_doubtful"] == 1.0
    pd.testing.assert_series_equal(before, after)


def test_last_pregame_status_wins_and_questionable_does_not_count() -> None:
    """The latest row before kickoff is the status. Questionable is not an absence."""
    games: pd.DataFrame = _game()
    starters: list[dict[str, object]] = [_starter("A", "wr", "WR")]
    cleared: pd.Series = _diffs(
        games,
        starters,
        [
            _report("A", "wr", "Out", "2016-09-09 12:00:00+00:00"),
            _report("A", "wr", "Questionable", "2016-09-10 12:00:00+00:00"),
        ],
    )
    listed: pd.Series = _diffs(
        games,
        starters,
        [
            _report("A", "wr", "Questionable", "2016-09-09 12:00:00+00:00"),
            _report("A", "wr", "Doubtful", "2016-09-10 12:00:00+00:00"),
        ],
    )
    assert cleared["diff_skill_starters_out_or_doubtful"] == 0.0
    assert listed["diff_skill_starters_out_or_doubtful"] == 1.0


def test_same_timestamp_keeps_out_over_questionable() -> None:
    """A tie on the clock keeps Out, which is the more severe listed status."""
    games: pd.DataFrame = _game()
    counted: pd.Series = _diffs(
        games,
        [_starter("A", "wr", "WR")],
        [
            _report("A", "wr", "Questionable", "2016-09-10 12:00:00+00:00"),
            _report("A", "wr", "Out", "2016-09-10 12:00:00+00:00"),
        ],
    )
    assert counted["diff_skill_starters_out_or_doubtful"] == 1.0


def test_groups_exclude_qb_fullback_specialists_and_backups() -> None:
    """QB, FB, K, P, LS, and depth 2 are not in the four counts."""
    games: pd.DataFrame = _game()
    counted: pd.Series = _diffs(
        games,
        [
            _starter("A", "qb", "QB"),
            _starter("A", "fb", "FB"),
            _starter("A", "k", "K"),
            _starter("A", "p", "P"),
            _starter("A", "ls", "LS"),
            _starter("A", "backup", "WR", depth=2),
            _starter("A", "t", "T"),
            _starter("A", "g", "G"),
            _starter("A", "de", "DE"),
            _starter("A", "lb", "LB"),
            _starter("A", "nt", "NT"),
            _starter("A", "cb", "CB"),
            _starter("A", "s", "S"),
            _starter("B", "wr", "WR"),
        ],
        [
            _report("A", player, "Out", "2016-09-10 12:00:00+00:00")
            for player in (
                "qb",
                "fb",
                "k",
                "p",
                "ls",
                "backup",
                "t",
                "g",
                "de",
                "lb",
                "nt",
                "cb",
                "s",
            )
        ]
        + [_report("B", "wr", "Doubtful", "2016-09-10 12:00:00+00:00")],
    )
    assert counted["diff_ol_starters_out_or_doubtful"] == 2.0
    assert counted["diff_skill_starters_out_or_doubtful"] == -1.0
    assert counted["diff_front_seven_starters_out_or_doubtful"] == 3.0
    assert counted["diff_db_starters_out_or_doubtful"] == 2.0


def test_duplicate_starter_counts_once() -> None:
    """Two chart rows for the same player in one group are one absence."""
    games: pd.DataFrame = _game()
    counted: pd.Series = _diffs(
        games,
        [_starter("A", "wr", "WR"), _starter("A", "wr", "WR")],
        [_report("A", "wr", "Out", "2016-09-10 12:00:00+00:00")],
    )
    assert counted["diff_skill_starters_out_or_doubtful"] == 1.0


def test_january_kickoff_uses_eastern_standard_time() -> None:
    """18:30 Eastern in February is 23:30 UTC. A later row is after kickoff."""
    games: pd.DataFrame = _game(gameday="2017-02-05", gametime="18:30")
    games["season"] = 2016
    games["week"] = 22
    games["game_type"] = "SB"
    starter: dict[str, object] = _starter("A", "wr", "WR")
    starter["week"] = 22
    starter["game_type"] = "SB"
    early: dict[str, object] = _report("A", "wr", "Out", "2017-02-05 23:00:00+00:00")
    early["week"] = 22
    early["game_type"] = "SB"
    late: dict[str, object] = _report("A", "wr", "Out", "2017-02-05 23:30:00+00:00")
    late["week"] = 22
    late["game_type"] = "SB"
    counted: pd.Series = _diffs(games, [starter], [early, late])
    assert counted["diff_skill_starters_out_or_doubtful"] == 1.0


def test_thursday_kickoff_excludes_a_friday_update() -> None:
    """A Friday timestamp is after a Thursday kickoff in the same week."""
    games: pd.DataFrame = _game(gameday="2019-11-28", gametime="20:20")
    counted: pd.Series = _diffs(
        games,
        [_starter("A", "te", "TE")],
        [_report("A", "te", "Out", "2019-11-29 13:28:25+00:00")],
    )
    assert counted["diff_skill_starters_out_or_doubtful"] == 0.0


def test_omitted_inputs_are_missing_and_one_input_is_refused() -> None:
    """Both frames omitted leave the columns missing. One frame is an error."""
    games: pd.DataFrame = _game()
    missing: pd.DataFrame = attach_injury_features(games, None, None)
    assert missing[list(INJURY_DIFF_COLUMNS)].isna().all().all()
    with pytest.raises(ValueError, match="both"):
        attach_injury_features(games, pd.DataFrame(), None)


def test_feature_sets_append_the_four_columns_in_order() -> None:
    """Both injury candidates use pkg_core plus the four differences."""
    assert PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ == (
        PREGAME_FEATURE_COLUMNS_PKG_CORE + INJURY_DIFF_COLUMNS
    )
    assert FEATURE_SET_PKG_CORE_INJ == "pkg_core_inj"
    assert FEATURE_SET_PKG_CORE_INJ_L2 == "pkg_core_inj_l2"
    groups: dict[str, list[str]] = feature_column_groups(list(INJURY_DIFF_COLUMNS))
    assert groups["diff"] == list(INJURY_DIFF_COLUMNS)


def test_build_ignores_a_report_dated_after_kickoff() -> None:
    """The pregame frame changes for a pre-kickoff Out and not for a later one."""
    games, epa = _multiweek_slate(weeks=2)
    parts: pd.Series = games["game_id"].astype(str).str.split("-", expand=True)
    games = games.copy()
    games["week"] = parts[1].astype(int) + 1
    games["game_type"] = "REG"
    games["gametime"] = "13:00"
    games["home_team"] = games["home_team_franchise"]
    games["away_team"] = games["away_team_franchise"]
    target: pd.Series = games.iloc[0]
    starters: pd.DataFrame = pd.DataFrame(
        [
            {
                "season": int(target["season"]),
                "week": int(target["week"]),
                "game_type": "REG",
                "team": target["home_team"],
                "gsis_id": "wr-1",
                "position": "WR",
            }
        ]
    )
    pregame: pd.DataFrame = pd.DataFrame(
        [
            {
                "season": int(target["season"]),
                "week": int(target["week"]),
                "game_type": "REG",
                "team": target["home_team"],
                "gsis_id": "wr-1",
                "report_status": "Out",
                "date_modified": "2016-09-01 12:00:00+00:00",
            }
        ]
    )
    later: pd.DataFrame = pd.concat(
        [
            pregame,
            pd.DataFrame(
                [
                    {
                        "season": int(target["season"]),
                        "week": int(target["week"]),
                        "game_type": "REG",
                        "team": target["home_team"],
                        "gsis_id": "wr-1",
                        "report_status": "Questionable",
                        "date_modified": "2016-09-20 12:00:00+00:00",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    quiet: pd.DataFrame = build_pregame_features(
        games,
        epa,
        depth_chart_starters=starters.iloc[0:0],
        injury_reports=pregame.iloc[0:0],
    )
    with_out: pd.DataFrame = build_pregame_features(
        games,
        epa,
        depth_chart_starters=starters,
        injury_reports=pregame,
    )
    with_later: pd.DataFrame = build_pregame_features(
        games,
        epa,
        depth_chart_starters=starters,
        injury_reports=later,
    )
    game_id: str = str(target["game_id"])
    quiet_row: pd.Series = quiet.loc[quiet["game_id"] == game_id, INJURY_DIFF_COLUMNS].iloc[0]
    out_row: pd.Series = with_out.loc[with_out["game_id"] == game_id, INJURY_DIFF_COLUMNS].iloc[0]
    later_row: pd.Series = with_later.loc[
        with_later["game_id"] == game_id, INJURY_DIFF_COLUMNS
    ].iloc[0]
    assert quiet_row["diff_skill_starters_out_or_doubtful"] == 0.0
    assert out_row["diff_skill_starters_out_or_doubtful"] == 1.0
    pd.testing.assert_series_equal(out_row, later_row)


def _dated_game(
    game_id: str,
    season: int,
    gameday: str,
    gametime: object,
    home: str = "A",
    away: str = "B",
) -> dict[str, object]:
    """One schedule row. ``gametime`` may be missing."""
    return {
        "game_id": game_id,
        "season": season,
        "week": 1,
        "game_type": "REG",
        "gameday": gameday,
        "gametime": gametime,
        "home_team": home,
        "away_team": away,
    }


def test_missing_gametime_before_coverage_stays_zero(caplog: pytest.LogCaptureFixture) -> None:
    """A 2000 game with no clock and no report is 0, and a later kickoff still counts."""
    games: pd.DataFrame = pd.DataFrame(
        [
            _dated_game("2000_01_TEN_BUF", 2000, "2000-09-03", None, home="TEN", away="BUF"),
            _dated_game("2000_01_SD_OAK", 2000, "2000-09-03", pd.NA, home="SD", away="OAK"),
            _dated_game("2016-1", 2016, "2016-09-11", "13:00"),
        ]
    )
    starters: list[dict[str, object]] = [_starter("A", "wr", "WR")]
    reports: list[dict[str, object]] = [_report("A", "wr", "Out", "2016-09-10 16:30:00+00:00")]
    with caplog.at_level(logging.WARNING, logger="endzone.features.injuries"):
        counted: pd.DataFrame = starter_absence_diffs(
            games,
            pd.DataFrame(starters),
            pd.DataFrame(reports),
        )
    early: pd.DataFrame = counted.loc[
        counted["game_id"].isin(["2000_01_TEN_BUF", "2000_01_SD_OAK"])
    ]
    assert (early[list(INJURY_DIFF_COLUMNS)] == 0.0).all().all()
    later: pd.Series = counted.loc[counted["game_id"] == "2016-1"].iloc[0]
    assert later["diff_skill_starters_out_or_doubtful"] == 1.0
    assert "no buildable kickoff" not in caplog.text


def test_injury_rows_without_a_kickoff_count_zero_and_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rows on a game with no clock are excluded. A neighboring kickoff still counts."""
    games: pd.DataFrame = pd.DataFrame(
        [
            _dated_game("1999_01_MIN_ATL", 1999, "1999-09-12", None, home="MIN", away="ATL"),
            _dated_game("no-clock", 2016, "2016-09-11", None),
            _dated_game("has-clock", 2016, "2016-09-18", "13:00", home="C", away="D"),
        ]
    )
    starters: list[dict[str, object]] = [
        _starter("A", "wr", "WR"),
        {
            "season": 2016,
            "week": 1,
            "game_type": "REG",
            "team": "C",
            "gsis_id": "wr2",
            "position": "WR",
            "depth_team": 1,
        },
    ]
    reports: list[dict[str, object]] = [
        _report("A", "wr", "Out", "2016-09-10 12:00:00+00:00"),
        {
            "season": 2016,
            "week": 1,
            "game_type": "REG",
            "team": "C",
            "gsis_id": "wr2",
            "report_status": "Out",
            "date_modified": "2016-09-17 16:00:00+00:00",
        },
    ]
    with caplog.at_level(logging.WARNING, logger="endzone.features.injuries"):
        counted: pd.DataFrame = starter_absence_diffs(
            games,
            pd.DataFrame(starters),
            pd.DataFrame(reports),
        )
    no_clock: pd.Series = counted.loc[counted["game_id"] == "no-clock"].iloc[0]
    has_clock: pd.Series = counted.loc[counted["game_id"] == "has-clock"].iloc[0]
    pre_coverage: pd.Series = counted.loc[counted["game_id"] == "1999_01_MIN_ATL"].iloc[0]
    assert no_clock["diff_skill_starters_out_or_doubtful"] == 0.0
    assert has_clock["diff_skill_starters_out_or_doubtful"] == 1.0
    assert pre_coverage["diff_skill_starters_out_or_doubtful"] == 0.0
    assert "1 games have injury rows and no buildable kickoff" in caplog.text


def test_build_keeps_training_when_an_early_game_has_no_gametime() -> None:
    """The feature build used by training does not raise on a 2000 game with no clock."""
    games, epa = _multiweek_slate(weeks=1)
    parts: pd.Series = games["game_id"].astype(str).str.split("-", expand=True)
    games = games.copy()
    games["week"] = parts[1].astype(int) + 1
    games["game_type"] = "REG"
    games["gametime"] = "13:00"
    games["home_team"] = games["home_team_franchise"]
    games["away_team"] = games["away_team_franchise"]
    early: pd.DataFrame = pd.DataFrame(
        [
            {
                "game_id": "2000_01_TEN_BUF",
                "season": 2000,
                "week": 1,
                "game_type": "REG",
                "gameday": "2000-09-03",
                "gametime": None,
                "home_team": "TEN",
                "away_team": "BUF",
                "home_team_franchise": "TEN",
                "away_team_franchise": "BUF",
                "home_won": 1,
                "home_score": 16,
                "away_score": 13,
                "home_rest": 7,
                "away_rest": 7,
                "home_moneyline": None,
                "away_moneyline": None,
            }
        ]
    )
    games = pd.concat([early, games], ignore_index=True)
    features: pd.DataFrame = build_pregame_features(
        games,
        epa,
        depth_chart_starters=pd.DataFrame(
            columns=["season", "week", "game_type", "team", "gsis_id", "position"]
        ),
        injury_reports=pd.DataFrame(
            columns=[
                "season",
                "week",
                "game_type",
                "team",
                "gsis_id",
                "report_status",
                "date_modified",
            ]
        ),
    )
    row: pd.Series = features.loc[
        features["game_id"] == "2000_01_TEN_BUF", INJURY_DIFF_COLUMNS
    ].iloc[0]
    assert row.tolist() == [0.0, 0.0, 0.0, 0.0]
