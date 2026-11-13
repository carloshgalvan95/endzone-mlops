"""Pregame home-field context. Each column is 0 or 1 and is never missing.

``is_divisional_game`` is 1 when the two franchises share an NFL division
in that season. 1999 through 2001 uses the six-division alignment. 2002 and
later uses the eight-division alignment. The mart stores the current
franchise abbreviation, so OAK, SD, and STL map onto LV, LAC, and LA.

``away_surface_mismatch`` is 1 when the current surface category differs
from the away team's most recent prior home surface. Home means
``location`` is Home. The prior ``gameday`` is strictly earlier. Grass is
``grass`` or ``dessograss``. Turf is ``a_turf``, ``astroplay``,
``astroturf``, ``fieldturf``, ``matrixturf``, or ``sportturf``. A blank or
unknown surface, or no prior home surface, is 0.

``west_coast_early_kickoff`` is 1 when the game is at home, the away
franchise is in the western set, the home franchise is not, and
``gametime`` in Eastern time is strictly before 14:00. A neutral site, a
blank clock, or a malformed clock is 0.
"""

from __future__ import annotations

import pandas as pd

HOME_FIELD_COLUMNS: tuple[str, ...] = (
    "is_divisional_game",
    "away_surface_mismatch",
    "west_coast_early_kickoff",
)
WESTERN_FRANCHISES: frozenset[str] = frozenset({"ARI", "DEN", "LA", "LAC", "LV", "SEA", "SF"})
_EARLY_KICKOFF_SECONDS: int = 14 * 60 * 60
_FRANCHISE_ALIASES: dict[str, str] = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
_GRASS_SURFACES: frozenset[str] = frozenset({"grass", "dessograss"})
_TURF_SURFACES: frozenset[str] = frozenset(
    {"a_turf", "astroplay", "astroturf", "fieldturf", "matrixturf", "sportturf"}
)


def _division_map(groups: dict[str, frozenset[str]]) -> dict[str, str]:
    """Map each franchise abbreviation to one division name."""
    mapping: dict[str, str] = {}
    for name, teams in groups.items():
        for team in teams:
            mapping[team] = name
    return mapping


_DIVISION_1999: dict[str, str] = _division_map(
    {
        "AFC East": frozenset({"BUF", "MIA", "NE", "NYJ", "IND"}),
        "AFC Central": frozenset({"BAL", "CIN", "CLE", "PIT", "TEN", "JAX"}),
        "AFC West": frozenset({"DEN", "KC", "LV", "LAC", "SEA"}),
        "NFC East": frozenset({"DAL", "NYG", "PHI", "WAS", "ARI"}),
        "NFC Central": frozenset({"CHI", "DET", "GB", "MIN", "TB"}),
        "NFC West": frozenset({"ATL", "CAR", "NO", "SF", "LA"}),
    }
)
_DIVISION_2002: dict[str, str] = _division_map(
    {
        "AFC East": frozenset({"BUF", "MIA", "NE", "NYJ"}),
        "AFC North": frozenset({"BAL", "CIN", "CLE", "PIT"}),
        "AFC South": frozenset({"HOU", "IND", "JAX", "TEN"}),
        "AFC West": frozenset({"DEN", "KC", "LV", "LAC"}),
        "NFC East": frozenset({"DAL", "NYG", "PHI", "WAS"}),
        "NFC North": frozenset({"CHI", "DET", "GB", "MIN"}),
        "NFC South": frozenset({"ATL", "CAR", "NO", "TB"}),
        "NFC West": frozenset({"ARI", "LA", "SF", "SEA"}),
    }
)


def attach_home_field_features(games: pd.DataFrame) -> pd.DataFrame:
    """Return the three home-field columns, one row per game.

    A schedule column that the builder does not require (``location``,
    ``surface``, or ``gametime``) leaves the column that needs it at 0.
    The three results are never missing.

    Args:
        games: One row per game. ``game_id``, ``season``, ``gameday``, and
            both franchise columns are required.

    Returns:
        ``game_id`` plus ``HOME_FIELD_COLUMNS``, each 0.0 or 1.0.

    Raises:
        ValueError: A required column is missing or ``game_id`` is duplicated.
    """
    required: tuple[str, ...] = (
        "game_id",
        "season",
        "gameday",
        "home_team_franchise",
        "away_team_franchise",
    )
    missing: list[str] = [column for column in required if column not in games.columns]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"games is missing columns: {missing_list}")
    game_ids: pd.Series = games["game_id"].astype("string")
    if game_ids.duplicated().any():
        raise ValueError("games game_id must be unique")
    frame: pd.DataFrame = games.reset_index(drop=True)
    output: pd.DataFrame = pd.DataFrame({"game_id": frame["game_id"].astype("string").to_numpy()})
    output["is_divisional_game"] = _divisional_flags(frame).to_numpy()
    output["away_surface_mismatch"] = _surface_mismatch(frame).to_numpy()
    output["west_coast_early_kickoff"] = _early_kickoff(frame).to_numpy()
    return output.loc[:, ["game_id", *HOME_FIELD_COLUMNS]]


def _franchise(values: pd.Series) -> pd.Series:
    """Uppercase a franchise code and apply the relocation aliases."""
    text: pd.Series = values.astype("string").str.strip().str.upper()
    return text.replace(_FRANCHISE_ALIASES)


def _division_name(season: pd.Series, franchise: pd.Series) -> pd.Series:
    """Return the division name, or missing when the era or team is unknown."""
    year: pd.Series = pd.to_numeric(season, errors="coerce")
    code: pd.Series = _franchise(franchise)
    names: pd.Series = pd.Series(pd.NA, index=season.index, dtype="string")
    early: pd.Series = year.between(1999, 2001)
    modern: pd.Series = year.ge(2002)
    names = names.mask(early, code.map(_DIVISION_1999))
    names = names.mask(modern, code.map(_DIVISION_2002))
    return names


def _divisional_flags(games: pd.DataFrame) -> pd.Series:
    """1.0 when both franchises share a division in that season."""
    home: pd.Series = _division_name(games["season"], games["home_team_franchise"])
    away: pd.Series = _division_name(games["season"], games["away_team_franchise"])
    same: pd.Series = home.notna() & away.notna() & home.eq(away)
    return same.fillna(False).astype(float)


def _surface_category(values: pd.Series) -> pd.Series:
    """Map a surface string to ``grass`` or ``turf``. Anything else is missing."""
    text: pd.Series = values.astype("string").str.strip().str.lower()
    category: pd.Series = pd.Series(pd.NA, index=values.index, dtype="string")
    category = category.mask(text.isin(_GRASS_SURFACES), "grass")
    category = category.mask(text.isin(_TURF_SURFACES), "turf")
    return category


def _surface_mismatch(games: pd.DataFrame) -> pd.Series:
    """1.0 when the current category differs from the away team's prior home."""
    zeros: pd.Series = pd.Series(0.0, index=games.index, dtype=float)
    if "surface" not in games.columns or "location" not in games.columns:
        return zeros
    current: pd.Series = _surface_category(games["surface"])
    prior: pd.Series = _prior_home_surface(games)
    mismatch: pd.Series = current.notna() & prior.notna() & current.ne(prior)
    return mismatch.fillna(False).astype(float)


def _prior_home_surface(games: pd.DataFrame) -> pd.Series:
    """Most recent known home surface for the away franchise before gameday."""
    empty: pd.Series = pd.Series(pd.NA, index=games.index, dtype="string")
    location: pd.Series = games["location"].astype("string").str.strip().str.casefold()
    category: pd.Series = _surface_category(games["surface"])
    gameday: pd.Series = pd.to_datetime(games["gameday"], errors="coerce")
    home_team: pd.Series = _franchise(games["home_team_franchise"])
    history: pd.DataFrame = pd.DataFrame(
        {
            "gameday": gameday.to_numpy(),
            "team": home_team.to_numpy(),
            "category": category.to_numpy(),
            "game_id": games["game_id"].astype("string").to_numpy(),
        }
    )
    history = history.loc[
        location.eq("home").to_numpy()
        & category.notna().to_numpy()
        & gameday.notna().to_numpy()
        & home_team.notna().to_numpy()
    ]
    if history.empty:
        return empty
    history = history.sort_values(["team", "gameday", "game_id"], kind="mergesort")
    current_rows: pd.DataFrame = pd.DataFrame(
        {
            "gameday": gameday.to_numpy(),
            "team": _franchise(games["away_team_franchise"]).to_numpy(),
            "row_id": games.index.to_numpy(),
        }
    )
    usable: pd.Series = current_rows["gameday"].notna() & current_rows["team"].notna()
    if not bool(usable.any()):
        return empty
    matched_rows: list[pd.DataFrame] = []
    lookup: pd.DataFrame = current_rows.loc[usable]
    for team, left in lookup.groupby("team", sort=False):
        right: pd.DataFrame = history.loc[
            history["team"].eq(team), ["gameday", "category"]
        ].sort_values("gameday", kind="mergesort")
        if right.empty:
            continue
        left_sorted: pd.DataFrame = left.sort_values("gameday", kind="mergesort")
        matched_rows.append(
            pd.merge_asof(
                left_sorted,
                right,
                on="gameday",
                direction="backward",
                allow_exact_matches=False,
            )
        )
    if not matched_rows:
        return empty
    merged: pd.DataFrame = pd.concat(matched_rows, ignore_index=True)
    aligned: pd.Series = merged.set_index("row_id")["category"].reindex(games.index)
    return aligned.astype("string")


def _before_fourteen(gametime: pd.Series) -> pd.Series:
    """True when the Eastern clock is strictly before 14:00."""
    clock: pd.Series = gametime.astype("string").str.strip()
    blank: pd.Series = clock.isna() | clock.str.lower().isin(["", "na", "nat", "none", "nan"])
    parts: pd.DataFrame = clock.str.split(":", expand=True, n=2)
    hour: pd.Series = pd.to_numeric(parts[0], errors="coerce")
    if int(parts.shape[1]) > 1:
        minute: pd.Series = pd.to_numeric(parts[1], errors="coerce")
    else:
        minute = pd.Series(pd.NA, index=gametime.index)
    if int(parts.shape[1]) > 2:
        second: pd.Series = pd.to_numeric(parts[2], errors="coerce").fillna(0)
    else:
        second = pd.Series(0.0, index=gametime.index)
    valid: pd.Series = (
        ~blank
        & hour.notna()
        & minute.notna()
        & hour.between(0, 23)
        & minute.between(0, 59)
        & second.between(0, 59)
    )
    total: pd.Series = hour.fillna(0) * 3600 + minute.fillna(0) * 60 + second.fillna(0)
    return valid & total.lt(_EARLY_KICKOFF_SECONDS)


def _early_kickoff(games: pd.DataFrame) -> pd.Series:
    """1.0 for a western visitor at a non-western home before 14:00 ET."""
    zeros: pd.Series = pd.Series(0.0, index=games.index, dtype=float)
    if "location" not in games.columns or "gametime" not in games.columns:
        return zeros
    location: pd.Series = games["location"].astype("string").str.strip().str.casefold()
    away: pd.Series = _franchise(games["away_team_franchise"])
    home: pd.Series = _franchise(games["home_team_franchise"])
    away_west: pd.Series = away.isin(WESTERN_FRANCHISES)
    home_known: pd.Series = home.notna() & home.ne("")
    home_west: pd.Series = home.isin(WESTERN_FRANCHISES)
    early: pd.Series = _before_fourteen(games["gametime"])
    flag: pd.Series = location.eq("home") & away_west & home_known & ~home_west & early
    return flag.fillna(False).astype(float)
