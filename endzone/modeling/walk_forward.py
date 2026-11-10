"""Walk-forward week backtest for logistic ``h2_full``.

For each completed week after the v1 cutoff, the candidate fits on games
before that week and the v1 fit uses only seasons through the cutoff.
Both score that week. Elo and the home win rate use the same cutoff.
The market is the no-vig closing line on the row.

Eval features drop the whole week, because the slate is published on
Tuesday before any game that week. Training rows for earlier games still
use ``gameday`` strictly before kickoff.

``python -m endzone.modeling.walk_forward`` writes the report under
``reports/``. It does not register a model and it does not publish a slate.
``--verify`` exits 1 when the candidate log loss is not lower than v1.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from endzone.features.pregame import (
    build_pregame_features,
    elo_home_probabilities,
    feature_columns_for,
    market_home_probabilities,
    prepare_training_data,
)
from endzone.modeling.dataset import load_data, validate_feature_columns
from endzone.modeling.evaluation import binary_classification_metrics, paired_log_loss_difference
from endzone.modeling.models import fit_logistic_pipeline
from endzone.modeling.reference import load_reference_metrics
from endzone.modeling.registry import (
    REFERENCE_FEATURE_SET,
    REFERENCE_MODEL_NAME,
    WALK_FORWARD_PROTOCOL,
    metrics_from_walk_forward_report,
    promotion_decision,
    read_walk_forward_report,
)
from endzone.modeling.weekly import (
    WEEKLY_FEATURE_SET,
    completed_games,
    games_before_week,
    team_games_before_week,
)
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

DEFAULT_OUTPUT_DIR: Path = Path("reports/walk_forward")
_PROTOCOL_NOTES: tuple[str, ...] = (
    "The candidate refits logistic h2_full on completed games before the scored week.",
    "v1 fits logistic h2_full on completed games before that week and on or before v1_through_season.",
    "A scored week is excluded from its own features, labels, and Elo update.",
    "The home baseline is the home win rate on games before the week.",
    "The market is the no-vig closing moneyline. It is not a training feature and not a betting pick.",
    "On the locked 2016-2023 reference the market log loss is lower than logistic h2_full. This report does not replace that table.",
    "Point differential is not a feature. Elo stays a baseline. Injury reports are not inputs.",
    "A later register fits every completed game before the slate, including the last scored week.",
)


class LabeledMatrix(NamedTuple):
    """One feature matrix and its home-win labels, in the same row order."""

    frame: pd.DataFrame
    target: pd.Series


class MarketRows(NamedTuple):
    """Closing-line probabilities and the labels of the games that have one."""

    probabilities: np.ndarray
    labels: np.ndarray


class WalkForwardSettings(NamedTuple):
    """Window and logistic settings for one walk-forward run."""

    v1_through_season: int
    start_season: int
    slate_season: int
    slate_week: int
    lookback_games: int = 10
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    random_state: int = 42
    n_bootstrap: int = 1000
    bootstrap_seed: int = 42
    max_null_fraction: float = 0.5


def default_slate(games: pd.DataFrame) -> dict[str, int]:
    """Return the week after the last completed week in the latest season.

    Raises:
        ValueError: No completed game has a season and a week.
    """
    played: pd.DataFrame = completed_games(games)
    seasons: pd.Series = pd.to_numeric(played["season"], errors="coerce")
    weeks: pd.Series = pd.to_numeric(played["week"], errors="coerce")
    if seasons.isna().any() or weeks.isna().any():
        raise ValueError("Completed games need season and week before a default slate.")
    latest: int = int(seasons.max())
    in_season: pd.Series = weeks.loc[seasons == latest]
    return {"season": latest, "week": int(in_season.max()) + 1}


def week_start_elo_probabilities(
    history: pd.DataFrame,
    week_games: pd.DataFrame,
) -> pd.Series:
    """Elo home probabilities for one week, with that week's results hidden.

    Same-week games do not update each other. A Tuesday slate does not know
    Thursday's score.

    Returns:
        Series indexed by ``game_id`` string, in ``week_games`` order.

    Raises:
        ValueError: A week game has no probability.
    """
    slate: pd.DataFrame = week_games.copy()
    slate["home_won"] = pd.NA
    if "home_score" in slate.columns:
        slate["home_score"] = pd.NA
    if "away_score" in slate.columns:
        slate["away_score"] = pd.NA
    combined: pd.DataFrame = pd.concat([history, slate], ignore_index=True)
    scored: dict[str, Any] = elo_home_probabilities(combined)
    by_id: pd.Series = pd.Series(
        scored["probabilities"].to_numpy(dtype=float),
        index=combined["game_id"].astype(str),
    )
    wanted_ids: list[str] = slate["game_id"].astype(str).tolist()
    chosen: pd.Series = by_id.reindex(wanted_ids).astype(float)
    if not np.isfinite(chosen.to_numpy(dtype=float)).all():
        raise ValueError("Elo did not score every game in the week.")
    return chosen


def run_walk_forward(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    settings: WalkForwardSettings,
) -> dict[str, Any]:
    """Fit each scored week on prior games and pool the log loss.

    Returns:
        The report object. Week rows and the aggregate agree.

    Raises:
        ValueError: The window is empty, a fit has one class, or a
            probability is missing.
    """
    if settings.start_season <= settings.v1_through_season:
        raise ValueError("start_season must be after the v1 window.")
    if settings.slate_week < 1:
        raise ValueError("slate_week must be >= 1.")
    if settings.n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0.")
    history: pd.DataFrame = games_before_week(games, settings.slate_season, settings.slate_week)
    history_ids: set[str] = set(history["game_id"].astype(str))
    prior_team_games: pd.DataFrame = team_games_before_week(
        team_games,
        settings.slate_season,
        settings.slate_week,
        history_ids,
    )
    features: pd.DataFrame = build_pregame_features(
        history,
        prior_team_games,
        lookback_games=settings.lookback_games,
    )
    validate_feature_columns(
        features,
        (WEEKLY_FEATURE_SET,),
        max_null_fraction=settings.max_null_fraction,
    )
    columns: list[str] = list(feature_columns_for(WEEKLY_FEATURE_SET))
    v1_games: pd.DataFrame = history.loc[
        pd.to_numeric(history["season"], errors="coerce") <= settings.v1_through_season
    ].copy()
    v1_matrix: LabeledMatrix = _labeled_matrix(v1_games, features, columns)
    v1_target: pd.Series = v1_matrix.target
    if v1_target.nunique() < 2:
        raise ValueError("The v1 window needs both home wins and home losses.")
    v1_pipeline: Pipeline = fit_logistic_pipeline(
        v1_matrix.frame,
        v1_target,
        C=settings.logistic_c,
        max_iter=settings.logistic_max_iter,
        random_state=settings.random_state,
    )
    v1_count: int = int(len(v1_target))
    week_rows: list[dict[str, Any]] = []
    labels: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    v1_parts: list[np.ndarray] = []
    elo_parts: list[np.ndarray] = []
    home_parts: list[np.ndarray] = []
    market_labels: list[np.ndarray] = []
    market_parts: list[np.ndarray] = []
    for season, week in _scored_week_keys(games, settings):
        week_games: pd.DataFrame = _games_in_week(games, season, week)
        week_history: pd.DataFrame = games_before_week(games, season, week)
        v1_ids: set[str] = set(v1_matrix.frame["game_id"].astype(str))
        week_v1_ids: set[str] = set(
            week_history.loc[
                pd.to_numeric(week_history["season"], errors="coerce")
                <= settings.v1_through_season,
                "game_id",
            ].astype(str)
        )
        if week_v1_ids != v1_ids:
            raise ValueError(
                f"{season} week {week} changes the v1 training set. "
                "Scored weeks have to sit entirely after that season."
            )
        train_matrix: LabeledMatrix = _labeled_matrix(week_history, features, columns)
        train_target: pd.Series = train_matrix.target
        if train_target.nunique() < 2:
            raise ValueError(f"{season} week {week} does not have both classes before it.")
        candidate: Pipeline = fit_logistic_pipeline(
            train_matrix.frame,
            train_target,
            C=settings.logistic_c,
            max_iter=settings.logistic_max_iter,
            random_state=settings.random_state,
        )
        eval_features: pd.DataFrame = _features_for_week(
            week_history,
            week_games,
            prior_team_games,
            settings.lookback_games,
        )
        eval_matrix: LabeledMatrix = _labeled_matrix(week_games, eval_features, columns)
        eval_frame: pd.DataFrame = eval_matrix.frame
        eval_target: pd.Series = eval_matrix.target
        if len(eval_frame) != len(week_games):
            raise ValueError(f"{season} week {week} dropped a game while building features.")
        candidate_p: np.ndarray = _positive_class_probabilities(candidate, eval_frame, columns)
        v1_p: np.ndarray = _positive_class_probabilities(v1_pipeline, eval_frame, columns)
        _require_finite(candidate_p, "candidate", season, week)
        _require_finite(v1_p, "v1", season, week)
        home_rate: float = float(train_target.mean())
        home_p: np.ndarray = np.full(len(eval_target), home_rate, dtype=float)
        elo_p: np.ndarray = _aligned(
            week_start_elo_probabilities(week_history, week_games),
            eval_frame["game_id"],
            "Elo",
            season,
            week,
        )
        y_true: np.ndarray = eval_target.to_numpy(dtype=int)
        market_rows: MarketRows = _market_rows(week_games, eval_frame["game_id"], y_true)
        market_p: np.ndarray = market_rows.probabilities
        market_y: np.ndarray = market_rows.labels
        row: dict[str, Any] = {
            "season": season,
            "week": week,
            "n": int(len(y_true)),
            "n_train_candidate": int(len(train_target)),
            "n_train_v1": v1_count,
            "candidate_log_loss": _mean_log_loss(y_true, candidate_p),
            "v1_log_loss": _mean_log_loss(y_true, v1_p),
            "elo_log_loss": _mean_log_loss(y_true, elo_p),
            "baseline_home_log_loss": _mean_log_loss(y_true, home_p),
            "market_n": int(len(market_y)),
            "market_log_loss": None if len(market_y) == 0 else _mean_log_loss(market_y, market_p),
        }
        week_rows.append(row)
        labels.append(y_true)
        candidate_parts.append(candidate_p)
        v1_parts.append(v1_p)
        elo_parts.append(elo_p)
        home_parts.append(home_p)
        if len(market_y) > 0:
            market_labels.append(market_y)
            market_parts.append(market_p)
        logger.info(
            "Scored %s week %s on %s games. Candidate %.4f, v1 %.4f.",
            season,
            week,
            row["n"],
            row["candidate_log_loss"],
            row["v1_log_loss"],
        )
    if not week_rows:
        raise ValueError("No completed week is after the v1 window and before the slate.")
    y_all: np.ndarray = np.concatenate(labels)
    candidate_all: np.ndarray = np.concatenate(candidate_parts)
    v1_all: np.ndarray = np.concatenate(v1_parts)
    compared: dict[str, float | int | None] = paired_log_loss_difference(
        y_all,
        candidate_all,
        v1_all,
        n_bootstrap=settings.n_bootstrap,
        seed=settings.bootstrap_seed,
    )
    candidate_loss: float = _mean_log_loss(y_all, candidate_all)
    v1_loss: float = _mean_log_loss(y_all, v1_all)
    market_loss: float | None = None
    market_n: int = 0
    if market_labels:
        market_y_all: np.ndarray = np.concatenate(market_labels)
        market_p_all: np.ndarray = np.concatenate(market_parts)
        market_n = int(len(market_y_all))
        market_loss = _mean_log_loss(market_y_all, market_p_all)
    report: dict[str, Any] = {
        "protocol": WALK_FORWARD_PROTOCOL,
        "model": REFERENCE_MODEL_NAME,
        "feature_set": REFERENCE_FEATURE_SET,
        "v1_through_season": int(settings.v1_through_season),
        "start_season": int(settings.start_season),
        "slate": {"season": int(settings.slate_season), "week": int(settings.slate_week)},
        "scored_from": {"season": week_rows[0]["season"], "week": week_rows[0]["week"]},
        "scored_through": {"season": week_rows[-1]["season"], "week": week_rows[-1]["week"]},
        "lookback_games": int(settings.lookback_games),
        "logistic_c": float(settings.logistic_c),
        "logistic_max_iter": int(settings.logistic_max_iter),
        "random_state": int(settings.random_state),
        "n_bootstrap": int(settings.n_bootstrap),
        "bootstrap_seed": int(settings.bootstrap_seed),
        "weeks": week_rows,
        "aggregate": {
            "n": int(len(y_all)),
            "candidate_log_loss": candidate_loss,
            "v1_log_loss": v1_loss,
            "elo_log_loss": _mean_log_loss(y_all, np.concatenate(elo_parts)),
            "baseline_home_log_loss": _mean_log_loss(y_all, np.concatenate(home_parts)),
            "market_log_loss": market_loss,
            "market_n": market_n,
            "log_loss_diff": float(candidate_loss - v1_loss),
            "log_loss_diff_ci_low": _optional_float(compared.get("log_loss_diff_ci_low")),
            "log_loss_diff_ci_high": _optional_float(compared.get("log_loss_diff_ci_high")),
            "beats_v1": bool(candidate_loss < v1_loss),
        },
        "notes": list(_PROTOCOL_NOTES),
    }
    metrics_from_walk_forward_report(report)
    return report


def write_walk_forward_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Write the JSON report and a markdown table. Returns both paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path: Path = output_dir / "walk_forward.json"
    markdown_path: Path = output_dir / "walk_forward.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_walk_forward_markdown(report), encoding="utf-8")
    return {"json": json_path.as_posix(), "markdown": markdown_path.as_posix()}


def render_walk_forward_markdown(report: dict[str, Any]) -> str:
    """Render the week table. The JSON file is the source for the gate."""
    aggregate: dict[str, Any] = report["aggregate"]
    lines: list[str] = [
        "# Walk-forward week backtest",
        "",
        "Logistic `h2_full`. The candidate trains on completed games before the week.",
        f"v1 trains only through season {report['v1_through_season']}.",
        "The market is a benchmark. Nothing here is a betting pick.",
        "On the locked 2016-2023 reference the market has the lower log loss.",
        "This file does not replace that table.",
        "",
        "| Season | Week | n | Candidate | v1 | Elo | Home | Market | Market n |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for week in report["weeks"]:
        lines.append(
            "| {season} | {week} | {n} | {candidate} | {v1} | {elo} | {home} | {market} | {market_n} |".format(
                season=week["season"],
                week=week["week"],
                n=week["n"],
                candidate=_format_loss(week["candidate_log_loss"]),
                v1=_format_loss(week["v1_log_loss"]),
                elo=_format_loss(week["elo_log_loss"]),
                home=_format_loss(week["baseline_home_log_loss"]),
                market=_format_loss(week["market_log_loss"]),
                market_n=week["market_n"],
            )
        )
    lines.extend(
        [
            "",
            "| Scope | n | Candidate | v1 | Elo | Home | Market | Market n | Diff |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| aggregate | {n} | {candidate} | {v1} | {elo} | {home} | {market} | {market_n} | {diff} |".format(
                n=aggregate["n"],
                candidate=_format_loss(aggregate["candidate_log_loss"]),
                v1=_format_loss(aggregate["v1_log_loss"]),
                elo=_format_loss(aggregate["elo_log_loss"]),
                home=_format_loss(aggregate["baseline_home_log_loss"]),
                market=_format_loss(aggregate["market_log_loss"]),
                market_n=aggregate["market_n"],
                diff=_format_loss(aggregate["log_loss_diff"]),
            ),
            "",
            f"beats_v1: {str(aggregate['beats_v1']).lower()}",
            "",
        ]
    )
    ci_low: object = aggregate.get("log_loss_diff_ci_low")
    ci_high: object = aggregate.get("log_loss_diff_ci_high")
    if ci_low is not None and ci_high is not None:
        lines.append(
            "Paired bootstrap of candidate minus v1, "
            f"{report['n_bootstrap']} resamples, seed {report['bootstrap_seed']}: "
            f"{_format_loss(ci_low)} to {_format_loss(ci_high)}."
        )
        lines.append("The promotion gate is the point estimate. The interval is descriptive.")
        lines.append("")
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def verify_walk_forward_report(path: str) -> dict[str, Any]:
    """Exit 1 unless the report beats v1. A valid loss is still a result.

    Raises:
        SystemExit: The report cannot be read, disagrees with itself, or
            does not beat v1.
    """
    try:
        payload: dict[str, Any] = read_walk_forward_report(path)
        tags: dict[str, str] = metrics_from_walk_forward_report(payload)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    decision: dict[str, Any] = promotion_decision(tags)
    aggregate: dict[str, Any] = payload["aggregate"]
    logger.info(
        "Walk-forward n=%s candidate=%s v1=%s diff=%s beats_v1=%s reason=%s.",
        aggregate["n"],
        aggregate["candidate_log_loss"],
        aggregate["v1_log_loss"],
        aggregate["log_loss_diff"],
        aggregate["beats_v1"],
        decision["reason"],
    )
    if not decision["allowed"]:
        raise SystemExit(1)
    return decision


def build_parser() -> argparse.ArgumentParser:
    """Build the walk-forward parser."""
    parser = argparse.ArgumentParser(
        description="Walk-forward week backtest of logistic h2_full against the through-2023 fit."
    )
    parser.add_argument("--slate-season", type=int, default=None)
    parser.add_argument("--slate-week", type=int, default=None)
    parser.add_argument("--v1-through-season", type=int, default=None)
    parser.add_argument("--start-season", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--max-null-fraction", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fixture-path", type=str, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--report", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Write a walk-forward report, or verify one. Does not register or publish.

    A report that loses to v1 is still a successful run. ``--verify`` is the
    gate and exits 1 in that case.
    """
    args: argparse.Namespace = build_parser().parse_args(argv)
    if args.verify:
        if not args.report:
            raise SystemExit("verify needs --report")
        return verify_walk_forward_report(args.report)
    if (args.slate_season is None) != (args.slate_week is None):
        raise SystemExit("slate-season and slate-week must be set together.")
    reference: dict[str, Any] = load_reference_metrics()
    games: pd.DataFrame
    team_games: pd.DataFrame
    games, team_games = load_data(args.fixture_path if args.fixture_path else None)
    if args.slate_season is None:
        slate: dict[str, int] = default_slate(games)
    else:
        slate = {"season": int(args.slate_season), "week": int(args.slate_week)}
    v1_through: int = (
        int(args.v1_through_season)
        if args.v1_through_season is not None
        else int(reference["cv_end_season"])
    )
    start_season: int = int(args.start_season) if args.start_season is not None else v1_through + 1
    n_bootstrap: int = (
        int(args.n_bootstrap) if args.n_bootstrap is not None else int(reference["n_bootstrap"])
    )
    bootstrap_seed: int = (
        int(args.bootstrap_seed)
        if args.bootstrap_seed is not None
        else int(reference["bootstrap_seed"])
    )
    settings = WalkForwardSettings(
        v1_through_season=v1_through,
        start_season=start_season,
        slate_season=slate["season"],
        slate_week=slate["week"],
        lookback_games=int(args.lookback),
        logistic_c=float(args.C),
        logistic_max_iter=int(args.max_iter),
        random_state=int(args.seed),
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        max_null_fraction=float(args.max_null_fraction),
    )
    report: dict[str, Any] = run_walk_forward(games, team_games, settings)
    paths: dict[str, str] = write_walk_forward_report(report, Path(args.output_dir))
    decision: dict[str, Any] = promotion_decision(metrics_from_walk_forward_report(report))
    aggregate: dict[str, Any] = report["aggregate"]
    logger.info(
        "Wrote %s. n=%s candidate=%s v1=%s elo=%s home=%s market=%s beats_v1=%s.",
        paths["json"],
        aggregate["n"],
        aggregate["candidate_log_loss"],
        aggregate["v1_log_loss"],
        aggregate["elo_log_loss"],
        aggregate["baseline_home_log_loss"],
        aggregate["market_log_loss"],
        aggregate["beats_v1"],
    )
    logger.info(
        "Market is a benchmark, not a betting pick. "
        "make verify-walk-forward is the gate. Reason now: %s.",
        decision["reason"],
    )
    return {"report": report, "paths": paths, "decision": decision}


def _scored_week_keys(
    games: pd.DataFrame,
    settings: WalkForwardSettings,
) -> list[tuple[int, int]]:
    """Completed weeks after the v1 season and strictly before the slate."""
    played: pd.DataFrame = completed_games(games)
    seasons: pd.Series = pd.to_numeric(played["season"], errors="coerce")
    weeks: pd.Series = pd.to_numeric(played["week"], errors="coerce")
    if seasons.isna().any() or weeks.isna().any():
        raise ValueError("Completed games need season and week.")
    slate_key: tuple[int, int] = (settings.slate_season, settings.slate_week)
    found: set[tuple[int, int]] = set()
    for season, week in zip(seasons.tolist(), weeks.tolist(), strict=True):
        key: tuple[int, int] = (int(season), int(week))
        if key[0] < settings.start_season or key >= slate_key:
            continue
        found.add(key)
    return sorted(found)


def _games_in_week(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Completed games in one season and week, ordered by kickoff."""
    played: pd.DataFrame = completed_games(games)
    seasons: pd.Series = pd.to_numeric(played["season"], errors="coerce")
    weeks: pd.Series = pd.to_numeric(played["week"], errors="coerce")
    mask: pd.Series = (seasons == int(season)) & (weeks == int(week))
    found: pd.DataFrame = played.loc[mask].copy()
    if found.empty:
        raise ValueError(f"No completed games in {season} week {week}.")
    return found.sort_values(["gameday", "game_id"], kind="mergesort").reset_index(drop=True)


def _features_for_week(
    history: pd.DataFrame,
    week_games: pd.DataFrame,
    team_games: pd.DataFrame,
    lookback_games: int,
) -> pd.DataFrame:
    """Pregame rows for one week. Same-week games are not results or EPA."""
    slate: pd.DataFrame = week_games.copy()
    slate["home_won"] = pd.NA
    season: int = int(pd.to_numeric(slate["season"]).iloc[0])
    week: int = int(pd.to_numeric(slate["week"]).iloc[0])
    history_ids: set[str] = set(history["game_id"].astype(str))
    prior: pd.DataFrame = team_games_before_week(team_games, season, week, history_ids)
    combined: pd.DataFrame = pd.concat([history, slate], ignore_index=True)
    return build_pregame_features(
        combined,
        prior,
        lookback_games=lookback_games,
        game_ids=set(slate["game_id"].astype(str)),
    )


def _labeled_matrix(
    games: pd.DataFrame,
    features: pd.DataFrame,
    columns: list[str],
) -> LabeledMatrix:
    """Feature matrix and home-win labels, ordered by kickoff then game id."""
    frame, target = prepare_training_data(
        games,
        features,
        feature_set=WEEKLY_FEATURE_SET,
    )
    labeled: pd.Series = target.notna()
    frame = frame.loc[labeled].copy()
    frame["_target"] = target.loc[labeled].astype(int).to_numpy()
    frame = frame.sort_values(["gameday", "game_id"], kind="mergesort").reset_index(drop=True)
    labels: pd.Series = frame.pop("_target").astype(int)
    frame[columns] = frame[columns].astype(float)
    return LabeledMatrix(frame=frame, target=labels)


def _positive_class_probabilities(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    """Home-win probability from the class labeled 1."""
    raw: np.ndarray = np.asarray(pipeline.predict_proba(frame[columns]), dtype=float)
    labels: list[int] = [int(label) for label in pipeline.named_steps["classifier"].classes_]
    if 1 not in labels:
        raise ValueError("Logistic fit has no home-win class.")
    return raw[:, labels.index(1)]


def _aligned(
    probabilities: pd.Series,
    game_ids: pd.Series,
    name: str,
    season: int,
    week: int,
) -> np.ndarray:
    """Reindex probabilities to ``game_ids`` and refuse a gap."""
    chosen: pd.Series = probabilities.reindex(game_ids.astype(str).to_numpy()).astype(float)
    values: np.ndarray = chosen.to_numpy(dtype=float)
    _require_finite(values, name, season, week)
    return values


def _market_rows(
    week_games: pd.DataFrame,
    game_ids: pd.Series,
    y_true: np.ndarray,
) -> MarketRows:
    """No-vig probabilities for games that have a closing line."""
    empty = MarketRows(
        probabilities=np.asarray([], dtype=float),
        labels=np.asarray([], dtype=int),
    )
    priced: dict[str, Any] = market_home_probabilities(week_games)
    if priced["probabilities"] is None:
        return empty
    by_id: pd.Series = pd.Series(
        priced["probabilities"].to_numpy(dtype=float),
        index=week_games["game_id"].astype(str),
    )
    aligned: pd.Series = by_id.reindex(game_ids.astype(str).to_numpy())
    finite: np.ndarray = aligned.notna().to_numpy()
    if not bool(finite.any()):
        return empty
    return MarketRows(
        probabilities=aligned.to_numpy(dtype=float)[finite],
        labels=y_true[finite],
    )


def _mean_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Pooled log loss. Raises when the metric is missing."""
    metrics: dict[str, float | int | None] = binary_classification_metrics(y_true, y_proba)
    loss: object = metrics["log_loss"]
    if loss is None:
        raise ValueError("log loss was not finite.")
    number: float = float(loss)
    if not math.isfinite(number):
        raise ValueError("log loss was not finite.")
    return number


def _require_finite(values: np.ndarray, name: str, season: int, week: int) -> None:
    """Raise when a probability vector has a gap."""
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} probability was not finite for {season} week {week}.")


def _optional_float(value: object) -> float | None:
    """Return a finite float, or None when the bootstrap interval is absent."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number: float = float(value)
    if not math.isfinite(number):
        return None
    return number


def _format_loss(value: object) -> str:
    """Four decimal places, matching the reference table. Null stays blank."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return ""
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
