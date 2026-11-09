"""Register logistic ``h2_full`` and promote a version to the ``prod`` alias.

``python -m endzone.modeling.registry_command register`` runs the reference
check, fits one model on completed games through the cutoff, and registers
that version. It does not set an alias.

``python -m endzone.modeling.registry_command promote --version N`` sets
alias ``prod`` only when that version's tags pass the promotion gate.

Neither command deletes a Databricks object. ``make reproduce`` does not
call them.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from endzone.config import configured_registry_uri, configured_tracking_uri
from endzone.features.pregame import (
    build_pregame_features,
    feature_columns_for,
    prepare_training_data,
)
from endzone.modeling.dataset import load_data, validate_feature_columns
from endzone.modeling.models import fit_logistic_pipeline
from endzone.modeling.pipeline import run_reference_check
from endzone.modeling.reference import format_reference_check
from endzone.modeling.registry import (
    PROD_ALIAS,
    REFERENCE_FEATURE_SET,
    assert_can_register,
    default_model_name,
    metrics_from_comparison_csv,
    metrics_from_reference_check,
    promote_registered_version,
    promotion_decision,
    register_model_version,
)
from endzone.modeling.weekly import WEEKLY_FEATURE_SET, games_before_week, team_games_before_week
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

DEFAULT_THROUGH_SEASON: int = 2023


def fit_reference_pipeline(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    through_season: int,
    lookback_games: int = 10,
    logistic_c: float = 1.0,
    logistic_max_iter: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit logistic ``h2_full`` on completed games through ``through_season``.

    The cutoff is every completed game before season ``through_season + 1``,
    week 1. Same-season weeks after that season stay out. Features for a
    game still use only earlier games.

    Returns:
        Dict with ``pipeline``, ``example``, ``n_train``, and ``through_season``.

    Raises:
        ValueError: There are no completed games, one class, or a bad column.
    """
    history: pd.DataFrame = games_before_week(games, int(through_season) + 1, 1)
    history_ids: set[str] = set(history["game_id"].astype(str))
    prior_team_games: pd.DataFrame = team_games_before_week(
        team_games,
        int(through_season) + 1,
        1,
        history_ids,
    )
    features: pd.DataFrame = build_pregame_features(
        history,
        prior_team_games,
        lookback_games=lookback_games,
    )
    validate_feature_columns(features, (WEEKLY_FEATURE_SET,))
    train_x, train_y = prepare_training_data(
        history,
        features,
        feature_set=WEEKLY_FEATURE_SET,
    )
    labeled: pd.Series = train_y.notna()
    train_x = train_x.loc[labeled].reset_index(drop=True)
    train_y = train_y.loc[labeled].astype(int).reset_index(drop=True)
    if train_y.nunique() < 2:
        raise ValueError("Completed games need both home wins and home losses to fit h2_full.")
    columns: list[str] = list(feature_columns_for(WEEKLY_FEATURE_SET))
    train_x[columns] = train_x[columns].astype(float)
    pipeline: Pipeline = fit_logistic_pipeline(
        train_x,
        train_y,
        C=logistic_c,
        max_iter=logistic_max_iter,
        random_state=random_state,
    )
    example: pd.DataFrame = train_x[columns].head(1)
    return {
        "pipeline": pipeline,
        "example": example,
        "n_train": int(len(train_y)),
        "through_season": int(through_season),
    }


def register_command(args: argparse.Namespace) -> dict[str, Any]:
    """Check the golden metrics, fit the reference, and register one version.

    The alias is not set. A failed check raises ``SystemExit`` before any
    registry write.

    Returns:
        Dict with ``model_name``, ``model_version``, ``alias`` (None), and
        ``through_season``.
    """
    tracking_uri: str = args.tracking_uri or configured_tracking_uri()
    registry_uri: str = args.registry_uri or configured_registry_uri()
    assert_can_register(tracking_uri, registry_uri)
    if args.comparison_csv:
        tags: dict[str, str] = metrics_from_comparison_csv(args.comparison_csv)
    else:
        fixture_path: str | None = args.fixture_path if args.fixture_path else None
        check: dict[str, Any] = run_reference_check(fixture_path=fixture_path)
        if not check["ok"]:
            logger.error(format_reference_check(check))
            raise SystemExit(1)
        tags = metrics_from_reference_check(check)
    decision: dict[str, Any] = promotion_decision(tags)
    if not decision["allowed"]:
        logger.error(
            "Refusing to register logistic %s: %s.",
            REFERENCE_FEATURE_SET,
            decision["reason"],
        )
        raise SystemExit(1)
    games, team_games = load_data(args.fixture_path if args.fixture_path else None)
    fitted: dict[str, Any] = fit_reference_pipeline(
        games,
        team_games,
        through_season=int(args.through_season),
        lookback_games=int(args.lookback),
        logistic_c=float(args.C),
        logistic_max_iter=int(args.max_iter),
        random_state=int(args.seed),
    )
    model_name: str = default_model_name(registry_uri)
    stamp: str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    logged = register_model_version(
        fitted["pipeline"],
        example=fitted["example"],
        model_name=model_name,
        tags=tags,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        run_name=f"register-h2-full-{int(args.through_season)}-{stamp}",
        through_season=int(args.through_season),
    )
    logger.info(
        "Registered %s version %s from %s training rows through season %s. "
        "Alias %s was not set. Promote it with make promote-model VERSION=%s.",
        model_name,
        logged.model_version,
        fitted["n_train"],
        int(args.through_season),
        PROD_ALIAS,
        logged.model_version,
    )
    return {
        "model_name": model_name,
        "model_version": logged.model_version,
        "alias": None,
        "through_season": int(args.through_season),
        "allowed": True,
        "reason": decision["reason"],
    }


def promote_command(args: argparse.Namespace) -> dict[str, Any]:
    """Set alias ``prod`` on one existing version when the gate passes.

    Raises:
        SystemExit: The version is not a positive integer, or the gate refuses.
    """
    version: str = _require_version(args.version)
    registry_uri: str = args.registry_uri or configured_registry_uri()
    model_name: str = default_model_name(registry_uri)
    decision: dict[str, Any] = promote_registered_version(model_name, version, registry_uri)
    if not decision["allowed"]:
        raise SystemExit(1)
    return decision


def build_parser() -> argparse.ArgumentParser:
    """Build the register and promote parser."""
    parser = argparse.ArgumentParser(description="Register logistic h2_full or set the prod alias.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser(
        "register",
        help="Fit h2_full through a season and register it. Does not set an alias.",
    )
    register.add_argument("--through-season", type=int, default=DEFAULT_THROUGH_SEASON)
    register.add_argument("--lookback", type=int, default=10)
    register.add_argument("--C", type=float, default=1.0)
    register.add_argument("--max-iter", type=int, default=1000)
    register.add_argument("--seed", type=int, default=42)
    register.add_argument(
        "--comparison-csv",
        type=str,
        default=None,
        help=(
            "Use the logistic h2_full row in this comparison CSV instead of "
            "running the reference check. The gate still applies."
        ),
    )
    register.add_argument("--fixture-path", type=str, default=None)
    register.add_argument("--tracking-uri", type=str, default=None)
    register.add_argument("--registry-uri", type=str, default=None)
    promote = subparsers.add_parser(
        "promote",
        help="Set the prod alias on one version when its CV tags pass the gate.",
    )
    promote.add_argument("--version", type=str, required=True)
    promote.add_argument("--registry-uri", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Register a version or promote one. Does not run from weekly scoring."""
    args: argparse.Namespace = build_parser().parse_args(argv)
    if args.command == "register":
        return register_command(args)
    if args.command == "promote":
        return promote_command(args)
    raise SystemExit(2)


def _require_version(value: str) -> str:
    """Return a positive integer version. ``prod`` is an alias, not a version.

    Raises:
        SystemExit: The text is blank or not a positive integer.
    """
    text: str = "" if value is None else str(value).strip()
    if not text.isdigit() or int(text) < 1:
        logger.error("promote needs a positive integer version. The prod alias is not a version.")
        raise SystemExit(1)
    return str(int(text))


if __name__ == "__main__":
    main()
