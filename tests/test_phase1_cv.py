"""Phase 1 rolling-origin CV, Elo is covered with the leakage tests."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.pregame import (
    DIFF_FEATURE_COLUMNS,
    PREGAME_FEATURE_COLUMNS,
    PREGAME_FEATURE_COLUMNS_H2_DEF,
    PREGAME_FEATURE_COLUMNS_H2_FULL,
    REGRESSED_EPA_COLUMNS,
    SEASON_DEFENSIVE_EPA_COLUMNS,
    american_odds_to_implied_probability,
    market_home_probabilities,
    no_vig_home_probability,
)
from src.train.train import (
    HYPOTHESIS_H1,
    HYPOTHESIS_H2,
    MIN_LOG_LOSS_IMPROVEMENT,
    PHASE1_EXPERIMENT_NAME,
    PHASE1_TAG,
    PHASE2_EXPERIMENT_NAME,
    PHASE2_H2_EXPERIMENT_NAME,
    PHASE2_TAG,
    attach_reference_comparison,
    binary_classification_metrics,
    compare_oof_to_reference,
    decide_log_loss_improvement,
    feature_column_groups,
    log_phase1_to_mlflow,
    logistic_feature_sets_for,
    main,
    make_rolling_origin_config,
    paired_log_loss_difference,
    phase_tracking_settings,
    resolve_reference_spec,
    rolling_origin_cv,
    run_phase1_baselines,
    write_phase1_outputs,
)

_NEUTRAL_TRACKING_URI: str = "sqlite:///:memory:"


def _release_mlflow_sqlalchemy_engines() -> None:
    """Dispose cached SQLAlchemy engines that keep a SQLite file open."""
    from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
    from mlflow.tracking._model_registry.utils import _get_store_registry
    from mlflow.tracking._tracking_service.utils import _tracking_store_registry

    for store_cls in (TrackingStore, RegistryStore, SqlAlchemyJobStore):
        engine_map = getattr(store_cls, "_engine_map", None)
        if not isinstance(engine_map, dict):
            continue
        for engine in list(engine_map.values()):
            engine.dispose()
        engine_map.clear()

    tracking_cache = _tracking_store_registry._get_store_with_resolved_uri
    tracking_cache.cache_clear()
    registry_cache = _get_store_registry()._get_store_with_resolved_uri
    registry_cache.cache_clear()


def _reset_mlflow_tracking() -> None:
    """End any open run and leave a non-file tracking URI."""
    import mlflow

    if mlflow.active_run() is not None:
        mlflow.end_run()
    _release_mlflow_sqlalchemy_engines()
    mlflow.set_tracking_uri(_NEUTRAL_TRACKING_URI)


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore MLflow's process-global tracking URI after each test."""
    import mlflow.tracking._tracking_service.utils as tracking_utils

    monkeypatch.setattr(tracking_utils, "_tracking_uri", tracking_utils._tracking_uri)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    yield
    _reset_mlflow_tracking()


def _schedule(with_moneyline: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Five seasons. 2021-2023 have both classes. 2024 and 2025 are present."""
    rows: list[dict[str, object]] = []
    game_index: int = 0
    for season in (2021, 2022, 2023, 2024, 2025):
        row_count: int = 8 if season <= 2023 else 4
        for index in range(row_count):
            game_index += 1
            home_won: int = index % 2
            rows.append(
                {
                    "game_id": f"G{game_index}",
                    "season": season,
                    "gameday": pd.Timestamp(year=season, month=9, day=1) + pd.Timedelta(days=index),
                    "home_team_franchise": "A" if index % 2 == 0 else "B",
                    "away_team_franchise": "B" if index % 2 == 0 else "A",
                    "home_score": 24 if home_won else 14,
                    "away_score": 17 if home_won else 21,
                    "home_won": home_won,
                    "home_rest": 7,
                    "away_rest": 7,
                }
            )
    games = pd.DataFrame(rows)
    if with_moneyline:
        games["home_moneyline"] = -150
        games["away_moneyline"] = 130
    epa_rows: list[dict[str, object]] = []
    for _, game in games.iterrows():
        for team, offense in (
            (game["home_team_franchise"], 0.2 if game["home_won"] == 1 else -0.1),
            (game["away_team_franchise"], -0.05),
        ):
            epa_rows.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "gameday": game["gameday"],
                    "team_franchise": team,
                    "avg_offensive_epa": offense,
                    "avg_defensive_epa_allowed": -offense,
                }
            )
    return games, pd.DataFrame(epa_rows)


def _fast_config(**overrides: object) -> object:
    """A two-season window so the logistic fits stay small."""
    base: dict[str, object] = {
        "cv_start_season": 2022,
        "cv_end_season": 2023,
        "n_bootstrap": 20,
        "bootstrap_seed": 1,
    }
    base.update(overrides)
    return make_rolling_origin_config(**base)


class TestRollingOriginConfig:
    """The CV window stops before the final validation season."""

    def test_defaults_are_2016_through_2023(self) -> None:
        """2024 and 2025 are outside the default window."""
        config = make_rolling_origin_config()
        assert config.cv_start_season == 2016
        assert config.cv_end_season == 2023

    def test_rejects_a_window_that_includes_2024_or_2025(self) -> None:
        """The final validation season and the test season cannot be scored."""
        with pytest.raises(ValueError, match="2024"):
            make_rolling_origin_config(cv_end_season=2024)
        with pytest.raises(ValueError, match="2024"):
            make_rolling_origin_config(cv_start_season=2023, cv_end_season=2025)

    def test_rejects_an_inverted_window_and_a_negative_bootstrap(self) -> None:
        """Start, end, and bootstrap count are checked before any fit."""
        with pytest.raises(ValueError, match="cv_start_season"):
            make_rolling_origin_config(cv_start_season=2023, cv_end_season=2020)
        with pytest.raises(ValueError, match="n_bootstrap"):
            make_rolling_origin_config(n_bootstrap=-1)
        with pytest.raises(TypeError):
            make_rolling_origin_config(not_a_field=1)


class TestFoldBoundaries:
    """Each scored season is trained only on earlier seasons."""

    def test_2024_and_2025_are_never_trained_or_scored(self) -> None:
        """Rows from the held-out seasons stay out of every fold."""
        games, _epa = _schedule(with_moneyline=False)
        frame = games[["game_id", "gameday", "season"]].copy()
        seen: list[dict[str, list[int]]] = []

        def spy(
            train_frame: pd.DataFrame,
            train_target: pd.Series,
            eval_frame: pd.DataFrame,
            eval_target: pd.Series,
        ) -> np.ndarray:
            del train_target, eval_target
            seen.append(
                {
                    "train": sorted(int(season) for season in train_frame["season"].unique()),
                    "eval": sorted(int(season) for season in eval_frame["season"].unique()),
                }
            )
            return np.full(len(eval_frame), 0.6)

        result = rolling_origin_cv(frame, games["home_won"], spy, _fast_config(n_bootstrap=0))
        assert seen == [
            {"train": [2021], "eval": [2022]},
            {"train": [2021, 2022], "eval": [2023]},
        ]
        assert set(result["oof"]["season"].tolist()) == {2022, 2023}
        assert result["pooled"]["n"] == 16
        assert result["pooled"]["log_loss"] is not None
        for fold in result["folds"]:
            assert fold["eval_season"] < 2024
            assert 2024 not in fold["train_seasons"]
            assert 2025 not in fold["train_seasons"]
            if fold["status"] == "scored":
                assert all(season < fold["eval_season"] for season in fold["train_seasons"])

    def test_skip_reasons_cover_empty_and_single_class_folds(self) -> None:
        """A fold that cannot be fit is recorded and the predictor is not called."""
        calls: list[int] = []

        def spy(
            train_frame: pd.DataFrame,
            train_target: pd.Series,
            eval_frame: pd.DataFrame,
            eval_target: pd.Series,
        ) -> np.ndarray:
            del train_frame, train_target, eval_target
            calls.append(len(eval_frame))
            return np.full(len(eval_frame), 0.55)

        frame = pd.DataFrame(
            {
                "game_id": [f"S{index}" for index in range(8)],
                "season": [2018, 2018, 2020, 2020, 2021, 2021, 2022, 2022],
                "gameday": pd.date_range("2018-09-01", periods=8, freq="7D"),
            }
        )
        target = pd.Series([0, 0, 1, 1, 0, 1, 1, 1])
        result = rolling_origin_cv(
            frame,
            target,
            spy,
            make_rolling_origin_config(
                cv_start_season=2018,
                cv_end_season=2022,
                n_bootstrap=0,
            ),
        )
        reasons = {fold["reason"] for fold in result["folds"] if fold["status"] == "skipped"}
        assert reasons == {
            "no_train_rows",
            "no_eval_rows",
            "single_class_train",
            "single_class_eval",
        }
        assert calls == [2]
        assert set(result["oof"]["season"].tolist()) == {2021}

    def test_rejects_a_frame_that_cannot_be_split(self) -> None:
        """Missing columns, bad labels, and duplicate ids fail before a fold."""
        frame = pd.DataFrame({"game_id": ["a", "b"], "season": [2022, 2023]})
        target = pd.Series([0, 1])
        config = _fast_config(n_bootstrap=0)
        with pytest.raises(ValueError, match="game_id and season"):
            rolling_origin_cv(frame.drop(columns=["season"]), target, lambda *args: None, config)
        with pytest.raises(ValueError, match="same number of rows"):
            rolling_origin_cv(frame, target.iloc[:1], lambda *args: None, config)
        with pytest.raises(ValueError, match="at least one row"):
            rolling_origin_cv(frame.iloc[:0], target.iloc[:0], lambda *args: None, config)
        with pytest.raises(ValueError, match="no nulls"):
            broken = target.copy()
            broken.iloc[0] = np.nan
            rolling_origin_cv(frame, broken, lambda *args: None, config)
        with pytest.raises(ValueError, match="only classes"):
            broken = target.copy()
            broken.iloc[0] = 2
            rolling_origin_cv(frame, broken, lambda *args: None, config)
        duplicated = pd.concat([frame, frame], ignore_index=True)
        with pytest.raises(ValueError, match="unique"):
            rolling_origin_cv(duplicated, pd.Series([0, 1, 0, 1]), lambda *args: None, config)

    def test_rejects_a_bad_probability_vector(self) -> None:
        """The fold predictor must return one finite probability per eval row."""
        frame = pd.DataFrame(
            {
                "game_id": ["a", "b", "c", "d"],
                "season": [2021, 2021, 2022, 2022],
            }
        )
        target = pd.Series([0, 1, 0, 1])
        config = _fast_config(n_bootstrap=0)

        def short(
            train_frame: pd.DataFrame,
            train_target: pd.Series,
            eval_frame: pd.DataFrame,
            eval_target: pd.Series,
        ) -> np.ndarray:
            del train_frame, train_target, eval_frame, eval_target
            return np.array([0.5])

        def missing(
            train_frame: pd.DataFrame,
            train_target: pd.Series,
            eval_frame: pd.DataFrame,
            eval_target: pd.Series,
        ) -> np.ndarray:
            del train_frame, train_target, eval_target
            return np.full(len(eval_frame), np.nan)

        with pytest.raises(ValueError, match="probabilities for"):
            rolling_origin_cv(frame, target, short, config)
        with pytest.raises(ValueError, match="non-finite"):
            rolling_origin_cv(frame, target, missing, config)


class TestPairedBootstrap:
    """The log loss gap resamples both models on the same games."""

    def test_a_better_model_has_a_negative_difference_and_a_stable_interval(self) -> None:
        """The point difference sits inside the percentile interval."""
        labels = np.array([0, 1, 0, 1])
        better = np.array([0.2, 0.8, 0.3, 0.7])
        worse = np.array([0.8, 0.2, 0.7, 0.3])
        result = paired_log_loss_difference(labels, better, worse, n_bootstrap=40, seed=3)
        assert result["log_loss_diff"] < 0
        assert result["log_loss_diff_ci_low"] <= result["log_loss_diff"]
        assert result["log_loss_diff"] <= result["log_loss_diff_ci_high"]
        again = paired_log_loss_difference(labels, better, worse, n_bootstrap=40, seed=3)
        assert again["log_loss_diff_ci_low"] == result["log_loss_diff_ci_low"]

    def test_edge_cases_skip_the_interval_or_raise(self) -> None:
        """Empty, one-row, zero-resample, and invalid inputs are explicit."""
        labels = np.array([0, 1, 0, 1])
        better = np.array([0.2, 0.8, 0.3, 0.7])
        worse = np.array([0.8, 0.2, 0.7, 0.3])
        zero = paired_log_loss_difference(labels, better, worse, n_bootstrap=0, seed=1)
        assert zero["log_loss_diff"] < 0
        assert zero["log_loss_diff_ci_low"] is None
        one_row = paired_log_loss_difference(
            np.array([1]),
            np.array([0.7]),
            np.array([0.4]),
            n_bootstrap=10,
            seed=1,
        )
        assert one_row["log_loss_diff_ci_high"] is None
        empty = paired_log_loss_difference(np.array([]), np.array([]), np.array([]), 5, 1)
        assert empty["n"] == 0
        assert empty["log_loss"] is None
        with pytest.raises(ValueError, match="same length"):
            paired_log_loss_difference(labels, better, worse[:-1], n_bootstrap=5, seed=1)
        with pytest.raises(ValueError, match="n_bootstrap"):
            paired_log_loss_difference(labels, better, worse, n_bootstrap=-1, seed=1)
        with pytest.raises(ValueError, match="finite"):
            paired_log_loss_difference(
                labels,
                np.array([np.nan, 0.5, 0.5, 0.5]),
                worse,
                n_bootstrap=5,
                seed=1,
            )
        one_class = binary_classification_metrics(np.array([1, 1]), np.array([0.6, 0.7]))
        assert one_class["auc"] is None
        assert one_class["accuracy"] == pytest.approx(1.0)
        assert binary_classification_metrics(np.array([]), np.array([]))["log_loss"] is None

    def test_empty_out_of_fold_tables_have_no_paired_difference(self) -> None:
        """A missing reference does not invent a confidence interval."""
        empty = compare_oof_to_reference(None, None, n_bootstrap=5, seed=1)
        assert empty["n"] == 0
        assert empty["log_loss_diff"] is None
        left = pd.DataFrame({"game_id": ["a"], "y_true": [1], "y_proba": [0.6]})
        right = pd.DataFrame({"game_id": ["b"], "y_true": [1], "y_proba": [0.4]})
        unmatched = compare_oof_to_reference(left, right, n_bootstrap=5, seed=1)
        assert unmatched["n"] == 0


class TestMarketProbability:
    """No-vig prices, and a clean skip when the columns are absent."""

    def test_even_prices_and_a_home_favorite(self) -> None:
        """Equal moneylines are 0.5. A favorite is the normalized raw price."""
        assert no_vig_home_probability(-110, -110) == pytest.approx(0.5)
        home_raw: float = 150.0 / 250.0
        away_raw: float = 100.0 / 230.0
        favorite: float = no_vig_home_probability(-150, 130)
        assert favorite == pytest.approx(home_raw / (home_raw + away_raw))
        assert favorite > 0.5
        assert no_vig_home_probability(150, -170) < 0.5
        assert american_odds_to_implied_probability(100) == pytest.approx(0.5)

    def test_invalid_odds_and_missing_columns(self) -> None:
        """Zero odds and a frame without moneylines do not raise into the CV."""
        with pytest.raises(ValueError, match="non-zero"):
            american_odds_to_implied_probability(0)
        with pytest.raises(ValueError, match="non-zero"):
            american_odds_to_implied_probability(np.inf)
        absent = market_home_probabilities(pd.DataFrame({"game_id": [1], "home_moneyline": [-110]}))
        assert absent["status"] == "skipped"
        assert absent["reason"] == "moneyline_columns_absent"
        assert absent["probabilities"] is None
        lines = pd.DataFrame(
            {
                "home_moneyline": [-110, np.nan, 0, 150],
                "away_moneyline": [-110, -110, -110, -170],
            }
        )
        partial = market_home_probabilities(lines)
        assert partial["status"] == "ok"
        assert partial["n_scored"] == 2
        assert partial["n_missing_lines"] == 2
        assert partial["probabilities"].iloc[0] == pytest.approx(0.5)
        empty = market_home_probabilities(
            pd.DataFrame({"home_moneyline": [np.nan], "away_moneyline": [-110]})
        )
        assert empty["reason"] == "no_valid_moneylines"


class TestPhase1Report:
    """Both feature sets are scored. The market is optional."""

    def test_drops_games_played_from_the_current_set_and_skips_a_missing_market(
        self,
        tmp_path: Path,
    ) -> None:
        """The legacy matrix is logged beside the default, and 2024 is not scored."""
        games, team_games = _schedule(with_moneyline=False)
        report = run_phase1_baselines(
            games,
            team_games,
            config=_fast_config(),
            lookback_games=2,
            logistic_max_iter=200,
        )
        table = report["comparison_table"]
        pairs = set(zip(table["model"], table["feature_set"], strict=True))
        assert ("logistic", "current") in pairs
        assert ("logistic", "with_games_played") in pairs
        assert ("logistic", "diff") not in pairs
        assert ("logistic", "diff_only") not in pairs
        assert ("baseline-home", "none") in pairs
        assert ("elo", "elo") in pairs
        assert report["phase"] == 1
        assert report["reference_model"] == "baseline-home"
        assert report["reference_feature_set"] == "none"
        assert report["hypothesis"] is None
        assert "h1-diff" not in report["markdown"]
        assert "Acceptance" not in report["markdown"]
        baseline = next(item for item in report["results"] if item["model"] == "baseline-home")
        assert baseline["comparison"]["log_loss_diff"] == pytest.approx(0.0)
        market = table.loc[table["model"] == "market"].iloc[0]
        assert market["status"] == "skipped"
        assert market["reason"] == "moneyline_columns_absent"
        current = next(item for item in report["results"] if item["feature_set"] == "current")
        legacy = next(
            item for item in report["results"] if item["feature_set"] == "with_games_played"
        )
        assert "home_games_played" not in current["params"]["features"]
        assert "away_games_played" not in current["params"]["features"]
        assert "home_games_played" in legacy["params"]["features"]
        assert (
            current["comparison"]["log_loss_diff_ci_low"] <= current["comparison"]["log_loss_diff"]
        )
        assert (
            current["comparison"]["log_loss_diff"] <= current["comparison"]["log_loss_diff_ci_high"]
        )
        assert "\u2014" not in report["markdown"]
        assert "2025 is not scored" in report["markdown"]
        assert "No model is promoted" in report["markdown"]
        for result in report["results"]:
            if result["oof"] is None:
                continue
            assert set(result["oof"]["season"]).isdisjoint({2024, 2025})
        paths = write_phase1_outputs(report, tmp_path / "phase1")
        assert Path(paths["markdown_path"]).is_file()
        loaded = pd.read_csv(paths["csv_path"])
        assert "log_loss_diff" in loaded.columns
        assert any(Path(path).is_file() for path in paths["oof_paths"].values())

    def test_scores_the_market_on_games_that_have_a_line(self) -> None:
        """A row with a null line is left out of the market out-of-fold table."""
        games, team_games = _schedule(with_moneyline=True)
        scored_2022 = games.index[games["season"] == 2022]
        games.loc[scored_2022[0], "home_moneyline"] = np.nan
        dropped_id: str = str(games.loc[scored_2022[0], "game_id"])
        report = run_phase1_baselines(
            games,
            team_games,
            config=_fast_config(),
            lookback_games=2,
            logistic_max_iter=200,
        )
        market = next(item for item in report["results"] if item["model"] == "market")
        baseline = next(item for item in report["results"] if item["model"] == "baseline-home")
        assert market["status"] == "ok"
        assert dropped_id not in set(market["oof"]["game_id"])
        assert dropped_id in set(baseline["oof"]["game_id"])
        assert market["comparison"]["n"] == len(market["oof"])
        assert market["comparison"]["n"] < len(baseline["oof"])
        assert "Skipped (moneyline_columns_absent)" not in report["markdown"]

    def test_logs_phase_feature_set_and_model_params(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each scored model is a phase=1 run. A skipped market is not logged."""
        import mlflow

        games, team_games = _schedule(with_moneyline=False)
        report = run_phase1_baselines(
            games,
            team_games,
            config=_fast_config(n_bootstrap=4),
            lookback_games=2,
            logistic_max_iter=200,
        )
        paths = write_phase1_outputs(report, tmp_path / "report")
        tracking_uri: str = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
        monkeypatch.setattr("src.train.train.LOCAL_ARTIFACT_DIR", str(tmp_path / "mlartifacts"))
        try:
            logged = log_phase1_to_mlflow(report, paths, tracking_uri=tracking_uri)
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            assert logged["experiment_name"] == PHASE1_EXPERIMENT_NAME
            logged_pairs = {(item["model"], item["feature_set"]) for item in logged["run_ids"]}
            assert ("market", "market") not in logged_pairs
            assert ("logistic", "current") in logged_pairs
            assert ("logistic", "with_games_played") in logged_pairs
            sample = client.get_run(logged["run_ids"][0]["run_id"])
            assert sample.data.tags["phase"] == PHASE1_TAG
            assert sample.data.tags["cv_scheme"] == "rolling_season_2016_2023"
            assert "git_commit" in sample.data.tags
            assert "feature_set" in sample.data.params
            assert "model_type" in sample.data.params
            assert "cv_log_loss" in sample.data.metrics
            assert "val_log_loss" not in sample.data.metrics
            artifact_names = {item.path for item in client.list_artifacts(sample.info.run_id)}
            assert "comparison.md" in artifact_names
            assert "comparison.csv" in artifact_names
        finally:
            _reset_mlflow_tracking()

    def test_cli_writes_the_report_from_a_fixture(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--cv`` reads fixtures, writes reports/phase1, and does not score 2025."""
        import mlflow

        games, team_games = _schedule(with_moneyline=True)
        fixture_dir = tmp_path / "fixtures"
        fixture_dir.mkdir()
        games.to_parquet(fixture_dir / "fct_games.parquet")
        team_games.to_parquet(fixture_dir / "fct_team_game_epa.parquet")
        tracking_uri: str = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setattr("src.train.train.LOCAL_ARTIFACT_DIR", str(tmp_path / "mlartifacts"))
        output_dir = tmp_path / "out"
        try:
            main(
                [
                    "--cv",
                    "--fixture-path",
                    str(fixture_dir),
                    "--output-dir",
                    str(output_dir),
                    "--cv-start-season",
                    "2022",
                    "--cv-end-season",
                    "2023",
                    "--n-bootstrap",
                    "4",
                    "--lookback",
                    "2",
                    "--max-iter",
                    "200",
                    "--seed",
                    "7",
                ]
            )
            markdown = (output_dir / "comparison.md").read_text(encoding="utf-8")
            assert "market" in markdown
            assert "2025 is not scored" in markdown
            assert "\u2014" not in markdown
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            experiment = client.get_experiment_by_name(PHASE1_EXPERIMENT_NAME)
            assert experiment is not None
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            assert runs
            assert all(run.data.tags.get("phase") == PHASE1_TAG for run in runs)
            assert all(run.data.tags.get("hypothesis") is None for run in runs)
        finally:
            _reset_mlflow_tracking()


def _oof(probabilities: list[float]) -> pd.DataFrame:
    """Four scored games with fixed labels and the given home probabilities."""
    return pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "season": [2022, 2022, 2023, 2023],
            "y_true": [1, 0, 1, 0],
            "y_proba": probabilities,
        }
    )


def _scored(model_name: str, feature_set: str, probabilities: list[float]) -> dict[str, object]:
    """One CV result row with an out-of-fold table and no comparison yet."""
    return {
        "model": model_name,
        "feature_set": feature_set,
        "status": "ok",
        "oof": _oof(probabilities),
        "comparison": None,
    }


class TestReferenceSelection:
    """The paired reference is chosen by phase, model, and feature set."""

    def test_phase_defaults_and_explicit_overrides(self) -> None:
        """Phase 1 stays on the home prior. Phase 2 uses logistic current."""
        phase1 = resolve_reference_spec(1)
        assert phase1 == {"model": "baseline-home", "feature_set": "none"}
        phase2 = resolve_reference_spec(2)
        assert phase2 == {"model": "logistic", "feature_set": "current"}
        elo = resolve_reference_spec(2, reference_model="elo")
        assert elo == {"model": "elo", "feature_set": "elo"}
        logistic_diff = resolve_reference_spec(
            1,
            reference_model="logistic",
            reference_feature_set="diff",
        )
        assert logistic_diff == {"model": "logistic", "feature_set": "diff"}
        assert logistic_feature_sets_for(1) == ("current", "with_games_played")
        assert logistic_feature_sets_for(2) == ("current", "diff", "diff_only")
        assert logistic_feature_sets_for(2, hypothesis="h2") == ("current", "h2_def", "h2_full")
        assert logistic_feature_sets_for(2, hypothesis="h2-season-def-regression") == (
            "current",
            "h2_def",
            "h2_full",
        )
        h1_settings = phase_tracking_settings(2)
        assert h1_settings["hypothesis"] == HYPOTHESIS_H1
        assert h1_settings["output_dir"] == "reports/phase2"
        assert h1_settings["experiment_name"] == PHASE2_EXPERIMENT_NAME
        h2_settings = phase_tracking_settings(2, hypothesis="h2")
        assert h2_settings["hypothesis"] == HYPOTHESIS_H2
        assert h2_settings["output_dir"] == "reports/phase2_h2"
        assert h2_settings["experiment_name"] == PHASE2_H2_EXPERIMENT_NAME
        assert h2_settings["phase_tag"] == PHASE2_TAG
        with pytest.raises(ValueError, match="Unknown phase"):
            resolve_reference_spec(3)
        with pytest.raises(ValueError, match="Unknown reference model"):
            resolve_reference_spec(2, reference_model="gbm")
        with pytest.raises(ValueError, match="does not take a hypothesis"):
            logistic_feature_sets_for(1, hypothesis="h2")
        with pytest.raises(ValueError, match="Unknown hypothesis"):
            logistic_feature_sets_for(2, hypothesis="h3")

    def test_attach_uses_the_named_reference_and_rejects_a_missing_one(self) -> None:
        """Log loss gaps are versus the selected row, which differs from itself by zero."""
        results = [
            _scored("baseline-home", "none", [0.55, 0.55, 0.55, 0.55]),
            _scored("logistic", "current", [0.70, 0.20, 0.80, 0.30]),
            _scored("logistic", "diff", [0.90, 0.10, 0.85, 0.15]),
        ]
        attach_reference_comparison(results, "logistic", "current", n_bootstrap=25, seed=1)
        current = results[1]["comparison"]
        diff = results[2]["comparison"]
        home = results[0]["comparison"]
        assert isinstance(current, dict)
        assert isinstance(diff, dict)
        assert isinstance(home, dict)
        assert current["log_loss_diff"] == pytest.approx(0.0)
        assert current["log_loss_diff_ci_low"] == pytest.approx(0.0)
        assert current["log_loss_diff_ci_high"] == pytest.approx(0.0)
        assert diff["reference_log_loss"] == pytest.approx(current["log_loss"])
        assert home["reference_log_loss"] == pytest.approx(current["log_loss"])
        assert diff["log_loss_diff"] != pytest.approx(0.0)
        with pytest.raises(ValueError, match="not in the CV results"):
            attach_reference_comparison(results, "logistic", "diff_only", n_bootstrap=0, seed=1)

    def test_acceptance_rule_requires_both_the_margin_and_an_interval_below_zero(self) -> None:
        """0.003 and a 95% interval entirely below 0 are both required."""
        minimum = MIN_LOG_LOSS_IMPROVEMENT
        accepted = decide_log_loss_improvement(-minimum, -0.01, -0.0001)
        assert accepted["decision"] == "accepted"
        short = decide_log_loss_improvement(-0.0029, -0.01, -0.001)
        assert short["decision"] == "rejected"
        includes_zero = decide_log_loss_improvement(-0.01, -0.02, 0.0)
        assert includes_zero["decision"] == "rejected"
        wrong_side = decide_log_loss_improvement(0.01, 0.001, 0.02)
        assert wrong_side["decision"] == "rejected"
        missing = decide_log_loss_improvement(-0.01, None, -0.001)
        assert missing["decision"] == "inconclusive"
        empty = decide_log_loss_improvement(None, None, None)
        assert empty["decision"] == "inconclusive"

    def test_difference_columns_impute_at_zero_and_level_win_rate_stays_at_half(self) -> None:
        """A missing differential win rate is not filled with a home edge."""
        groups = feature_column_groups(
            [
                "home_recent_offensive_epa_per_play",
                "home_recent_win_rate",
                "home_rest_days",
                "diff_recent_win_rate",
                "diff_rest_days",
            ]
        )
        assert groups["epa"] == ["home_recent_offensive_epa_per_play"]
        assert groups["win_rate"] == ["home_recent_win_rate"]
        assert groups["other"] == ["home_rest_days"]
        assert groups["diff"] == ["diff_recent_win_rate", "diff_rest_days"]
        h2_groups = feature_column_groups(list(PREGAME_FEATURE_COLUMNS_H2_FULL))
        for column in SEASON_DEFENSIVE_EPA_COLUMNS + REGRESSED_EPA_COLUMNS:
            assert column in h2_groups["epa"]
            assert column not in h2_groups["win_rate"]
            assert column not in h2_groups["diff"]


class TestPhase2Hypothesis:
    """H1 scores diff and diff_only against logistic current, plus the baselines."""

    def test_phase2_report_uses_logistic_current_as_the_reference(self) -> None:
        """The phase 1 sets stay out. The reference difference is zero."""
        games, team_games = _schedule(with_moneyline=False)
        report = run_phase1_baselines(
            games,
            team_games,
            config=_fast_config(n_bootstrap=8),
            lookback_games=2,
            logistic_max_iter=200,
            phase=2,
        )
        table = report["comparison_table"]
        pairs = set(zip(table["model"], table["feature_set"], strict=True))
        assert pairs == {
            ("baseline-home", "none"),
            ("logistic", "current"),
            ("logistic", "diff"),
            ("logistic", "diff_only"),
            ("elo", "elo"),
            ("market", "market"),
        }
        assert report["reference_model"] == "logistic"
        assert report["reference_feature_set"] == "current"
        assert report["phase"] == 2
        assert report["hypothesis"] == HYPOTHESIS_H1
        current = next(item for item in report["results"] if item["feature_set"] == "current")
        diff = next(item for item in report["results"] if item["feature_set"] == "diff")
        diff_only = next(item for item in report["results"] if item["feature_set"] == "diff_only")
        assert current["comparison"]["log_loss_diff"] == pytest.approx(0.0)
        assert diff["comparison"]["reference_log_loss"] == pytest.approx(
            current["comparison"]["log_loss"]
        )
        assert diff_only["comparison"]["n"] == current["comparison"]["n"]
        current_features = current["params"]["features"].split(",")
        diff_features = diff["params"]["features"].split(",")
        diff_only_features = diff_only["params"]["features"].split(",")
        assert current_features == list(PREGAME_FEATURE_COLUMNS)
        assert diff_features[: len(PREGAME_FEATURE_COLUMNS)] == list(PREGAME_FEATURE_COLUMNS)
        assert diff_features[len(PREGAME_FEATURE_COLUMNS) :] == list(DIFF_FEATURE_COLUMNS)
        assert diff_only_features == list(DIFF_FEATURE_COLUMNS)
        assert "home_games_played" not in diff_features
        decisions = {row["feature_set"]: row["decision"] for row in report["acceptance"]}
        assert decisions["current"] == "reference"
        assert decisions["diff"] in {"accepted", "rejected", "inconclusive"}
        assert decisions["diff_only"] in {"accepted", "rejected", "inconclusive"}
        assert "Hypothesis: `h1-diff`" in report["markdown"]
        assert "0.003" in report["markdown"]
        assert "No model is promoted" in report["markdown"]
        assert "\u2014" not in report["markdown"]
        for result in report["results"]:
            if result["oof"] is None:
                continue
            assert set(result["oof"]["season"]).isdisjoint({2024, 2025})

    def test_cli_phase2_tags_hypothesis_and_keeps_phase1_available(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--phase 2`` logs phase=2 and hypothesis=h1-diff. ``--reference-model`` selects."""
        import mlflow

        games, team_games = _schedule(with_moneyline=True)
        fixture_dir = tmp_path / "fixtures"
        fixture_dir.mkdir()
        games.to_parquet(fixture_dir / "fct_games.parquet")
        team_games.to_parquet(fixture_dir / "fct_team_game_epa.parquet")
        tracking_uri: str = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setattr("src.train.train.LOCAL_ARTIFACT_DIR", str(tmp_path / "mlartifacts"))
        output_dir = tmp_path / "phase2"
        try:
            main(
                [
                    "--cv",
                    "--phase",
                    "2",
                    "--reference-model",
                    "logistic",
                    "--reference-feature-set",
                    "current",
                    "--fixture-path",
                    str(fixture_dir),
                    "--output-dir",
                    str(output_dir),
                    "--cv-start-season",
                    "2022",
                    "--cv-end-season",
                    "2023",
                    "--n-bootstrap",
                    "4",
                    "--lookback",
                    "2",
                    "--max-iter",
                    "200",
                    "--seed",
                    "7",
                ]
            )
            table = pd.read_csv(output_dir / "comparison.csv")
            pairs = set(zip(table["model"], table["feature_set"], strict=True))
            assert ("logistic", "diff") in pairs
            assert ("logistic", "diff_only") in pairs
            assert ("logistic", "with_games_played") not in pairs
            assert ("elo", "elo") in pairs
            markdown = (output_dir / "comparison.md").read_text(encoding="utf-8")
            assert "on feature set `current`" in markdown
            assert "\u2014" not in markdown
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            experiment = client.get_experiment_by_name(PHASE2_EXPERIMENT_NAME)
            assert experiment is not None
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            assert runs
            assert all(run.data.tags.get("phase") == PHASE2_TAG for run in runs)
            assert all(run.data.tags.get("hypothesis") == HYPOTHESIS_H1 for run in runs)
            phase1 = client.get_experiment_by_name(PHASE1_EXPERIMENT_NAME)
            assert phase1 is None
            h2_experiment = client.get_experiment_by_name(PHASE2_H2_EXPERIMENT_NAME)
            assert h2_experiment is None
        finally:
            _reset_mlflow_tracking()


class TestPhase2H2Hypothesis:
    """H2 scores season defense and regressed EPA against logistic current."""

    def test_phase2_h2_report_keeps_the_same_reference_and_baselines(self) -> None:
        """h2_def and h2_full replace the diff sets. The home prior, Elo, and market stay."""
        games, team_games = _schedule(with_moneyline=False)
        report = run_phase1_baselines(
            games,
            team_games,
            config=_fast_config(n_bootstrap=8),
            lookback_games=2,
            logistic_max_iter=200,
            phase=2,
            hypothesis="h2",
        )
        table = report["comparison_table"]
        pairs = set(zip(table["model"], table["feature_set"], strict=True))
        assert pairs == {
            ("baseline-home", "none"),
            ("logistic", "current"),
            ("logistic", "h2_def"),
            ("logistic", "h2_full"),
            ("elo", "elo"),
            ("market", "market"),
        }
        assert report["reference_model"] == "logistic"
        assert report["reference_feature_set"] == "current"
        assert report["hypothesis"] == HYPOTHESIS_H2
        assert report["title"] == "Phase 2 H2 season defensive regression"
        current = next(item for item in report["results"] if item["feature_set"] == "current")
        h2_def = next(item for item in report["results"] if item["feature_set"] == "h2_def")
        h2_full = next(item for item in report["results"] if item["feature_set"] == "h2_full")
        assert current["comparison"]["log_loss_diff"] == pytest.approx(0.0)
        assert h2_def["comparison"]["reference_log_loss"] == pytest.approx(
            current["comparison"]["log_loss"]
        )
        assert h2_full["comparison"]["n"] == current["comparison"]["n"]
        assert h2_def["params"]["features"].split(",") == list(PREGAME_FEATURE_COLUMNS_H2_DEF)
        assert h2_full["params"]["features"].split(",") == list(PREGAME_FEATURE_COLUMNS_H2_FULL)
        assert h2_full["params"]["shrink_factor"] == pytest.approx(1.0 / 3.0)
        assert h2_full["params"]["blend_k"] == pytest.approx(4.0)
        assert "home_games_played" not in h2_full["params"]["features"]
        assert "diff_rest_days" not in h2_full["params"]["features"]
        decisions = {row["feature_set"]: row["decision"] for row in report["acceptance"]}
        assert decisions["current"] == "reference"
        assert decisions["h2_def"] in {"accepted", "rejected", "inconclusive"}
        assert decisions["h2_full"] in {"accepted", "rejected", "inconclusive"}
        assert "Hypothesis: `h2-season-def-regression`" in report["markdown"]
        assert "0.003" in report["markdown"]
        assert "n/(n+4)" in report["markdown"]
        assert "No model is promoted" in report["markdown"]
        assert "\u2014" not in report["markdown"]
        for result in report["results"]:
            if result["oof"] is None:
                continue
            assert set(result["oof"]["season"]).isdisjoint({2024, 2025})

    def test_cli_phase2_h2_tags_hypothesis_and_keeps_h1_available(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--hypothesis h2`` logs phase=2 and writes reports/phase2_h2."""
        import mlflow

        games, team_games = _schedule(with_moneyline=True)
        fixture_dir = tmp_path / "fixtures"
        fixture_dir.mkdir()
        games.to_parquet(fixture_dir / "fct_games.parquet")
        team_games.to_parquet(fixture_dir / "fct_team_game_epa.parquet")
        tracking_uri: str = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setattr("src.train.train.LOCAL_ARTIFACT_DIR", str(tmp_path / "mlartifacts"))
        output_dir = tmp_path / "phase2_h2"
        try:
            main(
                [
                    "--cv",
                    "--phase",
                    "2",
                    "--hypothesis",
                    "h2",
                    "--reference-model",
                    "logistic",
                    "--reference-feature-set",
                    "current",
                    "--fixture-path",
                    str(fixture_dir),
                    "--output-dir",
                    str(output_dir),
                    "--cv-start-season",
                    "2022",
                    "--cv-end-season",
                    "2023",
                    "--n-bootstrap",
                    "4",
                    "--lookback",
                    "2",
                    "--max-iter",
                    "200",
                    "--seed",
                    "7",
                ]
            )
            table = pd.read_csv(output_dir / "comparison.csv")
            pairs = set(zip(table["model"], table["feature_set"], strict=True))
            assert ("logistic", "h2_def") in pairs
            assert ("logistic", "h2_full") in pairs
            assert ("logistic", "current") in pairs
            assert ("logistic", "diff") not in pairs
            assert ("elo", "elo") in pairs
            assert ("baseline-home", "none") in pairs
            markdown = (output_dir / "comparison.md").read_text(encoding="utf-8")
            assert "Hypothesis: `h2-season-def-regression`" in markdown
            assert "on feature set `current`" in markdown
            assert "\u2014" not in markdown
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            experiment = client.get_experiment_by_name(PHASE2_H2_EXPERIMENT_NAME)
            assert experiment is not None
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            assert runs
            assert all(run.data.tags.get("phase") == PHASE2_TAG for run in runs)
            assert all(run.data.tags.get("hypothesis") == HYPOTHESIS_H2 for run in runs)
            h1_experiment = client.get_experiment_by_name(PHASE2_EXPERIMENT_NAME)
            assert h1_experiment is None
        finally:
            _reset_mlflow_tracking()
