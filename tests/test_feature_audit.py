"""Tests for the phase 0 feature input audit. Synthetic frames only."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.feature_audit import (
    FEATURE_TABLE_COLUMNS,
    PHASE0_EXPERIMENT_NAME,
    FeatureAuditConfig,
    _fold_skip_reason,
    _format_cell,
    bootstrap_binary_metrics,
    log_audit_to_mlflow,
    main,
    make_feature_audit_config,
    run_feature_audit,
    write_audit_outputs,
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


def _fast_config(**overrides: object) -> FeatureAuditConfig:
    """A small CV window so unit tests do not fit eight seasons."""
    base: dict[str, object] = {
        "cv_start_season": 2022,
        "cv_end_season": 2023,
        "n_bootstrap": 25,
        "permutation_repeats": 4,
        "random_state": 42,
        "bootstrap_seed": 42,
    }
    base.update(overrides)
    return make_feature_audit_config(**base)


def _labeled_frame(rows_per_season: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    """Five seasons. ``games_played`` walks forward. ``signal`` copies the label."""
    seasons = [2021, 2022, 2023, 2024, 2025]
    bases = {2021: 10, 2022: 40, 2023: 80, 2024: 200, 2025: 400}
    records: list[dict[str, object]] = []
    labels: list[int] = []
    for season in seasons:
        for index in range(rows_per_season):
            label = 0 if index % 2 else 1
            records.append(
                {
                    "game_id": f"{season}-{index}",
                    "gameday": pd.Timestamp(year=season, month=9, day=1) + pd.Timedelta(days=index),
                    "season": season,
                    "signal": float(label),
                    "signal_copy": float(label),
                    "noise": float((index % 3) - 1),
                    "week1_epa": np.nan if index == 0 else 7.0,
                    "games_played": float(bases[season] + index),
                }
            )
            labels.append(label)
    return pd.DataFrame(records), pd.Series(labels)


def _row(table: pd.DataFrame, feature: str) -> pd.Series:
    """Return one feature's audit row."""
    matched = table.loc[table["feature"] == feature]
    assert len(matched) == 1
    return matched.iloc[0]


class TestConfig:
    """Rejected settings fail before any model fit."""

    def test_rejects_inverted_cv_window(self) -> None:
        """The CV start season cannot be after the end season."""
        with pytest.raises(ValueError, match="cv_start_season"):
            make_feature_audit_config(cv_start_season=2024, cv_end_season=2020)

    def test_rejects_negative_bootstrap(self) -> None:
        """Bootstrap count cannot be negative."""
        with pytest.raises(ValueError, match="n_bootstrap"):
            make_feature_audit_config(n_bootstrap=-1)

    def test_rejects_zero_permutation_repeats(self) -> None:
        """Permutation needs at least one repeat."""
        with pytest.raises(ValueError, match="permutation_repeats"):
            make_feature_audit_config(permutation_repeats=0)

    def test_rejects_threshold_outside_unit_interval(self) -> None:
        """The correlation threshold is a positive number up to 1."""
        with pytest.raises(ValueError, match="correlation_threshold"):
            make_feature_audit_config(correlation_threshold=0)
        with pytest.raises(ValueError, match="correlation_threshold"):
            make_feature_audit_config(correlation_threshold=1.1)

    def test_rejects_unknown_field(self) -> None:
        """Unknown settings are a caller error."""
        with pytest.raises(TypeError):
            make_feature_audit_config(not_a_field=1)

    def test_threshold_of_one_is_valid(self) -> None:
        """A threshold of 1 keeps only perfect rank correlation."""
        config = make_feature_audit_config(correlation_threshold=1)
        assert config.correlation_threshold == 1


class TestInputValidation:
    """Frames that cannot be audited raise before fitting."""

    def test_requires_season_rows_features_and_a_binary_target(self) -> None:
        """Each invalid frame names the problem."""
        features, target = _labeled_frame(rows_per_season=4)
        config = _fast_config(
            n_bootstrap=0, permutation_repeats=1, cv_start_season=1999, cv_end_season=1999
        )

        with pytest.raises(ValueError, match="season column"):
            run_feature_audit(features.drop(columns=["season"]), target, config)
        with pytest.raises(ValueError, match="at least one row"):
            run_feature_audit(features.iloc[0:0], target.iloc[0:0], config)
        with pytest.raises(ValueError, match="same number of rows"):
            run_feature_audit(features, target.iloc[:-1], config)
        with pytest.raises(ValueError, match="no nulls"):
            broken = target.copy()
            broken.iloc[0] = np.nan
            run_feature_audit(features, broken, config)
        with pytest.raises(ValueError, match="only classes"):
            broken = target.copy()
            broken.iloc[0] = 2
            run_feature_audit(features, broken, config)
        with pytest.raises(ValueError, match="both classes"):
            run_feature_audit(features, pd.Series(np.ones(len(features), dtype=int)), config)
        identifiers = features[["season"]].copy()
        with pytest.raises(ValueError, match="at least one model column"):
            run_feature_audit(identifiers, target, config)
        text = features.copy()
        text["signal"] = "high"
        with pytest.raises(ValueError, match="must be numeric"):
            run_feature_audit(text, target, config)


class TestBootstrapMetrics:
    """Interval edge cases that a full audit frame does not produce."""

    def test_empty_sample_has_no_estimates(self) -> None:
        """No out-of-fold rows means every estimate is missing."""
        result = bootstrap_binary_metrics(np.array([]), np.array([]), n_bootstrap=5, seed=0)
        assert result["n_oof"] == 0
        assert result["log_loss"] is None
        assert result["auc"] is None

    def test_zero_resamples_keep_the_point_and_drop_the_interval(self) -> None:
        """n_bootstrap of 0 still reports the point estimates."""
        result = bootstrap_binary_metrics(
            np.array([0, 1, 0, 1]),
            np.array([0.2, 0.8, 0.3, 0.7]),
            n_bootstrap=0,
            seed=0,
        )
        assert result["log_loss"] is not None
        assert result["auc"] is not None
        assert result["log_loss_ci_low"] is None
        assert result["auc_ci_low"] is None

    def test_single_row_has_no_interval(self) -> None:
        """One row cannot be resampled into an interval."""
        result = bootstrap_binary_metrics(
            np.array([1]),
            np.array([0.6]),
            n_bootstrap=5,
            seed=0,
        )
        assert result["n_oof"] == 1
        assert result["auc"] is None
        assert result["log_loss_ci_low"] is None

    def test_single_class_sample_has_log_loss_and_no_auc(self) -> None:
        """AUC is undefined when every label is the same class."""
        result = bootstrap_binary_metrics(
            np.array([0, 0, 0, 0]),
            np.array([0.2, 0.3, 0.4, 0.1]),
            n_bootstrap=5,
            seed=0,
        )
        assert result["log_loss"] is not None
        assert result["auc"] is None
        assert result["auc_ci_low"] is None
        assert result["n_auc_resamples"] == 0

    def test_mixed_resamples_skip_single_class_draws_for_auc(self) -> None:
        """A two-row sample produces some one-class resamples."""
        result = bootstrap_binary_metrics(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            n_bootstrap=20,
            seed=0,
        )
        assert result["n_auc_resamples"] < 20
        assert result["auc_ci_low"] <= result["auc_ci_high"]


class TestFoldSkipReason:
    """Each skip reason is named from the fold's class counts."""

    def test_names_each_reason(self) -> None:
        """Empty folds and one-class folds are not fit."""
        assert _fold_skip_reason(pd.Series(dtype=int), pd.Series([0, 1])) == "no_train_rows"
        assert _fold_skip_reason(pd.Series([0, 1]), pd.Series(dtype=int)) == "no_eval_rows"
        assert _fold_skip_reason(pd.Series([0, 0]), pd.Series([0, 1])) == "single_class_train"
        assert _fold_skip_reason(pd.Series([0, 1]), pd.Series([1, 1])) == "single_class_eval"
        assert _fold_skip_reason(pd.Series([0, 1]), pd.Series([0, 1])) is None


class TestAuditReport:
    """Distribution flags, univariate signal, redundancy, and importance."""

    def test_profiles_missingness_range_signal_and_redundancy(self) -> None:
        """The clock, the copy, and the label feature show up in the table."""
        features, target = _labeled_frame()
        shifted = features.copy()
        shifted.index = range(100, 100 + len(features))
        target.index = range(len(target))
        report = run_feature_audit(shifted, target, _fast_config())
        table = report["feature_table"]

        assert list(table.columns) == list(FEATURE_TABLE_COLUMNS)
        assert "season" not in set(table["feature"])
        assert "\u2014" not in report["markdown"]
        for heading in (
            "## Missingness",
            "## Distribution",
            "## Univariate signal",
            "## Correlation and redundancy",
            "## Permutation importance",
        ):
            assert heading in report["markdown"]
        assert "not scored" in report["markdown"]

        rest = _row(table, "week1_epa")
        assert rest["missing_count"] == 5
        assert rest["missing_rate"] == pytest.approx(0.05)
        assert rest["train_missing_rate"] == pytest.approx(0.05)
        assert bool(rest["out_of_range"]) is False
        assert bool(rest["time_counter"]) is False

        clock = _row(table, "games_played")
        assert clock["val_out_of_range_count"] == 20
        assert clock["test_out_of_range_count"] == 20
        assert bool(clock["out_of_range"]) is True
        assert bool(clock["time_counter"]) is True
        assert clock["val_min"] > clock["train_max"]

        signal = _row(table, "signal")
        copy = _row(table, "signal_copy")
        assert signal["cv_auc"] > 0.95
        assert signal["cv_log_loss"] < _row(table, "noise")["cv_log_loss"]
        assert signal["cv_log_loss_ci_low"] <= signal["cv_log_loss_ci_high"]
        assert signal["cv_auc_ci_low"] <= signal["cv_auc"] <= signal["cv_auc_ci_high"]
        assert signal["cv_folds_used"] == 2
        assert bool(signal["redundant"]) is True
        assert signal["most_correlated_with"] == "signal_copy"
        assert copy["max_abs_spearman"] == pytest.approx(1.0)
        assert report["redundant_pairs"]
        assert report["permutation_status"] == "ok"

        again = run_feature_audit(features, target, _fast_config())
        assert again["feature_table"]["cv_auc"].tolist() == table["cv_auc"].tolist()

    def test_signal_beats_noise_on_permutation_importance(self) -> None:
        """Without a copy, the label feature outranks a column that ignores it."""
        features, target = _labeled_frame()
        kept = features.drop(columns=["signal_copy"])
        report = run_feature_audit(kept, target, _fast_config(permutation_repeats=6))
        table = report["feature_table"]
        signal = _row(table, "signal")["permutation_importance_mean"]
        noise = _row(table, "noise")["permutation_importance_mean"]
        assert signal > noise
        assert signal > 0

    def test_decreasing_series_outside_the_train_range_is_a_time_counter(self) -> None:
        """A strictly falling median that leaves the train range is flagged."""
        records: list[dict[str, object]] = []
        labels: list[int] = []
        levels = {2021: 50.0, 2022: 40.0, 2023: 30.0, 2024: 5.0, 2025: -5.0}
        for season, level in levels.items():
            for index in range(8):
                label = index % 2
                records.append(
                    {
                        "season": season,
                        "falling": level,
                        "signal": float(label),
                    }
                )
                labels.append(label)
        report = run_feature_audit(
            pd.DataFrame(records),
            pd.Series(labels),
            _fast_config(n_bootstrap=0, permutation_repeats=1),
        )
        falling = _row(report["feature_table"], "falling")
        assert bool(falling["time_counter"]) is True
        assert falling["test_max"] < falling["train_min"]

    def test_two_seasons_outside_the_range_are_not_a_time_counter(self) -> None:
        """Out of range without three monotone seasons is only an out-of-range flag."""
        features = pd.DataFrame(
            {
                "season": [2023] * 6 + [2024] * 6,
                "jump": [1.0] * 6 + [50.0] * 6,
                "signal": [0.0, 1.0] * 6,
            }
        )
        target = pd.Series([0, 1] * 6)
        report = run_feature_audit(
            features,
            target,
            _fast_config(
                n_bootstrap=0, permutation_repeats=1, cv_start_season=2024, cv_end_season=2024
            ),
        )
        jump = _row(report["feature_table"], "jump")
        assert bool(jump["out_of_range"]) is True
        assert bool(jump["time_counter"]) is False

    def test_all_null_train_values_do_not_invent_a_range(self) -> None:
        """No training observations means the out-of-range count is undefined."""
        seasons = [2021, 2022, 2023, 2024]
        records: list[dict[str, object]] = []
        labels: list[int] = []
        for season in seasons:
            for index in range(6):
                label = index % 2
                records.append(
                    {
                        "season": season,
                        "blank_epa": np.nan if season <= 2023 else 1.0,
                        "signal": float(label),
                    }
                )
                labels.append(label)
        report = run_feature_audit(
            pd.DataFrame(records),
            pd.Series(labels),
            _fast_config(n_bootstrap=0, permutation_repeats=1),
        )
        blank = _row(report["feature_table"], "blank_epa")
        assert pd.isna(blank["train_min"])
        assert pd.isna(blank["val_out_of_range_count"])
        assert bool(blank["out_of_range"]) is False
        assert bool(blank["time_counter"]) is False
        assert pd.isna(blank["max_abs_spearman"])

    def test_constant_and_single_feature_have_no_redundant_pair(self) -> None:
        """A constant column and a one-column matrix do not invent a partner."""
        single = pd.DataFrame(
            {
                "season": [2023] * 8 + [2024] * 8,
                "signal": [0.0, 1.0] * 8,
            }
        )
        target = pd.Series([0, 1] * 8)
        config = _fast_config(
            n_bootstrap=0, permutation_repeats=1, cv_start_season=2024, cv_end_season=2024
        )
        single_report = run_feature_audit(single, target, config)
        signal = _row(single_report["feature_table"], "signal")
        assert signal["most_correlated_with"] is None
        assert single_report["redundant_pairs"] == []
        assert "No pair reached" in single_report["markdown"]

        paired = single.copy()
        paired["flat"] = 1.0
        paired_report = run_feature_audit(paired, target, config)
        flat = _row(paired_report["feature_table"], "flat")
        assert flat["most_correlated_with"] is None
        assert paired_report["redundant_pairs"] == []

    def test_skipped_folds_cover_each_reason_and_2025_is_not_scored(self) -> None:
        """One-class 2025 rows are profiled and do not have to be class-balanced."""
        records: list[dict[str, object]] = []
        labels: list[int] = []
        plan = {
            2018: (0, 0),
            2020: (0, 1),
            2021: (1, 1),
            2024: (0, 1),
            2025: (1, 1),
        }
        for season, (first, second) in plan.items():
            for index, label in enumerate((first, second, first, second)):
                records.append(
                    {
                        "season": season,
                        "signal": float(label),
                        "level": float(season),
                    }
                )
                labels.append(label)
        report = run_feature_audit(
            pd.DataFrame(records),
            pd.Series(labels),
            _fast_config(
                cv_start_season=2018,
                cv_end_season=2021,
                n_bootstrap=0,
                permutation_repeats=1,
            ),
        )
        reasons = {item["reason"] for item in report["skipped_folds"]}
        assert reasons == {
            "no_train_rows",
            "no_eval_rows",
            "single_class_train",
            "single_class_eval",
        }
        assert "single_class_eval" in report["markdown"]
        level = _row(report["feature_table"], "level")
        assert level["test_min"] == pytest.approx(2025.0)
        assert "test_log_loss" not in report["feature_table"].columns
        assert pd.isna(level["cv_auc"])

    def test_empty_test_slice_leaves_test_distribution_blank(self) -> None:
        """A frame with no 2025 rows still audits train and 2024."""
        features, target = _labeled_frame(rows_per_season=8)
        kept = features.loc[features["season"] != 2025].copy()
        labels = target.loc[kept.index]
        report = run_feature_audit(kept, labels, _fast_config(n_bootstrap=0, permutation_repeats=1))
        signal = _row(report["feature_table"], "signal")
        assert pd.isna(signal["test_min"])
        assert pd.isna(signal["test_missing_rate"])
        assert "n/a" in report["markdown"]

    def test_default_config_is_used_when_omitted(self) -> None:
        """Omitting config uses the training split and the 2016-2023 window."""
        features = pd.DataFrame(
            {
                "season": [2022] * 6 + [2023] * 6 + [2024] * 6,
                "signal": [0.0, 1.0] * 9,
            }
        )
        target = pd.Series([0, 1] * 9)
        report = run_feature_audit(features, target, None)
        assert report["config"]["cv_start_season"] == 2016
        assert report["config"]["cv_end_season"] == 2023
        assert report["permutation_status"] == "ok"
        assert _row(report["feature_table"], "signal")["cv_folds_used"] == 1


class TestFitFailures:
    """A null in a column the logistic pipeline does not impute skips that fit."""

    def test_passthrough_nulls_are_reported_and_do_not_abort(self) -> None:
        """Rest-like nulls fail the current pipeline. The audit records that."""
        features = pd.DataFrame(
            {
                "season": [2023] * 8 + [2024] * 8,
                "signal": [0.0, 1.0] * 8,
                "rest_days": [np.nan, 7.0] * 8,
            }
        )
        target = pd.Series([0, 1] * 8)
        report = run_feature_audit(
            features,
            target,
            _fast_config(
                n_bootstrap=0,
                permutation_repeats=1,
                cv_start_season=2024,
                cv_end_season=2024,
            ),
        )
        assert report["permutation_reason"] == "fit_failed"
        assert "Skipped (fit_failed)" in report["markdown"]
        rest_reasons = {
            item["reason"] for item in report["skipped_folds"] if item["feature"] == "rest_days"
        }
        assert rest_reasons == {"fit_failed"}
        assert _row(report["feature_table"], "signal")["cv_folds_used"] == 1


class TestPermutationSkips:
    """The logistic model is not scored when train or 2024 cannot support it."""

    def _frame(
        self, seasons: list[int], label_for: dict[int, int]
    ) -> tuple[pd.DataFrame, pd.Series]:
        records: list[dict[str, object]] = []
        labels: list[int] = []
        for season in seasons:
            for index in range(6):
                label = label_for.get(season, index % 2)
                records.append({"season": season, "signal": float(index)})
                labels.append(label)
        return pd.DataFrame(records), pd.Series(labels)

    def test_skips_when_training_rows_are_missing(self) -> None:
        """A train window before the data has nothing to fit."""
        features, target = self._frame([2023, 2024], {})
        report = run_feature_audit(
            features,
            target,
            _fast_config(
                train_end_season=2010,
                n_bootstrap=0,
                permutation_repeats=1,
                cv_start_season=1999,
                cv_end_season=1999,
            ),
        )
        assert report["permutation_status"] == "skipped"
        assert report["permutation_reason"] == "no_training_rows"
        assert "Skipped (no_training_rows)" in report["markdown"]

    def test_skips_when_training_labels_are_one_class(self) -> None:
        """A one-class training slice is not fit."""
        features, target = self._frame([2023, 2024], {2023: 0})
        report = run_feature_audit(
            features,
            target,
            _fast_config(
                n_bootstrap=0, permutation_repeats=1, cv_start_season=1999, cv_end_season=1999
            ),
        )
        assert report["permutation_reason"] == "single_class_train"

    def test_skips_when_validation_rows_are_missing(self) -> None:
        """An empty 2024 slice skips permutation and still returns the table."""
        features, target = self._frame([2022, 2023], {})
        report = run_feature_audit(
            features,
            target,
            _fast_config(
                val_season=2099,
                n_bootstrap=0,
                permutation_repeats=1,
                cv_start_season=2023,
                cv_end_season=2023,
            ),
        )
        assert report["permutation_reason"] == "no_validation_rows"
        assert pd.isna(_row(report["feature_table"], "signal")["permutation_importance_mean"])

    def test_skips_when_validation_labels_are_one_class(self) -> None:
        """A one-class 2024 slice is not used to rank features."""
        features, target = self._frame([2023, 2024], {2024: 1})
        report = run_feature_audit(
            features,
            target,
            _fast_config(
                n_bootstrap=0, permutation_repeats=1, cv_start_season=1999, cv_end_season=1999
            ),
        )
        assert report["permutation_reason"] == "single_class_validation"


class TestOutputsAndCli:
    """CSV, markdown, MLflow, and the module entrypoint."""

    def test_writes_markdown_and_csv(self, tmp_path: Path) -> None:
        """Both artifacts exist and the CSV has the feature table columns."""
        features, target = _labeled_frame(rows_per_season=8)
        report = run_feature_audit(
            features, target, _fast_config(n_bootstrap=0, permutation_repeats=1)
        )
        paths = write_audit_outputs(report, tmp_path / "nested" / "out")
        write_audit_outputs(report, tmp_path / "nested" / "out")
        markdown = Path(paths["markdown_path"]).read_text(encoding="utf-8")
        loaded = pd.read_csv(paths["csv_path"])
        assert markdown.startswith("# Phase 0 feature input audit")
        assert list(loaded.columns) == list(FEATURE_TABLE_COLUMNS)
        assert "\u2014" not in markdown

    def test_format_cell_renders_each_kind(self) -> None:
        """Missing, bool, int, float, and text each have a stable rendering."""
        assert _format_cell(None) == "n/a"
        assert _format_cell(np.nan) == "n/a"
        assert _format_cell("epa") == "epa"
        assert _format_cell(True) == "yes"
        assert _format_cell(np.bool_(False)) == "no"
        assert _format_cell(3) == "3"
        assert _format_cell(np.int64(4)) == "4"
        assert _format_cell(np.float64(1.5)) == "1.5000"

    def test_logs_a_phase_tag_and_the_two_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The MLflow run is tagged phase=0-audit and stores the report files."""
        import mlflow

        features, target = _labeled_frame(rows_per_season=8)
        report = run_feature_audit(
            features, target, _fast_config(n_bootstrap=0, permutation_repeats=1)
        )
        paths = write_audit_outputs(report, tmp_path / "report")
        tracking_uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
        monkeypatch.setattr("src.train.train.LOCAL_ARTIFACT_DIR", str(tmp_path / "mlartifacts"))
        try:
            logged = log_audit_to_mlflow(report, paths, tracking_uri=tracking_uri)
            without_extra = log_audit_to_mlflow(report, paths, tracking_uri=tracking_uri)
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            run = client.get_run(logged["run_id"])
            assert logged["experiment_name"] == PHASE0_EXPERIMENT_NAME
            assert run.data.tags["phase"] == "0-audit"
            assert "git_commit" in run.data.tags
            assert "val_log_loss" not in run.data.metrics
            assert "test_log_loss" not in run.data.metrics
            assert "n_features_time_counter" in run.data.metrics
            artifact_names = {item.path for item in client.list_artifacts(logged["run_id"])}
            assert artifact_names == {"feature_audit.md", "feature_audit.csv"}
            extra_run = client.get_run(without_extra["run_id"])
            assert "lookback_games" not in extra_run.data.params
        finally:
            _reset_mlflow_tracking()

    def test_main_audits_a_fixture_and_logs_lookback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The CLI builds current features from fixtures and does not score 2025."""
        games, epa = _fixture_frames()
        fixture_dir = tmp_path / "fixtures"
        fixture_dir.mkdir()
        games.to_parquet(fixture_dir / "fct_games.parquet")
        epa.to_parquet(fixture_dir / "fct_team_game_epa.parquet")
        tracking_uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setattr("src.train.train.LOCAL_ARTIFACT_DIR", str(tmp_path / "mlartifacts"))
        try:
            result = main(
                [
                    "--fixture-path",
                    str(fixture_dir),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--lookback",
                    "2",
                    "--cv-start-season",
                    "2024",
                    "--cv-end-season",
                    "2024",
                    "--n-bootstrap",
                    "2",
                    "--permutation-repeats",
                    "2",
                    "--train-end-season",
                    "2023",
                    "--val-season",
                    "2024",
                    "--test-season",
                    "2025",
                ]
            )
            import mlflow

            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            run = client.get_run(result["run"]["run_id"])
            assert run.data.tags["phase"] == "0-audit"
            assert run.data.params["lookback_games"] == "2"
            assert Path(result["paths"]["markdown_path"]).is_file()
            assert Path(result["paths"]["csv_path"]).is_file()
            assert "test_log_loss" not in result["report"]["feature_table"].columns
        finally:
            _reset_mlflow_tracking()

    def test_main_without_a_fixture_path_uses_the_loader(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty fixture path calls the loader with None and does not hit Databricks."""
        games, epa = _fixture_frames()

        def _load(fixture_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
            assert fixture_path is None
            return games, epa

        monkeypatch.setattr("src.analysis.feature_audit.load_data", _load)
        tracking_uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        monkeypatch.setattr("src.train.train.LOCAL_ARTIFACT_DIR", str(tmp_path / "mlartifacts"))
        try:
            result = main(
                [
                    "--fixture-path",
                    "",
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--cv-start-season",
                    "2024",
                    "--cv-end-season",
                    "2024",
                    "--n-bootstrap",
                    "1",
                    "--permutation-repeats",
                    "1",
                ]
            )
            assert result["run"]["run_id"]
        finally:
            _reset_mlflow_tracking()


def _fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small games and team-game EPA frames with both classes in 2023 and 2024."""
    games = pd.DataFrame(
        {
            "game_id": [f"G{index}" for index in range(8)],
            "season": [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
            "gameday": pd.to_datetime(
                [
                    "2023-09-10",
                    "2023-09-17",
                    "2023-09-24",
                    "2023-10-01",
                    "2024-09-08",
                    "2024-09-15",
                    "2024-09-22",
                    "2024-09-29",
                ]
            ),
            "home_team_franchise": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "away_team_franchise": ["B", "A", "B", "A", "B", "A", "B", "A"],
            "home_won": [1, 0, 1, 0, 1, 0, 1, 0],
            "home_rest": [7, 7, 7, 7, 7, 7, 7, 7],
            "away_rest": [7, 7, 7, 7, 7, 7, 7, 7],
        }
    )
    epa_rows: list[dict[str, object]] = []
    for _, game in games.iterrows():
        for team, offense in (
            (game["home_team_franchise"], 0.2),
            (game["away_team_franchise"], -0.1),
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
