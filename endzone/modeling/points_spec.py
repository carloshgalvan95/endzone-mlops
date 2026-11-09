"""Load the pre-registered Sprint 2 margin and total hypothesis.

``configs/experiments/s2.yaml`` is the record. ADR-007 fixes the thresholds.
The classification catalog skips this file. A drift in the file fails here,
before a fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from endzone.features.pregame import FEATURE_SET_H2_FULL, FEATURE_SET_NAMES

SCORE_TASK: str = "score"
_TARGET_NAMES: tuple[str, ...] = ("margin", "total")
_BASELINES: dict[str, str] = {"margin": "elo_spread", "total": "league_mean"}
_SPEC_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "task",
        "feature_set",
        "model",
        "ridge_alpha",
        "random_state",
        "cv_start_season",
        "cv_end_season",
        "n_bootstrap",
        "bootstrap_seed",
        "min_seasons_improved",
        "season_count",
        "elo_points_per_spread",
        "interval_coverage",
        "winprob_log_loss_benchmark",
        "market_margin_multiplier",
        "experiment_name",
        "phase_tag",
        "hypothesis_tag",
        "output_dir",
        "title",
        "note",
        "targets",
    }
)


@dataclass(frozen=True)
class ScoreTargetSpec:
    """One regression target and the baseline it has to beat."""

    name: str
    baseline: str
    registered_model_name: str


@dataclass(frozen=True)
class ScoreExperimentSpec:
    """The Sprint 2 hypothesis. Fields match ``configs/experiments/s2.yaml``."""

    experiment_id: str
    task: str
    feature_set: str
    model: str
    ridge_alpha: float
    random_state: int
    cv_start_season: int
    cv_end_season: int
    n_bootstrap: int
    bootstrap_seed: int
    min_seasons_improved: int
    season_count: int
    elo_points_per_spread: float
    interval_coverage: float
    winprob_log_loss_benchmark: float
    market_margin_multiplier: float
    experiment_name: str
    phase_tag: str
    hypothesis_tag: str
    output_dir: str
    title: str
    note: str
    targets: tuple[ScoreTargetSpec, ...]

    def target(self, name: str) -> ScoreTargetSpec:
        """Return one target by name.

        Raises:
            KeyError: That target is not in the file.
        """
        for item in self.targets:
            if item.name == name:
                return item
        known: str = ", ".join(item.name for item in self.targets)
        raise KeyError(f"Unknown score target {name!r}. Known: {known}.")


def score_experiments_directory() -> Path:
    """Return ``configs/experiments`` at the repository root."""
    return Path(__file__).resolve().parents[2] / "configs" / "experiments"


def _require_mapping(payload: object, path: Path) -> dict[str, Any]:
    """Return a mapping with the score-experiment keys."""
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    unknown: list[str] = sorted(set(payload) - _SPEC_KEYS)
    if unknown:
        names: str = ", ".join(unknown)
        raise ValueError(f"{path.name} has unknown keys: {names}")
    missing: list[str] = sorted(_SPEC_KEYS - set(payload))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{path.name} is missing keys: {names}")
    return payload


def _require_str(payload: dict[str, Any], key: str, path: Path) -> str:
    """Return a non-empty string field."""
    value: object = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path.name} {key} must be a non-empty string")
    return value


def _require_int(payload: dict[str, Any], key: str, path: Path) -> int:
    """Return an int. Booleans are rejected."""
    value: object = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path.name} {key} must be an integer")
    return value


def _require_float(payload: dict[str, Any], key: str, path: Path) -> float:
    """Return a float. Integers are accepted. Booleans are not."""
    value: object = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path.name} {key} must be a number")
    return float(value)


def _parse_targets(payload: dict[str, Any], path: Path) -> tuple[ScoreTargetSpec, ...]:
    """Require margin then total, each with its pre-registered baseline."""
    raw: object = payload["targets"]
    if not isinstance(raw, list) or len(raw) != len(_TARGET_NAMES):
        raise ValueError(f"{path.name} targets must list margin and total")
    specs: list[ScoreTargetSpec] = []
    for expected, item in zip(_TARGET_NAMES, raw, strict=True):
        if not isinstance(item, dict):
            raise ValueError(f"{path.name} each target must be a mapping")
        extra: list[str] = sorted(set(item) - {"name", "baseline", "registered_model_name"})
        if extra:
            names: str = ", ".join(extra)
            raise ValueError(f"{path.name} target has unknown keys: {names}")
        name: object = item.get("name")
        baseline: object = item.get("baseline")
        model_name: object = item.get("registered_model_name")
        if name != expected or not isinstance(baseline, str) or not isinstance(model_name, str):
            raise ValueError(f"{path.name} targets must be margin then total")
        if baseline != _BASELINES[expected] or not model_name.strip():
            raise ValueError(f"{path.name} {expected} baseline must be {_BASELINES[expected]!r}")
        specs.append(ScoreTargetSpec(expected, baseline, model_name))
    return tuple(specs)


def _parse_score_experiment(payload: dict[str, Any], path: Path) -> ScoreExperimentSpec:
    """Validate one score file against the pre-registered shape."""
    if payload["task"] != SCORE_TASK:
        raise ValueError(f"{path.name} task must be {SCORE_TASK!r}")
    experiment_id: str = _require_str(payload, "id", path)
    if experiment_id != path.stem:
        raise ValueError(f"{path.name} id must match the file name, got {experiment_id!r}")
    feature_set: str = _require_str(payload, "feature_set", path)
    if feature_set != FEATURE_SET_H2_FULL or feature_set not in FEATURE_SET_NAMES:
        raise ValueError(f"{path.name} feature_set must be {FEATURE_SET_H2_FULL!r}")
    if _require_str(payload, "model", path) != "ridge":
        raise ValueError(f"{path.name} model must be 'ridge'")
    start: int = _require_int(payload, "cv_start_season", path)
    end: int = _require_int(payload, "cv_end_season", path)
    season_count: int = _require_int(payload, "season_count", path)
    if end - start + 1 != season_count:
        raise ValueError(f"{path.name} season_count must equal the CV window length")
    improved: int = _require_int(payload, "min_seasons_improved", path)
    if not 1 <= improved <= season_count:
        raise ValueError(f"{path.name} min_seasons_improved must be inside the season count")
    coverage: float = _require_float(payload, "interval_coverage", path)
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"{path.name} interval_coverage must be between 0 and 1")
    alpha: float = _require_float(payload, "ridge_alpha", path)
    if alpha <= 0.0:
        raise ValueError(f"{path.name} ridge_alpha must be positive")
    points_per_spread: float = _require_float(payload, "elo_points_per_spread", path)
    if points_per_spread <= 0.0:
        raise ValueError(f"{path.name} elo_points_per_spread must be positive")
    return ScoreExperimentSpec(
        experiment_id=experiment_id,
        task=SCORE_TASK,
        feature_set=feature_set,
        model="ridge",
        ridge_alpha=alpha,
        random_state=_require_int(payload, "random_state", path),
        cv_start_season=start,
        cv_end_season=end,
        n_bootstrap=_require_int(payload, "n_bootstrap", path),
        bootstrap_seed=_require_int(payload, "bootstrap_seed", path),
        min_seasons_improved=improved,
        season_count=season_count,
        elo_points_per_spread=points_per_spread,
        interval_coverage=coverage,
        winprob_log_loss_benchmark=_require_float(payload, "winprob_log_loss_benchmark", path),
        market_margin_multiplier=_require_float(payload, "market_margin_multiplier", path),
        experiment_name=_require_str(payload, "experiment_name", path),
        phase_tag=_require_str(payload, "phase_tag", path),
        hypothesis_tag=_require_str(payload, "hypothesis_tag", path),
        output_dir=_require_str(payload, "output_dir", path),
        title=_require_str(payload, "title", path),
        note=_require_str(payload, "note", path),
        targets=_parse_targets(payload, path),
    )


def load_score_spec(directory: Path | None = None) -> ScoreExperimentSpec:
    """Load the single ``task: score`` file.

    Args:
        directory: Folder of experiment YAML files. None uses the repo folder.

    Raises:
        ValueError: The folder does not contain exactly one score file, or
            the file breaks the pre-registered shape.
    """
    folder: Path = score_experiments_directory() if directory is None else directory
    matches: list[ScoreExperimentSpec] = []
    for path in sorted(folder.glob("*.yaml")):
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("task") != SCORE_TASK:
            continue
        matches.append(_parse_score_experiment(_require_mapping(payload, path), path))
    if len(matches) != 1:
        raise ValueError(
            f"{folder} must contain exactly one task: score file, found {len(matches)}"
        )
    return matches[0]
