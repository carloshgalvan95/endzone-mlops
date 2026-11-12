"""Load one YAML file per hypothesis and resolve CLI names to that record.

Adding a hypothesis is a new file in ``configs/experiments``. The home prior,
Elo, and the market are scored for every hypothesis and are not listed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from endzone.features.pregame import FEATURE_SET_NAMES

FINAL_VALIDATION_SEASON: int = 2024
HELD_OUT_TEST_SEASON: int = 2025
REPORT_PROFILES: frozenset[str] = frozenset(
    {
        "none",
        "season_regression",
        "pass_rush",
        "ridge",
        "clean_epa",
        "starting_qb",
        "h6",
        "pkg_core",
        "pkg_core_inj",
        "ensembles",
        "ensembles_r2",
    }
)
# These profiles label the market ``benchmark`` and write calibration plus
# the descriptive market-information regression. Other profiles do not.
MARKET_BENCHMARK_PROFILES: frozenset[str] = frozenset(
    {"h6", "pkg_core", "pkg_core_inj", "ensembles", "ensembles_r2"}
)
CV_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "logistic",
        "logistic_l2",
        "gbm_pkg_core_inj",
        "ens_mean",
        "ens_mean_elo",
        "stack_elo",
    }
)
MODEL_DEFAULT_FEATURE_SET: dict[str, str] = {
    "baseline-home": "none",
    "logistic": "current",
    "logistic_l2": "h2_full",
    "elo": "elo",
    "market": "market",
}
_EXPERIMENT_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "phase",
        "default",
        "aliases",
        "tag",
        "short_name",
        "experiment_name",
        "phase_tag",
        "output_dir",
        "title",
        "note",
        "reference_model",
        "reference_feature_set",
        "logistic_feature_sets",
        "model_specs",
        "log_shrinkage",
        "log_qb",
        "qb_subset",
        "qb_feature_set",
        "report_profile",
    }
)


@dataclass(frozen=True)
class CvModelSpec:
    """One model and feature set scored in a rolling-origin comparison.

    ``fixed_c`` overrides the command's C for that row. None keeps the
    command value. It is not the inner-fold search used by ``logistic_l2``.
    """

    model: str
    feature_set: str
    fixed_c: float | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    """One phase or hypothesis declared in ``configs/experiments``."""

    experiment_id: str
    phase: int
    default: bool
    aliases: tuple[str, ...]
    tag: str | None
    short_name: str
    experiment_name: str
    phase_tag: str
    output_dir: str
    title: str
    note: str
    reference_model: str
    reference_feature_set: str
    logistic_feature_sets: tuple[str, ...]
    model_specs: tuple[CvModelSpec, ...]
    log_shrinkage: bool
    log_qb: bool
    qb_subset: bool
    qb_feature_set: str | None
    report_profile: str


@dataclass(frozen=True)
class ExperimentCatalog:
    """Every experiment file, indexed by id, tag, and alias."""

    experiments: tuple[ExperimentSpec, ...]
    by_id_map: dict[str, ExperimentSpec]
    by_alias: dict[str, ExperimentSpec]
    by_tag_map: dict[str, ExperimentSpec]

    def by_id(self, experiment_id: str) -> ExperimentSpec:
        """Return the experiment with this file id.

        Raises:
            KeyError: No file declared that id.
        """
        return self.by_id_map[experiment_id]

    def by_tag(self, tag: str) -> ExperimentSpec | None:
        """Return the phase 2 experiment with this stored tag, if any."""
        return self.by_tag_map.get(tag)

    def phase1(self) -> ExperimentSpec:
        """Return the only phase 1 experiment."""
        matches: list[ExperimentSpec] = [item for item in self.experiments if item.phase == 1]
        if len(matches) != 1:
            raise ValueError("configs/experiments must contain exactly one phase 1 file")
        return matches[0]

    def default_phase2(self) -> ExperimentSpec:
        """Return the phase 2 experiment used when ``--hypothesis`` is omitted."""
        matches: list[ExperimentSpec] = [
            item for item in self.experiments if item.phase == 2 and item.default
        ]
        if len(matches) != 1:
            raise ValueError("configs/experiments must mark exactly one phase 2 default")
        return matches[0]

    def alias_names(self) -> tuple[str, ...]:
        """CLI hypothesis choices, sorted."""
        return tuple(sorted(self.by_alias))


def experiments_directory() -> Path:
    """Return ``configs/experiments`` at the repository root."""
    return Path(__file__).resolve().parents[2] / "configs" / "experiments"


def _require_mapping(payload: object, path: Path) -> dict[str, Any]:
    """Return a mapping, or raise when the file is not one."""
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    unknown: list[str] = sorted(set(payload) - _EXPERIMENT_KEYS)
    if unknown:
        names: str = ", ".join(unknown)
        raise ValueError(f"{path.name} has unknown keys: {names}")
    missing: list[str] = sorted(_EXPERIMENT_KEYS - set(payload))
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


def _require_bool(payload: dict[str, Any], key: str, path: Path) -> bool:
    """Return a boolean field. YAML ``yes`` and integers are rejected."""
    value: object = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{path.name} {key} must be true or false")
    return value


def _require_strings(payload: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    """Return a list of non-empty strings. An empty list is allowed."""
    value: object = payload[key]
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{path.name} {key} must be a list of strings")
    return tuple(value)


def _optional_str(value: object, key: str, path: Path) -> str | None:
    """Return a string or None."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path.name} {key} must be a string or null")
    return value


def _feature_set_name(name: str, path: Path) -> str:
    """Reject a feature set the pregame module does not define."""
    if name not in FEATURE_SET_NAMES:
        known: str = ", ".join(FEATURE_SET_NAMES)
        raise ValueError(f"{path.name} feature set {name!r} is unknown. Known: {known}.")
    return name


def _fixed_c(value: object, path: Path, model_name: str) -> float | None:
    """Return a positive finite C, or None when the row does not set one.

    Raises:
        ValueError: The value is not a positive finite number, or it is set
            on a model other than logistic.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path.name} fixed_c must be a positive number")
    c_value: float = float(value)
    if not math.isfinite(c_value) or c_value <= 0.0:
        raise ValueError(f"{path.name} fixed_c must be a positive finite number, got {value!r}")
    if model_name != "logistic":
        raise ValueError(f"{path.name} fixed_c is only valid on logistic, not {model_name}")
    return c_value


def _model_specs(payload: dict[str, Any], path: Path) -> tuple[CvModelSpec, ...]:
    """Parse the scored model rows."""
    raw: object = payload["model_specs"]
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"{path.name} model_specs must be a non-empty list")
    specs: list[CvModelSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{path.name} each model spec must be a mapping")
        model_name: object = item.get("model")
        feature_set: object = item.get("feature_set")
        extra: list[str] = sorted(set(item) - {"model", "feature_set", "fixed_c"})
        if extra or not isinstance(model_name, str) or not isinstance(feature_set, str):
            raise ValueError(f"{path.name} model specs need model and feature_set strings")
        if model_name not in CV_MODEL_NAMES:
            known: str = ", ".join(sorted(CV_MODEL_NAMES))
            raise ValueError(f"{path.name} model {model_name!r} is unknown. Known: {known}.")
        specs.append(
            CvModelSpec(
                model_name,
                _feature_set_name(feature_set, path),
                _fixed_c(item.get("fixed_c"), path, model_name),
            )
        )
    return tuple(specs)


def _parse_experiment(payload: dict[str, Any], path: Path) -> ExperimentSpec:
    """Validate one experiment file."""
    phase: object = payload["phase"]
    if phase not in (1, 2):
        raise ValueError(f"{path.name} phase must be 1 or 2")
    experiment_id: str = _require_str(payload, "id", path)
    if experiment_id != path.stem:
        raise ValueError(f"{path.name} id must match the file name, got {experiment_id!r}")
    aliases: tuple[str, ...] = _require_strings(payload, "aliases", path)
    for alias in aliases:
        if alias != alias.strip().lower():
            raise ValueError(f"{path.name} alias {alias!r} must be lowercase")
    tag: str | None = _optional_str(payload["tag"], "tag", path)
    logistic_sets: tuple[str, ...] = tuple(
        _feature_set_name(name, path)
        for name in _require_strings(payload, "logistic_feature_sets", path)
    )
    if not logistic_sets:
        raise ValueError(f"{path.name} logistic_feature_sets must not be empty")
    specs: tuple[CvModelSpec, ...] = _model_specs(payload, path)
    logistic_rows: tuple[str, ...] = tuple(
        spec.feature_set for spec in specs if spec.model == "logistic"
    )
    if logistic_sets != logistic_rows:
        raise ValueError(f"{path.name} logistic_feature_sets must match the logistic model rows")
    profile: str = _require_str(payload, "report_profile", path)
    if profile not in REPORT_PROFILES:
        known: str = ", ".join(sorted(REPORT_PROFILES))
        raise ValueError(f"{path.name} report_profile {profile!r} is unknown. Known: {known}.")
    log_qb: bool = _require_bool(payload, "log_qb", path)
    qb_subset: bool = _require_bool(payload, "qb_subset", path)
    qb_feature_set: str | None = _optional_str(payload["qb_feature_set"], "qb_feature_set", path)
    if log_qb or qb_subset:
        if qb_feature_set is None:
            raise ValueError(f"{path.name} qb_feature_set is required when QB fields are enabled")
        _feature_set_name(qb_feature_set, path)
    elif qb_feature_set is not None:
        raise ValueError(f"{path.name} qb_feature_set must be null when QB fields are off")
    if phase == 1:
        if tag is not None or aliases:
            raise ValueError(f"{path.name} phase 1 has no tag and no aliases")
    elif tag is None or tag not in aliases:
        raise ValueError(f"{path.name} tag must be one of its aliases")
    reference_feature_set: str = _require_str(payload, "reference_feature_set", path)
    if reference_feature_set != "none":
        _feature_set_name(reference_feature_set, path)
    return ExperimentSpec(
        experiment_id=experiment_id,
        phase=int(phase),
        default=_require_bool(payload, "default", path),
        aliases=aliases,
        tag=tag,
        short_name=_require_str(payload, "short_name", path),
        experiment_name=_require_str(payload, "experiment_name", path),
        phase_tag=_require_str(payload, "phase_tag", path),
        output_dir=_require_str(payload, "output_dir", path),
        title=_require_str(payload, "title", path),
        note=_require_str(payload, "note", path),
        reference_model=_require_str(payload, "reference_model", path),
        reference_feature_set=reference_feature_set,
        logistic_feature_sets=logistic_sets,
        model_specs=specs,
        log_shrinkage=_require_bool(payload, "log_shrinkage", path),
        log_qb=log_qb,
        qb_subset=qb_subset,
        qb_feature_set=qb_feature_set,
        report_profile=profile,
    )


def _load_catalog(directory: Path) -> ExperimentCatalog:
    """Read and cross-check every YAML file in ``directory``."""
    paths: list[Path] = sorted(directory.glob("*.yaml"))
    if not paths:
        raise ValueError(f"No experiment YAML files in {directory}")
    experiments: list[ExperimentSpec] = []
    for path in paths:
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Sprint 2 margin and total is a separate file. ADR-007.
        if isinstance(payload, dict) and payload.get("task") == "score":
            continue
        experiments.append(_parse_experiment(_require_mapping(payload, path), path))
    if not experiments:
        raise ValueError(f"No classification experiment YAML files in {directory}")
    by_id: dict[str, ExperimentSpec] = {}
    by_alias: dict[str, ExperimentSpec] = {}
    by_tag: dict[str, ExperimentSpec] = {}
    for spec in experiments:
        if spec.experiment_id in by_id:
            raise ValueError(f"Duplicate experiment id {spec.experiment_id}")
        by_id[spec.experiment_id] = spec
        for alias in spec.aliases:
            if alias in by_alias:
                raise ValueError(f"Duplicate hypothesis alias {alias}")
            by_alias[alias] = spec
        if spec.tag is not None:
            if spec.tag in by_tag:
                raise ValueError(f"Duplicate hypothesis tag {spec.tag}")
            by_tag[spec.tag] = spec
    catalog = ExperimentCatalog(tuple(experiments), by_id, by_alias, by_tag)
    catalog.phase1()
    catalog.default_phase2()
    return catalog


@lru_cache(maxsize=1)
def load_catalog() -> ExperimentCatalog:
    """Return the experiment catalog, loaded once per process."""
    return _load_catalog(experiments_directory())


def experiment_for(phase: int, hypothesis: str | None = None) -> ExperimentSpec:
    """Resolve a phase and optional CLI hypothesis to one experiment file.

    Args:
        phase: ``1`` or ``2``.
        hypothesis: Short or full hypothesis name. None keeps the phase default.

    Returns:
        The matching experiment. Phase 1 returns the phase 1 file.

    Raises:
        ValueError: The phase or hypothesis is unknown.
    """
    catalog: ExperimentCatalog = load_catalog()
    if phase == 1:
        if hypothesis is not None:
            raise ValueError("Phase 1 does not take a hypothesis. Omit --hypothesis.")
        return catalog.phase1()
    if phase != 2:
        raise ValueError(f"Unknown phase {phase}. Known phases: 1, 2.")
    if hypothesis is None:
        return catalog.default_phase2()
    resolved: ExperimentSpec | None = catalog.by_alias.get(hypothesis.strip().lower())
    if resolved is None:
        known: str = ", ".join(sorted(catalog.by_alias))
        raise ValueError(f"Unknown hypothesis {hypothesis!r}. Known hypotheses: {known}.")
    return resolved


def resolve_hypothesis(phase: int, hypothesis: str | None = None) -> str | None:
    """Map a CLI hypothesis name to the tag stored on the run.

    Phase 1 returns None. Phase 2 returns the tag from the experiment file.
    """
    spec: ExperimentSpec = experiment_for(phase, hypothesis)
    if phase == 1:
        return None
    return spec.tag


def logistic_feature_sets_for(phase: int, hypothesis: str | None = None) -> tuple[str, ...]:
    """Return the fixed-C logistic feature sets declared for this experiment."""
    return experiment_for(phase, hypothesis).logistic_feature_sets


def cv_model_specs_for(phase: int, hypothesis: str | None = None) -> tuple[CvModelSpec, ...]:
    """Return the model and feature-set rows declared for this experiment."""
    return experiment_for(phase, hypothesis).model_specs


def phase_tracking_settings(phase: int, hypothesis: str | None = None) -> dict[str, Any]:
    """MLflow experiment, tags, note, and default output directory for a phase.

    Returns:
        Dict with ``experiment_name``, ``phase_tag``, ``hypothesis``,
        ``note``, ``output_dir``, and ``title``. ``hypothesis`` is None
        for phase 1.
    """
    spec: ExperimentSpec = experiment_for(phase, hypothesis)
    return {
        "experiment_name": spec.experiment_name,
        "phase_tag": spec.phase_tag,
        "hypothesis": None if phase == 1 else spec.tag,
        "note": spec.note,
        "output_dir": spec.output_dir,
        "title": spec.title,
    }


def resolve_reference_spec(
    phase: int,
    reference_model: str | None = None,
    reference_feature_set: str | None = None,
    hypothesis: str | None = None,
) -> dict[str, str]:
    """Choose the paired-comparison reference for a CV phase.

    The experiment file supplies the default model and feature set. Passing
    either argument replaces that default. When only the model is set, its
    usual feature set is used, except when that model is already the file
    default: the file's feature set stays.

    Returns:
        Dict with ``model`` and ``feature_set``.

    Raises:
        ValueError: The phase is unknown, or the model has no default set.
    """
    spec: ExperimentSpec = experiment_for(phase, hypothesis)
    default_model: str = spec.reference_model
    default_feature_set: str = spec.reference_feature_set
    model_name: str = default_model if reference_model is None else reference_model
    if reference_feature_set is not None:
        feature_set: str = reference_feature_set
    elif reference_model is None or model_name == default_model:
        feature_set = default_feature_set
    elif model_name in MODEL_DEFAULT_FEATURE_SET:
        feature_set = MODEL_DEFAULT_FEATURE_SET[model_name]
    else:
        known_models: str = ", ".join(sorted(MODEL_DEFAULT_FEATURE_SET))
        raise ValueError(
            f"Unknown reference model {model_name!r}. "
            f"Pass a feature set, or use one of: {known_models}."
        )
    return {"model": model_name, "feature_set": feature_set}


def report_profile_for(hypothesis: str | None) -> str:
    """Return the markdown profile for a stored hypothesis tag.

    Phase 1 passes None and uses the phase 1 file. An unknown tag adds no
    extra paragraph.
    """
    catalog: ExperimentCatalog = load_catalog()
    if hypothesis is None:
        return catalog.phase1().report_profile
    spec: ExperimentSpec | None = catalog.by_tag(hypothesis)
    if spec is None:
        return "none"
    return spec.report_profile
