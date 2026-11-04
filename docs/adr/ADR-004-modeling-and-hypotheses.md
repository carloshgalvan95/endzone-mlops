# ADR-004: Modeling package and YAML hypotheses

**Date**: 2026-11-03
**Status**: Accepted

## Context

Training, metrics, MLflow, and the hypothesis list lived in one module. Each new hypothesis added constants, aliases, and another branch. The CLI flags and the MLflow tags are already recorded in the experiment notes.

## Decision

Training lives in `endzone/modeling/`. Each module has one job: the CLI (`train.py`), the single split (`holdout.py`), the comparison (`pipeline.py`), rolling-origin folds (`cv.py`), estimator factories (`models.py`), the YAML registry (`hypotheses.py`), metrics and the acceptance rule (`evaluation.py`), MLflow (`tracking.py`), the markdown report (`report.py`), one scored row (`scoring.py`), mart loads (`dataset.py`), and the reference check (`reference.py`).

`configs/experiments/*.yaml` is the hypothesis registry. Adding a hypothesis is a new file.

The canonical command is `python -m endzone.modeling.train`. `python -m endzone.train.train` is a shim with the same flags. MLflow tags do not embed the package name.

Modules group one responsibility. Size is a smell to review, not a gate.

## Consequences

Import-linter keeps the layer order in `pyproject.toml`. A new file has to sit on that graph. The golden CV file and the CLI flags stay as they are.
