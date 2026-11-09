# ADR-009: Model registry, the prod alias, and the track-record alert

**Date**: 2026-11-09
**Status**: Accepted

## Context

Tuesday scoring fits logistic `h2_full` on games before the week and registers `endzone-winprob` in a sqlite file on the runner. That version is thrown away with the runner. The page has no durable model to load, and nothing says when the published record is clearly worse than the validated baseline. Trailing the market by the reference gap is already the known state.

The reference table is still model log loss 0.6377 against market 0.6120. The market is closer. This ADR does not change that, and it does not move the 0.003 rule.

## Decision

### Registry name

Unity Catalog name: `workspace.nfl_dev.endzone_winprob`. Catalog `workspace` (`DATABRICKS_CATALOG`), schema `nfl_dev` (`ENDZONE_UC_SCHEMA`), model `endzone_winprob` (`ENDZONE_UC_MODEL`). Unity Catalog does not allow a hyphen in the model name. `ENDZONE_REGISTERED_MODEL` overrides the whole name.

A sqlite or file registry keeps the name `endzone-winprob`. Tests use sqlite for tracking and for the registry.

`MLFLOW_REGISTRY_URI` selects the registry. When it is unset, the registry is the tracking URI. Unity Catalog is `databricks-uc`. Registration to Unity Catalog requires `MLFLOW_TRACKING_URI=databricks`, because the registry has to read the artifact. A sqlite tracking URI with `databricks-uc` is refused before a write.

The first registration creates the model object in the existing schema `nfl_dev`. It does not create a schema, and it does not alter a table.

### Experiment path

The logical experiment name stays `nfl_pregame/registry`. SQLite tracking, including Pages and the tests, keeps that relative name.

Databricks tracking (`databricks` or `databricks://...`) rejects a relative name. `activate_tracking_experiment` is the only rewrite. A relative name is stored under `ENDZONE_MLFLOW_EXPERIMENT_ROOT`, which defaults to `/Shared/endzone-mlops`. `nfl_pregame/registry` becomes `/Shared/endzone-mlops/nfl_pregame/registry`. A name that already starts with `/` is unchanged. The same rule covers training and scoring when those commands use a Databricks tracking URI. It does not apply to `databricks-uc`, which is a registry URI.

`mlflow.set_experiment` does not create the parent workspace folder. The same function calls `WorkspaceClient.workspace.mkdirs` on the parent only. That call creates missing parents and does nothing when the folder already exists. It does not create a folder at the experiment path, because that path has to stay free for the experiment object. If `/Shared` is not writable, set the root to `/Users/<email>/endzone-mlops`.

### What gets registered

`make register-model` fits logistic `h2_full` on completed games through season 2023 (`games_before_week` of season 2024, week 1). The default cutoff is `--through-season 2023`.

The version tags are the rolling-origin CV of that recipe, the same numbers `make verify-reference` checks against `docs/experiments/reference_metrics.json`. They are not the in-sample loss of the registered fit. The fit is one model. The tags are the locked CV of the recipe.

The command does not set an alias. `make reproduce` does not call it.

### How a version gets `prod`

Promotion is a separate step: `make promote-model VERSION=<n>`, or the Model registry workflow with action `promote`. The version is a positive integer. `prod` is an alias, not a version.

Logistic `h2_full` may receive the alias when `cv_log_loss` and `cv_n` match the golden row within its tolerance (0.6377 and 2191, tolerance 1e-4). It does not have to beat itself, and it does not have to beat the market.

Any other feature set may receive the alias only when ADR-005 accepts it and its log loss is at least 0.003 below the golden `h2_full` log loss. The register command itself only registers logistic `h2_full`. The challenger path is the same gate, read from version tags.

A refusal leaves every alias as it was. Promotion does not delete a version. Nothing in weekly scoring sets the alias.

### What the scorer does

`python -m endzone.modeling.score` loads `models:/<name>@prod`.

When that load works, `model_version` on the document and on each game is `<version>@prod`, and `model_alias` is `prod`. The command does not register a new version.

When the alias is missing, the log says the alias is missing and this week will be fit. When the registry cannot be reached, or the loaded model cannot score the slate, the log names that failure. In every one of those cases the command keeps today's fit: logistic `h2_full` on games before the week, registered in the tracking store only. `model_alias` is null. `model_version` is the tracking-store version, or `local` when a sqlite register fails. A Databricks tracking URI still raises if that registration itself fails. Pages keeps tracking on `sqlite:///mlflow.db`, so the scheduled publish does not raise for a registry miss.

Pages sets `MLFLOW_REGISTRY_URI=databricks-uc` on the score step and leaves tracking on sqlite. The fallback fit, and the Sprint 2 margin and total models, are written to that sqlite file. They are not written to Unity Catalog.

Old week files omit `model_alias`. A present value is text or null. The results overlay freezes the field when it is present.

### Alert

After Results writes the track record, `python -m endzone.site.model_alert` compares the model and the no-vig market on the same clipped binary log loss as ADR-008. Ties and missing prices are out. The window is the last 64 decided games that have both prices. It does not trip unless at least 48 of those games are in the window.

The gap on that window is model log loss minus market log loss. The reference gap is the same difference on the locked CV table, read from `docs/experiments/reference_metrics.json` (logistic `h2_full` minus the market row). Today that is 0.6377 minus 0.6120. The alert trips when the window gap is greater than or equal to the reference gap plus a margin. The default margin is 0.02, so the bar is about 0.046. A window that looks like the reference table stays quiet. Trailing the market by the reference gap is the known, honest state. The alert means the published model is doing clearly worse than its own validated baseline.

Overrides: `ENDZONE_ALERT_WINDOW_GAMES`, `ENDZONE_ALERT_MIN_GAMES`, and `ENDZONE_ALERT_GAP_MARGIN`, or the CLI flags `--window-games`, `--min-games`, and `--margin`.

A trip opens or updates one GitHub issue, title "Published model is worse than its validated baseline", label `model-behind-market`. The issue is not a betting signal. The workflow uses `GITHUB_TOKEN`. An existing open issue with that label is updated. A pull request is ignored. The body has no timestamp, so an hourly run that sees the same window does not edit the issue again. A missing token, an HTTP error, or a short sample is logged. The step returns 0, and the workflow step is `continue-on-error`. The publish does not fail because of the alert. The alert does not edit a week file or the track record.

## Consequences

Until someone runs the register and then the promote, Tuesday logs that `prod` is missing and fits the week. The runbook has the one-time commands. This environment does not run them. They need the existing Databricks secrets, and they must not drop or alter an existing warehouse object.

The weekly publish still refuses to shrink a slate on `published-slates`. A registry failure is not a reason to skip a week or to replace one.
