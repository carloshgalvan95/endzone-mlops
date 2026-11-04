# ADR-006: Weekly predictions for v0.1

**Date**: 2026-11-04
**Status**: Accepted

## Context

The reference model is logistic `h2_full`. On rolling-origin CV, seasons 2016-2023, its log loss is 0.6377 and the no-vig market is 0.6120. The market wins. The page still has to show a batch of upcoming games without turning that gap into a betting product.

A week of predictions is a product file. Committing a fresh warehouse extract every week would put data in git. This sprint's GitHub Actions workflow does not receive Databricks secrets.

The visual system lives in `docs/design/site_identity.md`. The page copy is English.

## Decision

### Scope of v0.1

1. Fit logistic `h2_full` on every completed game. A feature for a game uses only rows with `gameday` strictly before that game. A row with a missing `home_won` is unplayed and is not a result.
2. Log the fitted pipeline to MLflow with a signature and an input example. Register it as `endzone-winprob`.
3. Score one season and week. The default is the earliest unplayed week in the earliest season that still has unplayed games. `make score SEASON=2026 WEEK=9` selects that slate.
4. When both moneylines are present, include the same no-vig home probability the market baseline uses. `diff` is model home win probability minus market home win probability. A missing, null, or non-finite line leaves that game's `market_home_wp` and `diff` null. It does not blank another game that has both prices. `fct_upcoming_games` selects `home_moneyline` and `away_moneyline` from staging and does not turn a null into zero.
5. Write `site/data/predictions/<season>-w<week>.json` and `site/data/latest.json`. `weeks_available` lists every non-example week file already in that directory for the same season, including the file just written. Older week files in the directory get the same list. A published site can be copied into the directory first (`--restore-published`) so a later score keeps those weeks. An example `latest.json` is not copied.

Document fields: `sample`, `season`, `week`, `weeks_available`, `generated_at`, `model_name`, `model_version`, `feature_set`, `threshold_pp`, `games`.

Each game: `game_id`, `kickoff`, `home`, `away`, `model_home_wp`, `market_home_wp`, `diff`, `model_version`, `generated_at`.

`kickoff` is an ISO timestamp in `America/New_York` when `gametime` is present. Otherwise it is the date. The page shows the instant in `America/Mexico_City`.

6. Team names, colors, and logo URLs come from nflverse `teams_colors_logos`. They are written to `site/data/teams.json`. Logo image files are not stored in the repo. The logo URL is `team_logo_espn`, except the light `500-dark` asset, which uses `team_logo_wikipedia`. `team_color` is used only as a 3px left border, and only when it matches `#RRGGBB`.
7. `site/` is a static page: plain HTML, CSS, and JavaScript. The honest banner is fixed: "Open experiment. Model log loss 0.6377 vs market 0.6120 (the market wins). Not betting picks." Disagreements highlight at 5 percentage points by default. The reader can choose 3, 5, 8, or 10. The badge reads "DISAGREES". Team logos sit on a cream disc and fall back to the abbreviation.
8. Generated prediction JSON is not committed. `site/data/sample.json` is an example slate: `sample: true`, no season, and no week. The page loads `latest.json`. It uses the sample only when that file is missing, and then the EXAMPLE label stays visible. The week selector lists the weeks in `weeks_available`, which are the week files for that season. `site/data/teams.json` is committed because the page needs public logo URLs and colors without a warehouse. It is not a game extract.
9. `.github/workflows/pages.yml` deploys `site/` to GitHub Pages. It runs on push to `main`, on `workflow_dispatch` (optional season and week), and on Tuesdays at 12:00 UTC. Concurrency group is `pages`. The job installs Python 3.12 with `uv sync --frozen`, refreshes the current season (`ingest --current-season --dataset all`), runs `dbt deps` and `dbt build --target databricks` (the same target as `make dbt-build-databricks`), then `python -m endzone.modeling.score`. Warehouse credentials are the `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, and `DATABRICKS_TOKEN` repository secrets, passed as environment variables. MLflow uses a sqlite store in the runner. If that local registry cannot register `endzone-winprob`, the job still writes the week file and sets `model_version` to `local`. The JSON is written only in the runner. A failed score does not upload the site, so the previous deployment stays up.

`make reproduce` still does not register a model. `make verify-reference` is unchanged.

### Later sprints

Sprint 2 is ADR-007. The acceptance rule was fixed before the fit. Home points are `(total + margin) / 2` and away points are `(total - margin) / 2`, rounded on the page only. Until both targets pass, the page leaves the estimate null and says it did not pass its pre-registered test. The win-probability box is unchanged.

The weekly Pages job is the publish step. A track record of past weeks, beyond keeping the week files the job already wrote, is still later work. The 0.003 rule is not moved because a page exists.

## Consequences

The score command needs `fct_games`, `fct_team_game_epa`, and `fct_upcoming_games`. A failure on a missing feature column still names the feature set and says to run `make dbt-build-databricks`.

The test workflow still compiles dbt on DuckDB and does not call the warehouse. The Pages workflow is the job that builds Databricks marts and publishes generated predictions. Those files stay out of git.

The banner stays on the page. A highlighted disagreement is not a pick. The footer states that team names and logos are trademarks of their owners and that the site is not affiliated with the NFL.
