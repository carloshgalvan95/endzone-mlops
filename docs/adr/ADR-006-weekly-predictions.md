# ADR-006: Weekly predictions for v0.1

**Date**: 2026-11-04
**Status**: Accepted

## Context

The reference model is logistic `h2_full`. On rolling-origin CV, seasons 2016-2023, its log loss is 0.6377 and the no-vig market is 0.6120. The market wins. The page still has to show a batch of upcoming games without turning that gap into a betting product.

A week of predictions is a product file. Committing a fresh warehouse extract every week would put data in git. This sprint's GitHub Actions workflow does not receive Databricks secrets.

The visual system lives in `docs/design/site_identity.md`. The page copy is English.

## Decision

### Scope of v0.1

1. Fit logistic `h2_full` on completed games strictly before the scored week. A row counts when its season is earlier, or the season matches and the week is earlier. Same-week finals are not training rows and are not features. A Sunday rebuild does not learn from a game that already finished that week. A feature for a game still uses only rows with `gameday` strictly before that game. A row with a missing `home_won` is unplayed and is not a result.
2. Log the fitted pipeline to MLflow with a signature and an input example. Register it as `endzone-winprob`.
3. Score one season and week. The default is the earliest unplayed week in the earliest season that still has unplayed games. `make score SEASON=2026 WEEK=10` selects that slate from `fct_upcoming_games`. If that week file is already on disk, the command keeps it and does not fit. `--replace-published` with both `--season` and `--week` scores every scheduled game that week, played and unplayed, from `fct_games` and `fct_upcoming_games`. `make rebuild-week SEASON=2026 WEEK=9` is that command followed by the results overlay.
4. When both moneylines are present, include the same no-vig home probability the market baseline uses. `diff` is model home win probability minus market home win probability. A missing, null, or non-finite line leaves that game's `market_home_wp` and `diff` null. It does not blank another game that has both prices. `fct_upcoming_games` selects `home_moneyline` and `away_moneyline` from staging and does not turn a null into zero.
5. Write `site/data/predictions/<season>-w<week>.json` and `site/data/latest.json`. `weeks_available` lists every non-example week file already in that directory for the same season, including the file just written. Older week files in the directory get the same list. A published site can be copied into the directory first (`--restore-published`) so a later score keeps those weeks. An example `latest.json` is not copied. A write that would drop a `game_id` already stored for that season and week fails. Adding games is allowed. The archive in decision 11 is checked the same way, including a `game_id` that appears only in an earlier snapshot. The job stops rather than deploying a shorter slate.

Document fields: `sample`, `season`, `week`, `weeks_available`, `generated_at`, `model_name`, `model_version`, `feature_set`, `threshold_pp`, `games`.

Each game: `game_id`, `kickoff`, `home`, `away`, `model_home_wp`, `market_home_wp`, `diff`, `model_version`, `generated_at`.

`kickoff` is an ISO timestamp in `America/New_York` when `gametime` is present. Otherwise it is the date. The page shows the instant in `America/Mexico_City`.

6. Team names, colors, and logo URLs come from nflverse `teams_colors_logos`. They are written to `site/data/teams.json`. Logo image files are not stored in the repo. The logo URL is `team_logo_espn`, except the light `500-dark` asset, which uses `team_logo_wikipedia`. `team_color` is used only as a 3px left border, and only when it matches `#RRGGBB`.
7. `site/` is a static page: plain HTML, CSS, and JavaScript. The honest banner is fixed: "Open experiment. Model log loss 0.6377 vs market 0.6120 (the market wins). Not betting picks." Disagreements highlight at 5 percentage points by default. The reader can choose 3, 5, 8, or 10. The badge reads "DISAGREES". Team logos sit on a cream disc and fall back to the abbreviation.
8. Generated prediction JSON is not committed on `main`. The orphan branch in decision 11 holds it. `site/data/sample.json` is an example slate: `sample: true`, no season, and no week. The page loads `latest.json`. It uses the sample only when that file is missing, and then the EXAMPLE label stays visible. The week selector lists the weeks in `weeks_available`, which are the week files for that season. `site/data/teams.json` is committed because the page needs public logo URLs and colors without a warehouse. It is not a game extract.
9. `.github/workflows/pages.yml` deploys `site/` to GitHub Pages. It runs on push to `main`, on `workflow_dispatch` (optional season, week, `replace_published`, and `backfill_archive`), and on Tuesdays at 12:00 UTC. Concurrency group is `pages`. A push installs Python 3.12 with `uv sync --frozen`, restores published week files from `published-slates` first (decision 11), overlays finals, archives those files, and deploys. A failed archive does not deploy. It does not refresh the warehouse and it does not fit a model. Tuesday and a dispatch without `replace_published` also refresh the current season (`ingest --current-season --dataset all`), run `dbt deps` and `dbt build --target databricks` (the same target as `make dbt-build-databricks`), restore published week files from the archive first, and run `python -m endzone.modeling.score`. That score keeps a week that is already published. `replace_published` requires season and week and passes `--replace-published`. Warehouse credentials are the `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, and `DATABRICKS_TOKEN` repository secrets, passed as environment variables on the ingest, dbt, and score steps only. MLflow uses a sqlite store in the runner. If that local registry cannot register `endzone-winprob`, the job still writes the week file and sets `model_version` to `local`. The JSON is written only in the runner. A failed score does not upload the site, so the previous deployment stays up.

`make reproduce` still does not register a model. `make verify-reference` is unchanged.

10. Final scores are a separate refresh. `python -m endzone.site.results` reads nflverse `games.parquet` (the schedules file ingest already downloads), matches `game_id`, and writes scores onto non-example week files and `latest.json`. It starts from those published files. It does not call `select_slate` or `load_upcoming_games`, because those drop finished games. It does not refit the model and it does not run dbt. Every game has `status` `scheduled` or `final`. `home_score` and `away_score` are present only when `status` is `final`. The winner is the higher score. Equal scores are a tie. Optional `result` is `home`, `away`, or `tie`. `generated_at` stays the prediction timestamp. `results_updated_at` is an ISO UTC timestamp with a Z suffix. `.github/workflows/refresh-results.yml` restores `published-slates` first, overlays scores, archives the files, and deploys `site/` on manual dispatch and on a UTC schedule covering Thursday night, Sunday, and Monday night. It has no Databricks secrets. `pages.yml` runs the same overlay on every deploy. On a push, the archive restore runs before the overlay. The overlay does not fetch again. After a score or a rebuild, the overlay does not restore again, so it does not copy the previous slate back over the file just written. Example files (`sample: true`) are left alone. The page keeps the yellow WINS box as the frozen model pick. When a game is final, the Kickoff cell shows the score and a cream check or cross for whether that pick matched the result. Sprint 2 point estimates are on the page. Both targets passed ADR-007.

11. Published week JSON has a durable copy outside the live Pages artifact. The copy is the orphan branch `published-slates`. It is not a parent of `main` and it is not merged. `main` stays free of these files.

`predictions/<season>-w<week>.json` and `latest.json` at the tip are the current files. `history/<season>-w<week>/<timestamp>.json` keeps every earlier file. The timestamp is `YYYYMMDDTHHMMSSZ`, with no colon. Replacing a week does not delete the earlier file. Git history on the branch is a second way to read the same versions. The push is not forced. Identical bytes do not add a commit.

A Pages or Results run restores that branch first. A week the branch does not have is copied from the live site. A missing branch, which is the state before the first backfill, falls back to the live site. Any other failure reading the remote does not fall back. An unreadable archived week fails the job instead of taking the live file.

After the overlay, and after a score or a rebuild, the workflow commits the exact week files it is about to deploy and pushes `published-slates`. The message is `chore(data): archive 2026 week 9`, or `weeks` when more than one file changed. A backfill uses `backfill` in place of `archive`. The commit is authored as `github-actions[bot]`. If the commit or the push fails, the job fails and Pages is not deployed.

The no-shrink rule applies to the archive. A file that drops a `game_id` present on the tip, or in any history file for that season and week, fails. Adding games is allowed. A deploy that omits a week the archive already has fails. `contents: write` is set only on the job that pushes the branch. That job also has `actions: read` so it can download the built site. The deploy job has `pages: write`, `id-token: write`, and `actions: read`. It does not have `contents: write`. Databricks secrets are set only on the ingest, dbt, and score steps.

The one-time seed is a Pages `workflow_dispatch` with `backfill_archive` set. It copies the live site onto the branch and does not score and does not deploy. `make backfill-archive PAGES_BASE=<pages url>` is the same command. Run it once after this change is on `main`, before the next Tuesday score. The runbook has the steps.

### Later sprints

Sprint 2 is ADR-007. The acceptance rule was fixed before the fit. Home points are `(total + margin) / 2` and away points are `(total - margin) / 2`, rounded on the page only. Until both targets pass, the page leaves the estimate null and says it did not pass its pre-registered test. The warehouse run passed both, so the page shows the estimates. The win-probability box is unchanged.

The weekly Pages job is the publish step. `published-slates` is the durable copy of past weeks. The track record on the page is ADR-008. It is computed from those archived files and does not rescore a week. The 0.003 rule is not moved because a page exists.

## Consequences

The score command needs `fct_games`, `fct_team_game_epa`, and `fct_upcoming_games`. A failure on a missing feature column still names the feature set and says to run `make dbt-build-databricks`.

The test workflow still compiles dbt on DuckDB and does not call the warehouse. The Pages workflow is the job that builds Databricks marts and publishes generated predictions. Those files stay off `main`. The orphan branch `published-slates` stores them, with history. See decision 11.

The banner stays on the page. A highlighted disagreement is not a pick. The footer states that team names and logos are trademarks of their owners and that the site is not affiliated with the NFL.
