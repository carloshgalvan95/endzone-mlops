# ADR-011: Site data contract

Date: 2026-11-17

Status: Accepted

## Context

The weekly page publishes JSON under `site/data`. GitHub Pages serves that directory as static files. A separate client, endzone-app, will read those files and nothing else. It is an Expo React Native app in its own repository, [carloshgalvan95/endzone-app](https://github.com/carloshgalvan95/endzone-app). It is not a package, a submodule, or a workflow in this repo.

The files a client needs are `latest.json`, `predictions/<season>-w<week>.json`, `teams.json`, `track_record.json`, and the example `sample.json`. ADR-006 is the publish decision. This ADR versions the bytes those files contain.

The page must keep showing the same fields. Files already on the `published-slates` branch are not rewritten. A deploy that drops a `game_id` still fails.

## Decision

The public JSON is a versioned read-only API for external clients. Clients do not get a warehouse credential, a model file, or a private endpoint. This remains a learning project. It is not betting advice. On the reference table the market still has the lower log loss.

### Where the schemas live

Draft 2020-12 JSON Schemas are in `contracts/site-data/v1/`:

- `slate.schema.json` covers `latest.json`, each week file, and `sample.json`.
- `teams.schema.json` covers `teams.json`.
- `track_record.schema.json` covers `track_record.json`.

The schema `$id` is the identity of that major version. This repository does not publish the schema files to GitHub Pages. A client pins a major version by reading `contracts/site-data/v1/` from this repo, at a git tag or a commit. It fetches the data from GitHub Pages.

### contract_version on the JSON

New documents from `build_prediction_document`, `team_catalog_document`, and the track record writer include `contract_version` `"1.0.0"`. `sample.json` and `teams.json` in git include it too. The field is optional in the schema. Week files already on `published-slates` do not have it, and this change does not rewrite them. A client treats a missing `contract_version` as major version 1. When the field is present, the v1 schemas accept only `1.x.y`.

I added the field because it does not change an existing field, the page does not render it, and the archive is not rewritten. Putting the version only in `$id` would hide it from a client that never clones the repo. Writing it into archived slates would change published files, so the field stays optional.

### Semantic versioning

- Patch (`1.0.x`): a description or a test change that does not change what validates.
- Minor (`1.x.0`): an additive optional field. Old clients still validate. The schema stays in `v1`.
- Major (`2.0.0`): a removal, a rename, or a type change. That ships as `contracts/site-data/v2/` with a new `$id`. The v1 schemas stay in the tree.

Required fields match what the writers emit today. An example slate has `sample: true`, null `season` and `week`, `is_example`, and `label`. A real slate has integer `season` and `week`. `model_alias` is present on new writes and absent on week files from before ADR-009, so it is optional. Final games carry integer scores and `result` of `home`, `away`, or `tie`. Equal scores, including 0-0, are a tie. Scheduled games omit scores and `result`. The score block's five target fields are always present. The longer ADR-007 metrics are present only after that test has been stored, so those keys are optional.

### How a client pins a major version

Read the schemas from `contracts/site-data/v1/`. Reject a document whose `contract_version`, when present, does not start with `1.`. Do not infer a newer major from an unknown field. An unknown field fails this repo's check before deploy, so a file that validates against v1 does not carry a removed or renamed field.

### Deprecation

A major version stays in the repository while any file on the live site or on `published-slates` was written against it. A breaking change is announced in a new ADR and in `contracts/site-data/README.md` before writers switch. v1 is not deleted in the same change that adds v2. Removal waits until the live site no longer serves a file that only v1 describes.

### Checks

`python -m endzone.site.contract --data-dir site/data` validates the files under that directory. CI runs it on the committed sample and team catalog. Pages and Results run it with `--require-published` in the build job, immediately before the site artifact upload. A failure stops that job, so the deploy does not publish a document that breaks v1. The command only reads. It does not score, overlay, archive, or edit a week file.

## Consequences

- endzone-app can pin v1 without importing this package.
- An optional field is a schema change in this repo before a writer emits it.
- A Pages or Results job fails closed when a writer and the schema drift.
- Archived slates from before this ADR stay valid because `contract_version` is optional.
- The page still renders the same fields. The new key is not shown.
