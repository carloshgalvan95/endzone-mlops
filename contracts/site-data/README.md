# Site data contract

The JSON that GitHub Pages publishes is a read-only API. Version 1 is [ADR-011](../../docs/adr/ADR-011-site-data-contract.md). Schemas are draft 2020-12 and live in `contracts/site-data/v1/`. This directory is not deployed to Pages. Pin a major version by reading `v1/` from this repo.

An additive optional field is a minor bump and stays in `v1`. A removal, a rename, or a type change is a major bump and a new `v2/` directory. v1 stays until the live site no longer needs it.

`contract_version` is `"1.0.0"` on files this repo writes after ADR-011. It is optional. Week files already on `published-slates` omit it. Treat a missing value as major version 1.

The native client is [endzone-app](https://github.com/carloshgalvan95/endzone-app), a separate repository. It only reads these URLs.

| File | Schema | URL |
| --- | --- | --- |
| `latest.json` | `v1/slate.schema.json` | https://carloshgalvan95.github.io/endzone-mlops/data/latest.json |
| `predictions/<season>-w<week>.json` | `v1/slate.schema.json` | https://carloshgalvan95.github.io/endzone-mlops/data/predictions/2026-w10.json |
| `sample.json` | `v1/slate.schema.json` | https://carloshgalvan95.github.io/endzone-mlops/data/sample.json |
| `teams.json` | `v1/teams.schema.json` | https://carloshgalvan95.github.io/endzone-mlops/data/teams.json |
| `track_record.json` | `v1/track_record.schema.json` | https://carloshgalvan95.github.io/endzone-mlops/data/track_record.json |

`latest.json` is the current slate. Each week file is the same slate shape. `sample.json` is an example (`sample: true`, no season or week) and is not a prediction. `teams.json` is names, `#RRGGBB` colors, and logo URLs. `track_record.json` is the derived record. A tie is `result: "tie"` with equal scores, including 0-0.

Check a checkout with:

```bash
python -m endzone.site.contract --data-dir site/data
```

Pages and Results add `--require-published` immediately before the site artifact upload. That command does not write a file.

Example client fetch:

```javascript
const base = "https://carloshgalvan95.github.io/endzone-mlops/data";
const response = await fetch(`${base}/latest.json`);
if (!response.ok) {
  throw new Error(`latest.json returned ${response.status}`);
}
const slate = await response.json();
const version = slate.contract_version || "1.0.0";
if (!version.startsWith("1.")) {
  throw new Error(`This client reads site data contract major version 1, got ${version}.`);
}
const teams = await fetch(`${base}/teams.json`).then((result) => result.json());
const record = await fetch(`${base}/track_record.json`).then((result) => result.json());
```
