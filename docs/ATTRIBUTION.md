# Data Source Attribution

This document provides full attribution for all data sources used in the endzone-mlops project.

## nflverse

**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

**Source**: https://github.com/nflverse

**Coverage**: NFL play-by-play, game schedules, rosters, and related data from 1999-present

**Citation**:
```
nflverse (2024). NFL data aggregation and distribution.
GitHub repository: https://github.com/nflverse
Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
```

**Files Used**:
- Play-by-play data: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet`
- Games/schedules: `https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet`
- Player data: `https://github.com/nflverse/nflverse-data/releases/download/player_stats/players.parquet`

**Terms of Use**:
- Attribution required: Yes (this document fulfills requirement)
- Commercial use: Allowed
- Modifications: Allowed
- Distribution: Allowed with attribution

**Data Quality Notes**:
- Clean version: Updated nightly after games complete
- Raw version: Available ~15 minutes after game ends
- Corrections: Historical data may receive updates; re-pull recommended periodically
- Coverage: ~370 columns per play; closing odds in games table since 1999

## BALLDONTLIE (Future, Not Active)

**Status**: Documented for future v0.2+ integration; NOT used in v0.1

**API**: https://nfl.balldontlie.io/

**Pricing**: GOAT tier ~$39.99/month (600 req/min)

**License**: Commercial API with terms of service
- Machine learning: Allowed
- Reselling raw feed: Prohibited
- Product use: Allowed with attribution

**Decision Gate**: Activation requires explicit owner approval and is post-Day 14

**Attribution**:
```
BALLDONTLIE NFL API (when active in v0.2+)
https://nfl.balldontlie.io/
Unofficial NFL data provider, near-real-time updates.
Not affiliated with or endorsed by the NFL.
```

## Third-Party Libraries

This project uses various open-source Python libraries. See `requirements.txt` and `pyproject.toml` for full dependency list. Key libraries:

- **pandas**: BSD 3-Clause License
- **dbt-core**: Apache License 2.0
- **MLflow**: Apache License 2.0
- **pytest**: MIT License
- **ruff**: MIT License

## Intellectual Property Statement

The code and documentation in this repository (excluding data sources) are original work by Carlos Galvan, licensed under MIT License (see [LICENSE](../LICENSE)).

Data sources retain their original licenses as documented above. This project does not claim ownership of nflverse data or any third-party APIs.

## No Official Affiliation

This project is **not affiliated with, endorsed by, or sponsored by**:
- The National Football League (NFL)
- nflverse project (we are grateful users)
- BALLDONTLIE (future API vendor)
- Databricks, Inc.
- dbt Labs
- Any team, player, or league depicted in the data

This is an independent learning portfolio demonstrating MLOps practices with publicly available sports data.

## Compliance

All data usage in this project complies with:
- CC BY 4.0 attribution requirements (nflverse)
- API terms of service (when BALLDONTLIE becomes active)
- Fair use for educational and portfolio purposes
- No redistribution of raw commercial API responses in repository

## Updates

This attribution document will be updated when:
- New data sources are added
- License terms change
- Live API integration activates (v0.2+)

Last updated: 2026-10-06
