# Working on this repo

Rules live in `.cursor/rules/` (`git-and-commits`, `quality`, `repo-map`, `ml-experiments`, `portfolio-quality`). Read those before editing.

## Verify

```bash
uv python install 3.12
uv sync --frozen
. .venv/bin/activate
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -v
test -f dbt/profiles.yml || cp dbt/profiles.yml.example dbt/profiles.yml
( cd dbt && dbt deps --profiles-dir . && dbt seed --profiles-dir . --target local && dbt compile --profiles-dir . --target local )
make train-sweep
```

`make setup` is `uv python install 3.12` and `uv sync --frozen`. `make lint` and `make test` match the ruff and pytest steps. `make dbt-compile` compiles with the local profile. `make train-sweep` runs the five training configs.
