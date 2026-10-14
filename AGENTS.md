# Working on this repo

Rules live in `.cursor/rules/` (`git-and-commits`, `quality`, `repo-map`, `ml-experiments`, `portfolio-quality`). Read those before editing.

## Verify

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -v
test -f dbt/profiles.yml || cp dbt/profiles.yml.example dbt/profiles.yml
( cd dbt && dbt deps --profiles-dir . && dbt seed --profiles-dir . --target local && dbt compile --profiles-dir . --target local )
make train-sweep
```

`make setup`, `make lint`, and `make test` match the venv, ruff, and pytest steps. `make dbt-compile` compiles with the local profile. `make train-sweep` runs the five training configs.
