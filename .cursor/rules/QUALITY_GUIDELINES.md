# QUALITY_GUIDELINES.md

**Project:** Portfolio endzone-mlops (MLOps / DevOps) for Carlos Galvan  
**Stack primario:** Databricks Free Edition + dbt-databricks + MLflow + GitHub Actions + Python modular  
**Plan B:** Snowflake trial + dbt Core + MLflow sidecar, solo si el spike D1 de CE falla  
**Outline ancla:** the 14-day outline is not in this repo. Do not look for `outline_14d_2026-10-06.md` here.  
**Idioma de trabajo:** español (términos técnicos en inglés cuando el dominio los usa: dbt, MLflow, DoD, ADR, MoSCoW, CI, ASR, etc.)  
**Regla tipográfica permanente:** nunca usar el carácter em dash (U+2014). Usar coma, punto, dos puntos o guion corto (-).

Este documento es la guía canónica permanente de calidad y proceso para el portfolio. Cada PR, milestone (D1 / D7 / D14) y release (v0.1.0) se valida contra estas reglas.

---

## 1. Purpose and how to use

### Para qué existe

Codificar un quality bar accionable, extraído de:
- Notas SWEBOK (requirements, architecture, design, management, process)
- Patrones de proceso TC5062-25 (SRS-lite, backlog, DoD, sprint planning lite)
- Notas MLOps (Sculley, Treveil, DevOps vs MLOps, MLflow, Wilson 9/10/14)
- Outline Idaho 14 días

### Cómo usarlo

- Antes de abrir un PR: recorrer la DoD de la sección 4 y la checklist de la sección tocada
- Al cerrar D1, D7 o D14: marcar el DoD del milestone en la sección 5
- Al aceptar scope nuevo: scrubbing + change control; si cambia un ASR, abrir o actualizar un ADR
- Código generado con IA: quien lo integra lo lee completo, lo ejecuta y responde por él

---

## 2. North star / honesty

### North star

Construir un learning project público con forma de producción (lakehouse Spark/SQL + dbt + experiment tracking + CI), reproducible desde README, batch-first.

### Honesty (no negociable)

El README no lleva una sección Honesty, y no nombra empleadores, un CV, ni análogos de producción. El encuadre público es una o dos líneas: learning project with production-shaped patterns, batch only, not betting advice. No afirmar que el modelo le gana al mercado. En la tabla de referencia el mercado tiene menor log loss.

### Git authorship (hard permanent rule)

**Bajo ninguna circunstancia, por ningún motivo**, incluir Cursor, Grok Bot, Idaho, ni ningún AI agent/assistant como:
- author o co-author de commits Git (Author, Committer, trailers Co-authored-by:)
- entrada en CONTRIBUTORS, AUTHORS, o créditos equivalentes
- badge de contributors en README
- cuenta bot en el grafo GitHub Contributors

Los commits deben quedar authored como Carlos (su identidad GitHub). La IA puede ayudar a escribir código; nunca debe aparecer en git history metadata como author/co-author. No hay excepciones.

### Authorship voice (hard permanent rule)

**Commit messages, PR titles, PR descriptions, and PR comments MUST use first person as Carlos** (imperative or 'I ...'). Never refer to Carlos in third person. Never use agent voice or mention Cursor/AI/agents in any git content.

Examples:
- Good: "Add cross-platform setup", "I develop on both Windows and macOS"
- Bad: "Ready for Carlos to review", "Carlos can now use this feature", "Cursor agent added this"

---

## 3. Change workflow

### Starting work

- Always fetch the latest base branch: `git fetch origin` and branch from current `origin/main`
- Verify merge base matches current main: `git merge-base HEAD origin/main` should equal `git rev-parse origin/main`
- Before creating new modules, search the repository first to extend existing ones instead of recreating from scratch

### During development

- Verify external data schemas by inspecting the real file before writing models
- One focused change per PR; avoid mixing unrelated changes
- Before any push, the local suite passes: `ruff check endzone/ tests/`, `ruff format --check endzone/ tests/`, `make lint-structure` (Windows: `.\scripts\dev.ps1 lint-structure`), and `pytest tests/ -v`. Tests that read or write text pass `encoding="utf-8"` so Windows does not pick a legacy code page. When dbt files change, also run `make dbt-compile`.

### Committing changes

- Git identity must be Carlos Galván <carloshgalvan95@gmail.com> as both author and committer
- Branch names use a conventional prefix (`feat/`, `fix/`, `docs/`, `chore/`, `ci/`, `test/`, `refactor/`, `perf/`, `build/`). Never use the `cursor/` prefix or any other tool-generated prefix.
- No Co-authored-by trailers (not even self-attribution)
- Commit messages in first person or imperative voice
- No em dash character (U+2014) anywhere; use comma, period, colon, or regular hyphen

### Pull requests

- The coding agent pushes a `feat/`, `fix/`, or other conventional branch and does not open the pull request. Carlos opens the pull request from his account (`carloshgalvan95`). The title is a Conventional Commit. The body is first person. No AI traces. No Co-authored-by.
- A branch with an open pull request does not receive new commits. Close that pull request, push the fix, then open a new pull request from Carlos's account.
- Before committing, disable the `commit-msg.cursor.co-author` hook or any other trailer hook. Identity, trailers, and the em dash rule are in `git-and-commits.mdc`.
- PR title must follow Conventional Commit format: `type(scope): description`
- PR body in first person or imperative voice
- Complete the PR template sections: Summary, Changes, Verification (with at least one item checked), Risks
- PR must be up to date with base branch before merge

### Automated guardrails

The PR Guard workflow validates:
- Base branch freshness (not stale)
- Head branch uses a conventional prefix. A tool-generated prefix is rejected.
- Git authorship (Carlos only, no bots in metadata)
- Pull request opened by `carloshgalvan95`
- No Co-authored-by trailers or AI agent mentions in commit messages
- Conventional Commit PR title
- No tool-generated footers or em dash in PR body
- No em dash in changed files
- No data/secret files committed (except in tests/fixtures)
- PR template sections present and verification checklist not entirely unchecked

If PR Guard fails, close the pull request, fix the reported issues, push the branch, and open a new pull request from the `carloshgalvan95` account.

---

## 4. Quality bar (checklists accionables)

### 4.1 Requirements

- Problema de negocio o analítico escrito en 1 párrafo en README
- Fuentes de datos nombradas (URL, licencia, versión)
- Exclusiones explícitas: streaming, feature store real, K8s, multi-cloud, UI rica, dos lakes
- MoSCoW: Must (sin esto no hay DoD), Should, Could, Won't (14d)

### 4.2 Architecture

- ADR-001 con spike pass/fail
- docs/architecture.md: C4 L1/L2, flujo raw -> stg -> marts -> features -> train/score -> predicciones
- Trade-offs documentados

### 4.3 Design (Wilson 9/10)

- Entrypoint real: `endzone/modeling/train.py`. `python -m endzone.train.train` es el comando compatible. El score semanal es `python -m endzone.modeling.score`. Config inyectada.
- Nombres descriptivos; retornos dict/namedtuple
- Cada módulo agrupa una responsabilidad. El tamaño es un smell para revisar, no un gate. No hay chequeo de líneas en CI.
- Sin except desnudo; sin estado global mutable
- Contratos dbt schema.yml + test de negocio

### 4.4 MLOps

- Combatir glue code, pipeline jungles, config debt
- MLflow: train dentro de start_run(); log params/metrics/artifacts/model; seed fijo
- Registro: `make register-model` no pone el alias. `make promote-model VERSION=n` lo pone solo si las tags pasan el golden. El score semanal carga `@prod` y, si falta, ajusta la semana. ADR-009.
- Alerta: log loss publicado contra el mercado, un issue, no tumba el deploy. ADR-009.
- Reproducibilidad: deps pinneadas, make reproduce (no registra un modelo)

---

## 5. Definition of Done (portfolio-specific)

Una historia/PR está Done solo si cumple todos los puntos aplicables:

### Código
- [ ] PR a main con Issue; lint limpio; sin secretos
- [ ] Conventional Commits; author = Carlos (sin AI/Cursor/Grok Bot/Idaho en metadata Git ni créditos)

### Pruebas
- [ ] Evidencia de AC (pytest, dbt test, o guion + captura)
- [ ] CI verde
- [ ] Refactors: el golden CV (`tests/test_golden_cv.py`) pasa, y Carlos corre `make verify-reference`

### Docs
- [ ] README/ADR/runbook/architecture al día; el encuadre de learning project sigue en el README, sin empleadores ni CV

### Milestone
- [ ] Cumplir DoD D1 o D7 o D14 (sección 6)

---

## 6. Incremental process (D1 / D7 / D14)

### DoD Día 1 - Foundations + spike
- [x] Repo público + README + ADR-001
- [x] Skeleton + Makefile + CI smoke (lint/pytest)
- [x] Spike CE documentado en runbook (pass/fail): cluster, dbt, MLflow, GH secrets
- [x] Si pass: 1 raw + 1 stg_* + 1 test dbt
- [x] Si fail: Plan B Snowflake mismo día; ADR-001b
- [x] Actions verde

### DoD Día 7 - Pipeline + MLOps mínimo
- [x] dbt: staging + >=2 marts + schema tests + >=1 test de negocio
- [x] Python: features + `endzone/modeling/train.py` (argparse, >=3 HP, seed) + MLflow (>=5 runs, params/metrics/artifact/model). `python -m endzone.train.train` sigue funcionando.
- [x] CI: pytest + dbt compile (DuckDB). dbt build sigue en el target Databricks, no en CI.
- [x] Docs: C4 L1 + secuencia en `docs/architecture.md`
- [x] `make reproduce` instala el lock, construye marts en Databricks y reajusta la referencia (`verify-reference`). No descarga nflverse y no registra un modelo. Badge CI sí está.

### DoD Día 14 - Production-shaped + demo
- [x] `python -m endzone.modeling.score` escribe el JSON de una semana en `site/data` (ADR-006). No hay tabla de predicciones en el warehouse.
- [x] Observability lite: logging + 1 señal de calidad (log loss publicado vs mercado, ADR-009). No es un monitor de drift de features.
- [x] Job Databricks o GHA schedule
- [ ] README + capturas + Loom 3-5 min
- [ ] Tag v0.1.0

---

## 7. Anti-patterns (bloquear)

1. Notebook-as-product
2. Secrets en git
3. Two lakes en paralelo
4. Logos de producción inventados
5. Scope creep API/K8s/streaming antes del DoD batch
6. Prototipo sin refactor (Wilson 9)
7. Runs MLflow no reproducibles
8. DoD theater
9. AI/Cursor/Grok Bot/Idaho como Git author, Co-authored-by, CONTRIBUTORS/AUTHORS, badges o grafo Contributors (commits solo como Carlos)

---

Este documento es vivo. Dueño: Carlos Galvan. Cambios a ASRs requieren ADR. El estado de los experimentos está en `ml-experiments.mdc`.
