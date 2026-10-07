# QUALITY_GUIDELINES.md

**Project:** Portfolio endzone-mlops (MLOps / DevOps) for Carlos Galvan  
**Stack primario:** Databricks CE/Free + dbt-databricks + MLflow + GitHub Actions + Python modular  
**Plan B:** Snowflake trial + dbt Core + MLflow sidecar, solo si el spike D1 de CE falla  
**Outline ancla:** outline_14d_2026-10-06.md (in project uploads/)  
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

## 2. North star / honesty banner

### North star

Construir un learning project público con forma de producción (lakehouse Spark/SQL + dbt + experiment tracking + CI), reproducible desde README, batch-first, honesto en CV/LinkedIn.

**Frase análoga obligatoria (README / CV):**

> Lakehouse Spark/SQL + dbt + experiment tracking (MLflow) + CI; en producción el análogo cercano sería Athena/Redshift Spectrum + Glue + Prefect (o Airflow).

### Honesty banner (no negociable)

Pegar desde commit 1 en README.md (y mantenerlo):

```markdown
## Honesty

Public learning portfolio. Production analogues for CV/LinkedIn: Athena / Glue / Redshift Spectrum / Prefect (role-dependent).
Do not invent Databricks, dbt, MLflow, or Snowflake logos on PEMEX or Kavak bullets.
```

### Git authorship (hard permanent rule)

**Bajo ninguna circunstancia, por ningún motivo**, incluir Cursor, Grok Bot, Idaho, ni ningún AI agent/assistant como:
- author o co-author de commits Git (Author, Committer, trailers Co-authored-by:)
- entrada en CONTRIBUTORS, AUTHORS, o créditos equivalentes
- badge de contributors en README
- cuenta bot en el grafo GitHub Contributors

Los commits deben quedar authored como Carlos (su identidad GitHub). La IA puede ayudar a escribir código; nunca debe aparecer en git history metadata como author/co-author. No hay excepciones.

---

## 3. Quality bar (checklists accionables)

### 3.1 Requirements

- Problema de negocio o analítico escrito en 1 párrafo en README
- Fuentes de datos nombradas (URL, licencia, versión)
- Exclusiones explícitas: streaming, feature store real, K8s, multi-cloud, UI rica, dos lakes
- MoSCoW: Must (sin esto no hay DoD), Should, Could, Won't (14d)

### 3.2 Architecture

- ADR-001 con spike pass/fail
- docs/architecture.md: C4 L1/L2, flujo raw -> stg -> marts -> features -> train/score -> predicciones
- Trade-offs documentados

### 3.3 Design (Wilson 9/10)

- Entrypoints modulares (train.py, score.py); config inyectada
- Nombres descriptivos; retornos dict/namedtuple
- Sin except desnudo; sin estado global mutable
- Contratos dbt schema.yml + test de negocio

### 3.4 MLOps

- Combatir glue code, pipeline jungles, config debt
- MLflow: train dentro de start_run(); log params/metrics/artifacts/model; seed fijo
- Reproducibilidad: deps pinneadas, make reproduce

---

## 4. Definition of Done (portfolio-specific)

Una historia/PR está Done solo si cumple todos los puntos aplicables:

### Código
- [ ] PR a main con Issue; lint limpio; sin secretos
- [ ] Conventional Commits; author = Carlos (sin AI/Cursor/Grok Bot/Idaho en metadata Git ni créditos)

### Pruebas
- [ ] Evidencia de AC (pytest, dbt test, o guion + captura)
- [ ] CI verde

### Docs
- [ ] README/ADR/runbook/architecture al día; honesty intacta

### Milestone
- [ ] Cumplir DoD D1 o D7 o D14 (sección 5)

---

## 5. Incremental process (D1 / D7 / D14)

### DoD Día 1 - Foundations + spike
- [x] Repo público + README con honesty banner + ADR-001
- [x] Skeleton + Makefile + CI smoke (lint/pytest)
- [x] Spike CE documentado en runbook (pass/fail): cluster, dbt, MLflow, GH secrets
- [x] Si pass: 1 raw + 1 stg_* + 1 test dbt
- [x] Si fail: Plan B Snowflake mismo día; ADR-001b
- [x] Actions verde

### DoD Día 7 - Pipeline + MLOps mínimo
- [ ] dbt: staging + >=2 marts + schema tests + >=1 test de negocio
- [ ] Python: features + train.py (argparse, >=3 HP, seed) + MLflow (>=5 runs, params/metrics/artifact/model)
- [ ] CI: pytest + dbt compile/build
- [ ] Docs: C4 L1 + secuencia raw -> mart -> score
- [ ] make reproduce deja marts + modelo en MLflow; badge CI

### DoD Día 14 - Production-shaped + demo
- [ ] score.py batch -> tabla de predicciones
- [ ] Observability lite: logging + 1 señal drift/calidad
- [ ] Job Databricks o GHA schedule
- [ ] README + capturas + Loom 3-5 min
- [ ] Tag v0.1.0

---

## 6. Anti-patterns (bloquear)

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

Este documento es vivo. Dueño: Carlos Galvan + agente Idaho. Cambios a ASRs requieren ADR.
