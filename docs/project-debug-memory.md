# Lean Agent Builder Project Debug Memory

## Scope

Operational memory for reproducing and validating Blueprint Pro and ACP issues in local without re-learning the environment on every session.

Last updated: 2026-08-27

## Repositories

- Backend repo: `C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend`
- Frontend repo: `C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\frontend`
- Backend remote: `backendLab.git`
- Frontend remote: `LABfrontend.git`

## Local runtime

- Backend base URL: `http://127.0.0.1:8000`
- Frontend base URL: `http://127.0.0.1:3200`
- Frontend proxies `/api/*` to `BACKEND_API_ORIGIN`
- Local admin defaults are resolved from:
  - `LOCAL_ADMIN_EMAIL` in `backend/.env`
  - `NEXT_PUBLIC_LOCAL_ADMIN_EMAIL` in `frontend/.env.local`
  - `NEXT_PUBLIC_LOCAL_ADMIN_PASSWORD` in `frontend/.env.local`

## Local database

- Local database is standard PostgreSQL, not local Supabase.
- Current backend `.env` points to the local Postgres database used for LAB debugging.
- Do not copy secrets into docs or commits. Read connection values from `backend/.env` when needed.
- Useful local verification commands:

```powershell
cd C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend
.venv\Scripts\python.exe -c "from app.core.config import settings; print(settings.database_url)"
```

## Production project snapshot to local

Use the skill-backed deterministic flow:

- Skill: `C:\Users\Messi\.agents\skills\supabase-project-prod-to-local\SKILL.md`
- Workflow: `C:\Users\Messi\.agents\skills\supabase-project-prod-to-local\references\workflow.md`

Standard command:

```powershell
C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend\.venv\Scripts\python.exe `
  C:\Users\Messi\.agents\skills\supabase-project-prod-to-local\scripts\copy_project_snapshot.py `
  --source-database-url "<PROD_SUPABASE_DATABASE_URL>" `
  --project-id "<PROJECT_UUID>"
```

Notes:

- Treat `/projects/<uuid>` as the root `sessions.id` unless code inspection proves otherwise.
- This is a project-scoped snapshot, not a full environment clone.
- By default it copies session-scoped rows such as session, logs, artifacts, stage operations, diagrams and LLM ledger.
- It does not guarantee parity for encrypted secrets, platform seeds or `knowledge_documents`.

### SQL Editor fallback

When the production extraction comes from a manual Supabase SQL Editor export instead of a direct `DATABASE_URL`, use:

- `backend/scripts/materialize_sql_editor_snapshot.py`
- `backend/scripts/import_materialized_project_snapshot.py`
- `backend/scripts/sql/export_project_snapshot.sql`

Suggested sequence:

```powershell
cd C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend
.venv\Scripts\python.exe scripts/materialize_sql_editor_snapshot.py --input <raw-json-from-sql-editor> --output-dir runtime\manual-snapshot
.venv\Scripts\python.exe scripts/import_materialized_project_snapshot.py --snapshot-dir runtime\manual-snapshot
```

## Main validation script

End-to-end local validation script:

- `C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend\scripts\release_local_validation.py`

What it covers now:

- health
- login
- session creation
- discovery, canvas and blueprint generation
- approval resolution
- evaluation bootstrap and evaluation
- async estimate completion
- ACP tier enablement
- ACP preview, generation, question answering, regeneration and export

Evidence directory from the latest successful run on 2026-08-27:

- `C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend\runtime\release-stage6\20260827-031845`

Important artifacts from that run:

- `acp-preview-initial.json`
- `acp-generated-initial.json`
- `acp-readiness-final.json`
- `acp-validation-final.json`
- final ACP ZIP export

## Key behavioral findings

### ACP questions are dynamic

- The ACP question set is not immutable.
- After answering or deferring a question, the current preview can be recalculated.
- A question that disappears from the active preview is treated as resolved history, not as an open question that must still accept input.
- Tests and UI logic must follow the live question set, not the initial snapshot only.

### Deferred questions

- Deferred ACP questions are valid non-open outcomes.
- They must not continue blocking export when validation allows package generation.
- Deferred decisions should remain traceable for implementation.

### Readiness gating

- `can_start_build` depends on validation plus unanswered blocking items.
- If validation does not allow build, readiness must stay blocked even if question counts look clean.

### Export expectations

- ACP exports now expose paths like:
  - `ACP/construction-readiness/required-api-contracts.yaml`
  - `ACP/tools/external/*`
  - `ACP/tools/permissions.yaml`
- Do not rely on the previous `ACP/tools/contracts/*` path pattern.

## Commands used most often

Backend targeted tests:

```powershell
cd C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend
.venv\Scripts\python.exe -m pytest tests/test_acp_continuity.py tests/test_attention_v2_aggregator.py tests/test_deliverable_quality_dependency.py tests/test_product_processing_policy.py tests/test_stage_operations_api.py -x -vv
.venv\Scripts\python.exe -m pytest tests/test_sessions_api.py -k "backend_smoke_flow_covers_login_discovery_blueprint_evaluation_and_acp_export" -x -vv
.venv\Scripts\python.exe -m pytest tests/test_sessions_api.py -k "acp_routes_generate_preview_validate_file_and_zip or acp_workspace_autobootstraps_missing_evaluation_assets or acp_questions_can_be_answered_and_reinjected_into_regeneration or acp_questions_can_be_delegated_without_manual_answer_text or package_preview_and_canonical_exports_detect_consistency_drift_against_approved_chain or acp_design_only_profile_closes_independently_from_extended_profile" -x -vv
```

Frontend targeted tests:

```powershell
cd C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\frontend
npm.cmd test -- --run src/components/lean/auth-pages.test.tsx src/features/acp/acp-adapter.test.ts src/features/product-experience/saas/saas-views.test.tsx --reporter=verbose
```

## Safe working rules

- Never modify production data when reproducing bugs.
- If local behavior differs from the imported project snapshot, check seed drift, workspace runtime settings and knowledge corpus parity before blaming the project row data.
- Do not commit `.env`, runtime snapshots or imported production secrets.
