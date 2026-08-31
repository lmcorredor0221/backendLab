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

## Local backend against Supabase

Use this mode when the current production state itself matters more than safe replay, for example when the project `84e2cdc7-5352-40d1-bd06-7795021d4b2d` shows blockers that are not reproduced from a stale local snapshot.

Files:

- Example env: `C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend\.env.supabase.local.example`
- Starter script: `C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend\scripts\start-supabase-debug.ps1`

Recommended sequence:

```powershell
cd C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend
Copy-Item .env.supabase.local.example .env.supabase.local
# Edit only the values that are secret or environment-specific.
.\scripts\start-supabase-debug.ps1 -CheckOnly
.\scripts\start-supabase-debug.ps1
```

Operational rules:

- This mode points the local backend to the shared Supabase database through `DATABASE_URL`.
- Keep `SCHEMA_MANAGEMENT_MODE=alembic`.
- Keep `RUNTIME_BOOTSTRAP_ENABLED=false`.
- Keep `KNOWLEDGE_REPO_AUTOSYNC_ENABLED=false`.
- For browser-driven local debugging against Supabase, start with `DATABASE_POOL_SIZE=5` and `DATABASE_MAX_OVERFLOW=5` so auth, snapshot, attention and export requests do not starve each other.
- If you only need isolated SQL inspection scripts, you can lower the pool again deliberately.
- Do not use this mode for destructive cleanup or broad reprocesamientos unless you intentionally want to mutate shared production data.

What this mode is good for:

- inspect the exact live state of blockers, checkpoints, jobs, approvals and runtime operations
- validate whether a production project is actually resuming `Memoria` after `Herramientas`
- compare frontend representation against the real backend state without first copying rows into local

## Production project snapshot to local

Use this mode when you need safe local writes and repeatable reprocesamiento after capturing the current production state.

Files:

- Wrapper script: `C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend\scripts\copy-project-snapshot.ps1`
- Skill: `C:\Users\Messi\.agents\skills\supabase-project-prod-to-local\SKILL.md`
- Workflow: `C:\Users\Messi\.agents\skills\supabase-project-prod-to-local\references\workflow.md`

Wrapper example:

```powershell
cd C:\Users\Messi\OneDrive\Documentos\Agentes\Asistente\backend
.\scripts\copy-project-snapshot.ps1 `
  -ProjectId "84e2cdc7-5352-40d1-bd06-7795021d4b2d" `
  -SourceDatabaseUrl "<PROD_SUPABASE_DATABASE_URL>"
```

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
- If the live issue depends on current blockers in Attention, workspace-wide runtime state or product build rows not present in the snapshot scope, diagnose first with the live Supabase mode and only then replay locally.

## Suggested path for project `84e2cdc7-5352-40d1-bd06-7795021d4b2d`

When the symptom is "the ReAct correction from `Memoria` is not being resumed correctly and four blockers remain visible", follow this order:

1. Start the backend locally against Supabase with `.\scripts\start-supabase-debug.ps1`.
2. Open the local frontend and inspect the same production project state through the local backend.
3. Capture the exact blockers, stage operations, approvals and current snapshot payloads.
4. Copy the project into the local database with `.\scripts\copy-project-snapshot.ps1`.
5. Re-run the scenario locally only after the live state has been captured, so local reprocesamientos do not touch the shared production rows.

This split avoids a common debugging trap:

- live mode tells us whether production is currently wrong
- snapshot mode lets us iterate fixes safely after we already know what production looked like

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
