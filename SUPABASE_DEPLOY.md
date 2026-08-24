# Supabase Deploy Guide

This backend is built with:

- `FastAPI`
- `SQLModel`
- `Alembic`
- `psycopg`

That means the recommended production architecture is:

- `Supabase` for PostgreSQL
- `GitHub` for the source repository
- `A container host` for the FastAPI backend

Examples of compatible backend hosts include any platform that deploys a Dockerfile from GitHub.

## What goes to Supabase

Supabase is a strong fit for:

- PostgreSQL
- Backups
- SQL editor
- connection pooling

Supabase is **not** the natural place to run this exact FastAPI app as-is. To run the backend inside the Supabase platform, you would need to rewrite the API into Supabase-native functions. That is not the shortest path for this repository.

## Recommended path

1. Create a new Supabase project.
2. Copy the database password and project reference.
3. Choose the correct Postgres connection string.
4. Deploy this backend from GitHub using the included `Dockerfile`.
5. Run `alembic upgrade head` before the first production boot.
6. Start the API with `SCHEMA_MANAGEMENT_MODE=alembic`.

## Connection string for this repository

This repository uses SQLAlchemy with the `psycopg` driver, so the URL must start with:

```text
postgresql+psycopg://
```

Do not paste a raw `postgresql://` string without adapting it if your host only has `psycopg` installed.

### Option A: direct connection

Use this when your backend host supports IPv6 and you want the backend talking directly to Supabase Postgres.

```text
postgresql+psycopg://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-REF].supabase.co:5432/postgres?sslmode=require
```

### Option B: Supavisor session pooler

Use this when your backend host is IPv4-only.

```text
postgresql+psycopg://postgres.[YOUR-PROJECT-REF]:[YOUR-PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres?sslmode=require
```

For this long-lived FastAPI backend, prefer a direct connection or the session pooler. Avoid the transaction pooler unless you explicitly validate that your SQLAlchemy usage is compatible with it.

## Production environment variables

Use `backend/.env.supabase.example` as the base.

Important production values:

- `APP_DEBUG=false`
- `DATABASE_URL=...`
- `SCHEMA_MANAGEMENT_MODE=alembic`
- `CORS_ORIGINS=["https://leanagentbuilder.com","https://www.leanagentbuilder.com"]`
- `COMMERCE_PUBLIC_BASE_URL=https://leanagentbuilder.com`
- `OPENAI_API_KEY=...`

## Database migrations

This backend already contains Alembic migrations.

Before the first production start, run:

```bash
alembic upgrade head
```

This matters because the application startup intentionally checks for Alembic state when:

```text
SCHEMA_MANAGEMENT_MODE=alembic
```

If the database has not been initialized, the API will fail fast and tell you to run `alembic upgrade head`.

## Legacy schema recovery without `alembic_version`

If production already has the core application tables but is missing `alembic_version`, do not assume the safest move is a blind `alembic upgrade head`.

Recommended recovery sequence:

1. Verify the legacy schema already contains the baseline objects expected before the missing revisions.
2. If only the newest revisions are absent, apply the missing DDL in controlled SQL blocks.
3. Stamp `alembic_version` only after the schema matches the expected head.

For the Supabase production alignment executed on August 24, 2026, the reusable SQL script is:

```text
backend/scripts/prod_schema_alignment_20260824.sql
```

That recovery aligned a production database that:

- already had legacy core tables such as `workspaces`, `users`, `sessions`, `commercial_orders`, and `hotmart_webhook_events`
- already had the `stage_operations` control columns introduced in revision `20260816_0014`
- was missing `alembic_version`
- was missing the tables introduced by revisions `20260823_0015`, `20260823_0016`, and `20260824_0017`

The script:

- creates the missing commercial quota, catalog, debt, and Hotmart pending activation tables
- normalizes the Hotmart webhook unique constraint to `(event_id, event_type)`
- stamps `alembic_version` to `20260824_0017`

When running that script in Supabase SQL Editor, choose `Run without RLS` to match the backend-managed schema expected by the Alembic migrations.

## GitHub deployment flow

Because your code is already in GitHub, the simplest path is:

1. Push the backend folder with the new `Dockerfile`.
2. In your backend host, create a new service from the GitHub repository.
3. Point the service root to `backend/` if the host supports monorepos.
4. Add the production environment variables.
5. Set a release or pre-start command to:

```bash
alembic upgrade head
```

6. Start the app with the Dockerfile default command, or equivalent:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Recommended DNS and frontend alignment

If your frontend stays on Vercel under:

- `https://leanagentbuilder.com`
- `https://www.leanagentbuilder.com`

then your backend can live on a separate API host such as:

- `https://api.leanagentbuilder.com`

and `CORS_ORIGINS` should include the frontend domains.

## First-production checklist

- Supabase project created
- Production database password saved
- `DATABASE_URL` converted to `postgresql+psycopg://...`
- `sslmode=require` present
- `SCHEMA_MANAGEMENT_MODE=alembic`
- `alembic upgrade head` executed
- `OPENAI_API_KEY` configured
- `CORS_ORIGINS` updated for the real domain
- Backend host connected to GitHub
