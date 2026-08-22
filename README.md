# Lean Agent Builder Backend

Backend FastAPI para Lean Agent Builder.

## Setup local

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Tests

```powershell
.venv\Scripts\python -m pytest
```

## Notas

- `.env`, bases locales, logs, `runtime/`, `tmp/` y `.venv/` no se versionan.
- Las migraciones Alembic y pruebas forman parte del repositorio.
- Para usar Supabase como Postgres de produccion, revisa [SUPABASE_DEPLOY.md](./SUPABASE_DEPLOY.md).

## Deploy desde GitHub

La arquitectura recomendada para este backend es:

- `Supabase`: base de datos PostgreSQL
- `GitHub`: fuente del codigo
- `Host de contenedores`: despliegue del backend FastAPI

Supabase no es el host natural de este backend FastAPI largo-vivo. Lo normal es desplegar el contenedor desde GitHub en un proveedor externo y apuntar `DATABASE_URL` a Supabase.
