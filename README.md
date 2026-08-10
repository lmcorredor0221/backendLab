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
