# Lean Agent Builder Backend

Backend FastAPI del proyecto. La documentación funcional y técnica central está en [../Docs/README.md](../Docs/README.md), especialmente [manual-tecnico.md](../Docs/manual-tecnico.md) y [pasarela-pagos.md](../Docs/pasarela-pagos.md).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Las migraciones están en `alembic/versions/`; los secretos no forman parte del repositorio.
