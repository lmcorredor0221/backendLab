from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    auth,
    commerce,
    diagram_center,
    estimation_calibration,
    health,
    knowledge_memory,
    platform_runtime,
    productization,
    runtime_settings,
    runtime_status,
    session_acp,
    session_diagrams,
    session_operations,
    sessions,
)
from app.core.config import get_settings
from app.db import create_db_and_tables
from app.services.auth_service import seed_default_user


settings = get_settings()


def run_startup_tasks() -> None:
    create_db_and_tables()
    seed_default_user()


@asynccontextmanager
async def lifespan(_: FastAPI):
    run_startup_tasks()
    yield


app = FastAPI(title=settings.app_name, debug=settings.app_debug, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(commerce.router, prefix=settings.api_v1_prefix)
app.include_router(diagram_center.router, prefix=settings.api_v1_prefix.removesuffix("/v1"))
app.include_router(health.router)
app.include_router(platform_runtime.router, prefix=settings.api_v1_prefix)
app.include_router(productization.router, prefix=settings.api_v1_prefix)
app.include_router(runtime_settings.router, prefix=settings.api_v1_prefix)
app.include_router(runtime_status.router, prefix=settings.api_v1_prefix)
app.include_router(estimation_calibration.router, prefix=settings.api_v1_prefix)
app.include_router(knowledge_memory.router, prefix=settings.api_v1_prefix)
app.include_router(sessions.router, prefix=settings.api_v1_prefix)
app.include_router(session_operations.router, prefix=settings.api_v1_prefix)
app.include_router(session_acp.router, prefix=settings.api_v1_prefix)
app.include_router(session_diagrams.router, prefix=settings.api_v1_prefix)
