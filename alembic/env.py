from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

from app.core.config import get_settings
from app import models  # noqa: F401
from app.services.deliverable_catalog import persistence as deliverable_catalog_persistence  # noqa: F401
from app.services.diagram_center import persistence as diagram_center_persistence  # noqa: F401
from app.services.product_processing import persistence as product_processing_persistence  # noqa: F401


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def get_url() -> str:
    return get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        compare_type=True,
        literal_binds=True,
        target_metadata=target_metadata,
        url=get_url(),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(compare_type=True, connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
