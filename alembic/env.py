"""Alembic environment.

The database URL comes from application settings rather than alembic.ini,
so migrations run against the same database the service uses and there is
one place to change it. Credentials therefore stay in `.env` and out of a
committed config file.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from edgar_rag.config import get_settings
from edgar_rag.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)

# Autogenerate compares the live schema against these models.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL without connecting, for review or manual application."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Without this a column type change is silently ignored, and the
            # migration that "ran fine" leaves the schema wrong.
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
