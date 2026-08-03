"""Database engine and session management."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from edgar_rag.config import Settings, get_settings
from edgar_rag.db.models import Base

logger = logging.getLogger(__name__)


@lru_cache
def get_engine(database_url: str | None = None) -> Engine:
    """Engine for `database_url`, cached per URL."""
    settings = get_settings()
    url = database_url or settings.database_url

    # SQLite (used by the test suite) needs its own connect args and does
    # not support pool sizing.
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})

    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)


def create_tables(engine: Engine | None = None) -> None:
    """Create any missing tables.

    Enough for development and tests; Alembic owns schema changes once the
    database holds data worth migrating.
    """
    Base.metadata.create_all(bind=engine or get_engine())


@contextmanager
def session_scope(
    settings: Settings | None = None, engine: Engine | None = None
) -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on error."""
    settings = settings or get_settings()
    factory = get_session_factory(engine or get_engine(settings.database_url))
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
