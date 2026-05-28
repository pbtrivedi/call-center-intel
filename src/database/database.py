from __future__ import annotations

import threading
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base

_lock = threading.Lock()
_SessionFactory: sessionmaker | None = None


def get_session() -> Session:
    """
    Return a new SQLAlchemy session from the cached factory.

    Creates the engine, tables, and sessionmaker on the first call; subsequent
    calls reuse the cached factory. Thread-safe via a module-level lock.
    """
    global _SessionFactory
    if _SessionFactory is None:
        with _lock:
            if _SessionFactory is None:
                _SessionFactory = _build_factory()
    return _SessionFactory()


def _build_factory() -> sessionmaker:
    from src.config.loader import get_settings
    db_path = get_settings().db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _reset_session_factory() -> None:
    """Reset the cached factory. For use in tests only."""
    global _SessionFactory
    _SessionFactory = None
