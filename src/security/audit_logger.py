from __future__ import annotations

import json
from datetime import datetime, timezone

from src.common.logger import get_logger

_logger = get_logger(__name__)


def log_event(call_id: str, action: str, details: dict | None = None) -> None:
    """
    Record a pipeline audit event to SQLite (and mirror to the application log).

    The SQLite write is best-effort — a DB failure never interrupts the pipeline.
    """
    # Always write to the application log as a fast, synchronous record
    entry = {
        "call_id": call_id,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }
    _logger.info("audit event=%s", json.dumps(entry))

    # Persist to SQLite via the repository layer
    try:
        from src.database.database import get_session
        from src.database.repository import log_audit_event
        with get_session() as session:
            log_audit_event(session, call_id, action, details)
    except Exception as exc:
        _logger.warning("audit DB write failed call_id=%s action=%s error=%s", call_id, action, exc)
