from __future__ import annotations

import json
from datetime import datetime, timezone

from src.common.logger import get_logger

_logger = get_logger(__name__)


def log_event(call_id: str, action: str, details: dict | None = None) -> None:
    """
    Record a pipeline audit event.

    Phase 3: logs to the application logger (structured key=value).
    Phase 6 will swap this stub for an INSERT-only SQLite write via the repository layer.
    """
    entry = {
        "call_id": call_id,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    _logger.info("audit event=%s", json.dumps(entry))
