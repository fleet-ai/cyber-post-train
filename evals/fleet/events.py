"""Append-only, secret-free receipts for paid Fleet launches."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": "fleet_eval_event_v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        **event,
    }
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, (json.dumps(envelope, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)
