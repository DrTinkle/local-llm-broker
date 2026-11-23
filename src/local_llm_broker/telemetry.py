from __future__ import annotations

import json
import logging
import re
from collections import deque
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_DROP_KEYS = {
    "prompt",
    "prompts",
    "result",
    "output",
    "content",
    "messages",
    "file",
    "files",
    "authorization",
    "token",
    "access_token",
    "argv",
    "args",
}


class _EventFileHandler(RotatingFileHandler):
    """Records a failed file write instead of letting logging print and forget it."""

    def __init__(self, path: Path, *, max_bytes: int, backup_count: int) -> None:
        super().__init__(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        self.failed = False

    def handleError(self, record: logging.LogRecord) -> None:
        self.failed = True


class EventWriter:
    def __init__(self, path: Path, *, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._logger = logging.getLogger(f"local_llm_broker.events.{self.path}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        for existing in list(self._logger.handlers):
            self._logger.removeHandler(existing)
            existing.close()
        handler = _EventFileHandler(self.path, max_bytes=max_bytes, backup_count=backup_count)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._handler: _EventFileHandler | None = handler
        self._healthy = True

    @property
    def healthy(self) -> bool:
        """False after a failed write, until a later write succeeds."""
        return self._healthy

    def close(self) -> None:
        handler = self._handler
        if handler is None:
            return
        self._handler = None
        self._logger.removeHandler(handler)
        handler.close()

    def emit(self, event: dict[str, Any]) -> None:
        """Write one event. Never raises; a failed write marks the writer unhealthy."""
        handler = self._handler
        try:
            safe = self._sanitize(event)
            safe.setdefault("timestamp", datetime.now(UTC).isoformat())
            line = json.dumps(safe, sort_keys=True, ensure_ascii=False)
            if handler is not None:
                handler.failed = False
            self._logger.info(line)
            for current in self._logger.handlers:
                current.flush()
            self._healthy = handler is not None and not handler.failed
        except Exception:
            self._healthy = False

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: deque[dict[str, Any]] = deque(maxlen=max(1, min(limit, 1000)))
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return list(rows)

    def _sanitize(self, value: Any, key: str | None = None) -> Any:
        if key and key.lower() in _DROP_KEYS:
            return None
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for child_key, child_value in value.items():
                cleaned = self._sanitize(child_value, str(child_key))
                if cleaned is not None:
                    result[str(child_key)] = cleaned
            if "error" in result:
                result["error_code"] = "backend_error"
                result.pop("error", None)
            return result
        if isinstance(value, (list, tuple)):
            return [self._sanitize(item) for item in value]
        if isinstance(value, str):
            value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~-]+", "Bearer [redacted]", value)
            value = re.sub(r"(?i)(api[-_ ]?key|secret|password)\s*[:=]\s*\S+", r"\1=[redacted]", value)
            if len(value) > 256:
                return value[:256] + "…"
            if "sentinel" in value.lower():
                return "[redacted]"
        return value

