from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import BinaryIO


class LockHeldError(RuntimeError):
    pass


_local_guard = threading.Lock()
_held_paths: set[str] = set()


def common_backend_lock_path(backend_url: str) -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    digest = hashlib.sha256(backend_url.rstrip("/").encode("utf-8")).hexdigest()[:24]
    return root / "LocalLLMBroker" / f"backend-{digest}.lock"


class BackendOwnerLock:
    def __init__(self, path: Path, *, common_path: Path | None = None) -> None:
        self.path = Path(path)
        self.common_path = Path(common_path) if common_path else self.path
        self._handles: list[BinaryIO] = []
        self._paths: list[str] = []

    def acquire(self) -> None:
        paths = [self.path, self.common_path]
        normalized = [os.path.normcase(os.path.abspath(path)) for path in paths]
        with _local_guard:
            if any(path in _held_paths for path in normalized):
                raise LockHeldError("another broker already owns the LM Studio backend")
            handles: list[BinaryIO] = []
            try:
                for path in paths:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    handle = open(path, "a+b")
                    self._lock_file(handle)
                    handles.append(handle)
            except Exception as exc:
                for locked_handle in handles:
                    self._unlock_file(locked_handle)
                    locked_handle.close()
                if isinstance(exc, LockHeldError):
                    raise
                raise LockHeldError("another broker already owns the LM Studio backend") from exc
            self._handles = handles
            self._paths = normalized
            _held_paths.update(normalized)

    def release(self) -> None:
        with _local_guard:
            for handle in reversed(self._handles):
                self._unlock_file(handle)
                handle.close()
            _held_paths.difference_update(self._paths)
            self._handles = []
            self._paths = []

    @staticmethod
    def _lock_file(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise LockHeldError("lock is already held") from exc
        else:  # pragma: no cover - exercised on non-Windows CI
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]

    @staticmethod
    def _unlock_file(handle: BinaryIO) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        except OSError:
            pass

