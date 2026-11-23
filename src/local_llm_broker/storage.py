from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .contracts import TERMINAL_STATES, JobSnapshot, JobSpec, JobState, utc_now


class IdempotencyConflict(Exception):
    pass


class MetadataStore:
    def __init__(
        self,
        path: Path,
        *,
        idempotency_ttl: timedelta = timedelta(hours=24),
        digest_key: bytes | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.idempotency_ttl = idempotency_ttl
        self.digest_key = digest_key or b"local-llm-broker-test-key"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
                INSERT INTO schema_version(version)
                    SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    task_label TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    state TEXT NOT NULL,
                    backend_state TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result_expired INTEGER NOT NULL DEFAULT 0,
                    queue_seconds REAL,
                    load_seconds REAL,
                    inference_seconds REAL,
                    time_to_first_token REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    finish_reason TEXT,
                    error_json TEXT
                );
                CREATE TABLE IF NOT EXISTS transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    state TEXT NOT NULL,
                    at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS idempotency (
                    client_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(client_id, key_hash)
                );
                """
            )

    def create_job(self, job_id: str, spec: JobSpec, idempotency_key: str | None = None) -> JobSnapshot:
        now = utc_now()
        snapshot = JobSnapshot(
            job_id=job_id,
            client_id=spec.client_id,
            project=spec.project,
            task_label=spec.task_label,
            profile=spec.profile,
            state=JobState.QUEUED,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            if idempotency_key:
                key_hash = self._hash_key(idempotency_key)
                row = connection.execute(
                    "SELECT payload_digest, job_id, created_at FROM idempotency WHERE client_id=? AND key_hash=?",
                    (spec.client_id, key_hash),
                ).fetchone()
                if row and self._parse_time(row["created_at"]) + self.idempotency_ttl > now:
                    if row["payload_digest"] != spec.payload_digest(self.digest_key):
                        raise IdempotencyConflict("idempotency key was reused with a different job")
                    existing = self.get_job(str(row["job_id"]))
                    if existing is None:
                        raise IdempotencyConflict("idempotency record points to missing job")
                    return existing
            connection.execute(
                """INSERT INTO jobs(job_id, client_id, project, task_label, profile, state,
                   backend_state, request_id, created_at, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    snapshot.job_id,
                    snapshot.client_id,
                    snapshot.project,
                    snapshot.task_label,
                    snapshot.profile,
                    snapshot.state.value,
                    snapshot.backend_state,
                    snapshot.request_id,
                    snapshot.created_at.isoformat(),
                    snapshot.updated_at.isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO transitions(job_id, state, at) VALUES(?,?,?)",
                (job_id, JobState.QUEUED.value, now.isoformat()),
            )
            if idempotency_key:
                connection.execute(
                    "INSERT OR REPLACE INTO idempotency(client_id,key_hash,payload_digest,job_id,created_at) VALUES(?,?,?,?,?)",
                    (spec.client_id, self._hash_key(idempotency_key), spec.payload_digest(self.digest_key), job_id, now.isoformat()),
                )
        return snapshot

    def get_job(self, job_id: str) -> JobSnapshot | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._snapshot(row)

    def transition(self, job_id: str, state: JobState, **updates: Any) -> JobSnapshot:
        now = utc_now()
        allowed = {
            "backend_state",
            "result_expired",
            "queue_seconds",
            "load_seconds",
            "inference_seconds",
            "time_to_first_token",
            "input_tokens",
            "output_tokens",
            "finish_reason",
            "error_json",
        }
        fields = {key: value for key, value in updates.items() if key in allowed}
        fields.update({"state": state.value, "updated_at": now.isoformat()})
        assignments = ", ".join(f"{key}=?" for key in fields)
        values = list(fields.values()) + [job_id]
        with self._connect() as connection:
            connection.execute(f"UPDATE jobs SET {assignments} WHERE job_id=?", values)
            connection.execute("INSERT INTO transitions(job_id,state,at) VALUES(?,?,?)", (job_id, state.value, now.isoformat()))
        result = self.get_job(job_id)
        if result is None:
            raise KeyError(job_id)
        return result

    def get_idempotency(self, client_id: str, key: str, spec: JobSpec | None = None) -> JobSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM idempotency WHERE client_id=? AND key_hash=?",
                (client_id, self._hash_key(key)),
            ).fetchone()
        if row is None or self._parse_time(row["created_at"]) + self.idempotency_ttl <= utc_now():
            return None
        if spec is not None and row["payload_digest"] != spec.payload_digest(self.digest_key):
            raise IdempotencyConflict("idempotency key was reused with a different job")
        return self.get_job(str(row["job_id"]))

    def set_idempotency_created_at(self, client_id: str, key: str, at: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE idempotency SET created_at=? WHERE client_id=? AND key_hash=?",
                (at.isoformat(), client_id, self._hash_key(key)),
            )

    def mark_unfinished_interrupted(self) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id FROM jobs WHERE state NOT IN (?,?,?,?)",
                tuple(state.value for state in TERMINAL_STATES),
            ).fetchall()
        for row in rows:
            self.transition(str(row["job_id"]), JobState.INTERRUPTED, backend_state="unknown")
        return len(rows)

    def prune(self, before: datetime) -> int:
        """Delete terminal jobs last updated before `before`, with their transitions and idempotency rows.

        Child rows go first because transitions and idempotency reference jobs. Expired idempotency rows go too.
        Returns the number of jobs deleted.
        """
        marks = ",".join("?" for _ in TERMINAL_STATES)
        doomed = f"SELECT job_id FROM jobs WHERE state IN ({marks}) AND updated_at < ?"
        params = (*[state.value for state in TERMINAL_STATES], before.isoformat())
        with self._connect() as connection:
            connection.execute(f"DELETE FROM transitions WHERE job_id IN ({doomed})", params)
            connection.execute(f"DELETE FROM idempotency WHERE job_id IN ({doomed})", params)
            cursor = connection.execute(f"DELETE FROM jobs WHERE job_id IN ({doomed})", params)
            connection.execute(
                "DELETE FROM idempotency WHERE created_at < ?",
                ((datetime.now(UTC) - self.idempotency_ttl).isoformat(),),
            )
            return cursor.rowcount

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            counts = connection.execute("SELECT state, COUNT(*) AS count FROM jobs GROUP BY state").fetchall()
            timings = connection.execute(
                "SELECT AVG(queue_seconds), AVG(load_seconds), AVG(inference_seconds) FROM jobs WHERE state=?",
                (JobState.SUCCEEDED.value,),
            ).fetchone()
        return {
            "counts": {str(row["state"]): int(row["count"]) for row in counts},
            "successful_average_seconds": {
                "queue": timings[0],
                "load": timings[1],
                "inference": timings[2],
            },
        }

    def _snapshot(self, row: sqlite3.Row) -> JobSnapshot:
        error = json.loads(row["error_json"]) if row["error_json"] else None
        return JobSnapshot(
            job_id=str(row["job_id"]),
            client_id=str(row["client_id"]),
            project=str(row["project"]),
            task_label=str(row["task_label"]),
            profile=str(row["profile"]),
            state=JobState(str(row["state"])),
            created_at=self._parse_time(row["created_at"]),
            updated_at=self._parse_time(row["updated_at"]),
            backend_state=str(row["backend_state"]),
            result_expired=bool(row["result_expired"]),
            queue_seconds=row["queue_seconds"],
            load_seconds=row["load_seconds"],
            inference_seconds=row["inference_seconds"],
            time_to_first_token=row["time_to_first_token"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            finish_reason=row["finish_reason"],
            error=error,
            request_id=str(row["request_id"]),
        )

    @staticmethod
    def _parse_time(value: str) -> datetime:
        return datetime.fromisoformat(value)

    @staticmethod
    def _hash_key(key: str) -> str:
        import hashlib

        return hashlib.sha256(key.encode("utf-8")).hexdigest()

