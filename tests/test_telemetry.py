from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from local_llm_broker.telemetry import EventWriter


def test_events_redact_prompt_result_credentials_and_backend_bodies(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path / "events.jsonl")
    writer.emit(
        {
            "event": "failed",
            "job_id": "job-1",
            "prompt": "SENTINEL PROMPT",
            "result": "SENTINEL RESULT",
            "error": "backend said SENTINEL SECRET",
            "authorization": "Bearer top-secret-token",
            "message": "safe status",
        }
    )

    row = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    text = json.dumps(row)
    assert "SENTINEL" not in text
    assert "top-secret-token" not in text
    assert row["event"] == "failed"
    assert row["error_code"] == "backend_error"


def test_events_rotate_and_retain_five_files(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path / "events.jsonl", max_bytes=200, backup_count=5)
    for index in range(100):
        writer.emit({"event": "tick", "job_id": f"job-{index}", "value": "x" * 100})

    files = sorted(tmp_path.glob("events.jsonl*"))
    assert len(files) <= 6
    assert (tmp_path / "events.jsonl").exists()


def test_two_writers_on_same_path_keep_one_handler_and_write_each_event_once(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = EventWriter(path)
    second = EventWriter(path)
    logger = logging.getLogger(f"local_llm_broker.events.{path}")
    assert len(logger.handlers) == 1

    first.emit({"event": "first"})
    second.emit({"event": "second"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["first", "second"]
    first.close()
    second.close()


def test_close_removes_handlers_and_releases_file_for_deletion(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    writer.emit({"event": "tick"})
    logger = logging.getLogger(f"local_llm_broker.events.{path}")

    writer.close()

    assert logger.handlers == []
    path.unlink()
    assert not path.exists()


def test_close_twice_does_not_raise(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path / "events.jsonl")
    writer.close()
    writer.close()


def test_failed_handler_write_marks_writer_unhealthy_and_next_success_restores_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    handler = writer._handler
    assert handler is not None
    original_emit = handler.emit

    def failing_emit(record: logging.LogRecord) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(handler, "emit", failing_emit)
    writer.emit({"event": "lost"})  # must not raise
    assert writer.healthy is False

    monkeypatch.setattr(handler, "emit", original_emit)
    writer.emit({"event": "kept"})
    assert writer.healthy is True
    assert [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()] == ["kept"]
    writer.close()


def test_rotation_error_inside_handler_is_recorded_not_printed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = EventWriter(tmp_path / "events.jsonl")
    handler = writer._handler
    assert handler is not None
    original_should_roll = handler.shouldRollover

    def broken_rollover(record: logging.LogRecord) -> int:
        raise OSError("cannot rotate")

    monkeypatch.setattr(handler, "shouldRollover", broken_rollover)
    writer.emit({"event": "lost"})  # logging hands the OSError to handleError; emit still returns
    assert writer.healthy is False

    monkeypatch.setattr(handler, "shouldRollover", original_should_roll)
    writer.emit({"event": "kept"})
    assert writer.healthy is True
    writer.close()

