from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_llm_broker.watch import describe_event, read_new_events, recent_events

BASE = {
    "timestamp": "2026-11-25T12:00:05.250000+00:00",
    "client_id": "agent-1",
    "project": "billing",
    "task_label": "triage-log",
    "profile": "4b-nothink",
    "job_id": "abc",
}


def line(**fields: Any) -> str:
    text = describe_event({**BASE, **fields})
    assert text is not None
    return text


def test_a_queued_job_names_who_asked_for_what() -> None:
    text = line(event="queued", state="queued")
    assert "queued" in text
    assert "agent-1 / billing / triage-log" in text
    assert "4b-nothink" in text


def test_a_finished_job_shows_time_and_tokens() -> None:
    text = line(event="inference_finished", state="succeeded", inference_seconds=6.43, input_tokens=1200, output_tokens=85)
    assert "done" in text
    assert "6.4s" in text
    assert "1200 in" in text and "85 out" in text


def test_a_failed_job_shows_the_error_code() -> None:
    text = line(event="failure", state="failed", error_code="model_load_failed")
    assert "FAILED" in text
    assert "model_load_failed" in text


def test_model_loads_are_shown_with_their_time() -> None:
    assert "loading" in line(event="load_started", tier="low")
    loaded = line(event="load_finished", tier="low", load_seconds=11.86)
    assert "loaded" in loaded and "11.9s" in loaded


def test_embedding_requests_show_text_count_and_time() -> None:
    text = line(event="embedding_finished", inputs=8, seconds=0.12)
    assert "embedded" in text
    assert "8 texts" in text
    assert "0.12s" in text
    failed = line(event="embedding_failed", inputs=1, seconds=0.5, error_code="backend_http_error")
    assert "FAILED" in failed and "backend_http_error" in failed


def test_the_time_is_shown_in_local_time() -> None:
    text = line(event="queued")
    assert text[2] == ":" and text[5] == ":"  # HH:MM:SS first


@pytest.mark.parametrize("event", ["first_token", "accepted", "tier_raised", "metadata_pruned", "something_new"])
def test_noise_events_are_skipped(event: str) -> None:
    assert describe_event({**BASE, "event": event}) is None


def test_events_without_labels_still_print() -> None:
    text = describe_event({"timestamp": BASE["timestamp"], "event": "queued", "client_id": "old", "profile": "4b"})
    assert text is not None and "old" in text


def test_read_new_events_returns_only_what_was_added(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"event": "queued", "n": 1}) + "\n", encoding="utf-8")
    events, offset = read_new_events(path, 0)
    assert [e["n"] for e in events] == [1]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "queued", "n": 2}) + "\n")
    events, offset = read_new_events(path, offset)
    assert [e["n"] for e in events] == [2]
    assert read_new_events(path, offset)[0] == []


def test_read_new_events_waits_for_a_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "queued", "n": 1}\n{"event": "que', encoding="utf-8")
    events, offset = read_new_events(path, 0)
    assert [e["n"] for e in events] == [1]
    with path.open("a", encoding="utf-8") as handle:
        handle.write('ued", "n": 2}\n')
    events, _ = read_new_events(path, offset)
    assert [e["n"] for e in events] == [2]


def test_read_new_events_starts_over_after_the_file_rotates(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"event": "queued", "n": 1}) + "\n" * 1 + json.dumps({"event": "queued", "n": 2}) + "\n", encoding="utf-8")
    _, offset = read_new_events(path, 0)
    path.write_text(json.dumps({"event": "queued", "n": 3}) + "\n", encoding="utf-8")  # a new, shorter file
    events, _ = read_new_events(path, offset)
    assert [e["n"] for e in events] == [3]


def test_a_missing_file_has_no_events(tmp_path: Path) -> None:
    assert read_new_events(tmp_path / "none.jsonl", 0) == ([], 0)


def test_recent_events_returns_the_last_few_and_the_end_offset(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps({"event": "queued", "n": n}) + "\n" for n in range(10)), encoding="utf-8")
    events, offset = recent_events(path, 3)
    assert [e["n"] for e in events] == [7, 8, 9]
    assert offset == path.stat().st_size


def test_watch_prints_recent_activity_and_exits_with_no_follow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from local_llm_broker.cli import EXIT_OK, main

    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    rows = [
        {**BASE, "event": "queued", "state": "queued"},
        {**BASE, "event": "first_token"},
        {**BASE, "event": "inference_finished", "state": "succeeded", "inference_seconds": 2.0},
    ]
    (data / "logs" / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(f'[broker]\ndata_dir = "{data.as_posix()}"\n', encoding="utf-8")
    assert main(["--config", str(config), "watch", "--no-follow"]) == EXIT_OK
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2  # first_token is noise
    assert "queued" in lines[0] and "done" in lines[1]


def test_watch_says_so_when_there_is_no_log_yet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from local_llm_broker.cli import EXIT_OK, main

    config = tmp_path / "config.toml"
    config.write_text(f'[broker]\ndata_dir = "{(tmp_path / "empty").as_posix()}"\n', encoding="utf-8")
    assert main(["--config", str(config), "watch", "--no-follow"]) == EXIT_OK
    assert "No activity yet" in capsys.readouterr().out
