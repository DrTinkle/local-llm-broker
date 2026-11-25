"""A live, human-readable view of what the broker is doing, for `llm-broker watch`.

It reads the event log (`logs/events.jsonl` in the data folder) and prints one line per request step: who asked
(client / project / task), which profile, and how it went. The log never holds prompts or answers, so neither does
this view.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


def describe_event(event: dict[str, Any]) -> str | None:
    """One line for an event worth showing, or None for events that only add noise."""
    name = str(event.get("event", ""))
    who = _who(event)
    profile = str(event.get("profile", ""))
    stamp = _clock(event.get("timestamp"))
    if name == "queued":
        text = f"queued    {who}  ->  {profile}"
    elif name == "load_started":
        text = f"loading   {profile}" + _tier(event)
    elif name == "load_finished":
        text = f"loaded    {profile}" + _tier(event) + f" in {_seconds(event.get('load_seconds'))}"
    elif name == "inference_started":
        text = f"started   {who}  ->  {profile}"
    elif name == "inference_finished":
        state = str(event.get("state", ""))
        label = "done     " if state == "succeeded" else f"{state:<9} "
        text = f"{label}{who}  ->  {profile}  {_seconds(event.get('inference_seconds'))}{_tokens(event)}"
    elif name == "failure":
        text = f"FAILED    {who}  ->  {profile}  {event.get('error_code', '')}"
    elif name == "cancellation":
        text = f"cancelled {who}  ->  {profile}  ({event.get('phase', '')})"
    elif name == "embedding_finished":
        text = f"embedded  {who}  ->  {profile}  {event.get('inputs', '?')} texts {_seconds(event.get('seconds'))}"
    elif name == "embedding_failed":
        text = f"FAILED    {who}  ->  {profile}  {event.get('inputs', '?')} texts  {event.get('error_code', '')}"
    else:
        return None
    return f"{stamp}  {text}".rstrip()


def _who(event: dict[str, Any]) -> str:
    parts = [str(event.get(key) or "") for key in ("client_id", "project", "task_label")]
    return " / ".join(part for part in parts if part) or "unknown"


def _tier(event: dict[str, Any]) -> str:
    return f" ({event['tier']} tier)" if event.get("tier") else ""


def _seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "?"
    return f"{value:.2f}s" if value < 1 else f"{value:.1f}s"


def _tokens(event: dict[str, Any]) -> str:
    if event.get("input_tokens") is None and event.get("output_tokens") is None:
        return ""
    return f"  {event.get('input_tokens', '?')} in / {event.get('output_tokens', '?')} out"


def _clock(value: Any) -> str:
    try:
        return datetime.fromisoformat(str(value)).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return "--:--:--"


def read_new_events(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Complete events written after `offset`, and the new offset.

    The file is opened and closed on each call, so the broker can rotate it freely. A file shorter than the offset
    was replaced, so reading starts again from its beginning. A half-written last line is left for the next call.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return [], 0
    if size < offset:
        offset = 0
    if size == offset:
        return [], offset
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    complete = data.rfind(b"\n") + 1
    events: list[dict[str, Any]] = []
    for raw in data[:complete].splitlines():
        try:
            value = json.loads(raw)
        except ValueError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events, offset + complete


def recent_events(path: Path, count: int) -> tuple[list[dict[str, Any]], int]:
    """The last `count` events, and the offset where new ones will start."""
    events, offset = read_new_events(path, 0)
    return list(deque(events, maxlen=max(0, count))), offset
