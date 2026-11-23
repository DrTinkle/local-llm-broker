from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from support import PRESET

from local_llm_broker.app import create_app
from local_llm_broker.backend import LoadedModel
from local_llm_broker.config import load_config
from local_llm_broker.contracts import JobSnapshot, JobSpec, JobState
from local_llm_broker.routes_openai import iter_sse_events
from local_llm_broker.scheduler import Scheduler
from local_llm_broker.storage import MetadataStore


class NonReadingStreamBackend:
    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def unload(self, instance_id: str) -> None:
        return None

    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        return LoadedModel(profile, profile, "instance", 16_000, 4)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("streamed jobs must not use the non-stream path")

    async def _stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        for _ in range(10_000):
            await asyncio.sleep(0)
            yield b'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'

    def stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        return self._stream(payload)


class CompatibilityBackend:
    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def unload(self, instance_id: str) -> None:
        return None

    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        return LoadedModel(profile, profile, "instance", 32_000, 1)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": payload["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }

    async def _stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        body = (
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"hé"}}]}\n\n'
            'data: [DONE]\n\n'
        ).encode()
        for index in range(0, len(body), 3):
            await asyncio.sleep(0)
            yield body[index : index + 3]

    def stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        return self._stream(payload)


@pytest.mark.asyncio
async def test_sse_parser_handles_arbitrary_byte_and_utf8_boundaries() -> None:
    async def source() -> AsyncIterator[bytes]:
        for chunk in [b"data: {\"text\":\"h", "é".encode(), b"\"}\n\n", b"data: [DONE]\n\n"]:
            yield chunk

    chunks = [piece async for piece in iter_sse_events(source())]
    assert chunks == ['data: {"text":"hé"}', "data: [DONE]"]


@pytest.mark.asyncio
async def test_sse_parser_rejects_trailing_partial_frame() -> None:
    async def source() -> AsyncIterator[bytes]:
        yield b'data: {"incomplete":true}'

    with pytest.raises(ValueError, match="incomplete"):
        async for _ in iter_sse_events(source()):
            pass


@pytest.mark.asyncio
async def test_openai_non_stream_and_stream_use_same_compatibility_route(tmp_path: Path) -> None:
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    app = create_app(config, adapter=CompatibilityBackend())
    await app.state.scheduler.start()
    try:
        token = app.state.auth_token
        headers = {"Authorization": f"Bearer {token}"}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:1235") as client:
            payload = {"model": "4b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
            non_stream = await client.post("/v1/chat/completions", headers=headers, json=payload)
            assert non_stream.status_code == 200
            assert non_stream.json()["choices"][0]["message"]["content"] == "hello"

            streamed = await client.post("/v1/chat/completions", headers=headers, json={**payload, "stream": True})
            assert streamed.status_code == 200
            assert "hé" in streamed.text
            assert "[DONE]" in streamed.text
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_openai_route_rejects_unsupported_fields_with_openai_error(tmp_path: Path) -> None:
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    app = create_app(config, adapter=CompatibilityBackend())
    await app.state.scheduler.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {app.state.auth_token}"},
                json={"model": "4b", "messages": [{"role": "user", "content": "hi"}], "unsupported_semantics": True},
            )
            assert response.status_code == 400
            assert response.json()["error"]["type"] == "invalid_request_error"
    finally:
        await app.state.scheduler.shutdown()



@pytest.mark.asyncio
async def test_stalled_stream_consumer_hits_deadline_instead_of_holding_gpu_slot(tmp_path: Path) -> None:
    config = replace(
        load_config(PRESET),
        data_dir=tmp_path,
        max_stream_buffer_bytes=64,
        stalled_stream_timeout=0.2,
        generation_timeout=5,
    )
    scheduler = Scheduler(config, NonReadingStreamBackend(), store=MetadataStore(tmp_path / "state.sqlite3"))
    await scheduler.start()
    job = await scheduler.submit(
        JobSpec("client-a", "project", "stalled", "4b", [{"role": "user", "content": "hi"}], {"stream": True}),
    )
    try:
        # The consumer never reads, so the 64-byte buffer fills on the second chunk.
        snapshot = await scheduler.wait_for_terminal(job.job_id, timeout=3)
        assert snapshot.state is JobState.FAILED
        assert snapshot.backend_state == "unknown"
        assert snapshot.error is not None
        assert snapshot.error["code"] == "generation_timeout"
    finally:
        await scheduler.cancel(job.job_id)
        await scheduler.shutdown()


class ScriptedStreamBackend:
    """Streams a fixed SSE body in small pieces, the way the upstream backend delivers it."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def unload(self, instance_id: str) -> None:
        return None

    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        return LoadedModel(profile, profile, "instance", 32_000, 1)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("streamed jobs must not use the non-stream path")

    async def _stream(self) -> AsyncIterator[bytes]:
        for index in range(0, len(self.body), 7):
            await asyncio.sleep(0)
            yield self.body[index : index + 7]

    def stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        return self._stream()


def _sse(*frames: dict[str, Any] | str) -> bytes:
    lines = []
    for frame in frames:
        data = frame if isinstance(frame, str) else json.dumps(frame, ensure_ascii=False)
        lines.append(f"data: {data}\n\n")
    return "".join(lines).encode("utf-8")


def _chunk(delta: dict[str, Any] | None = None, finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }


async def _collect(scheduler: Scheduler, job_id: str) -> bytes:
    return b"".join([piece async for piece in scheduler.stream_events(job_id)])


async def _run_streamed_job(tmp_path: Path, body: bytes) -> tuple[JobSnapshot, bytes]:
    config = replace(load_config(PRESET), data_dir=tmp_path)
    scheduler = Scheduler(config, ScriptedStreamBackend(body), store=MetadataStore(tmp_path / "state.sqlite3"))
    await scheduler.start()
    try:
        job = await scheduler.submit(
            JobSpec("client-a", "project", "task", "4b", [{"role": "user", "content": "hi"}], {"stream": True}),
        )
        # The queue record exists only while the job is known, so start reading before the job can finish.
        collected = asyncio.create_task(_collect(scheduler, job.job_id))
        await asyncio.sleep(0)
        snapshot = await scheduler.wait_for_terminal(job.job_id, timeout=5)
        return snapshot, await collected
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_streamed_job_cut_off_at_length_fails_output_truncated(tmp_path: Path) -> None:
    body = _sse(_chunk({"role": "assistant"}), _chunk({"content": "partial answer"}), _chunk({}, "length"), "[DONE]")
    snapshot, _ = await _run_streamed_job(tmp_path, body)
    assert snapshot.state is JobState.FAILED
    assert snapshot.finish_reason == "length"
    assert snapshot.error is not None
    assert snapshot.error["code"] == "output_truncated"
    assert "max_output_tokens" in snapshot.error["message"]


@pytest.mark.asyncio
async def test_streamed_job_with_only_reasoning_fails_empty_output_and_keeps_usage(tmp_path: Path) -> None:
    body = _sse(
        _chunk({"role": "assistant", "reasoning_content": "thinking about it"}),
        _chunk({"reasoning_content": " still thinking"}),
        _chunk({"content": "   "}, "stop"),
        _usage(9, 40),
        "[DONE]",
    )
    snapshot, _ = await _run_streamed_job(tmp_path, body)
    assert snapshot.state is JobState.FAILED
    assert snapshot.error is not None
    assert snapshot.error["code"] == "empty_output"
    assert snapshot.finish_reason == "stop"
    assert snapshot.input_tokens == 9
    assert snapshot.output_tokens == 40


@pytest.mark.asyncio
async def test_normal_streamed_job_succeeds_and_client_bytes_match_backend(tmp_path: Path) -> None:
    body = _sse(
        _chunk({"role": "assistant"}),
        _chunk({"content": "hé"}),
        _chunk({"content": " there"}),
        _chunk({}, "stop"),
        _usage(5, 2),
        "[DONE]",
    )
    snapshot, forwarded = await _run_streamed_job(tmp_path, body)
    assert snapshot.state is JobState.SUCCEEDED
    assert snapshot.error is None
    assert snapshot.finish_reason == "stop"
    assert snapshot.input_tokens == 5
    assert snapshot.output_tokens == 2
    assert forwarded == body


@pytest.mark.asyncio
async def test_malformed_sse_frame_is_ignored_by_completion_checks(tmp_path: Path) -> None:
    body = _sse(_chunk({"content": "ok"}), "{not json", _chunk({}, "stop"), "[DONE]")
    snapshot, forwarded = await _run_streamed_job(tmp_path, body)
    assert snapshot.state is JobState.SUCCEEDED
    assert forwarded == body
