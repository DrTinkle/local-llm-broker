from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from support import PRESET

from local_llm_broker.app import create_app
from local_llm_broker.backend import LoadedModel
from local_llm_broker.config import load_config
from local_llm_broker.contracts import ProfileConfig
from local_llm_broker.lm_studio import LMStudioAdapter

BASE = "http://127.0.0.1:1235"


class GatedBackend:
    """Fake LM Studio: completions wait for `release`, health() calls are counted."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.health_calls = 0

    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        return LoadedModel(profile, profile, "instance", 32_000, 1)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        await self.release.wait()
        return {"id": "completion", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    async def health(self) -> dict[str, Any]:
        self.health_calls += 1
        return {"state": "ready", "installed_profiles": ["4b"], "loaded_instances": 0}


class StatusStream:
    """Drives GET /status/stream on the ASGI app directly, because httpx's ASGI transport buffers whole responses."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token
        self.disconnected = asyncio.Event()
        self.chunks: asyncio.Queue[bytes] = asyncio.Queue()
        self.buffer = b""
        self.status: int | None = None
        self.headers: dict[bytes, bytes] = {}
        self.task: asyncio.Task[None] | None = None

    def start(self) -> StatusStream:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/status/stream",
            "raw_path": b"/status/stream",
            "root_path": "",
            "query_string": b"",
            "headers": [(b"host", b"127.0.0.1:1235"), (b"authorization", f"Bearer {self.token}".encode())],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 1235),
        }
        self.task = asyncio.create_task(self.app(scope, self._receive, self._send))
        return self

    async def _receive(self) -> dict[str, Any]:
        await self.disconnected.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = dict(message.get("headers", []))
        elif message["type"] == "http.response.body" and message.get("body"):
            await self.chunks.put(message["body"])

    async def next_frame(self, timeout: float = 2.0) -> str:
        while b"\n\n" not in self.buffer:
            self.buffer += await asyncio.wait_for(self.chunks.get(), timeout)
        frame, self.buffer = self.buffer.split(b"\n\n", 1)
        return frame.decode()

    async def next_snapshot(self, timeout: float = 2.0) -> dict[str, Any]:
        while True:
            frame = await self.next_frame(timeout)
            if frame.startswith("event: status_snapshot"):
                snapshot: dict[str, Any] = json.loads(frame.split("data: ", 1)[1])
                return snapshot

    async def close(self) -> None:
        self.disconnected.set()
        if self.task is not None:
            await asyncio.wait_for(self.task, 2.0)


def _app(tmp_path: Path, backend: Any, **kwargs: Any) -> Any:
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    return create_app(config, adapter=backend, **kwargs)


async def _submit(app: Any) -> str:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as client:
        response = await client.post(
            "/jobs",
            headers={"Authorization": f"Bearer {app.state.auth_token}"},
            json={
                "client_id": "client",
                "profile": "4b",
                "messages": [{"role": "user", "content": "hello"}],
                "generation": {"max_tokens": 8},
            },
        )
        assert response.status_code == 202
        return str(response.json()["job_id"])


@pytest.mark.asyncio
async def test_stream_sends_a_snapshot_at_once_then_one_per_change(tmp_path: Path) -> None:
    backend = GatedBackend()
    app = _app(tmp_path, backend, status_keepalive_seconds=60)
    await app.state.scheduler.start()
    stream = StatusStream(app, app.state.auth_token).start()
    try:
        first = await stream.next_snapshot()
        assert stream.status == 200
        assert stream.headers[b"content-type"].startswith(b"text/event-stream")
        assert first["queue_count"] == 0 and first["active_count"] == 0
        assert first["backend_health"]["state"] == "ready"
        assert app.state.status_listeners == 1

        await _submit(app)
        running = await stream.next_snapshot()
        while running["active_count"] == 0:  # the queued frame may come first
            running = await stream.next_snapshot()
        assert running["active_count"] == 1
        assert running["active_profile"] == "4b"

        backend.release.set()
        finished = await stream.next_snapshot()
        while finished["active_count"]:
            finished = await stream.next_snapshot()
        assert finished["queue_count"] == 0
    finally:
        backend.release.set()
        await stream.close()
        await app.state.scheduler.shutdown()
    assert app.state.status_listeners == 0


@pytest.mark.asyncio
async def test_quiet_stream_sends_only_keep_alive_comments_and_does_not_ask_the_backend(tmp_path: Path) -> None:
    backend = GatedBackend()
    app = _app(tmp_path, backend, status_keepalive_seconds=0.05)
    await app.state.scheduler.start()
    stream = StatusStream(app, app.state.auth_token).start()
    try:
        await stream.next_snapshot()
        calls_after_first = backend.health_calls
        for _ in range(3):
            assert await stream.next_frame() == ": keep-alive"
        assert backend.health_calls == calls_after_first
    finally:
        await stream.close()
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_client_disconnect_ends_the_stream_and_drops_the_listener(tmp_path: Path) -> None:
    app = _app(tmp_path, GatedBackend(), status_keepalive_seconds=60)
    await app.state.scheduler.start()
    streams = [StatusStream(app, app.state.auth_token).start() for _ in range(3)]
    try:
        for stream in streams:
            await stream.next_snapshot()
        assert app.state.status_listeners == 3
        await streams[0].close()
        assert streams[0].task is not None and streams[0].task.done()
        assert app.state.status_listeners == 2
    finally:
        for stream in streams[1:]:
            await stream.close()
        await app.state.scheduler.shutdown()
    assert app.state.status_listeners == 0


@pytest.mark.asyncio
async def test_broker_shutdown_ends_open_streams(tmp_path: Path) -> None:
    app = _app(tmp_path, GatedBackend(), status_keepalive_seconds=60)
    await app.state.scheduler.start()
    stream = StatusStream(app, app.state.auth_token).start()
    try:
        await stream.next_snapshot()
        await app.state.scheduler.shutdown()
        assert stream.task is not None
        await asyncio.wait_for(asyncio.shield(stream.task), 2.0)
    finally:
        await stream.close()
    assert app.state.status_listeners == 0


@pytest.mark.asyncio
async def test_stream_requires_the_bearer_token(tmp_path: Path) -> None:
    app = _app(tmp_path, GatedBackend())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as client:
        response = await client.get("/status/stream")
    assert response.status_code == 401


# --- LM Studio health cache --------------------------------------------------------------------------------------


def _listing() -> dict[str, Any]:
    return {"models": [{"type": "llm", "key": "model-4b", "loaded_instances": []}]}


def _counting_adapter(ttl: float = 30.0, fail: bool = False) -> tuple[LMStudioAdapter, list[str]]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if fail:
            return httpx.Response(503, json={}, request=request)
        if request.url.path == "/api/v1/models/unload":
            return httpx.Response(200, json={}, request=request)
        return httpx.Response(200, json=_listing(), request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    adapter = LMStudioAdapter(
        "http://test",
        {"4b": ProfileConfig("4b", "model-4b", 16_000, 4)},
        client=client,
        token_counter=lambda instance, text: asyncio.sleep(0, result=1),
        health_ttl_seconds=ttl,
    )
    return adapter, calls


def _listings(calls: list[str]) -> int:
    return sum(1 for call in calls if call == "GET /api/v1/models")


@pytest.mark.asyncio
async def test_health_asks_lm_studio_once_per_period_however_many_callers() -> None:
    adapter, calls = _counting_adapter()
    results = await asyncio.gather(*(adapter.health() for _ in range(20)))
    assert _listings(calls) == 1
    assert all(result["state"] == "ready" and result["installed_profiles"] == ["4b"] for result in results)
    await adapter.health()
    assert _listings(calls) == 1


@pytest.mark.asyncio
async def test_health_cache_expires_and_is_cleared_by_an_unload() -> None:
    adapter, calls = _counting_adapter(ttl=0.05)
    await adapter.health()
    await asyncio.sleep(0.08)
    await adapter.health()
    assert _listings(calls) == 2

    adapter.health_ttl_seconds = 30.0
    await adapter.health()
    assert _listings(calls) == 2
    await adapter.unload("instance-a")
    await adapter.health()
    assert _listings(calls) == 3


@pytest.mark.asyncio
async def test_lifecycle_listings_refresh_the_cache_and_failures_are_cached_too() -> None:
    adapter, calls = _counting_adapter()
    await adapter.list_models()  # as a job's load check would
    await adapter.health()
    assert _listings(calls) == 1

    failing, failed_calls = _counting_adapter(fail=True)
    first = await failing.health()
    second = await failing.health()
    assert first["state"] == second["state"] == "unavailable"
    assert _listings(failed_calls) == 1


@pytest.mark.asyncio
async def test_polling_status_while_idle_reaches_lm_studio_once(tmp_path: Path) -> None:
    adapter, calls = _counting_adapter()
    app = _app(tmp_path, adapter)
    await app.state.scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as client:
            for _ in range(25):
                response = await client.get("/status", headers=headers)
                assert response.status_code == 200
                assert response.json()["backend_health"]["state"] == "ready"
    finally:
        await app.state.scheduler.shutdown()
    assert _listings(calls) == 1
