from __future__ import annotations

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
from local_llm_broker.routes_openai import _resolve_profile
from local_llm_broker.scheduler import SchedulerError


class ApiBackend:
    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        return LoadedModel(profile, profile, "instance", 32_000, 1)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "completion", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


@pytest.mark.asyncio
async def test_idempotent_replay_of_post_jobs_emits_one_accepted_event(tmp_path: Path) -> None:
    config = load_config(PRESET)
    config = replace(config, data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    app = create_app(config, adapter=ApiBackend())
    scheduler = app.state.scheduler
    await scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}", "Idempotency-Key": "replay-key"}
        body = {
            "client_id": "client",
            "project": "project",
            "task_label": "task",
            "profile": "4b",
            "messages": [{"role": "user", "content": "hello"}],
            "generation": {"max_tokens": 8},
        }
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            first = await client.post("/jobs", headers=headers, json=body)
            second = await client.post("/jobs", headers=headers, json=body)
        assert first.status_code == 202
        assert second.status_code == 202
        assert second.json()["job_id"] == first.json()["job_id"]
        accepted = [event["job_id"] for event in app.state.telemetry.recent(limit=500) if event.get("event") == "accepted"]
        assert accepted == [first.json()["job_id"]]
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_native_api_health_is_public_and_jobs_require_bearer(tmp_path: Path) -> None:
    config = load_config(PRESET)
    config = replace(config, data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    app = create_app(config, adapter=ApiBackend())
    scheduler = app.state.scheduler
    await scheduler.start()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:1235") as client:
            health = await client.get("/health")
            assert health.status_code == 200
            unauthorized = await client.get("/status")
            assert unauthorized.status_code == 401
            token = app.state.auth_token
            submitted = await client.post(
                "/jobs",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "client_id": "client",
                    "project": "project",
                    "task_label": "task",
                    "profile": "4b",
                    "messages": [{"role": "user", "content": "hello"}],
                    "generation": {"max_tokens": 8},
                },
            )
            assert submitted.status_code == 202
            job_id = submitted.json()["job_id"]
            job = await client.get(f"/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
            assert job.status_code == 200
            assert "hello" not in job.text
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_api_rejects_non_loopback_host_and_oversized_body(tmp_path: Path) -> None:
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token", max_body_bytes=64)
    app = create_app(config, adapter=ApiBackend())
    await app.state.scheduler.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            bad_host = await client.get("/health", headers={"host": "evil.example"})
            assert bad_host.status_code == 400
            oversized = await client.post("/health", content="x" * 100, headers={"content-type": "text/plain"})
            assert oversized.status_code == 413

            async def chunks() -> AsyncIterator[bytes]:
                yield b"x" * 100

            chunked = await client.post("/health", content=chunks(), headers={"content-type": "text/plain"})
            assert chunked.status_code == 413
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_status_reports_degraded_reason_and_null_when_healthy(tmp_path: Path) -> None:
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    app = create_app(config, adapter=ApiBackend())
    scheduler = app.state.scheduler
    await scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            healthy = await client.get("/status", headers=headers)
            assert healthy.status_code == 200
            assert "degraded" in healthy.json()
            assert healthy.json()["degraded"] is None

            scheduler.degraded_reason = "storage_unavailable"
            degraded = await client.get("/status", headers=headers)
            assert degraded.json()["degraded"] == "storage_unavailable"
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_openai_vision_message_is_accepted_on_9b_and_rejected_on_4b(tmp_path: Path) -> None:
    # A 3 MB request body: larger than the old 2 MiB limit, well under the 10 MiB limit.
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    app = create_app(config, adapter=ApiBackend())
    await app.state.scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}"}
        image = "data:image/png;base64," + "A" * (3 * 1024 * 1024)
        content = [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": image}}]
        message = {"role": "user", "content": content}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            accepted = await client.post("/v1/chat/completions", headers=headers, json={"model": "9b", "messages": [message], "max_tokens": 8})
            assert accepted.status_code == 200
            assert accepted.json()["choices"][0]["message"]["content"] == "ok"
            rejected = await client.post("/v1/chat/completions", headers=headers, json={"model": "4b", "messages": [message], "max_tokens": 8})
            assert rejected.status_code == 400
            assert rejected.json()["error"]["code"] == "unsupported_input"
            assert "does not support vision inputs" in rejected.json()["error"]["message"]
    finally:
        await app.state.scheduler.shutdown()


class RecordingBackend(ApiBackend):
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        return await super().complete(payload)


@pytest.mark.asyncio
async def test_canonical_9b_model_id_with_an_image_uses_the_thinking_profile(tmp_path: Path) -> None:
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    backend = RecordingBackend()
    app = create_app(config, adapter=backend)
    await app.state.scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}"}
        content = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"model": "qwen/qwen3.5-9b", "messages": [{"role": "user", "content": content}], "max_tokens": 8},
            )
            assert response.status_code == 200
    finally:
        await app.state.scheduler.shutdown()
    assert len(backend.payloads) == 1
    # The thinking profile sends no reasoning_effort; the -nothink twin always sends "none".
    assert "reasoning_effort" not in backend.payloads[0]


@pytest.mark.asyncio
async def test_malformed_content_length_returns_400_and_bearer_must_match(tmp_path: Path) -> None:
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    app = create_app(config, adapter=ApiBackend())
    await app.state.scheduler.start()
    try:
        token = app.state.auth_token
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            malformed = await client.post("/health", content=b"x", headers={"content-length": "abc"})
            assert malformed.status_code == 400
            assert malformed.json()["error"]["code"] == "invalid_content_length"
            negative = await client.post("/health", content=b"x", headers={"content-length": "-5"})
            assert negative.status_code == 400
            assert negative.json()["error"]["code"] == "invalid_content_length"

            wrong = await client.get("/status", headers={"Authorization": "Bearer not-the-token"})
            assert wrong.status_code == 401
            correct = await client.get("/status", headers={"Authorization": f"Bearer {token}"})
            assert correct.status_code == 200
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_openai_null_generation_fields_fall_back_to_defaults(tmp_path: Path) -> None:
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    app = create_app(config, adapter=ApiBackend())
    await app.state.scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}"}
        message = {"role": "user", "content": "hello"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"model": "9b", "messages": [message], "max_tokens": None, "temperature": None},
            )
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "ok"
    finally:
        await app.state.scheduler.shutdown()


def test_canonical_model_ids_resolve_to_thinking_profiles_and_aliases_pick_the_exact_profile(tmp_path: Path) -> None:
    config = load_config(PRESET)

    assert _resolve_profile(config, "qwen/qwen3.5-9b") == "9b"
    assert _resolve_profile(config, "qwen/qwen3.6-35b-a3b") == "35b"
    assert _resolve_profile(config, "qwen/qwen3-4b-thinking-2507") == "4b"
    assert _resolve_profile(config, "qwen/qwen3-4b-2507") == "4b-nothink"
    assert _resolve_profile(config, "9b") == "9b"
    assert _resolve_profile(config, "9b-nothink") == "9b-nothink"
    assert _resolve_profile(config, "35b-nothink") == "35b-nothink"
    with pytest.raises(SchedulerError) as unknown:
        _resolve_profile(config, "qwen/qwen3.5-27b")
    assert unknown.value.code == "model_not_found"


@pytest.mark.asyncio
async def test_models_endpoint_reports_thinks_vision_description(tmp_path: Path) -> None:
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    app = create_app(config, adapter=ApiBackend())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
        response = await client.get("/v1/models", headers={"Authorization": f"Bearer {app.state.auth_token}"})
    assert response.status_code == 200
    entries = {entry["id"]: entry for entry in response.json()["data"]}
    assert entries["9b"]["thinks"] is True
    assert entries["9b"]["vision"] is True
    assert entries["9b"]["description"]
    assert entries["4b-nothink"]["thinks"] is False
    assert entries["4b-nothink"]["vision"] is False
    assert {"model_id", "parallel", "tiers", "owned_by", "object"} <= set(entries["9b"])
