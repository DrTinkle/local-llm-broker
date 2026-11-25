from __future__ import annotations

import asyncio
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
from local_llm_broker.lm_studio import BackendFailure
from local_llm_broker.scheduler import SchedulerError


class EmbedBackend:
    def __init__(self, delay: float = 0.0, fail: BackendFailure | None = None) -> None:
        self.delay = delay
        self.fail = fail
        self.active = 0
        self.max_active = 0
        self.calls: list[tuple[str, list[str]]] = []

    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        return LoadedModel(profile, profile, "instance", 32_000, 1)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "completion", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    async def embed(self, profile: str, inputs: list[str]) -> dict[str, Any]:
        self.calls.append((profile, inputs))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.fail is not None:
                raise self.fail
            return {
                "object": "list",
                "model": "embed-model",
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
                "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i, _ in enumerate(inputs)],
            }
        finally:
            self.active -= 1


def make_app(tmp_path: Path, backend: EmbedBackend, parallel: int = 2) -> Any:
    config = load_config(PRESET)
    profiles = dict(config.profiles)
    profiles["embed"] = ProfileConfig("embed", "embed-model", 2048, parallel, kind="embedding", description="Text embeddings.")
    config = replace(config, data_dir=tmp_path, auth_token_path=tmp_path / "auth.token", profiles=profiles)
    return create_app(config, adapter=backend)


async def post(app: Any, body: dict[str, Any], path: str = "/v1/embeddings") -> httpx.Response:
    headers = {"Authorization": f"Bearer {app.state.auth_token}"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
        return await client.post(path, headers=headers, json=body)


@pytest.mark.asyncio
async def test_embeddings_returns_openai_shaped_vectors(tmp_path: Path) -> None:
    backend = EmbedBackend()
    app = make_app(tmp_path, backend)
    await app.state.scheduler.start()
    try:
        response = await post(app, {"model": "embed", "input": ["one", "two"]})
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        assert body["model"] == "embed"
        assert [row["index"] for row in body["data"]] == [0, 1]
        assert body["data"][0]["embedding"] == [0.1, 0.2]
        assert backend.calls == [("embed", ["one", "two"])]
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_a_single_string_input_is_accepted(tmp_path: Path) -> None:
    backend = EmbedBackend()
    app = make_app(tmp_path, backend)
    await app.state.scheduler.start()
    try:
        response = await post(app, {"model": "embed", "input": "just one"})
        assert response.status_code == 200
        assert backend.calls == [("embed", ["just one"])]
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_embeddings_requires_the_bearer_token(tmp_path: Path) -> None:
    app = make_app(tmp_path, EmbedBackend())
    await app.state.scheduler.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            response = await client.post("/v1/embeddings", json={"model": "embed", "input": "x"})
        assert response.status_code == 401
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"model": "embed"},
        {"model": "embed", "input": []},
        {"model": "embed", "input": [1, 2]},
        {"model": "embed", "input": ["x"], "encoding_format": "base64"},
        {"model": "embed", "input": ["x"], "dimensions": 64},
        {"input": ["x"]},
    ],
)
async def test_bad_embedding_requests_are_rejected(tmp_path: Path, body: dict[str, Any]) -> None:
    backend = EmbedBackend()
    app = make_app(tmp_path, backend)
    await app.state.scheduler.start()
    try:
        response = await post(app, body)
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"
        assert backend.calls == []
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_a_chat_profile_cannot_embed_and_an_embedding_profile_cannot_chat(tmp_path: Path) -> None:
    app = make_app(tmp_path, EmbedBackend())
    await app.state.scheduler.start()
    try:
        wrong = await post(app, {"model": "4b", "input": ["x"]})
        assert wrong.status_code == 400
        assert wrong.json()["error"]["code"] == "wrong_profile_kind"
        chat = await post(
            app, {"model": "embed", "messages": [{"role": "user", "content": "hi"}]}, path="/v1/chat/completions"
        )
        assert chat.status_code == 400
        assert chat.json()["error"]["code"] == "wrong_profile_kind"
        unknown = await post(app, {"model": "nope", "input": ["x"]})
        assert unknown.status_code == 404
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_native_jobs_reject_an_embedding_profile(tmp_path: Path) -> None:
    app = make_app(tmp_path, EmbedBackend())
    await app.state.scheduler.start()
    try:
        response = await post(
            app,
            {
                "client_id": "c", "project": "p", "task_label": "t", "profile": "embed",
                "messages": [{"role": "user", "content": "hi"}], "generation": {"max_tokens": 8},
            },
            path="/jobs",
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "wrong_profile_kind"
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_requests_beyond_the_profile_limit_wait(tmp_path: Path) -> None:
    backend = EmbedBackend(delay=0.05)
    app = make_app(tmp_path, backend, parallel=2)
    await app.state.scheduler.start()
    try:
        responses = await asyncio.gather(*[post(app, {"model": "embed", "input": [f"t{i}"]}) for i in range(6)])
        assert all(r.status_code == 200 for r in responses)
        assert backend.max_active == 2
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_backend_failures_become_openai_errors(tmp_path: Path) -> None:
    backend = EmbedBackend(fail=BackendFailure("model_load_failed", "LM Studio rejected the embedding model load", retryable=True))
    app = make_app(tmp_path, backend)
    await app.state.scheduler.start()
    try:
        response = await post(app, {"model": "embed", "input": ["x"]})
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "model_load_failed"
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_embedding_requests_are_logged_without_their_text(tmp_path: Path) -> None:
    app = make_app(tmp_path, EmbedBackend())
    await app.state.scheduler.start()
    try:
        await post(app, {"model": "embed", "input": ["secret sentence about nothing"]})
        events = [e for e in app.state.telemetry.recent(limit=100) if e.get("event") == "embedding_finished"]
        assert len(events) == 1
        assert events[0]["profile"] == "embed"
        assert events[0]["inputs"] == 1
        assert "secret sentence" not in str(app.state.telemetry.recent(limit=100))
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_models_lists_the_kind_and_status_lists_resident_profiles(tmp_path: Path) -> None:
    app = make_app(tmp_path, EmbedBackend())
    await app.state.scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            models = (await client.get("/v1/models", headers=headers)).json()["data"]
            await client.post("/v1/embeddings", headers=headers, json={"model": "embed", "input": ["x"]})
            status = (await client.get("/status", headers=headers)).json()
        kinds = {m["id"]: m["kind"] for m in models}
        assert kinds["embed"] == "embedding"
        assert kinds["4b"] == "chat"
        assert status["embedding"] == {"embed": {"active": 0, "limit": 2, "requests": 1}}
    finally:
        await app.state.scheduler.shutdown()


def test_chat_resolution_never_picks_an_embedding_profile() -> None:
    config = load_config(PRESET)
    profiles = dict(config.profiles)
    profiles["embed"] = ProfileConfig("embed", "embed-model", 2048, 2, kind="embedding")
    config = replace(config, profiles=profiles)
    from local_llm_broker.routes_openai import _resolve_profile

    with pytest.raises(SchedulerError) as raised:
        _resolve_profile(config, "embed")
    assert raised.value.code == "wrong_profile_kind"
