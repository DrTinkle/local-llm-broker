from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_llm_broker.backend import LoadedModel
from local_llm_broker.contracts import ProfileConfig
from local_llm_broker.lm_studio import BackendFailure, LMStudioAdapter

EMBED_KEY = "embed-model"
PROFILES = {
    "4b": ProfileConfig("4b", "chat-model", 16_000, 4),
    "embed": ProfileConfig("embed", EMBED_KEY, 2048, 8, kind="embedding"),
}


class FakeLmStudio:
    """A tiny LM Studio: a listing of loaded instances plus load, unload and embeddings calls."""

    def __init__(self, loaded: list[tuple[str, str]] | None = None) -> None:
        self.loaded = list(loaded or [])  # (instance_id, model key)
        self.calls: list[tuple[str, str]] = []
        self.embedding_bodies: list[dict[str, Any]] = []
        self.fail_embeddings = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        if path == "/api/v1/models":
            models: list[dict[str, Any]] = [
                {"type": "llm", "key": "chat-model", "loaded_instances": []},
                {"type": "embedding", "key": EMBED_KEY, "loaded_instances": []},
            ]
            for instance_id, key in self.loaded:
                entry = next(m for m in models if m["key"] == key)
                config = {"context_length": 2048} if key == EMBED_KEY else {"context_length": 16_000, "parallel": 4}
                entry["loaded_instances"].append({"id": instance_id, "config": config})
            return httpx.Response(200, json={"models": models})
        body = json.loads(request.content) if request.content else {}
        if path == "/api/v1/models/load":
            self.loaded.append((body["model"], body["model"]))
            return httpx.Response(200, json={"instance_id": body["model"], "load_time_seconds": 0.5, "status": "loaded"})
        if path == "/api/v1/models/unload":
            self.loaded = [item for item in self.loaded if item[0] != body["instance_id"]]
            return httpx.Response(200, json={})
        if path == "/v1/embeddings":
            self.embedding_bodies.append(body)
            if self.fail_embeddings:
                self.fail_embeddings -= 1
                return httpx.Response(404, json={"error": "model not loaded"})
            items = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return httpx.Response(
                200,
                json={"object": "list", "model": body["model"], "usage": {"prompt_tokens": 0, "total_tokens": 0},
                      "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]} for i, _ in enumerate(items)]},
            )
        return httpx.Response(404, json={})


def make(fake: FakeLmStudio) -> LMStudioAdapter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake), base_url="http://test")

    async def idle(instance_id: str) -> dict[str, Any] | None:
        return {"status": "idle", "queued": 0}

    return LMStudioAdapter("http://test", PROFILES, client=client, status_reader=idle, busy_poll_interval=0.0, post_load_retry_seconds=0.0)


@pytest.mark.asyncio
async def test_embed_loads_the_model_once_and_keeps_it_loaded() -> None:
    fake = FakeLmStudio()
    adapter = make(fake)
    first = await adapter.embed("embed", ["a", "b"])
    second = await adapter.embed("embed", ["c"])
    assert [row["index"] for row in first["data"]] == [0, 1]
    assert len(second["data"]) == 1
    assert [call for call in fake.calls if call[1] == "/api/v1/models/load"] == [("POST", "/api/v1/models/load")]
    assert fake.embedding_bodies[0] == {"model": EMBED_KEY, "input": ["a", "b"]}


@pytest.mark.asyncio
async def test_embed_adopts_an_instance_that_is_already_loaded() -> None:
    fake = FakeLmStudio(loaded=[(EMBED_KEY, EMBED_KEY)])
    await make(fake).embed("embed", ["a"])
    assert all(path != "/api/v1/models/load" for _, path in fake.calls)


@pytest.mark.asyncio
async def test_embed_does_not_unload_the_chat_model() -> None:
    fake = FakeLmStudio(loaded=[("chat-1", "chat-model")])
    await make(fake).embed("embed", ["a"])
    assert ("chat-1", "chat-model") in fake.loaded
    assert all(path != "/api/v1/models/unload" for _, path in fake.calls)


@pytest.mark.asyncio
async def test_embed_loads_again_after_the_model_was_unloaded_outside_the_broker() -> None:
    fake = FakeLmStudio()
    adapter = make(fake)
    await adapter.embed("embed", ["a"])
    fake.loaded.clear()
    fake.fail_embeddings = 1
    await adapter.embed("embed", ["b"])
    loads = [call for call in fake.calls if call[1] == "/api/v1/models/load"]
    assert len(loads) == 2


@pytest.mark.asyncio
async def test_embed_rejects_a_chat_profile() -> None:
    with pytest.raises(BackendFailure) as raised:
        await make(FakeLmStudio()).embed("4b", ["a"])
    assert raised.value.code == "wrong_profile_kind"


@pytest.mark.asyncio
async def test_embed_reports_a_model_that_is_not_installed() -> None:
    fake = FakeLmStudio()
    profiles = {"embed": ProfileConfig("embed", "missing", 2048, 8, kind="embedding")}
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake), base_url="http://test")
    adapter = LMStudioAdapter("http://test", profiles, client=client)
    with pytest.raises(BackendFailure) as raised:
        await adapter.embed("embed", ["a"])
    assert raised.value.code == "model_unavailable"


@pytest.mark.asyncio
async def test_chat_load_leaves_the_resident_embedding_model_loaded() -> None:
    fake = FakeLmStudio(loaded=[(EMBED_KEY, EMBED_KEY)])
    adapter = make(fake)
    # The chat load path lists the models and loads the chat model; it must not unload the embedding instance.
    try:
        await adapter.ensure_loaded("4b")
    except BackendFailure:
        pass  # the fake gives no caps for a REST load; only the unload behavior matters here
    assert (EMBED_KEY, EMBED_KEY) in fake.loaded
    assert all(path != "/api/v1/models/unload" for _, path in fake.calls)


@pytest.mark.asyncio
async def test_still_loaded_alone_ignores_the_resident_embedding_model() -> None:
    fake = FakeLmStudio(loaded=[("chat-1", "chat-model"), (EMBED_KEY, EMBED_KEY)])
    adapter = make(fake)
    loaded = LoadedModel(profile="4b", model_id="chat-model", instance_id="chat-1", context_limit=16_000, parallel_limit=4)
    assert await adapter.still_loaded_alone(loaded) is True
