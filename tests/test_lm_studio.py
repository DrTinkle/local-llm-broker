from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from local_llm_broker import lm_studio
from local_llm_broker.backend import LoadedModel
from local_llm_broker.contracts import ProfileConfig
from local_llm_broker.lm_studio import (
    BackendFailure,
    LMStudioAdapter,
    _parse_lms_status,
)


def response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload, request=httpx.Request("GET", "http://test"))


async def idle_status(instance_id: str) -> dict[str, Any] | None:
    return {"status": "idle", "queued": 0}


def status_sequence(events: list[str], *statuses: dict[str, Any] | None) -> Callable[[str], Awaitable[dict[str, Any] | None]]:
    # Each reading is recorded in `events`; the last reading repeats once the list runs out.
    queue = list(statuses)

    async def reader(instance_id: str) -> dict[str, Any] | None:
        events.append("status")
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return reader


def make_adapter(
    handler: Callable[[httpx.Request], Any],
    profiles: dict[str, ProfileConfig] | None = None,
    status_reader: Callable[[str], Awaitable[dict[str, Any] | None]] | None = None,
    busy_wait_timeout: float = 120.0,
    post_load_retry_seconds: float = 0.0,
) -> LMStudioAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return LMStudioAdapter(
        "http://test",
        profiles or {"4b": ProfileConfig("4b", "model-4b", 16_000, 4)},
        client=client,
        status_reader=status_reader or idle_status,
        busy_wait_timeout=busy_wait_timeout,
        busy_poll_interval=0.0,
        post_load_retry_seconds=post_load_retry_seconds,
    )


SWITCH_PROFILES = {
    "4b": ProfileConfig("4b", "model-a", 16_000, 4),
    "9b": ProfileConfig("9b", "model-b", 32_000, 2),
}


@pytest.mark.asyncio
async def test_list_and_reuse_loaded_instance_with_effective_caps() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        assert request.url.path == "/api/v1/models"
        return response(
            {
                "models": [
                    {
                        "type": "llm",
                        "key": "model-4b",
                        "loaded_instances": [
                            {"id": "instance-a", "config": {"context_length": 8192, "parallel": 2}}
                        ],
                        "max_context_length": 32768,
                    }
                ]
            }
        )

    adapter = make_adapter(handler)
    loaded = await adapter.ensure_loaded("4b")

    assert loaded.instance_id == "instance-a"
    assert loaded.context_limit == 8192
    assert loaded.parallel_limit == 2
    assert calls == [("GET", "/api/v1/models")]
    await adapter.close()


@pytest.mark.asyncio
async def test_switch_unloads_managed_instance_then_loads_exact_model() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    current = {"model-a"}
    loaded_new = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal loaded_new
        payload = json.loads(request.content or b"{}")
        calls.append((request.method, request.url.path, payload))
        if request.url.path == "/api/v1/models":
            models = [
                {"key": "model-a", "loaded_instances": [{"id": "old", "config": {"context_length": 4096, "parallel": 1}}] if current else []},
                {"key": "model-b", "loaded_instances": [{"id": "new", "config": {"context_length": 32000, "parallel": 2}}] if loaded_new else []},
            ]
            return response({"models": models})
        if request.url.path == "/api/v1/models/unload":
            current.clear()
            return response({"instance_id": "old"})
        if request.url.path == "/api/v1/models/load":
            loaded_new = True
            return response({"type": "llm", "instance_id": "new", "load_time_seconds": 1.25, "load_config": {"context_length": 16000, "parallel": 1}})
        raise AssertionError(request.url.path)

    profiles = {
        "4b": ProfileConfig("4b", "model-a", 16_000, 4),
        "9b": ProfileConfig("9b", "model-b", 32_000, 2),
    }
    adapter = make_adapter(handler, profiles)
    loaded = await adapter.ensure_loaded("9b")

    assert loaded.model_id == "model-b"
    assert loaded.load_seconds == 1.25
    # Caps come from the listed instance (32000 / 2), not the load echo (16000 / 1).
    assert (loaded.context_limit, loaded.parallel_limit) == (32000, 2)
    assert [item[1] for item in calls] == [
        "/api/v1/models",
        "/api/v1/models/unload",
        "/api/v1/models",
        "/api/v1/models/load",
        "/api/v1/models",
    ]
    assert calls[3][2] == {"model": "model-b", "context_length": 32000, "echo_load_config": True}
    await adapter.close()


@pytest.mark.asyncio
async def test_foreign_instance_is_unloaded_before_the_model_loads() -> None:
    loaded: dict[str, str] = {"someone-elses-model": "foreign"}
    events: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            models = [
                {
                    "key": key,
                    "loaded_instances": [{"id": loaded[key], "config": {"context_length": 16000, "parallel": 4}}] if key in loaded else [],
                }
                for key in ("someone-elses-model", "model-4b")
            ]
            return response({"models": models})
        if request.url.path == "/api/v1/models/unload":
            events.append("unload")
            loaded.clear()
            return response({"instance_id": "foreign"})
        if request.url.path == "/api/v1/models/load":
            events.append("load")
            loaded["model-4b"] = "inst-4b"
            return response({"instance_id": "inst-4b", "load_time_seconds": 0.1})
        raise AssertionError(request.url.path)

    adapter = make_adapter(handler)
    model = await adapter.ensure_loaded("4b")
    assert events == ["unload", "load"]
    assert model.instance_id == "inst-4b"
    await adapter.close()


@pytest.mark.asyncio
async def test_load_failure_message_is_sanitized() -> None:
    def oom_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response({"models": [{"key": "model-4b", "loaded_instances": []}]})
        return response({"error": "SENTINEL BACKEND BODY"}, status=507)

    adapter = make_adapter(oom_handler)
    with pytest.raises(BackendFailure) as error:
        await adapter.ensure_loaded("4b")
    assert error.value.code == "model_load_failed"
    assert "SENTINEL" not in str(error.value)
    await adapter.close()


@pytest.mark.asyncio
async def test_concurrent_ensure_loaded_performs_one_load() -> None:
    load_count = 0
    loaded_ids: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal load_count
        if request.url.path == "/api/v1/models":
            instances = [{"id": "loaded", "config": {"context_length": 16000, "parallel": 4}} for _ in loaded_ids]
            return response({"models": [{"key": "model-4b", "loaded_instances": instances}]})
        if request.url.path == "/api/v1/models/load":
            load_count += 1
            await asyncio.sleep(0.01)
            loaded_ids.append("loaded")
            return response({"instance_id": "loaded", "load_time_seconds": 0.1, "load_config": {"context_length": 16000, "parallel": 4}})
        raise AssertionError(request.url.path)

    adapter = make_adapter(handler)
    first, second = await asyncio.gather(adapter.ensure_loaded("4b"), adapter.ensure_loaded("4b"))
    assert first.instance_id == second.instance_id == "loaded"
    assert load_count == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_adapter_uses_separate_load_and_generation_deadlines() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response({"models": [{"key": "model-4b", "loaded_instances": []}]})
        return response({"instance_id": "loaded", "load_config": {"context_length": 16000, "parallel": 4}})

    adapter = LMStudioAdapter(
        "http://test",
        {"4b": ProfileConfig("4b", "model-4b", 16000, 4)},
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test"),
        load_timeout=7.0,
        generation_timeout=99.0,
    )
    assert adapter.load_timeout == 7.0
    assert adapter.generation_timeout == 99.0
    await adapter.close()


@pytest.mark.asyncio
async def test_configured_variant_must_be_selected_and_loads_by_model_key() -> None:
    loaded: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response(
                {
                    "models": [
                        {
                            "key": "model-4b",
                            "variants": ["model-4b@q4_k_m"],
                            "selected_variant": "model-4b@q4_k_m",
                            "loaded_instances": loaded,
                        }
                    ]
                }
            )
        # Live finding: LM Studio answers 404 model_not_found for a variant-suffixed load target.
        model = json.loads(request.content)["model"]
        assert model == "model-4b"
        loaded.append({"id": "variant-instance", "config": {"context_length": 16000, "parallel": 4}})
        return response({"instance_id": "variant-instance", "load_config": {"context_length": 16000, "parallel": 4}})

    adapter = make_adapter(
        handler,
        {"4b": ProfileConfig("4b", "model-4b", 16000, 4, variant="model-4b@q4_k_m")},
    )
    loaded_model = await adapter.ensure_loaded("4b")
    assert loaded_model.instance_id == "variant-instance"
    await adapter.close()


@pytest.mark.asyncio
async def test_unselected_variant_fails_load_instead_of_loading_another_variant() -> None:
    load_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response(
                {
                    "models": [
                        {
                            "key": "model-4b",
                            "variants": ["model-4b@q4_k_m", "model-4b@q8_0"],
                            "selected_variant": "model-4b@q8_0",
                            "loaded_instances": [],
                        }
                    ]
                }
            )
        load_calls.append(request.url.path)
        raise AssertionError("no load may be attempted for an unselected variant")

    adapter = make_adapter(
        handler,
        {"4b": ProfileConfig("4b", "model-4b", 16000, 4, variant="model-4b@q4_k_m")},
    )
    with pytest.raises(BackendFailure) as error:
        await adapter.ensure_loaded("4b")
    assert error.value.code == "variant_mismatch"
    assert load_calls == []
    await adapter.close()


@pytest.mark.asyncio
async def test_fresh_load_takes_caps_from_listed_instance_when_echo_omits_parallel() -> None:
    loaded: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response({"models": [{"key": "model-4b", "loaded_instances": loaded}]})
        if request.url.path == "/api/v1/models/load":
            loaded.append({"id": "instance-x", "config": {"context_length": 8192, "parallel": 3}})
            return response({"instance_id": "instance-x", "load_config": {"context_length": 8192}})
        raise AssertionError(request.url.path)

    adapter = make_adapter(handler)
    loaded_model = await adapter.ensure_loaded("4b")
    assert loaded_model.instance_id == "instance-x"
    assert loaded_model.context_limit == 8192
    assert loaded_model.parallel_limit == 3
    await adapter.close()


@pytest.mark.asyncio
async def test_unverified_parallel_cap_unloads_and_fails_fresh_and_reused_instances() -> None:
    loaded: list[dict[str, Any]] = []
    unloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response({"models": [{"key": "model-4b", "loaded_instances": loaded}]})
        if request.url.path == "/api/v1/models/load":
            loaded.append({"id": "instance-x", "config": {"context_length": 8192}})
            # The echo reports parallel, but only the listed instance is authoritative.
            return response({"instance_id": "instance-x", "load_config": {"context_length": 8192, "parallel": 4}})
        if request.url.path == "/api/v1/models/unload":
            unloads.append(json.loads(request.content)["instance_id"])
            loaded.clear()
            return response({"instance_id": unloads[-1]})
        raise AssertionError(request.url.path)

    adapter = make_adapter(handler)
    with pytest.raises(BackendFailure) as fresh:
        await adapter.ensure_loaded("4b")
    assert fresh.value.code == "caps_unverified"
    assert fresh.value.retryable is True
    assert unloads == ["instance-x"]

    loaded.append({"id": "stale", "config": {"context_length": 8192}})
    with pytest.raises(BackendFailure) as reused:
        await adapter.ensure_loaded("4b")
    assert reused.value.code == "caps_unverified"
    assert unloads == ["instance-x", "stale"]
    await adapter.close()


@pytest.mark.asyncio
async def test_failed_switch_does_not_leave_previous_model_marked_loaded() -> None:
    loaded: dict[str, str] = {}
    load_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            models = [
                {
                    "key": key,
                    "loaded_instances": [{"id": loaded[key], "config": {"context_length": 16000, "parallel": 4}}] if key in loaded else [],
                }
                for key in ("model-a", "model-b")
            ]
            return response({"models": models})
        if request.url.path == "/api/v1/models/unload":
            loaded.clear()
            return response({"instance_id": "unloaded"})
        if request.url.path == "/api/v1/models/load":
            model = json.loads(request.content)["model"]
            load_calls.append(model)
            if model == "model-b":
                return response({"error": "out of memory"}, status=507)
            loaded[model] = f"inst-{model}"
            return response({"instance_id": f"inst-{model}", "load_time_seconds": 0.1})
        raise AssertionError(request.url.path)

    profiles = {
        "4b": ProfileConfig("4b", "model-a", 16_000, 4),
        "9b": ProfileConfig("9b", "model-b", 32_000, 2),
    }
    adapter = make_adapter(handler, profiles)
    await adapter.ensure_loaded("4b")

    with pytest.raises(BackendFailure):
        await adapter.ensure_loaded("9b")
    assert loaded == {}

    # model-a was unloaded by the failed switch, so it must be loaded again rather than reused.
    again = await adapter.ensure_loaded("4b")
    assert again.instance_id == "inst-model-a"
    assert load_calls == ["model-a", "model-b", "model-a"]
    await adapter.close()


@pytest.mark.asyncio
async def test_35b_profile_loads_automatically_with_only_context_sent() -> None:
    loaded: list[dict[str, Any]] = []
    load_bodies: list[dict[str, Any]] = []
    unloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response(
                {
                    "models": [
                        {
                            "key": "qwen/qwen3.6-35b-a3b",
                            "variants": ["qwen/qwen3.6-35b-a3b@q4_k_m"],
                            "selected_variant": "qwen/qwen3.6-35b-a3b@q4_k_m",
                            "loaded_instances": loaded,
                        }
                    ]
                }
            )
        if request.url.path == "/api/v1/models/unload":
            unloads.append(json.loads(request.content)["instance_id"])
            return response({"instance_id": unloads[-1]})
        if request.url.path == "/api/v1/models/load":
            load_bodies.append(json.loads(request.content))
            # Live 2026-10-25: LM Studio applied the saved preset (28160 context, 1 parallel, 23 CPU expert layers).
            loaded.append({"id": "qwen/qwen3.6-35b-a3b", "config": {"context_length": 28160, "parallel": 1}})
            return response({"instance_id": "qwen/qwen3.6-35b-a3b", "load_time_seconds": 14.0})
        raise AssertionError(request.url.path)

    adapter = make_adapter(
        handler,
        {"35b": ProfileConfig("35b", "qwen/qwen3.6-35b-a3b", 28_000, 1, variant="qwen/qwen3.6-35b-a3b@q4_k_m")},
    )
    loaded_model = await adapter.ensure_loaded("35b")
    assert load_bodies == [{"model": "qwen/qwen3.6-35b-a3b", "context_length": 28000, "echo_load_config": True}]
    assert (loaded_model.context_limit, loaded_model.parallel_limit) == (28000, 1)
    assert unloads == []
    await adapter.close()


@pytest.mark.asyncio
async def test_listed_context_above_preset_unloads_and_fails_retryably_fresh_and_reused() -> None:
    loaded: list[dict[str, Any]] = []
    unloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response({"models": [{"key": "model-4b", "loaded_instances": loaded}]})
        if request.url.path == "/api/v1/models/load":
            # A load that ignored the requested context would allocate far more KV cache than the preset.
            loaded.append({"id": "oversized", "config": {"context_length": 262144, "parallel": 4}})
            return response({"instance_id": "oversized", "load_config": {"context_length": 16000, "parallel": 4}})
        if request.url.path == "/api/v1/models/unload":
            unloads.append(json.loads(request.content)["instance_id"])
            loaded.clear()
            return response({"instance_id": unloads[-1]})
        raise AssertionError(request.url.path)

    adapter = make_adapter(handler)
    with pytest.raises(BackendFailure) as fresh:
        await adapter.ensure_loaded("4b")
    assert fresh.value.code == "context_above_preset"
    assert fresh.value.retryable is True
    assert unloads == ["oversized"]

    loaded.append({"id": "stale-oversized", "config": {"context_length": 262144, "parallel": 4}})
    with pytest.raises(BackendFailure) as reused:
        await adapter.ensure_loaded("4b")
    assert reused.value.code == "context_above_preset"
    assert unloads == ["oversized", "stale-oversized"]
    await adapter.close()


@pytest.mark.asyncio
async def test_switch_waits_for_idle_before_unloading_the_previous_model() -> None:
    events: list[str] = []
    loaded = {"model-a": "old"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            models = [
                {
                    "key": key,
                    "loaded_instances": [{"id": loaded[key], "config": {"context_length": 16000, "parallel": 4}}] if key in loaded else [],
                }
                for key in ("model-a", "model-b")
            ]
            return response({"models": models})
        if request.url.path == "/api/v1/models/unload":
            events.append("unload")
            loaded.clear()
            return response({"instance_id": "old"})
        if request.url.path == "/api/v1/models/load":
            events.append("load")
            loaded["model-b"] = "new"
            return response({"instance_id": "new", "load_time_seconds": 0.1})
        raise AssertionError(request.url.path)

    reader = status_sequence(
        events,
        {"status": "generating", "queued": 0},
        {"status": "idle", "queued": 1},
        {"status": "idle", "queued": 0},
    )
    adapter = make_adapter(handler, SWITCH_PROFILES, status_reader=reader)
    loaded_model = await adapter.ensure_loaded("9b")
    assert loaded_model.model_id == "model-b"
    assert events == ["status", "status", "status", "unload", "load"]
    await adapter.close()


@pytest.mark.asyncio
async def test_switch_fails_retryably_and_unloads_nothing_while_instance_stays_busy() -> None:
    lifecycle_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response(
                {"models": [{"key": "model-a", "loaded_instances": [{"id": "old", "config": {"context_length": 16000, "parallel": 4}}]}, {"key": "model-b", "loaded_instances": []}]}
            )
        lifecycle_calls.append(request.url.path)
        return response({"instance_id": "x"})

    async def busy(instance_id: str) -> dict[str, Any] | None:
        return {"status": "generating", "queued": 2}

    adapter = make_adapter(handler, SWITCH_PROFILES, status_reader=busy, busy_wait_timeout=0.05)
    with pytest.raises(BackendFailure) as error:
        await adapter.ensure_loaded("9b")
    assert error.value.code == "backend_busy"
    assert error.value.retryable is True
    assert lifecycle_calls == []
    await adapter.close()


@pytest.mark.asyncio
async def test_unreadable_instance_status_fails_switch_without_unloading() -> None:
    lifecycle_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response(
                {"models": [{"key": "model-a", "loaded_instances": [{"id": "old", "config": {"context_length": 16000, "parallel": 4}}]}, {"key": "model-b", "loaded_instances": []}]}
            )
        lifecycle_calls.append(request.url.path)
        return response({"instance_id": "x"})

    async def unknown(instance_id: str) -> dict[str, Any] | None:
        return None

    adapter = make_adapter(handler, SWITCH_PROFILES, status_reader=unknown)
    with pytest.raises(BackendFailure) as error:
        await adapter.ensure_loaded("9b")
    assert error.value.code == "backend_status_unknown"
    assert lifecycle_calls == []
    await adapter.close()


@pytest.mark.asyncio
async def test_reusing_the_loaded_model_does_not_check_status_or_unload() -> None:
    checks: list[str] = []

    async def reader(instance_id: str) -> dict[str, Any] | None:
        checks.append(instance_id)
        return {"status": "generating", "queued": 3}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/models"
        return response({"models": [{"key": "model-a", "loaded_instances": [{"id": "warm", "config": {"context_length": 16000, "parallel": 4}}]}]})

    adapter = make_adapter(handler, SWITCH_PROFILES, status_reader=reader)
    loaded_model = await adapter.ensure_loaded("4b")
    assert loaded_model.instance_id == "warm"
    assert checks == []
    await adapter.close()


def test_parse_lms_status_reads_only_the_matching_instance() -> None:
    stdout = json.dumps(
        [
            {"identifier": "qwen/qwen3.5-9b", "status": "idle", "queued": 0, "parallel": 2, "type": "llm"},
            {"identifier": "qwen/qwen3-4b-thinking-2507", "status": "generating", "queued": 1, "parallel": 4, "type": "llm"},
        ]
    )
    assert _parse_lms_status(stdout, "qwen/qwen3-4b-thinking-2507") == {"status": "generating", "queued": 1}
    assert _parse_lms_status(stdout, "not-loaded") is None
    with pytest.raises(BackendFailure) as error:
        _parse_lms_status("not json", "x")
    assert error.value.code == "backend_status_unknown"


@pytest.mark.asyncio
async def test_status_uses_configured_lms_path(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str | None] = []
    started: list[tuple[Any, ...]] = []

    def fake_find(configured: str | None) -> str | None:
        asked.append(configured)
        return "C:/tools/lms.exe"

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return json.dumps([{"identifier": "inst", "status": "idle", "queued": 0}]).encode(), b""

    async def fake_exec(*command: Any, **_: Any) -> FakeProcess:
        started.append(command)
        return FakeProcess()

    monkeypatch.setattr(lm_studio, "find_lms", fake_find)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    adapter = LMStudioAdapter("http://test", {}, lms_path="D:/lm/lms.exe")

    assert await adapter._status_reader("inst") == {"status": "idle", "queued": 0}
    assert asked == ["D:/lm/lms.exe"]
    assert started == [("C:/tools/lms.exe", "ps", "--json")]
    await adapter.close()


@pytest.mark.asyncio
async def test_status_without_lms_names_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm_studio, "find_lms", lambda configured: None)
    adapter = LMStudioAdapter("http://test", {})

    with pytest.raises(BackendFailure) as failure:
        await adapter._status_reader("inst")

    assert failure.value.code == "backend_status_unknown"
    assert failure.value.retryable is True
    assert "lms_path" in str(failure.value)
    await adapter.close()


@pytest.mark.asyncio
async def test_stream_preserves_upstream_sse_frame_delimiters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data: {}\n\ndata: [DONE]\n\n", request=request)

    adapter = make_adapter(handler)
    chunks = [chunk async for chunk in adapter.stream({"model": "model-4b", "stream": True})]
    assert b"\n\n" in b"".join(chunks)
    await adapter.close()


@pytest.mark.asyncio
async def test_load_timeout_fails_only_the_job_and_next_job_adopts_the_finished_load() -> None:
    loaded: list[dict[str, Any]] = []
    load_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response({"models": [{"key": "model-4b", "loaded_instances": loaded}]})
        if request.url.path == "/api/v1/models/load":
            load_calls.append("load")
            # LM Studio keeps loading after the broker's deadline; the instance appears once that load finishes.
            loaded.append({"id": "slow-instance", "config": {"context_length": 16000, "parallel": 4}})
            raise httpx.ReadTimeout("load deadline exceeded", request=request)
        raise AssertionError(request.url.path)

    adapter = make_adapter(handler)
    with pytest.raises(BackendFailure) as first:
        await adapter.ensure_loaded("4b")
    assert first.value.code == "load_in_flight"
    assert first.value.retryable is True

    second = await adapter.ensure_loaded("4b")
    assert second.instance_id == "slow-instance"
    assert load_calls == ["load"]
    await adapter.close()


@pytest.mark.asyncio
async def test_unload_timeout_fails_retryably_and_next_job_retries_the_unload() -> None:
    loaded = [{"id": "old", "config": {"context_length": 16000, "parallel": 4}}]
    unload_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal unload_calls
        if request.url.path == "/api/v1/models":
            models = [
                {"key": "model-a", "loaded_instances": loaded},
                {"key": "model-b", "loaded_instances": []},
            ]
            return response({"models": models})
        if request.url.path == "/api/v1/models/unload":
            unload_calls += 1
            if unload_calls == 1:
                raise httpx.ReadTimeout("unload deadline exceeded", request=request)
            loaded.clear()
            return response({"instance_id": "old"})
        if request.url.path == "/api/v1/models/load":
            loaded.append({"id": "new", "config": {"context_length": 32000, "parallel": 2}})
            return response({"instance_id": "new", "load_time_seconds": 0.1})
        raise AssertionError(request.url.path)

    adapter = make_adapter(handler, SWITCH_PROFILES)
    with pytest.raises(BackendFailure) as first:
        await adapter.ensure_loaded("9b")
    assert first.value.code == "unload_in_flight"
    assert first.value.retryable is True

    second = await adapter.ensure_loaded("9b")
    assert second.instance_id == "new"
    assert unload_calls == 2
    await adapter.close()


@pytest.mark.asyncio
async def test_duplicate_instances_of_the_target_are_reduced_to_one() -> None:
    loaded = [
        {"id": "first", "config": {"context_length": 32000, "parallel": 2}},
        {"id": "second", "config": {"context_length": 32000, "parallel": 2}},
    ]
    unloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return response({"models": [{"key": "model-b", "loaded_instances": loaded}]})
        if request.url.path == "/api/v1/models/unload":
            instance_id = json.loads(request.content)["instance_id"]
            unloads.append(instance_id)
            loaded[:] = [item for item in loaded if item["id"] != instance_id]
            return response({"instance_id": instance_id})
        raise AssertionError(request.url.path)

    adapter = make_adapter(handler, SWITCH_PROFILES)
    model = await adapter.ensure_loaded("9b")
    assert model.instance_id == "first"
    assert unloads == ["second"]
    await adapter.close()


@pytest.mark.asyncio
async def test_still_loaded_alone_is_false_when_another_model_or_nothing_is_loaded() -> None:
    state: dict[str, list[dict[str, Any]]] = {
        "model-a": [],
        "model-b": [{"id": "inst-b", "config": {"context_length": 32000, "parallel": 2}}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        models = [{"key": key, "loaded_instances": items} for key, items in state.items()]
        return response({"models": models})

    adapter = make_adapter(handler, SWITCH_PROFILES)
    loaded = LoadedModel("9b", "model-b", "inst-b", 32_000, 2)
    assert await adapter.still_loaded_alone(loaded) is True

    state["model-a"] = [{"id": "owner-model", "config": {"context_length": 4096, "parallel": 1}}]
    assert await adapter.still_loaded_alone(loaded) is False

    state["model-a"] = []
    state["model-b"] = []
    assert await adapter.still_loaded_alone(loaded) is False
    await adapter.close()



def post_load_handler(*, drops: int = 0, unlisted: int = 0) -> tuple[Callable[[httpx.Request], httpx.Response], dict[str, Any]]:
    # The model loads, then the first listings after the load either drop the connection or leave the instance out.
    state: dict[str, Any] = {"loaded": False, "listings": 0, "drops": drops, "unlisted": unlisted}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            if not state["loaded"]:
                return response({"models": [{"key": "model-b", "loaded_instances": []}]})
            state["listings"] += 1
            if state["drops"] > 0:
                state["drops"] -= 1
                raise httpx.ConnectError("connection dropped", request=request)
            if state["unlisted"] > 0:
                state["unlisted"] -= 1
                return response({"models": [{"key": "model-b", "loaded_instances": []}]})
            instances = [{"id": "new", "config": {"context_length": 32000, "parallel": 2}}]
            return response({"models": [{"key": "model-b", "loaded_instances": instances}]})
        if request.url.path == "/api/v1/models/load":
            state["loaded"] = True
            return response({"type": "llm", "instance_id": "new", "load_time_seconds": 1.0, "load_config": {}})
        raise AssertionError(request.url.path)

    return handler, state


@pytest.mark.asyncio
async def test_connection_drop_right_after_a_load_is_retried_and_the_load_succeeds() -> None:
    handler, state = post_load_handler(drops=1)
    adapter = make_adapter(handler, {"9b": ProfileConfig("9b", "model-b", 32_000, 2)})

    loaded = await adapter.ensure_loaded("9b")

    assert loaded.instance_id == "new"
    assert (loaded.context_limit, loaded.parallel_limit) == (32000, 2)
    assert state["listings"] == 2
    await adapter.close()


@pytest.mark.asyncio
async def test_instance_not_listed_yet_after_a_load_is_looked_for_again() -> None:
    handler, state = post_load_handler(unlisted=1)
    adapter = make_adapter(handler, {"9b": ProfileConfig("9b", "model-b", 32_000, 2)})

    loaded = await adapter.ensure_loaded("9b")

    assert loaded.instance_id == "new"
    assert state["listings"] == 2
    await adapter.close()


@pytest.mark.asyncio
async def test_post_load_listing_that_keeps_failing_still_fails_the_job_retryably() -> None:
    handler, state = post_load_handler(drops=10)
    adapter = make_adapter(handler, {"9b": ProfileConfig("9b", "model-b", 32_000, 2)})

    with pytest.raises(BackendFailure) as failure:
        await adapter.ensure_loaded("9b")

    assert failure.value.code == "backend_unavailable"
    assert failure.value.retryable is True
    assert state["listings"] == 3
    await adapter.close()
