from __future__ import annotations

from typing import Any

import httpx
import pytest
from support import PRESET

from local_llm_broker import setup_discover
from local_llm_broker.config import load_config
from local_llm_broker.setup_discover import (
    ModelInfo,
    detect_hardware,
    discover_models,
    fetch_listing,
    preflight,
)


def llm(key: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "llm",
        "publisher": "qwen",
        "key": key,
        "display_name": key,
        "architecture": "qwen3",
        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4},
        "size_bytes": 2497459753,
        "params_string": "4B",
        "loaded_instances": [],
        "max_context_length": 262144,
        "format": "gguf",
        "capabilities": {"vision": False, "trained_for_tool_use": True},
        "description": None,
        "variants": [f"{key}@q4_k_m"],
        "selected_variant": f"{key}@q4_k_m",
    }
    entry.update(overrides)
    return entry


EMBEDDING: dict[str, Any] = {
    "type": "embedding", "publisher": "nomic-ai", "key": "text-embedding-nomic-embed-text-v1.5",
    "display_name": "Nomic Embed Text v1.5", "architecture": None, "params_string": None, "size_bytes": 84106624,
    "loaded_instances": [], "max_context_length": 2048, "format": "gguf",
}


def test_discover_keeps_llms_and_skips_embeddings() -> None:
    models = discover_models({"models": [llm("qwen/qwen3-4b-2507"), EMBEDDING]})

    assert [m.key for m in models] == ["qwen/qwen3-4b-2507"]


def test_discover_reads_fields() -> None:
    big = llm(
        "qwen/qwen3.5-35b-a3b", display_name="Qwen3.5 35B A3B", architecture="qwen35moe", params_string="35B-A3B",
        size_bytes=21_000_000_000, capabilities={"vision": True, "trained_for_tool_use": True},
        selected_variant="qwen/qwen3.5-35b-a3b@q4_k_m",
    )

    [model] = discover_models({"models": [big]})

    assert model == ModelInfo(
        key="qwen/qwen3.5-35b-a3b", display_name="Qwen3.5 35B A3B", architecture="qwen35moe", params="35B-A3B",
        size_bytes=21_000_000_000, max_context=262144, vision=True, is_moe=True, variant="qwen/qwen3.5-35b-a3b@q4_k_m",
    )
    assert model.variant == big["selected_variant"]


@pytest.mark.parametrize(
    ("params", "architecture", "expected"),
    [
        ("35B-A3B", "qwen35moe", True),
        ("30B-A3B", "qwen3", True),
        ("8x7B", "llama", False),
        ("8x7B", "somethingmoe", True),
        ("9B", "qwen35", False),
        ("9B", "SomeMoE", True),
        (None, None, False),
    ],
)
def test_moe_detection(params: str | None, architecture: str | None, expected: bool) -> None:
    [model] = discover_models({"models": [llm("m/x", params_string=params, architecture=architecture)]})

    assert model.is_moe is expected


def test_discover_sorts_by_size_then_key() -> None:
    listing = {"models": [
        llm("b/mid", size_bytes=5), llm("z/small", size_bytes=1), llm("a/mid", size_bytes=5), llm("c/large", size_bytes=9),
    ]}

    assert [m.key for m in discover_models(listing)] == ["z/small", "a/mid", "b/mid", "c/large"]


def test_discover_tolerates_missing_fields() -> None:
    listing = {"models": [
        {"type": "llm", "key": "bare/model"},
        {"type": "llm", "key": "odd/model", "size_bytes": "big", "max_context_length": None, "capabilities": "none",
         "params_string": 7, "architecture": 3, "selected_variant": 5, "display_name": None},
        {"type": "llm"},
        "not a dict",
    ]}

    models = {m.key: m for m in discover_models(listing)}

    assert set(models) == {"bare/model", "odd/model"}
    bare = models["bare/model"]
    assert (bare.size_bytes, bare.max_context, bare.vision, bare.is_moe) == (0, 0, False, False)
    assert (bare.architecture, bare.params, bare.variant) == (None, None, None)
    odd = models["odd/model"]
    assert (odd.size_bytes, odd.max_context, odd.vision, odd.is_moe) == (0, 0, False, False)
    assert (odd.architecture, odd.params, odd.variant) == (None, None, None)
    assert odd.display_name == "odd/model"
    assert discover_models({}) == []


def test_hardware_without_nvidia_smi_has_unknown_vram() -> None:
    hw = detect_hardware(nvidia_smi=lambda: None, ram_reader=lambda: 65536)

    assert hw.vram_mib is None
    assert hw.gpu_name is None
    assert hw.ram_mib == 65536


def test_hardware_reads_gpu() -> None:
    hw = detect_hardware(nvidia_smi=lambda: ("Some GPU", 12288), ram_reader=lambda: 32768)

    assert (hw.gpu_name, hw.vram_mib, hw.ram_mib) == ("Some GPU", 12288, 32768)


def client_for(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_preflight_reports_lmstudio_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_discover, "find_lms", lambda configured: "C:/lms.exe")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    config = load_config(PRESET)
    async with client_for(handler) as client:
        result = await preflight(config, client)
        with pytest.raises(RuntimeError, match="LM Studio is not answering at http://127.0.0.1:1234"):
            await fetch_listing(config, client)

    assert result.lmstudio_ok is False
    assert "LM Studio is not answering at http://127.0.0.1:1234" in result.problems[0]
    assert result.broker_running is False


async def test_preflight_reports_missing_lms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_discover, "find_lms", lambda configured: None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"models": []})
        raise httpx.ConnectError("refused", request=request)

    async with client_for(handler) as client:
        result = await preflight(load_config(PRESET), client)

    assert result.lmstudio_ok is True
    assert result.lms_path is None
    assert any("lms_path" in problem for problem in result.problems)


async def test_preflight_detects_running_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_discover, "find_lms", lambda configured: "C:/lms.exe")
    config = load_config(PRESET)
    seen: list[str] = []

    def answering(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(401 if request.url.path == "/health" else 404)

    def silent_broker(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"models": []})
        raise httpx.ConnectError("refused", request=request)

    async with client_for(answering) as client:
        running = await preflight(config, client)
    async with client_for(silent_broker) as client:
        stopped = await preflight(config, client)

    assert running.broker_running is True
    assert running.problems == []
    assert running.lms_path == "C:/lms.exe"
    assert f"{config.service_url}/health" in seen
    assert stopped.broker_running is False
    assert stopped.problems == []


async def test_preflight_treats_non_200_listing_as_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_discover, "find_lms", lambda configured: "C:/lms.exe")

    async with client_for(lambda request: httpx.Response(500)) as client:
        result = await preflight(load_config(PRESET), client)

    assert result.lmstudio_ok is False
    assert "LM Studio is not answering at" in result.problems[0]
