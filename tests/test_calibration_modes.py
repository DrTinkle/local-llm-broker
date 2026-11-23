from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from support import PRESET

from local_llm_broker import calibrate_direct
from local_llm_broker.calibrate import default_profiles, resolve_models
from local_llm_broker.calibrate_broker_modes import _tokens, depths_for, format_tokens
from local_llm_broker.calibrate_direct import (
    CONTEXT_STEP,
    BrokerRunning,
    DirectSession,
    format_search,
    format_sweep,
    search_combos,
    search_context,
    slot_candidates,
    sweep_experts,
    sweep_slots,
)
from local_llm_broker.calibrate_probe import JobResult, content_samples, png_data_url
from local_llm_broker.config import load_config
from local_llm_broker.contracts import TierConfig


def test_samples_are_deterministic_and_about_the_requested_size() -> None:
    first, second = content_samples(), content_samples()
    assert first == second
    assert all(15_500 <= len(text.encode("utf-8")) <= 16_500 for text in first.values())
    from local_llm_broker.calibrate import synthetic_prompt

    # The size count includes the sentence periods, so the broker's estimate matches the request.
    assert 32_000 <= len(synthetic_prompt(32_000).encode("utf-8")) // 4 <= 32_010
    assert png_data_url(8, 8).startswith("data:image/png;base64,iVBORw0KGgo")


def test_search_combos() -> None:
    assert search_combos(False) == (("f16", None), ("q8_0", None))
    assert search_combos(True) == (("f16", 0.5), ("q8_0", 0.5), ("q8_0", 0.75))


def test_slot_candidates() -> None:
    assert slot_candidates(1) == (1, 2)
    assert slot_candidates(2) == (1, 2, 3)
    assert slot_candidates(3) == (2, 3, 4)
    assert slot_candidates(4) == (4, 6, 8)


def test_default_profiles_picks_one_non_thinking_profile_per_model(tmp_path: Path) -> None:
    assert default_profiles(load_config(PRESET)) == ["4b-nothink", "9b-nothink", "35b-nothink"]
    only_thinking = tmp_path / "config.toml"
    only_thinking.write_text('[profiles.deep]\nmodel_id = "org/deep"\nthinks = true\n', encoding="utf-8")
    assert default_profiles(load_config(only_thinking)) == ["deep"]
    assert default_profiles(load_config(tmp_path / "missing.toml")) == []


def test_calibrate_direct_models_accepts_profile_alias_or_model_id() -> None:
    config = load_config(PRESET)
    assert resolve_models(config, ["9b-nothink"]) == ["qwen/qwen3.5-9b"]
    assert resolve_models(config, ["qwen/qwen3.5-9b", "35b"]) == ["qwen/qwen3.5-9b", "qwen/qwen3.6-35b-a3b"]
    assert resolve_models(config, ["org/not-configured"]) == ["org/not-configured"]
    assert resolve_models(config, ["9b", "9b-nothink"]) == ["qwen/qwen3.5-9b"]
    assert resolve_models(config, None) == [
        "qwen/qwen3-4b-thinking-2507",
        "qwen/qwen3-4b-2507",
        "qwen/qwen3.5-9b",
        "qwen/qwen3.6-35b-a3b",
    ]


def test_depths_skip_prompts_that_would_not_fit_the_tier() -> None:
    assert depths_for(16_384, 1_000_000) == [2_000, 8_000]
    assert depths_for(32_768, 1_000_000) == [2_000, 8_000, 16_000, 32_000]
    assert depths_for(262_144, 32_768) == [2_000, 8_000, 16_000, 32_000]


@pytest.mark.asyncio
async def test_tokens_mode_reports_bytes_per_token_and_image_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        if isinstance(content, list):
            images = [part for part in content if part["type"] == "image_url"]
            tokens = 20 + (len(images[0]["image_url"]["url"]) // 1000 if images else 0)
        else:
            text = content.removesuffix("\n\nReply with OK.")
            # The baseline is 20 tokens; everything else tokenizes at 2 bytes per token.
            tokens = 20 if content == "Reply with OK." else 20 + len(text.encode("utf-8")) // 2
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}], "usage": {"prompt_tokens": tokens}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1235") as client:
        report = await _tokens(client, ["4b-nothink"], ["9b-nothink"])
    assert report.errors == []
    assert all(value == pytest.approx(2.0, rel=0.01) for value in report.text["4b-nothink"].values())
    assert sorted(report.images["9b-nothink"]) == [256, 512, 1024, 1536]
    assert "2.0x the tokens" in format_tokens(report)


# --- direct modes ---------------------------------------------------------------------------------------------


class FakeLMStudio:
    """Loads succeed up to `max_fit` context; above it they load but run slowly, like a VRAM overflow."""

    def __init__(self, max_fit: int, base_gen: float = 100.0) -> None:
        self.max_fit = max_fit
        self.base_gen = base_gen
        self.current: TierConfig | None = None
        self.parallel = 0
        self.loads: list[tuple[int, str, float | None, int]] = []

    async def loader(self, model: str, tier: TierConfig, parallel: int) -> tuple[str, float]:
        self.current, self.parallel = tier, parallel
        self.loads.append((tier.context_length, tier.kv_cache, tier.cpu_expert_ratio, parallel))
        return model, 1.0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            loaded = [{"id": "m"}] if self.current else []
            return httpx.Response(200, json={"models": [{"key": "m", "loaded_instances": loaded}]})
        if request.url.path == "/api/v1/models/unload":
            self.current = None
            return httpx.Response(200, json={})
        raise AssertionError(request.url.path)

    def speed(self) -> float:
        assert self.current is not None
        slow = self.current.context_length > self.max_fit
        ratio = self.current.cpu_expert_ratio or 0.0
        return (self.base_gen * (1.2 - ratio)) * (0.5 if slow else 1.0)

    async def stream_job(self, client: httpx.AsyncClient, body: dict[str, Any]) -> JobResult:
        gen = self.speed()
        return JobResult(True, 6_000, 129, started_at=0.0, first_token_at=6_000 / (gen * 40), ended_at=6_000 / (gen * 40) + 128 / gen)


def session_for(fake: FakeLMStudio, monkeypatch: pytest.MonkeyPatch, vram: tuple[int, int] | None = None) -> DirectSession:
    monkeypatch.setattr(calibrate_direct, "stream_job", fake.stream_job)
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler), base_url="http://127.0.0.1:1234")
    return DirectSession("http://127.0.0.1:1234", "http://127.0.0.1:1", loader=fake.loader, client=client, vram_reader=lambda: vram, settle_seconds=0)


@pytest.mark.asyncio
async def test_search_finds_the_largest_context_that_keeps_full_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLMStudio(max_fit=100_000)
    session = session_for(fake, monkeypatch)
    result = await search_context(session, "m", "q8_0", None, 4)
    assert result.max_context is not None
    assert result.max_context <= 100_000 < result.max_context + CONTEXT_STEP
    assert result.max_context % CONTEXT_STEP == 0
    assert all(parallel == 4 for *_, parallel in fake.loads)
    assert fake.current is None  # everything unloaded afterwards
    assert "largest context that fits" in format_search([result])
    await session.close()


@pytest.mark.asyncio
async def test_search_stops_at_the_native_maximum_when_everything_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    session = session_for(FakeLMStudio(max_fit=10**9), monkeypatch)
    result = await search_context(session, "m", "f16", 1.0, 1)
    assert result.max_context == 262_144
    assert len(result.probes) == 1
    await session.close()


@pytest.mark.asyncio
async def test_search_context_accepts_baseline_top_and_step(monkeypatch: pytest.MonkeyPatch) -> None:
    everything = FakeLMStudio(max_fit=10**9)
    session = session_for(everything, monkeypatch)
    result = await search_context(session, "m", "f16", None, 1, baseline_context=8_192, top_context=131_072, step=16_384)
    assert result.max_context == 131_072
    assert [load[0] for load in everything.loads] == [8_192, 131_072]
    await session.close()

    limited = FakeLMStudio(max_fit=65_536)
    session = session_for(limited, monkeypatch)
    result = await search_context(session, "m", "f16", None, 1, baseline_context=8_192, top_context=131_072, step=16_384)
    assert result.max_context == 65_536
    assert limited.loads[0][0] == 8_192
    assert all(load[0] % 16_384 == 0 for load in limited.loads[1:])
    await session.close()


@pytest.mark.asyncio
async def test_too_little_free_vram_counts_as_not_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    session = session_for(FakeLMStudio(max_fit=10**9), monkeypatch, vram=(12_100, 12_288))
    result = await search_context(session, "m", "f16", None, 1)
    assert result.max_context is None
    assert result.error is not None
    await session.close()


@pytest.mark.asyncio
async def test_failed_load_is_recorded_as_not_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLMStudio(max_fit=10**9)

    async def refuse(model: str, tier: TierConfig, parallel: int) -> tuple[str, float]:
        raise RuntimeError("out of memory")

    session = session_for(fake, monkeypatch)
    session._loader = refuse
    probe = await session.try_config("m", TierConfig("t", 16_384), 1)
    assert (probe.loaded, probe.fits(None)) == (False, False)
    assert "out of memory" in (probe.error or "")
    await session.close()


@pytest.mark.asyncio
async def test_expert_sweep_picks_the_fastest_ratio_that_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLMStudio(max_fit=10**9)
    readings = iter([(12_200, 12_288), (11_000, 12_288), (10_500, 12_288)])  # 0.5 leaves too little VRAM
    session = session_for(fake, monkeypatch)
    session._vram = lambda: next(readings)
    result = await sweep_experts(session, "m", TierConfig("low", 32_768, "f16", cpu_expert_ratio=0.625, try_mmap=False), [0.5, 0.6, 0.7])
    assert [ratio for *_, ratio, _ in fake.loads] == [0.5, 0.6, 0.7]
    assert "fastest that fits: ratio 0.60" in format_sweep([result], "experts")
    await session.close()


@pytest.mark.asyncio
async def test_slot_sweep_loads_each_slot_count_and_runs_that_many_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLMStudio(max_fit=10**9)
    session = session_for(fake, monkeypatch)
    sent: list[int] = []

    async def fake_round(client: Any, model: str, tier: Any, prompts: list[str]) -> Any:
        from local_llm_broker.calibrate_concurrency import Round

        sent.append(len(prompts))
        jobs = [JobResult(True, 1_000, 201, first_token_at=1.0, ended_at=1.0 + 2.0 * len(prompts) ** 0.5) for _ in prompts]
        return Round(jobs, 0.0)

    monkeypatch.setattr(calibrate_direct, "_round", fake_round)
    result = await sweep_slots(session, "m", TierConfig("low", 16_384), [2, 3, 4])
    assert sent == [2, 3, 4]
    assert [parallel for *_, parallel in fake.loads] == [2, 3, 4]
    assert "fastest that fits: 4 slots" in format_sweep([result], "slots")
    await session.close()


@pytest.mark.asyncio
async def test_direct_session_refuses_while_the_broker_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def answering(self: httpx.AsyncClient, url: str, **_: Any) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    session = DirectSession("http://127.0.0.1:1234", "http://127.0.0.1:1235")
    monkeypatch.setattr(httpx.AsyncClient, "get", answering)
    with pytest.raises(BrokerRunning):
        await session.ensure_broker_stopped()
    await session.close()
