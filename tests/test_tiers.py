from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import httpx
import pytest
from support import PRESET

from local_llm_broker.backend import LoadedModel
from local_llm_broker.calibrate import TierMeasurement, measure_stream, verdicts
from local_llm_broker.config import load_config
from local_llm_broker.contracts import JobSpec, JobState, ProfileConfig, TierConfig
from local_llm_broker.lm_studio import BackendFailure, LMStudioAdapter
from local_llm_broker.lms_sdk import CPU_EXPERT_RATIO_KEY, PARALLEL_KEY, load_config_stack
from local_llm_broker.routes_openai import _resolve_profile_and_tier
from local_llm_broker.scheduler import Scheduler, SchedulerError
from local_llm_broker.storage import MetadataStore

_T = TypeVar("_T")


def present(value: _T | None) -> _T:
    assert value is not None
    return value


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


# --- config -------------------------------------------------------------------------------------


def test_thinking_and_plain_profiles_of_one_model_share_its_tiers(tmp_path: Path) -> None:
    config = load_config(PRESET)
    assert config.tiers_for("9b") is config.tiers_for("9b-nothink")
    assert config.tiers_for("35b")["high"].cpu_expert_ratio == 0.775
    assert config.tiers_for("35b")["low"].try_mmap is False
    # The 4B models do full attention in every layer, so their faster tiers keep an f16 cache.
    assert [tier.kv_cache for tier in config.tiers_for("4b").values()] == ["f16", "f16", "q8_0"]


def nine(low: str = "", medium: str = "", high: str = "", profile: str = "") -> str:
    return (
        f'[profiles.9b]\nmodel_id = "qwen/qwen3.5-9b"\n{profile}'
        f'[tiers."qwen/qwen3.5-9b".low]\ncontext_length = 32768\nkv_cache = "f16"\n{low}'
        f'[tiers."qwen/qwen3.5-9b".medium]\ncontext_length = 131072\nkv_cache = "q8_0"\n{medium}'
        f'[tiers."qwen/qwen3.5-9b".high]\ncontext_length = 163840\nkv_cache = "q8_0"\n{high}'
    )


def test_a_tier_can_be_defined_with_only_the_settings_it_needs(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, nine(high="flash_attention = true\n").replace("163840", "196608")))
    high = config.tiers_for("9b")["high"]
    assert (high.context_length, high.kv_cache) == (196_608, "q8_0")
    assert config.profiles["9b"].context_ceiling == 196_608


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (nine().replace(".low]", ".huge]"), "tier names"),
        (nine().replace("context_length = 32768", "context_length = 200000"), "strictly increasing"),
        (nine().replace('kv_cache = "f16"', 'kv_cache = "q2"'), "kv_cache"),
        (nine(medium="flash_attention = false\n"), "needs flash_attention"),
        (nine(low="cpu_expert_ratio = 1.5\n"), "cpu_expert_ratio"),
        (nine(low='speed = "fast"\n'), "unknown settings"),
        (nine(profile="context_ceiling = 20000\n"), "is replaced by the model"),
    ],
)
def test_invalid_tier_settings_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, text))


# --- SDK load stack -----------------------------------------------------------------------------


def stack_fields(tier: TierConfig, parallel: int) -> dict[str, Any]:
    fields = load_config_stack(tier, parallel)["layers"][0]["config"]["fields"]
    return {item["key"]: item["value"] for item in fields}


def test_tier_load_stack_carries_cache_type_mmap_expert_ratio_and_parallel() -> None:
    fields = stack_fields(TierConfig("high", 262_144, "q8_0", cpu_expert_ratio=0.775, try_mmap=False), 1)
    assert fields["llm.load.contextLength"] == 262_144
    assert fields["llm.load.llama.kCacheQuantizationType"] == {"checked": True, "value": "q8_0"}
    assert fields["llm.load.llama.vCacheQuantizationType"] == {"checked": True, "value": "q8_0"}
    assert fields["llm.load.llama.flashAttention"] is True
    assert fields["llm.load.llama.tryMmap"] is False
    assert fields["llm.load.llama.acceleration.offloadRatio"] == 1.0
    assert fields[CPU_EXPERT_RATIO_KEY] == 0.775
    assert fields[PARALLEL_KEY] == 1


def test_f16_tier_sends_an_explicit_f16_cache_and_no_expert_ratio() -> None:
    fields = stack_fields(TierConfig("low", 16_384, "f16"), 4)
    assert fields["llm.load.llama.kCacheQuantizationType"] == {"checked": True, "value": "f16"}
    assert CPU_EXPERT_RATIO_KEY not in fields
    assert "llm.load.llama.tryMmap" not in fields
    assert fields[PARALLEL_KEY] == 4


# --- scheduler ----------------------------------------------------------------------------------


def job(label: str, profile: str, *, tier: str | None = None, prompt_bytes: int = 4, max_tokens: int = 8) -> JobSpec:
    content = label + "x" * max(0, prompt_bytes - len(label))
    return JobSpec("client", "project", label, profile, [{"role": "user", "content": content}], {"max_tokens": max_tokens}, tier=tier)


@dataclass
class TierBackend:
    tiers: dict[str, int]
    loads: list[tuple[str, str | None, bool]] = field(default_factory=list)
    current: tuple[str, str] | None = None
    started: list[str] = field(default_factory=list)

    async def ensure_loaded(self, profile: str, *, tier: str | None = None, accept_higher: bool = False) -> LoadedModel:
        self.loads.append((profile, tier, accept_higher))
        assert tier is not None
        if self.current and self.current[0] == profile and accept_higher and list(self.tiers).index(self.current[1]) > list(self.tiers).index(tier):
            tier = self.current[1]
        self.current = (profile, tier)
        return LoadedModel(profile, profile, f"inst-{tier}", self.tiers[tier], 1, 0.01, tier=tier)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.started.append(str(payload["metadata"]["task_label"]))
        await asyncio.sleep(0)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {}}


def scheduler_for(tmp_path: Path, backend: Any) -> Scheduler:
    return Scheduler(load_config(PRESET), backend, store=MetadataStore(tmp_path / "state.sqlite3"))


async def finish(scheduler: Scheduler, ids: list[str]) -> list[JobState]:
    for _ in range(400):
        states = [present(scheduler.get(job_id)).state for job_id in ids]
        if all(state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED} for state in states):
            return states
        await asyncio.sleep(0.005)
    raise AssertionError("jobs did not finish")


def test_auto_tier_is_the_smallest_that_fits(tmp_path: Path) -> None:
    scheduler = scheduler_for(tmp_path, TierBackend({}))
    assert scheduler.resolve_tier(job("a", "35b")) == ("low", True)
    # About 50,000 prompt tokens plus output does not fit low (32,768) but fits medium (131,072).
    assert scheduler.resolve_tier(job("b", "35b", prompt_bytes=200_000, max_tokens=4096)) == ("medium", True)
    assert scheduler.resolve_tier(job("c", "35b", tier="high")) == ("high", False)
    assert scheduler.resolve_tier(job("d", "35b", tier="auto")) == ("low", True)


def test_an_explicit_tier_too_small_for_the_job_is_rejected_with_a_hint(tmp_path: Path) -> None:
    scheduler = scheduler_for(tmp_path, TierBackend({}))
    with pytest.raises(SchedulerError) as caught:
        scheduler.resolve_tier(job("a", "4b", tier="low", prompt_bytes=200_000))
    assert caught.value.code == "context_exceeded"
    assert "higher tier" in str(caught.value)
    with pytest.raises(SchedulerError) as unknown:
        scheduler.resolve_tier(job("b", "4b", tier="max"))
    assert unknown.value.code == "unknown_tier"


@pytest.mark.asyncio
async def test_auto_jobs_run_on_a_larger_loaded_tier_but_explicit_ones_reload(tmp_path: Path) -> None:
    backend = TierBackend({"low": 32_768, "medium": 131_072, "high": 262_144})
    scheduler = scheduler_for(tmp_path, backend)
    await scheduler.start()
    try:
        ids = [
            (await scheduler.submit(job("big", "35b", tier="high"))).job_id,
            (await scheduler.submit(job("small-auto", "35b"))).job_id,
            (await scheduler.submit(job("small-low", "35b", tier="low"))).job_id,
        ]
        assert await finish(scheduler, ids) == [JobState.SUCCEEDED] * 3
    finally:
        await scheduler.shutdown()
    # The auto job ran on the loaded high tier without a load call; the explicit low job caused one.
    assert backend.loads == [("35b", "high", False), ("35b", "low", False)]
    assert backend.started == ["big", "small-auto", "small-low"]


@pytest.mark.asyncio
async def test_status_reports_the_loaded_tier(tmp_path: Path) -> None:
    backend = TierBackend({"low": 16_384, "medium": 32_768, "high": 65_536})
    scheduler = scheduler_for(tmp_path, backend)
    await scheduler.start()
    try:
        await finish(scheduler, [(await scheduler.submit(job("a", "4b", tier="medium"))).job_id])
        assert scheduler.status()["active_tier"] == "medium"
    finally:
        await scheduler.shutdown()


# --- adapter ------------------------------------------------------------------------------------

TIERS = {
    "model-x": {
        "low": TierConfig("low", 16_384, "f16"),
        "high": TierConfig("high", 98_304, "q8_0", cpu_expert_ratio=0.75),
    }
}
PROFILES = {"x": ProfileConfig("x", "model-x", 98_304, 2)}


def tier_adapter(state: dict[str, Any], loads: list[tuple[str, TierConfig, int]], unloads: list[str]) -> LMStudioAdapter:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models":
            instances = [{"id": state["id"], "config": {"context_length": state["context"], "parallel": 2}}] if state.get("id") else []
            return httpx.Response(200, json={"models": [{"key": "model-x", "loaded_instances": instances}]})
        if request.url.path == "/api/v1/models/unload":
            unloads.append(json.loads(request.content)["instance_id"])
            state.clear()
            return httpx.Response(200, json={})
        raise AssertionError(f"unexpected REST call {request.url.path}")

    async def loader(model_key: str, tier: TierConfig, parallel: int) -> tuple[str, float]:
        loads.append((model_key, tier, parallel))
        state.update(id="model-x", context=tier.context_length)
        return "model-x", 2.5

    async def idle(instance_id: str) -> dict[str, Any]:
        return {"status": "idle", "queued": 0}

    return LMStudioAdapter(
        "http://test",
        PROFILES,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test"),
        status_reader=idle,
        busy_poll_interval=0.0,
        post_load_retry_seconds=0.0,
        tiers=TIERS,
        tier_loader=loader,
    )


@pytest.mark.asyncio
async def test_tier_load_goes_through_the_sdk_loader_with_the_tier_values() -> None:
    loads: list[tuple[str, TierConfig, int]] = []
    adapter = tier_adapter({}, loads, [])
    loaded = await adapter.ensure_loaded("x", tier="high")
    assert [(key, tier.name, parallel) for key, tier, parallel in loads] == [("model-x", "high", 2)]
    assert (loaded.tier, loaded.context_limit, loaded.load_seconds) == ("high", 98_304, 2.5)
    await adapter.close()


@pytest.mark.asyncio
async def test_loaded_instance_at_the_same_tier_is_reused_and_another_tier_reloads() -> None:
    loads: list[tuple[str, TierConfig, int]] = []
    unloads: list[str] = []
    state = {"id": "model-x", "context": 16_384}
    adapter = tier_adapter(state, loads, unloads)
    reused = await adapter.ensure_loaded("x", tier="low")
    assert (reused.tier, loads, unloads) == ("low", [], [])
    switched = await adapter.ensure_loaded("x", tier="high")
    assert switched.tier == "high"
    assert unloads == ["model-x"]
    assert [tier.name for _, tier, _ in loads] == ["high"]
    await adapter.close()


@pytest.mark.asyncio
async def test_accept_higher_adopts_a_larger_loaded_tier_without_a_reload() -> None:
    loads: list[tuple[str, TierConfig, int]] = []
    adapter = tier_adapter({"id": "model-x", "context": 98_304}, loads, [])
    loaded = await adapter.ensure_loaded("x", tier="low", accept_higher=True)
    assert (loaded.tier, loaded.context_limit, loads) == ("high", 98_304, [])
    await adapter.close()


@pytest.mark.asyncio
async def test_hand_loaded_instance_at_an_unknown_context_is_reloaded_at_the_tier() -> None:
    loads: list[tuple[str, TierConfig, int]] = []
    unloads: list[str] = []
    adapter = tier_adapter({"id": "model-x", "context": 50_000}, loads, unloads)
    loaded = await adapter.ensure_loaded("x", tier="low", accept_higher=True)
    assert (loaded.tier, unloads, [tier.name for _, tier, _ in loads]) == ("low", ["model-x"], ["low"])
    await adapter.close()


@pytest.mark.asyncio
async def test_a_failed_sdk_load_fails_retryably_and_names_the_tier() -> None:
    async def broken(model_key: str, tier: TierConfig, parallel: int) -> tuple[str, float]:
        raise RuntimeError("socket closed")

    adapter = tier_adapter({}, [], [])
    adapter._tier_loader = broken
    with pytest.raises(BackendFailure) as caught:
        await adapter.ensure_loaded("x", tier="low")
    assert (caught.value.code, caught.value.retryable) == ("model_load_failed", True)
    assert "'low'" in str(caught.value)
    await adapter.close()


# --- OpenAI route and calibration ---------------------------------------------------------------


def test_openai_model_suffix_or_field_selects_the_tier(tmp_path: Path) -> None:
    config = load_config(PRESET)
    assert _resolve_profile_and_tier(config, "35b:high", None) == ("35b", "high")
    assert _resolve_profile_and_tier(config, "35b", "medium") == ("35b", "medium")
    assert _resolve_profile_and_tier(config, "9b-nothink:auto", None) == ("9b-nothink", None)
    with pytest.raises(SchedulerError):
        _resolve_profile_and_tier(config, "35b:high", "low")


def test_calibration_measures_from_the_first_output_token() -> None:
    times = iter([2.0, 12.0])  # the first content chunk arrives at 2 s and the stream ends at 12 s
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        'data: {"choices":[{"delta":{"content":" there"}}],"usage":{"prompt_tokens":8000,"completion_tokens":301}}',
        "data: [DONE]",
    ]
    ttft, generation, usage = measure_stream(lines, started=0.0, clock=lambda: next(times))
    item = TierMeasurement("35b", "low", 32_768, usage["prompt_tokens"], usage["completion_tokens"], ttft, generation)
    assert (ttft, generation) == (2.0, 10.0)
    assert item.prefill_tps == 4000.0
    assert item.generation_tps == 30.0


def test_overflow_needs_slower_prompt_processing_too_otherwise_it_is_a_slower_tier() -> None:
    def m(profile: str, tier: str, prefill: float, tps: float) -> TierMeasurement:
        return TierMeasurement(profile, tier, 1, int(prefill), int(tps * 10) + 1, 1.0, 10.0)

    # Measured: 9B at 192K on 2026-11-03, and the 4B on 2026-11-04 (96K overflowed; 64K q8_0 only generates slower).
    marks = verdicts([
        m("9b", "low", 3500, 91.0), m("9b", "medium", 3740, 85.7), m("9b", "high", 1313, 60.0),
        m("4b", "low", 4950, 120.2), m("4b", "medium", 4700, 72.5), m("4b", "high", 700, 43.5),
    ])
    assert marks == {
        ("9b", "low"): "ok", ("9b", "medium"): "ok", ("9b", "high"): "likely VRAM overflow",
        ("4b", "low"): "ok", ("4b", "medium"): "slower generation", ("4b", "high"): "likely VRAM overflow",
    }


# --- idle unload --------------------------------------------------------------------------------


@dataclass
class ReleasingBackend:
    loads: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)
    fail_release: bool = False

    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        self.loads.append(profile)
        return LoadedModel(profile, profile, f"inst-{len(self.loads)}", 32_000, 1, 0.01)

    async def release(self, loaded: LoadedModel) -> bool:
        if self.fail_release:
            raise BackendFailure("backend_busy", "busy", retryable=True)
        self.released.append(loaded.instance_id)
        return True

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {}}


def idle_scheduler(tmp_path: Path, backend: Any, seconds: float) -> Scheduler:
    from dataclasses import replace

    config = replace(load_config(PRESET), idle_unload_seconds=seconds)
    return Scheduler(config, backend, store=MetadataStore(tmp_path / "state.sqlite3"))


async def wait_until(predicate: Any, seconds: float = 2.0) -> None:
    for _ in range(int(seconds / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


@pytest.mark.asyncio
async def test_idle_model_is_released_and_the_next_job_loads_it_again(tmp_path: Path) -> None:
    backend = ReleasingBackend()
    scheduler = idle_scheduler(tmp_path, backend, 0.05)
    await scheduler.start()
    try:
        await finish(scheduler, [(await scheduler.submit(job("a", "4b"))).job_id])
        await wait_until(lambda: backend.released == ["inst-1"])
        assert scheduler.status()["active_profile"] is None
        await finish(scheduler, [(await scheduler.submit(job("b", "4b"))).job_id])
        assert backend.loads == ["4b", "4b"]
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_idle_unload_is_off_at_zero_and_a_failed_release_keeps_the_model(tmp_path: Path) -> None:
    for seconds, fail in ((0.0, False), (0.05, True)):
        backend = ReleasingBackend(fail_release=fail)
        scheduler = idle_scheduler(tmp_path / f"run-{seconds}", backend, seconds)
        await scheduler.start()
        try:
            await finish(scheduler, [(await scheduler.submit(job("a", "4b"))).job_id])
            await asyncio.sleep(0.2)
            assert backend.released == []
            assert scheduler.status()["active_profile"] == "4b"
        finally:
            await scheduler.shutdown()


@pytest.mark.asyncio
async def test_release_unloads_only_the_brokers_instance_once_idle() -> None:
    unloads: list[str] = []
    adapter = tier_adapter({"id": "model-x", "context": 16_384}, [], unloads)
    loaded = LoadedModel("x", "model-x", "model-x", 16_384, 2, tier="low")
    assert await adapter.release(loaded) is True
    assert unloads == ["model-x"]
    # Already gone (for example unloaded by hand): nothing to unload.
    assert await adapter.release(loaded) is True
    assert unloads == ["model-x"]
    await adapter.close()


# --- concurrency calibration --------------------------------------------------------------------


def test_usable_slots_is_the_largest_count_that_still_clearly_raises_throughput() -> None:
    from local_llm_broker.calibrate_concurrency import recommend_slots

    assert recommend_slots({1: 120.0, 2: 210.0, 3: 290.0, 4: 300.0}) == 3
    assert recommend_slots({1: 37.0, 2: 38.0}) == 1
    # A failed or missing round does not count as a gain.
    assert recommend_slots({1: 90.0, 2: None}) == 1


def test_capacity_prompts_fill_the_shared_context_by_the_given_share() -> None:
    from local_llm_broker.calibrate_concurrency import OUTPUT_TOKENS, capacity_prompt_tokens

    per_job = capacity_prompt_tokens(16_384, 4, 1.5)
    assert 4 * (per_job + OUTPUT_TOKENS + 512) == pytest.approx(16_384 * 1.5, abs=4)
    # Each job alone still fits the tier, so only the shared pool overfills.
    assert per_job + OUTPUT_TOKENS + 512 <= 16_384


def test_combined_speed_counts_all_tokens_over_the_shared_generation_span() -> None:
    from local_llm_broker.calibrate_concurrency import Round, _outcome
    from local_llm_broker.calibrate_probe import JobResult

    jobs = [
        JobResult(True, 2000, 201, first_token_at=1.0, ended_at=3.0, finish_reason="stop"),
        JobResult(True, 2000, 201, first_token_at=1.5, ended_at=5.0, finish_reason="length"),
        JobResult(False, error="inference_failed"),
    ]
    round_ = Round(jobs, started_at=0.0)
    assert round_.combined_tps == pytest.approx(402 / 4.0)
    assert round_.per_job_tps == pytest.approx((100.0 + 200 / 3.5) / 2)
    assert _outcome(round_) == "1 of 3 failed: inference_failed"


@pytest.mark.asyncio
async def test_concurrency_round_sends_jobs_at_the_same_time_with_the_tier() -> None:
    from local_llm_broker.calibrate_concurrency import measure_tier

    seen: list[dict[str, Any]] = []
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        body = json.loads(request.content)
        seen.append(body)
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        stream = (
            'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=stream.encode(), headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1235") as client:
        result = await measure_tier(client, "4b-nothink", "low", 16_384, 4)
    assert result.error is None
    assert sorted(result.throughput) == [1, 2, 3, 4]
    assert sorted(result.capacity) == [0.5, 0.9, 1.2, 1.5, 1.8]
    assert peak == 4
    assert {body["tier"] for body in seen} == {"low"}
    assert all(not r.failures for r in result.throughput.values())
