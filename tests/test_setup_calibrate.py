from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from local_llm_broker.calibrate_concurrency import Round
from local_llm_broker.calibrate_direct import DirectSession, Probe
from local_llm_broker.calibrate_probe import JobResult
from local_llm_broker.contracts import TierConfig
from local_llm_broker.setup_calibrate import (
    CalibrationResult,
    Progress,
    calibrate_model,
    floor_step,
    probe_thinking,
    tiers_from_max,
)
from local_llm_broker.setup_discover import Hardware, ModelInfo

HW = Hardware("Test GPU", 12_288, 65_536)


def model_info(key: str = "org/m", *, moe: bool = False, max_context: int = 262_144) -> ModelInfo:
    return ModelInfo(key, key, "arch", "9B", 6_000_000_000, max_context, False, moe, None)


def thinking_reply() -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": "42", "reasoning_content": "17 + 25 is 42"}}]}


def plain_reply() -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": "42"}}]}


class FakeSession:
    """Stands in for DirectSession: no LM Studio, no network. `limit` says the largest context that loads."""

    def __init__(
        self,
        limit: Callable[[TierConfig, int], int] | None = None,
        *,
        gen: Callable[[TierConfig], float] | None = None,
        combined: dict[int, float] | None = None,
        chat: Callable[[httpx.Request], httpx.Response] | None = None,
    ) -> None:
        self.limit = limit or (lambda tier, parallel: 10**9)
        self.gen = gen or (lambda tier: 100.0)
        self.combined = combined or {}
        self.calls: list[tuple[int, str, float | None, int, int]] = []
        self.loader_calls: list[tuple[str, TierConfig, int]] = []
        self.requests: list[dict[str, Any]] = []
        self.unloads = 0
        self.fail_on: BaseException | None = None
        self.fail_after = 10**9
        self.chat = chat or (lambda request: httpx.Response(200, json=plain_reply()))
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler), base_url="http://127.0.0.1:1234")

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        self.requests.append(json.loads(request.content))
        return self.chat(request)

    async def _loader(self, model: str, tier: TierConfig, parallel: int) -> tuple[str, float]:
        self.loader_calls.append((model, tier, parallel))
        return "inst-1", 0.1

    async def unload_all(self) -> None:
        self.unloads += 1

    async def try_config(self, model: str, tier: TierConfig, parallel: int, *, probe_tokens: int = 8_000, jobs: int = 0) -> Probe:
        self.calls.append((tier.context_length, tier.kv_cache, tier.cpu_expert_ratio, parallel, jobs))
        if len(self.calls) > self.fail_after and self.fail_on is not None:
            raise self.fail_on
        probe = Probe(model, tier, parallel)
        if tier.context_length > self.limit(tier, parallel):
            probe.error = "load failed: out of memory"
            return probe
        probe.loaded = True
        gen = self.gen(tier)
        run = JobResult(True, 6_000, 129, started_at=0.0, first_token_at=1.0, ended_at=1.0 + 128 / gen)
        probe.run = run
        if jobs:
            speed = self.combined[jobs]
            ended = 1.0 + jobs * 129 / speed
            probe.round = Round([JobResult(True, 1_500, 129, started_at=0.0, first_token_at=1.0, ended_at=ended) for _ in range(jobs)], 0.0)
            probe.run = probe.round.jobs[0]
        return probe

    def as_session(self) -> DirectSession:
        return cast(DirectSession, self)


# --- tier rules -------------------------------------------------------------------------------------------------


def test_floor_step() -> None:
    assert floor_step(20_480) == 20_480
    assert floor_step(20_479.9) == 16_384
    assert floor_step(100) == 0


def test_tiers_from_max() -> None:
    tiers = tiers_from_max(57_344, 102_400, None)
    assert {name: (tier.context_length, tier.kv_cache) for name, tier in tiers.items()} == {
        "low": (16_384, "f16"),
        "medium": (40_960, "f16"),
        "high": (86_016, "q8_0"),
    }
    assert all(tier.name == name and tier.cpu_expert_ratio is None and tier.try_mmap is None for name, tier in tiers.items())
    assert list(tiers) == ["low", "medium", "high"]

    # f16 is the larger maximum, so high stays f16; medium is f16 too.
    only_f16 = tiers_from_max(65_536, None, None)
    assert [(t.context_length, t.kv_cache) for t in only_f16.values()] == [(12_288, "f16"), (24_576, "f16"), (53_248, "f16")]

    # medium is q8_0 when f16 cannot reach it.
    wide = tiers_from_max(20_480, 131_072, None)
    assert wide["high"].kv_cache == "q8_0"
    assert wide["medium"].context_length == floor_step(0.5 * wide["high"].context_length)
    assert wide["medium"].kv_cache == "q8_0"


def test_tiers_from_max_collapses_duplicates_and_small_models() -> None:
    # medium would equal low (both 4096), so it is dropped.
    collapsed = tiers_from_max(15_000, None, None)
    assert [(name, tier.context_length) for name, tier in collapsed.items()] == [("low", 4_096), ("high", 12_288)]

    small = tiers_from_max(12_288, None, None)
    assert list(small) == ["low"]
    assert 4_096 <= small["low"].context_length <= 12_288

    tiny = tiers_from_max(8_192, None, None)
    assert list(tiny) == ["low"]
    assert tiny["low"].context_length == 4_096

    for best in range(4_096, 140_000, 997):
        result = tiers_from_max(best, None, None)
        contexts = [tier.context_length for tier in result.values()]
        assert result and contexts == sorted(set(contexts)) and max(contexts) <= best


def test_tiers_from_max_copies_ratio_and_handles_nothing() -> None:
    tiers = tiers_from_max(65_536, 98_304, 0.75)
    assert all(tier.cpu_expert_ratio == 0.75 and tier.try_mmap is False for tier in tiers.values())
    assert tiers_from_max(None, None, None) == {}
    assert tiers_from_max(0, None, 0.5) == {}


# --- levels -----------------------------------------------------------------------------------------------------


async def test_quick_level() -> None:
    fake = FakeSession(limit=lambda tier, parallel: 65_536, chat=lambda request: httpx.Response(200, json=thinking_reply()))
    result = await calibrate_model(fake.as_session(), model_info(), HW, "quick", 3)
    assert result.failed is None
    assert result.tiers == tiers_from_max(65_536, None, None)
    assert result.parallel == 3
    assert len(fake.calls) <= 8
    assert all(kv == "f16" and jobs == 0 for _, kv, _, _, jobs in fake.calls)  # f16 only, no slot sweep
    assert fake.calls[0][0] == 8_192
    assert len(fake.loader_calls) == 1  # the thinking probe loads once, at the low tier
    assert fake.loader_calls[0][1] == result.tiers["low"]
    assert (result.thinks, result.can_disable_thinking) == (True, False)  # a reply that always thinks
    assert fake.unloads >= 1


async def test_quick_level_on_a_moe_model_uses_the_default_ratio() -> None:
    fake = FakeSession(limit=lambda tier, parallel: 65_536)
    result = await calibrate_model(fake.as_session(), model_info(moe=True), HW, "quick", 1)
    assert result.tiers == tiers_from_max(65_536, None, 0.5)
    assert all(tier.try_mmap is False for tier in result.tiers.values())
    assert (result.thinks, result.can_disable_thinking) == (False, False)


async def test_quick_level_caps_the_search_at_the_models_own_maximum() -> None:
    fake = FakeSession()
    result = await calibrate_model(fake.as_session(), model_info(max_context=32_768), HW, "quick", 1)
    assert max(call[0] for call in fake.calls) == 32_768
    assert result.tiers["high"].context_length <= 32_768


async def test_thorough_level_runs_all_combos_slots_and_experts() -> None:
    def limit(tier: TierConfig, parallel: int) -> int:
        if tier.kv_cache == "f16":
            return 57_344
        return 114_688 if tier.cpu_expert_ratio == 0.75 else 98_304

    ratios = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    fake = FakeSession(limit=limit, gen=lambda tier: 100.0 - 100.0 * abs((tier.cpu_expert_ratio or 0.0) - 0.9), combined={1: 50.0, 2: 70.0})
    result = await calibrate_model(fake.as_session(), model_info(moe=True), HW, "thorough", 1)

    assert result.failed is None
    searched = {(kv, ratio) for _, kv, ratio, _, jobs in fake.calls if jobs == 0}
    assert {("f16", 0.5), ("q8_0", 0.5), ("q8_0", 0.75)} <= searched
    assert any(ctx == 262_144 for ctx, *_ in fake.calls)
    # Experts: every sweep ratio at or above the searched one is loaded, one slot, at the low tier.
    swept = {ratio for _, _, ratio, parallel, jobs in fake.calls if parallel == 1 and jobs == 0 and ratio in ratios}
    assert {0.9, 1.0} <= swept
    # The fastest ratio that fits (0.9) is used for every tier and recorded.
    assert result.tiers == tiers_from_max(57_344, 114_688, 0.9)
    assert any("0.9" in note for note in result.notes)
    # Slots: 2 slots are at least 15% faster than 1, so the result is 2, and the speeds are recorded.
    assert {jobs for *_, jobs in fake.calls} >= {1, 2}
    assert result.parallel == 2
    assert any("slot" in note for note in result.notes)
    assert len(fake.loader_calls) == 1


async def test_thorough_slots_keep_the_smaller_count_without_a_clear_gain() -> None:
    fake = FakeSession(combined={1: 50.0, 2: 55.0})
    result = await calibrate_model(fake.as_session(), model_info(), HW, "thorough", 1)
    assert result.parallel == 1
    assert result.tiers


async def test_moe_that_does_not_fit_tries_expert_ratios() -> None:
    fake = FakeSession(limit=lambda tier, parallel: 40_960 if tier.cpu_expert_ratio == 1.0 else 0)
    result = await calibrate_model(fake.as_session(), model_info(moe=True), HW, "quick", 1)
    baselines = [ratio for ctx, _, ratio, _, _ in fake.calls if ctx == 8_192]
    assert baselines[:3] == [0.5, 0.75, 1.0]
    assert result.failed is None
    assert result.tiers == tiers_from_max(32_768, None, 1.0)


@pytest.mark.parametrize("level", ["quick", "thorough"])
async def test_model_that_never_loads_is_reported_not_guessed(level: str) -> None:
    for moe in (False, True):
        fake = FakeSession(limit=lambda tier, parallel: 0)
        result = await calibrate_model(fake.as_session(), model_info(moe=moe), HW, level, 2)  # type: ignore[arg-type]
        assert result.failed == "did not load at 8,192 context"
        assert result.tiers == {}
        assert (result.thinks, result.can_disable_thinking) == (None, None)
        assert result.parallel == 2
        assert fake.loader_calls == []  # no thinking probe either


# --- thinking probe ---------------------------------------------------------------------------------------------


def chat_handler(first: dict[str, Any], second: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=second if body.get("reasoning_effort") == "none" else first)

    return handler


async def test_probe_thinking() -> None:
    tier = TierConfig("low", 8_192)

    fake = FakeSession(chat=chat_handler(thinking_reply(), plain_reply()))
    assert await probe_thinking(fake.as_session(), "org/m", tier) == (True, True)
    assert fake.loader_calls == [("org/m", tier, 1)]
    first, second = fake.requests
    assert first == {
        "model": "inst-1",
        "messages": [{"role": "user", "content": "What is 17 plus 25? Answer with just the number."}],
        "max_tokens": 512,
        "temperature": 0,
    }
    assert second == {**first, "reasoning_effort": "none"}
    assert fake.unloads == 1

    plain = FakeSession(chat=chat_handler(plain_reply(), plain_reply()))
    assert await probe_thinking(plain.as_session(), "org/m", tier) == (False, False)
    assert plain.unloads == 1

    reasoning_field = {"choices": [{"message": {"content": "42", "reasoning": "adding"}}]}
    stubborn = FakeSession(chat=chat_handler(reasoning_field, {"choices": [{"message": {"content": "<think>hm</think>42"}}]}))
    assert await probe_thinking(stubborn.as_session(), "org/m", tier) == (True, False)

    failing = FakeSession(chat=lambda request: httpx.Response(500, json={"error": "boom"}))
    assert await probe_thinking(failing.as_session(), "org/m", tier) == (False, False)
    assert failing.unloads == 1

    refusing = FakeSession()

    async def no_load(model: str, tier: TierConfig, parallel: int) -> tuple[str, float]:
        raise RuntimeError("out of memory")

    refusing._loader = no_load  # type: ignore[method-assign]
    assert await probe_thinking(refusing.as_session(), "org/m", tier) == (False, False)
    assert refusing.unloads == 1


# --- resume -----------------------------------------------------------------------------------------------------


def sample_result(key: str = "org/a", failed: str | None = None) -> CalibrationResult:
    return CalibrationResult(key, tiers_from_max(65_536, None, 0.5), 2, True, False, ["note"], failed)


def test_progress_resume(tmp_path: Path) -> None:
    path = tmp_path / "setup-progress.json"
    assert Progress.load(path, "quick").done("org/a") is None  # no file yet

    progress = Progress.load(path, "quick")
    saved = sample_result("org/a")
    progress.save(saved)
    progress.save(sample_result("org/b"))
    assert not list(tmp_path.glob("*.tmp")) and not [p for p in tmp_path.iterdir() if p.name != path.name]

    again = Progress.load(path, "quick")
    assert again.done("org/a") == saved
    assert again.done("org/c") is None
    assert isinstance(again.done("org/a"), CalibrationResult)
    assert Progress.load(path, "thorough").done("org/a") is None

    # Saving at another level keeps the first level's entries.
    other = Progress.load(path, "thorough")
    other.save(sample_result("org/a"))
    assert Progress.load(path, "quick").done("org/b") is not None
    assert Progress.load(path, "thorough").done("org/a") is not None

    # A failed result is not saved, so a rerun retries it.
    failed = Progress.load(path, "quick")
    failed.save(sample_result("org/f", failed="did not load at 8,192 context"))
    assert Progress.load(path, "quick").done("org/f") is None

    path.write_text("{not json", encoding="utf-8")
    assert Progress.load(path, "quick").done("org/a") is None
    path.write_text(json.dumps({"results": {"quick:org/a": {"model_key": 3}}}), encoding="utf-8")
    assert Progress.load(path, "quick").done("org/a") is None
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert Progress.load(path, "quick").done("org/a") is None


def test_interrupt_unloads(tmp_path: Path) -> None:
    path = tmp_path / "setup-progress.json"
    Progress.load(path, "quick").save(sample_result("org/done"))

    fake = FakeSession()
    fake.fail_on, fake.fail_after = KeyboardInterrupt(), 1
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(calibrate_model(fake.as_session(), model_info("org/next"), HW, "quick", 1))
    assert fake.unloads >= 1
    assert Progress.load(path, "quick").done("org/done") == sample_result("org/done")
    assert Progress.load(path, "quick").done("org/next") is None


async def test_cancellation_unloads() -> None:
    fake = FakeSession()
    fake.fail_on, fake.fail_after = asyncio.CancelledError(), 1
    with pytest.raises(asyncio.CancelledError):
        await calibrate_model(fake.as_session(), model_info(), HW, "quick", 1)
    assert fake.unloads >= 1
