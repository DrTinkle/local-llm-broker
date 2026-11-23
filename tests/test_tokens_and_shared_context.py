from __future__ import annotations

import asyncio
import base64
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from support import PRESET

from local_llm_broker.backend import LoadedModel
from local_llm_broker.calibrate_probe import content_samples, png_data_url
from local_llm_broker.config import load_config
from local_llm_broker.contracts import JobSpec, JobState, ProfileConfig
from local_llm_broker.lm_studio import BackendFailure, LMStudioAdapter
from local_llm_broker.scheduler import Scheduler, SchedulerError
from local_llm_broker.storage import MetadataStore
from local_llm_broker.tokens import (
    TOOLS_OVERHEAD,
    UNKNOWN_IMAGE_TOKENS,
    estimate_prompt_tokens,
    estimate_text_tokens,
    image_tokens,
)

# Real token counts measured on 2026-11-04 (the higher of the 4B and 9B tokenizers) for content_samples().
MEASURED = {
    "English prose": 3002, "Finnish prose": 5999, "Python code": 5900, "JSON": 8421,
    "CSV numbers": 16026, "Log lines": 11374, "Markdown table": 9200,
}


def test_the_estimate_errs_high_but_not_wildly_on_every_measured_content_type() -> None:
    for name, text in content_samples().items():
        ratio = estimate_text_tokens(text) / MEASURED[name]
        assert 1.0 <= ratio <= 1.3, (name, ratio)


def test_image_tokens_follow_the_measured_pixel_rule_for_each_format() -> None:
    assert [image_tokens(png_data_url(side, side)) for side in (256, 512, 1024, 1536)] == [66, 258, 1026, 2306]
    gif = b"GIF89a" + struct.pack("<HH", 640, 480) + b"\x00" * 8
    jpeg = b"\xff\xd8\xff\xe0" + struct.pack(">H", 16) + b"\x00" * 14 + b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 300, 400) + b"\x00" * 12
    webp = b"RIFF" + b"\x00" * 4 + b"WEBPVP8X" + b"\x00" * 8 + (799).to_bytes(3, "little") + (599).to_bytes(3, "little")
    for data, expected in ((gif, 20 * 15 + 2), (jpeg, 13 * 10 + 2), (webp, 25 * 19 + 2)):
        assert image_tokens("data:image/x;base64," + base64.b64encode(data).decode()) == expected
    assert image_tokens("data:image/png;base64," + "A" * 400) == UNKNOWN_IMAGE_TOKENS


def test_tools_are_counted_as_text_plus_a_fixed_preamble() -> None:
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    assert estimate_prompt_tokens(messages, {"tools": tools}) - estimate_prompt_tokens(messages, {}) > TOOLS_OVERHEAD


@pytest.mark.asyncio
async def test_adapter_counts_with_the_loaded_instance_and_reports_failures_retryably() -> None:
    seen: list[tuple[str, str]] = []

    async def counter(identifier: str, text: str) -> int:
        seen.append((identifier, text))
        if text == "boom":
            raise RuntimeError("socket closed")
        return 42

    adapter = LMStudioAdapter(
        "http://test", {"x": ProfileConfig("x", "model-x", 16_000, 1)},
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})), base_url="http://test"),
        token_counter=counter,
    )
    loaded = LoadedModel("x", "model-x", "inst-1", 16_000, 1)
    assert await adapter.count_tokens(loaded, "hello") == 42
    assert seen == [("inst-1", "hello")]
    with pytest.raises(BackendFailure) as caught:
        await adapter.count_tokens(loaded, "boom")
    assert (caught.value.code, caught.value.retryable) == ("token_count_failed", True)
    await adapter.close()


# --- shared context in the scheduler --------------------------------------------------------------------------


@dataclass
class CountingBackend:
    """Tiered fake: contexts from the default config, counts tokens as characters // `chars_per_token`."""

    contexts: dict[str, int]
    parallel: int = 4
    chars_per_token: int = 1
    fail_count: bool = False
    loads: list[tuple[str, str | None]] = field(default_factory=list)
    active: int = 0
    peak: int = 0
    counted: int = 0

    async def ensure_loaded(self, profile: str, *, tier: str | None = None, accept_higher: bool = False) -> LoadedModel:
        self.loads.append((profile, tier))
        assert tier is not None
        return LoadedModel(profile, f"model-{profile}", f"inst-{tier}", self.contexts[tier], self.parallel, 0.01, tier=tier)

    async def count_tokens(self, loaded: LoadedModel, text: str) -> int:
        self.counted += 1
        if self.fail_count:
            raise BackendFailure("token_count_failed", "no", retryable=True)
        return len(text) // self.chars_per_token

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.03)
        self.active -= 1
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {}}


def job(label: str, profile: str, chars: int, *, tier: str | None = None, max_tokens: int = 500) -> JobSpec:
    # Digits estimate at about 1.3 tokens each, which keeps the estimate above the fake's exact count.
    return JobSpec("client", "project", label, profile, [{"role": "user", "content": "1" * chars}], {"max_tokens": max_tokens}, tier=tier)


def scheduler_for(tmp_path: Path, backend: Any) -> Scheduler:
    return Scheduler(load_config(PRESET), backend, store=MetadataStore(tmp_path / "state.sqlite3"))


async def finish(scheduler: Scheduler, ids: list[str]) -> list[JobState]:
    for _ in range(600):
        states = [scheduler.get(job_id).state for job_id in ids]  # type: ignore[union-attr]
        if all(state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED} for state in states):
            return states
        await asyncio.sleep(0.005)
    raise AssertionError("jobs did not finish")


@pytest.mark.asyncio
async def test_parallel_jobs_that_do_not_fit_the_shared_context_together_wait(tmp_path: Path) -> None:
    # 4 slots on a 16,384 context; each job needs about 5,000 + 500 + 512 tokens, so only two fit at once.
    backend = CountingBackend({"low": 16_384, "medium": 32_768, "high": 65_536})
    scheduler = scheduler_for(tmp_path, backend)
    await scheduler.start()
    try:
        ids = [(await scheduler.submit(job(f"j{i}", "4b-nothink", 5_000, tier="low"))).job_id for i in range(4)]
        assert await finish(scheduler, ids) == [JobState.SUCCEEDED] * 4
    finally:
        await scheduler.shutdown()
    assert backend.peak == 2
    assert backend.counted == 4


@pytest.mark.asyncio
async def test_small_parallel_jobs_still_run_together(tmp_path: Path) -> None:
    backend = CountingBackend({"low": 16_384, "medium": 32_768, "high": 65_536})
    scheduler = scheduler_for(tmp_path, backend)
    await scheduler.start()
    try:
        ids = [(await scheduler.submit(job(f"j{i}", "4b-nothink", 1_000, tier="low"))).job_id for i in range(4)]
        assert await finish(scheduler, ids) == [JobState.SUCCEEDED] * 4
    finally:
        await scheduler.shutdown()
    assert backend.peak == 4


@pytest.mark.asyncio
async def test_an_auto_job_whose_exact_count_is_too_big_moves_to_a_larger_tier(tmp_path: Path) -> None:
    # Letters estimate low (about 7 characters per token); the fake tokenizer counts one token per character.
    backend = CountingBackend({"low": 16_384, "medium": 32_768, "high": 65_536})
    scheduler = scheduler_for(tmp_path, backend)
    spec = JobSpec("client", "project", "big", "4b-nothink", [{"role": "user", "content": "x" * 20_000}], {"max_tokens": 500})
    assert scheduler.resolve_tier(spec) == ("low", True)
    await scheduler.start()
    try:
        assert await finish(scheduler, [(await scheduler.submit(spec)).job_id]) == [JobState.SUCCEEDED]
    finally:
        await scheduler.shutdown()
    assert backend.loads == [("4b-nothink", "low"), ("4b-nothink", "medium")]


@pytest.mark.asyncio
async def test_an_explicit_tier_job_too_big_by_exact_count_fails_with_the_numbers(tmp_path: Path) -> None:
    backend = CountingBackend({"low": 16_384, "medium": 32_768, "high": 65_536})
    scheduler = scheduler_for(tmp_path, backend)
    spec = JobSpec("client", "project", "big", "4b-nothink", [{"role": "user", "content": "x" * 20_000}], {"max_tokens": 500}, tier="low")
    await scheduler.start()
    try:
        job_id = (await scheduler.submit(spec)).job_id
        assert await finish(scheduler, [job_id]) == [JobState.FAILED]
        error = scheduler.get(job_id).error  # type: ignore[union-attr]
    finally:
        await scheduler.shutdown()
    assert error is not None and error["code"] == "context_exceeded"
    assert "16384-token context" in error["message"]
    assert backend.loads == [("4b-nothink", "low")]


@pytest.mark.asyncio
async def test_a_failed_count_falls_back_to_the_estimate(tmp_path: Path) -> None:
    backend = CountingBackend({"low": 16_384, "medium": 32_768, "high": 65_536}, fail_count=True)
    scheduler = scheduler_for(tmp_path, backend)
    await scheduler.start()
    try:
        ids = [(await scheduler.submit(job(f"j{i}", "4b-nothink", 1_000))).job_id for i in range(2)]
        assert await finish(scheduler, ids) == [JobState.SUCCEEDED] * 2
    finally:
        await scheduler.shutdown()


def test_a_job_the_estimate_puts_over_the_largest_tier_but_may_fit_is_accepted_at_it(tmp_path: Path) -> None:
    scheduler = scheduler_for(tmp_path, CountingBackend({}))
    largest = scheduler.config.profiles["4b-nothink"].context_ceiling
    # Digits estimate at 1.109 x 1.15 tokens each: choose a size whose estimate is just over the largest tier.
    chars = int((largest - 500 - 512 - 40) / (1.109 * 1.15)) + 200
    spec = job("edge", "4b-nothink", chars)
    assert scheduler._estimate_prompt_tokens(spec) + 500 + 512 > largest
    assert scheduler.resolve_tier(spec) == ("high", True)
    with pytest.raises(SchedulerError):
        scheduler.resolve_tier(job("huge", "4b-nothink", largest * 2))


# --- the long-lived SDK token counter -------------------------------------------------------------------------


class FakeSdk:
    def __init__(self, *, fail_connect: bool = False, fail_on: str | None = None) -> None:
        self.connects = 0
        self.fail_connect = fail_connect
        self.fail_on = fail_on
        sdk = self

        class Llm:
            async def _count_tokens(self, identifier: str, text: str) -> int:
                if text == sdk.fail_on:
                    sdk.fail_on = None
                    raise RuntimeError("socket closed")
                return len(text)

        class Client:
            def __init__(self, host: str) -> None:
                self.llm = Llm()

            async def __aenter__(self) -> Any:
                sdk.connects += 1
                if sdk.fail_connect:
                    raise OSError("connection refused")
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

        self.Client = Client


@pytest.mark.asyncio
async def test_token_counter_reuses_one_connection_and_reconnects_after_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import lmstudio

    from local_llm_broker.lms_sdk import SdkTokenCounter

    fake = FakeSdk(fail_on="bad")
    monkeypatch.setattr(lmstudio, "AsyncClient", fake.Client)
    counter = SdkTokenCounter("http://127.0.0.1:1234")
    assert [await counter.count("inst", text) for text in ("a", "bb", "ccc")] == [1, 2, 3]
    assert fake.connects == 1
    with pytest.raises(RuntimeError):
        await counter.count("inst", "bad")
    assert await counter.count("inst", "dddd") == 4
    assert fake.connects == 2
    await counter.close()


@pytest.mark.asyncio
async def test_token_counter_that_cannot_connect_fails_the_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import lmstudio

    from local_llm_broker.lms_sdk import SdkTokenCounter

    monkeypatch.setattr(lmstudio, "AsyncClient", FakeSdk(fail_connect=True).Client)
    counter = SdkTokenCounter("http://127.0.0.1:1234")
    with pytest.raises(OSError):
        await asyncio.wait_for(counter.count("inst", "a"), timeout=2)
    await counter.close()


@pytest.mark.asyncio
async def test_parallel_jobs_start_together_once_counted(tmp_path: Path) -> None:
    # Jobs submitted together must start together, not one per count.
    backend = CountingBackend({"low": 16_384, "medium": 32_768, "high": 65_536})
    scheduler = scheduler_for(tmp_path, backend)
    await scheduler.start()
    try:
        specs = [job(f"j{i}", "4b-nothink", 500, tier="low") for i in range(4)]
        ids = [snapshot.job_id for snapshot in await asyncio.gather(*(scheduler.submit(spec) for spec in specs))]
        assert await finish(scheduler, ids) == [JobState.SUCCEEDED] * 4
    finally:
        await scheduler.shutdown()
    assert backend.peak == 4
