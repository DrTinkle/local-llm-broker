from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import pytest
from support import PRESET

from local_llm_broker import scheduler as scheduler_module
from local_llm_broker.backend import LoadedModel
from local_llm_broker.config import load_config
from local_llm_broker.contracts import JobSpec, JobState, ProfileConfig
from local_llm_broker.lm_studio import BackendFailure
from local_llm_broker.scheduler import ResultExpired, Scheduler, SchedulerError
from local_llm_broker.storage import MetadataStore
from local_llm_broker.telemetry import EventWriter
from local_llm_broker.tokens import (
    MESSAGE_OVERHEAD,
    REQUEST_OVERHEAD,
    UNKNOWN_IMAGE_TOKENS,
    estimate_text_tokens,
)

_T = TypeVar("_T")


def present(value: _T | None) -> _T:
    assert value is not None
    return value


@dataclass
class FakeBackend:
    limits: dict[str, int] = field(default_factory=lambda: {"4b": 2, "35b": 1})
    started: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    payloads: list[dict[str, Any]] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    active: int = 0
    max_active: int = 0

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        raise NotImplementedError
        yield b""

    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def unload(self, instance_id: str) -> None:
        return None

    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        self.loaded.append(profile)
        # Like the real adapter, the loaded model carries the profile's model_id (twin profiles share it).
        model_id = load_config(PRESET).profiles[profile].model_id
        return LoadedModel(profile, model_id, f"instance-{profile}", 32_000, self.limits[profile], 0.01)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        label = str(payload["metadata"]["task_label"])
        profile = str(payload["model"])
        self.started.append(label)
        self.payloads.append(payload)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        self.completed.append(label)
        return {
            "id": f"completion-{label}",
            "choices": [{"message": {"role": "assistant", "content": f"answer-{label}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "model": profile,
        }


def spec(label: str, profile: str, client: str = "client-a") -> JobSpec:
    return JobSpec(client, "project", label, profile, [{"role": "user", "content": label}], {"max_tokens": 8})


async def wait_for_all(scheduler: Scheduler, ids: list[str]) -> None:
    for _ in range(300):
        snapshots = [scheduler.get(job_id) for job_id in ids]
        if all(snapshot and snapshot.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.INTERRUPTED} for snapshot in snapshots):
            return
        await asyncio.sleep(0.005)
    raise AssertionError("jobs did not reach terminal state")


@pytest.mark.asyncio
async def test_fifo_model_barrier_prevents_later_same_model_job_from_jumping(tmp_path: Path) -> None:
    config = load_config(PRESET)
    backend = FakeBackend()
    scheduler = Scheduler(config, backend, store=MetadataStore(tmp_path / "scheduler.sqlite3"))
    await scheduler.start()
    try:
        ids = [
            (await scheduler.submit(spec("A", "4b"))).job_id,
            (await scheduler.submit(spec("B", "4b"))).job_id,
            (await scheduler.submit(spec("C", "35b"))).job_id,
            (await scheduler.submit(spec("D", "4b"))).job_id,
        ]
        await wait_for_all(scheduler, ids)
        assert backend.started == ["A", "B", "C", "D"]
        assert backend.loaded == ["4b", "35b", "4b"]
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_scheduler_uses_actual_loaded_parallel_cap(tmp_path: Path) -> None:
    config = load_config(PRESET)
    backend = FakeBackend(limits={"4b": 1, "35b": 1})
    scheduler = Scheduler(config, backend, store=MetadataStore(tmp_path / "scheduler-caps.sqlite3"))
    await scheduler.start()
    try:
        ids = [(await scheduler.submit(spec(f"job-{i}", "4b"))).job_id for i in range(2)]
        await wait_for_all(scheduler, ids)
        assert backend.max_active == 1
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_queued_cancel_and_result_expiry_are_distinguishable(tmp_path: Path) -> None:
    config = load_config(PRESET)
    backend = FakeBackend()
    scheduler = Scheduler(config, backend, store=MetadataStore(tmp_path / "scheduler-cancel.sqlite3"), result_ttl_seconds=0.01)
    await scheduler.start()
    try:
        first = await scheduler.submit(spec("first", "4b"))
        second = await scheduler.submit(spec("second", "4b"))
        cancelled = await scheduler.cancel(second.job_id)
        assert cancelled.state is JobState.CANCELLED
        await wait_for_all(scheduler, [first.job_id])
        await asyncio.sleep(0.02)
        with pytest.raises(ResultExpired):
            scheduler.result(first.job_id)
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_generation_timeout_fails_only_that_job_and_the_next_job_runs(tmp_path: Path) -> None:
    class SlowFirstBackend(FakeBackend):
        calls: int = 0

        async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(0.5)
            return await super().complete(payload)

    base = load_config(PRESET)
    config = base.__class__(**{**{field: getattr(base, field) for field in base.__dataclass_fields__}, "generation_timeout": 0.1})
    scheduler = Scheduler(config, SlowFirstBackend(), store=MetadataStore(tmp_path / "state.sqlite3"))
    await scheduler.start()
    try:
        slow = (await scheduler.submit(spec("slow", "4b"))).job_id
        fast = (await scheduler.submit(spec("fast", "4b"))).job_id
        await wait_for_all(scheduler, [slow, fast])
        assert present(scheduler.get(slow)).state is JobState.FAILED
        assert present(scheduler.get(slow)).backend_state == "unknown"
        assert present(scheduler.get(fast)).state is JobState.SUCCEEDED
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_effective_context_limit_rejects_prompt_plus_output_budget(tmp_path: Path) -> None:
    base = load_config(PRESET)
    config = replace(base, profiles={"4b": ProfileConfig("4b", "model-4b", 64, 2)})
    scheduler = Scheduler(config, FakeBackend(), store=MetadataStore(tmp_path / "state.sqlite3"))

    with pytest.raises(Exception, match="context"):
        await scheduler.submit(
            JobSpec(
                "client",
                "project",
                "too-large",
                "4b",
                [{"role": "user", "content": "x" * 200}],
                {"max_tokens": 16},
            )
        )


@pytest.mark.asyncio
async def test_preparing_cancel_does_not_dispatch_after_load_finishes(tmp_path: Path) -> None:
    load_started = asyncio.Event()
    release_load = asyncio.Event()

    class SlowLoadBackend(FakeBackend):
        async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
            load_started.set()
            await release_load.wait()
            return await super().ensure_loaded(profile)

    backend = SlowLoadBackend()
    scheduler = Scheduler(load_config(PRESET), backend, store=MetadataStore(tmp_path / "state.sqlite3"))
    await scheduler.start()
    try:
        submitted = await scheduler.submit(spec("preparing", "4b"))
        await asyncio.wait_for(load_started.wait(), timeout=1)
        cancelling = await scheduler.cancel(submitted.job_id)
        assert cancelling.state is JobState.CANCELLING
        release_load.set()
        await asyncio.sleep(0.05)
        assert present(scheduler.get(submitted.job_id)).state is JobState.CANCELLED
        assert backend.started == []
    finally:
        release_load.set()
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_scheduler_emits_operational_events_without_payloads(tmp_path: Path) -> None:
    events = EventWriter(tmp_path / "events.jsonl")
    scheduler = Scheduler(
        load_config(PRESET),
        FakeBackend(),
        store=MetadataStore(tmp_path / "state.sqlite3"),
        events=events,
    )
    await scheduler.start()
    try:
        submitted = await scheduler.submit(spec("telemetry", "4b"))
        await wait_for_all(scheduler, [submitted.job_id])
    finally:
        await scheduler.shutdown()
    text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert '"event": "inference_started"' in text
    assert '"event": "inference_finished"' in text
    assert "telemetry" not in text



@dataclass
class FailingSwitchBackend(FakeBackend):
    fail_once: set[str] = field(default_factory=set)

    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        if profile in self.fail_once:
            self.fail_once.discard(profile)
            self.loaded.append(profile)
            raise BackendFailure("model_load_failed", "LM Studio rejected the configured model load", retryable=True)
        return await super().ensure_loaded(profile)


@pytest.mark.asyncio
async def test_failed_switch_fails_only_that_job_and_the_queued_job_behind_it_runs(tmp_path: Path) -> None:
    config = replace(load_config(PRESET), data_dir=tmp_path)
    backend = FailingSwitchBackend(fail_once={"35b"})
    scheduler = Scheduler(config, backend, store=MetadataStore(tmp_path / "state.sqlite3"))
    await scheduler.start()
    try:
        first = (await scheduler.submit(spec("A", "4b"))).job_id
        await wait_for_all(scheduler, [first])
        failed = (await scheduler.submit(spec("B", "35b"))).job_id
        behind = (await scheduler.submit(spec("C", "4b"))).job_id
        await wait_for_all(scheduler, [failed, behind])
        assert present(scheduler.get(failed)).state is JobState.FAILED
        assert present(scheduler.get(behind)).state is JobState.SUCCEEDED
        assert backend.loaded == ["4b", "35b", "4b"]
    finally:
        await scheduler.shutdown()


@dataclass
class ResidencyBackend(FakeBackend):
    verdicts: list[bool] = field(default_factory=list)
    checks: int = 0

    async def still_loaded_alone(self, loaded: LoadedModel) -> bool:
        self.checks += 1
        return self.verdicts.pop(0) if self.verdicts else True


@pytest.mark.asyncio
async def test_model_unloaded_outside_the_broker_is_loaded_again_before_the_next_batch(tmp_path: Path) -> None:
    backend = ResidencyBackend(verdicts=[False])
    scheduler = Scheduler(load_config(PRESET), backend, store=MetadataStore(tmp_path / "state.sqlite3"))
    await scheduler.start()
    try:
        first = (await scheduler.submit(spec("A", "4b"))).job_id
        await wait_for_all(scheduler, [first])
        # The next job must start a new batch: wait until the scheduler has released the first one. A job that
        # arrives while another is still active joins its batch, and a slow machine would hit that case.
        for _ in range(200):
            if scheduler.status()["backend_state"] == "idle":
                break
            await asyncio.sleep(0.05)
        second = (await scheduler.submit(spec("B", "4b"))).job_id
        await wait_for_all(scheduler, [second])
        assert present(scheduler.get(second)).state is JobState.SUCCEEDED
        assert backend.checks == 1
        assert backend.loaded == ["4b", "4b"]
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_job_that_hits_the_output_limit_fails_instead_of_succeeding(tmp_path: Path) -> None:
    class TruncatedBackend(FakeBackend):
        async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
            result = await super().complete(payload)
            result["choices"][0]["finish_reason"] = "length"
            return result

    scheduler = Scheduler(load_config(PRESET), TruncatedBackend(), store=MetadataStore(tmp_path / "state.sqlite3"))
    await scheduler.start()
    try:
        job_id = (await scheduler.submit(spec("truncated", "4b"))).job_id
        snapshot = await scheduler.wait_for_terminal(job_id, timeout=5)
        assert snapshot.state is JobState.FAILED
        assert snapshot.error is not None
        assert snapshot.error["code"] == "output_truncated"
        assert snapshot.error["retryable"] is False
        assert snapshot.finish_reason == "length"
        assert snapshot.output_tokens == 2
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_empty_answer_fails_instead_of_succeeding(tmp_path: Path) -> None:
    class EmptyBackend(FakeBackend):
        async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
            result = await super().complete(payload)
            result["choices"][0]["message"]["content"] = "   "
            return result

    scheduler = Scheduler(load_config(PRESET), EmptyBackend(), store=MetadataStore(tmp_path / "state.sqlite3"))
    await scheduler.start()
    try:
        job_id = (await scheduler.submit(spec("empty", "4b"))).job_id
        snapshot = await scheduler.wait_for_terminal(job_id, timeout=5)
        assert snapshot.state is JobState.FAILED
        assert snapshot.error is not None
        assert snapshot.error["code"] == "empty_output"
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_non_thinking_profile_sends_reasoning_effort_none_and_thinking_profile_does_not(tmp_path: Path) -> None:
    config = load_config(PRESET)
    backend = FakeBackend(limits={"9b": 2, "9b-nothink": 2})
    scheduler = Scheduler(config, backend, store=MetadataStore(tmp_path / "scheduler.sqlite3"))
    await scheduler.start()
    try:
        ids = [
            (await scheduler.submit(spec("thinking", "9b"))).job_id,
            (await scheduler.submit(spec("plain", "9b-nothink"))).job_id,
        ]
        await wait_for_all(scheduler, ids)
    finally:
        await scheduler.shutdown()
    payloads = {str(payload["metadata"]["task_label"]): payload for payload in backend.payloads}
    assert "reasoning_effort" not in payloads["thinking"]
    assert payloads["plain"]["reasoning_effort"] == "none"


@dataclass
class TwinOverlapBackend(FakeBackend):
    arrived: int = 0
    both_running: asyncio.Event = field(default_factory=asyncio.Event)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Hold each job until two are in flight, so the overlap is observed rather than assumed.
        self.arrived += 1
        if self.arrived == 2:
            self.both_running.set()
        await asyncio.wait_for(self.both_running.wait(), timeout=2)
        return await super().complete(payload)


@pytest.mark.asyncio
async def test_twin_profile_runs_beside_its_loaded_twin_without_reload_or_drain(tmp_path: Path) -> None:
    backend = TwinOverlapBackend(limits={"9b": 2})
    scheduler = Scheduler(load_config(PRESET), backend, store=MetadataStore(tmp_path / "scheduler.sqlite3"))
    await scheduler.start()
    try:
        ids = [
            (await scheduler.submit(spec("thinking", "9b"))).job_id,
            (await scheduler.submit(spec("plain", "9b-nothink"))).job_id,
        ]
        await wait_for_all(scheduler, ids)
        assert [present(scheduler.get(job_id)).state for job_id in ids] == [JobState.SUCCEEDED, JobState.SUCCEEDED]
    finally:
        await scheduler.shutdown()
    assert backend.loaded == ["9b"]
    assert backend.max_active == 2


@dataclass
class DrainCheckBackend(TwinOverlapBackend):
    active_at_load: list[int] = field(default_factory=list)

    async def ensure_loaded(self, profile: str, **kwargs: object) -> LoadedModel:
        self.active_at_load.append(self.active)
        return await super().ensure_loaded(profile, **kwargs)


@pytest.mark.asyncio
async def test_other_model_waits_for_twin_jobs_to_drain_before_loading(tmp_path: Path) -> None:
    backend = DrainCheckBackend(limits={"9b": 2, "4b": 2})
    scheduler = Scheduler(load_config(PRESET), backend, store=MetadataStore(tmp_path / "scheduler.sqlite3"))
    await scheduler.start()
    try:
        ids = [
            (await scheduler.submit(spec("A", "9b"))).job_id,
            (await scheduler.submit(spec("B", "9b-nothink"))).job_id,
            (await scheduler.submit(spec("C", "4b"))).job_id,
        ]
        await wait_for_all(scheduler, ids)
        assert all(present(scheduler.get(job_id)).state is JobState.SUCCEEDED for job_id in ids)
    finally:
        await scheduler.shutdown()
    assert sorted(backend.started[:2]) == ["A", "B"]
    assert backend.started[2] == "C"
    assert backend.loaded == ["9b", "4b"]
    assert backend.active_at_load == [0, 0]
    assert backend.max_active == 2


@dataclass
class CountingBackend(FakeBackend):
    counted: list[str] = field(default_factory=list)

    async def count_tokens(self, loaded: LoadedModel, text: str) -> int:
        self.counted.append(text)
        return 1


@pytest.mark.asyncio
async def test_count_pass_skips_queued_job_cancelled_while_waiting(tmp_path: Path) -> None:
    backend = CountingBackend()
    scheduler = Scheduler(load_config(PRESET), backend, store=MetadataStore(tmp_path / "scheduler.sqlite3"))
    # Not started: one counting pass is driven by hand. The second job is cancelled without leaving the queue,
    # the state a cancel reaches while an earlier job's count is awaited.
    scheduler._loaded = LoadedModel("4b", load_config(PRESET).profiles["4b"].model_id, "instance-4b", 32_000, 4, 0.01)
    await scheduler.submit(spec("first", "4b"))
    second = (await scheduler.submit(spec("second", "4b"))).job_id
    scheduler._records[second].snapshot = scheduler.store.transition(second, JobState.CANCELLED)
    await scheduler._count_ready()
    assert len(backend.counted) == 1
    assert "first" in backend.counted[0]
    assert "second" not in backend.counted[0]


# A 2 MB base64 image. Its length must not be counted as text when the job is checked or estimated.
IMAGE_URL = "data:image/png;base64," + "A" * 2_000_000


def vision_spec(label: str, profile: str) -> JobSpec:
    content = [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": IMAGE_URL}}]
    return JobSpec("client-a", "project", label, profile, [{"role": "user", "content": content}], {"max_tokens": 8})


@pytest.mark.asyncio
async def test_multimodal_message_is_accepted_on_a_vision_profile_and_reaches_the_backend(tmp_path: Path) -> None:
    config = load_config(PRESET)
    backend = FakeBackend(limits={"9b": 2})
    scheduler = Scheduler(config, backend, store=MetadataStore(tmp_path / "scheduler.sqlite3"))
    await scheduler.start()
    try:
        job_id = (await scheduler.submit(vision_spec("look", "9b"))).job_id
        await wait_for_all(scheduler, [job_id])
        assert present(scheduler.get(job_id)).state is JobState.SUCCEEDED
    finally:
        await scheduler.shutdown()
    sent = next(payload for payload in backend.payloads if payload["metadata"]["task_label"] == "look")
    assert sent["messages"][0]["content"][1]["image_url"]["url"] == IMAGE_URL


@pytest.mark.asyncio
async def test_image_is_rejected_on_a_text_only_profile(tmp_path: Path) -> None:
    config = load_config(PRESET)
    scheduler = Scheduler(config, FakeBackend(), store=MetadataStore(tmp_path / "scheduler.sqlite3"))
    with pytest.raises(SchedulerError) as rejected:
        await scheduler.submit(vision_spec("look", "4b"))
    assert rejected.value.code == "unsupported_input"
    assert "does not support vision inputs" in str(rejected.value)


@pytest.mark.asyncio
async def test_text_parts_are_accepted_on_a_text_only_profile(tmp_path: Path) -> None:
    config = load_config(PRESET)
    backend = FakeBackend()
    scheduler = Scheduler(config, backend, store=MetadataStore(tmp_path / "scheduler.sqlite3"))
    await scheduler.start()
    try:
        spec = JobSpec(
            "client-a", "project", "parts", "4b", [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], {"max_tokens": 8}
        )
        job_id = (await scheduler.submit(spec)).job_id
        await wait_for_all(scheduler, [job_id])
        assert present(scheduler.get(job_id)).state is JobState.SUCCEEDED
    finally:
        await scheduler.shutdown()


def test_image_base64_is_not_counted_as_text_and_an_unreadable_image_adds_a_fixed_budget() -> None:
    # The base64 is not counted as text. This test image has no readable size, so it counts as the fixed budget.
    expected = estimate_text_tokens("hi") + REQUEST_OVERHEAD + MESSAGE_OVERHEAD + UNKNOWN_IMAGE_TOKENS
    assert Scheduler._estimate_prompt_tokens(vision_spec("hi", "9b")) == expected


class EmptyAnswerBackend(FakeBackend):
    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = await super().complete(payload)
        result["choices"][0]["message"]["content"] = ""
        return result


async def wait_until_released(scheduler: Scheduler, job_id: str) -> None:
    for _ in range(300):
        if job_id not in scheduler._records:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("record was not released")


@pytest.mark.asyncio
async def test_expired_result_drops_prompt_and_evicts_record(tmp_path: Path) -> None:
    scheduler = Scheduler(
        load_config(PRESET),
        FakeBackend(),
        store=MetadataStore(tmp_path / "scheduler-evict.sqlite3"),
        result_ttl_seconds=0.01,
    )
    await scheduler.start()
    try:
        job_id = (await scheduler.submit(spec("SENTINEL-EVICT", "4b"))).job_id
        record = scheduler._records[job_id]
        await wait_for_all(scheduler, [job_id])
        await wait_until_released(scheduler, job_id)
        assert record.spec.messages == []
        assert record.result is None
        snapshot = scheduler.get(job_id)
        assert snapshot is not None and snapshot.state is JobState.SUCCEEDED
        with pytest.raises(ResultExpired):
            scheduler.result(job_id)
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_failed_and_cancelled_jobs_do_not_keep_their_prompt(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "scheduler-terminal.sqlite3")
    failing = Scheduler(load_config(PRESET), EmptyAnswerBackend(), store=store)
    await failing.start()
    try:
        job_id = (await failing.submit(spec("SENTINEL-FAILED", "4b"))).job_id
        record = failing._records[job_id]
        snapshot = await failing.wait_for_terminal(job_id, timeout=5)
        assert snapshot.state is JobState.FAILED
        await wait_until_released(failing, job_id)
        assert record.spec.messages == []
        assert present(failing.get(job_id)).state is JobState.FAILED
    finally:
        await failing.shutdown()

    # Not started: the job stays queued until cancelled, so the cancel path is exercised directly.
    cancelling = Scheduler(load_config(PRESET), FakeBackend(), store=store)
    job_id = (await cancelling.submit(spec("SENTINEL-CANCELLED", "4b"))).job_id
    record = cancelling._records[job_id]
    cancelled = await cancelling.cancel(job_id)
    assert cancelled.state is JobState.CANCELLED
    assert record.spec.messages == []
    assert job_id not in cancelling._records
    assert present(cancelling.get(job_id)).state is JobState.CANCELLED
    assert (await cancelling.cancel(job_id)).state is JobState.CANCELLED


@pytest.mark.asyncio
async def test_periodic_metadata_prune_runs_at_most_once_an_hour(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "scheduler-prune.sqlite3")
    befores: list[datetime] = []
    original_prune = store.prune

    def recording_prune(before: datetime) -> int:
        befores.append(before)
        pruned: int = original_prune(before)
        return pruned

    store.prune = recording_prune  # type: ignore[method-assign]
    config = replace(load_config(PRESET), metadata_ttl_days=30)
    scheduler = Scheduler(config, FakeBackend(), store=store)
    scheduler._prune_metadata_if_due()
    scheduler._prune_metadata_if_due()
    assert len(befores) == 1
    age = datetime.now(UTC) - befores[0]
    assert timedelta(days=30) <= age < timedelta(days=30, minutes=1)

    disabled = Scheduler(replace(config, metadata_ttl_days=0), FakeBackend(), store=store)
    befores.clear()
    disabled._prune_metadata_if_due()
    assert befores == []

    def failing_prune(before: datetime) -> int:
        raise sqlite3.OperationalError("database is locked")

    store.prune = failing_prune  # type: ignore[method-assign]
    failing = Scheduler(config, FakeBackend(), store=store)
    failing._prune_metadata_if_due()  # a failed prune is swallowed; the loop keeps running
    assert failing._next_prune_at > 0.0


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_store_failure_refuses_submissions_until_a_probe_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(scheduler_module, "time", clock)
    store = MetadataStore(tmp_path / "scheduler-degraded.sqlite3")
    scheduler = Scheduler(load_config(PRESET), FakeBackend(), store=store)
    real_create_job = store.create_job
    real_summary = store.summary
    down = {"value": True}

    def create_job(job_id: str, spec_: JobSpec, idempotency_key: str | None = None) -> Any:
        if down["value"]:
            raise sqlite3.OperationalError("database is locked")
        return real_create_job(job_id, spec_, idempotency_key=idempotency_key)

    def summary() -> dict[str, Any]:
        if down["value"]:
            raise sqlite3.OperationalError("database is locked")
        current: dict[str, Any] = real_summary()
        return current

    monkeypatch.setattr(store, "create_job", create_job)
    monkeypatch.setattr(store, "summary", summary)

    with pytest.raises(SchedulerError) as failed:
        await scheduler.submit(spec("A", "4b"))
    assert (failed.value.code, failed.value.status_code, failed.value.retryable) == ("broker_degraded", 503, True)
    assert scheduler.status()["degraded"] == "storage_unavailable"

    clock.now += 1  # inside the probe interval: refused without probing
    with pytest.raises(SchedulerError, match="degraded"):
        await scheduler.submit(spec("B", "4b"))

    clock.now += 5  # probe due, store still down: refused
    with pytest.raises(SchedulerError, match="degraded"):
        await scheduler.submit(spec("C", "4b"))

    down["value"] = False
    clock.now += 5  # probe due and store answers: the submission proceeds and degraded clears
    snapshot = await scheduler.submit(spec("D", "4b"))
    assert snapshot.state is JobState.QUEUED
    assert scheduler.status()["degraded"] is None


@pytest.mark.asyncio
async def test_event_sink_failure_refuses_submissions_until_a_write_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(scheduler_module, "time", clock)
    events = EventWriter(tmp_path / "events.jsonl")
    handler = events._handler
    assert handler is not None
    original_should_roll = handler.shouldRollover
    store = MetadataStore(tmp_path / "scheduler-sink.sqlite3")
    scheduler = Scheduler(load_config(PRESET), FakeBackend(), store=store, events=events)

    await scheduler.submit(spec("A", "4b"))
    assert events.healthy

    def broken_rollover(record: Any) -> int:
        raise OSError("cannot rotate")

    monkeypatch.setattr(handler, "shouldRollover", broken_rollover)
    await scheduler.submit(spec("B", "4b"))  # accepted; its queued event is the write that fails
    assert not events.healthy

    with pytest.raises(SchedulerError, match="degraded") as refused:
        await scheduler.submit(spec("C", "4b"))
    assert refused.value.code == "broker_degraded"
    assert scheduler.status()["degraded"] == "event_sink_unavailable"

    monkeypatch.setattr(handler, "shouldRollover", original_should_roll)
    clock.now += 5
    await scheduler.submit(spec("D", "4b"))  # the probe write succeeds, so submissions resume
    assert events.healthy
    assert scheduler.status()["degraded"] is None
