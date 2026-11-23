from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from support import PRESET

from local_llm_broker.backend import LoadedModel
from local_llm_broker.config import load_config
from local_llm_broker.contracts import JobSpec, JobState
from local_llm_broker.scheduler import Scheduler
from local_llm_broker.storage import MetadataStore


def test_restart_marks_unfinished_metadata_interrupted(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "state.sqlite3")
    spec = JobSpec("client", "project", "task", "4b", [{"role": "user", "content": "private"}], {})
    store.create_job("job-1", spec)
    store.transition("job-1", JobState.PREPARING, backend_state="unknown")
    assert store.mark_unfinished_interrupted() == 1
    snapshot = store.get_job("job-1")
    assert snapshot is not None
    assert snapshot.state is JobState.INTERRUPTED
    assert snapshot.backend_state == "unknown"


@pytest.mark.asyncio
async def test_scheduler_start_does_not_block_after_restart_with_interrupted_job(tmp_path: Path) -> None:
    class Backend:
        async def stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
            raise NotImplementedError
            yield b""

        async def list_models(self) -> list[dict[str, Any]]:
            return []

        async def unload(self, instance_id: str) -> None:
            return None

        async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
            return LoadedModel(profile, profile, "instance", 32_000, 1)

        async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    store = MetadataStore(tmp_path / "state.sqlite3")
    store.create_job("job-1", JobSpec("client", "project", "task", "4b", [{"role": "user", "content": "x"}], {}))
    store.transition("job-1", JobState.RUNNING, backend_state="unknown")
    scheduler = Scheduler(load_config(PRESET), Backend(), store=store)
    await scheduler.start()
    try:
        snapshot = store.get_job("job-1")
        assert snapshot is not None and snapshot.state is JobState.INTERRUPTED
        submitted = await scheduler.submit(JobSpec("client", "project", "next", "4b", [{"role": "user", "content": "x"}], {"max_tokens": 8}))
        finished = await scheduler.wait_for_terminal(submitted.job_id, timeout=2)
        assert finished.state is JobState.SUCCEEDED
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_scheduler_start_does_not_block_when_backend_has_loaded_instances(tmp_path: Path) -> None:
    class Backend:
        async def inspect_instances(self) -> list[dict[str, str]]:
            return [{"model_id": "qwen/qwen3-4b-thinking-2507", "id": "warm"}]

        async def stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
            raise NotImplementedError
            yield b""

        async def list_models(self) -> list[dict[str, Any]]:
            return []

        async def unload(self, instance_id: str) -> None:
            return None

        async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
            return LoadedModel(profile, profile, "instance", 16_000, 4)

        async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    scheduler = Scheduler(load_config(PRESET), Backend(), store=MetadataStore(tmp_path / "state.sqlite3"))
    await scheduler.start()
    try:
        submitted = await scheduler.submit(JobSpec("client", "project", "next", "4b", [{"role": "user", "content": "x"}], {"max_tokens": 8}))
        finished = await scheduler.wait_for_terminal(submitted.job_id, timeout=2)
        assert finished.state is JobState.SUCCEEDED
    finally:
        await scheduler.shutdown()

