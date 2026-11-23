from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

from .backend import BackendAdapter, LoadedModel
from .contracts import (
    TERMINAL_STATES,
    TIER_ORDER,
    BrokerConfig,
    JobSnapshot,
    JobSpec,
    JobState,
    tier_rank,
    utc_now,
)
from .lm_studio import BackendFailure
from .storage import IdempotencyConflict, MetadataStore
from .telemetry import EventWriter
from .tokens import SAFETY, estimate_prompt_tokens, fixed_overhead, job_texts

# Old terminal jobs and their idempotency rows are pruned from the metadata store at most this often.
METADATA_PRUNE_INTERVAL_SECONDS = 3600.0
# While degraded, new submissions are refused; the store and sink are probed at most this often.
DEGRADED_PROBE_SECONDS = 5.0

_TRUNCATED_MESSAGE = "the answer reached max_output_tokens before it finished; raise the output budget"
_EMPTY_MESSAGE = "the model returned no visible answer; if it reasoned, raise the output budget"
# A single SSE line longer than this is not parsed for completion checks; forwarding is never affected.
_MAX_SSE_LINE_BYTES = 1024 * 1024


def _completion_failure(finish_reason: Any, visible_answer: bool) -> tuple[str, str] | None:
    """Return (error code, message) when a finished completion must be recorded as FAILED, else None."""
    if finish_reason == "length":
        return "output_truncated", _TRUNCATED_MESSAGE
    if not visible_answer:
        return "empty_output", _EMPTY_MESSAGE
    return None


class _StreamOutcome:
    """Reads OpenAI chat.completion.chunk frames as they are forwarded. It never changes the bytes it is fed."""

    def __init__(self) -> None:
        self._partial = b""
        self.visible_answer = False
        self.finish_reason: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def feed(self, chunk: bytes) -> None:
        try:
            *lines, partial = (self._partial + chunk).split(b"\n")
            self._partial = partial if len(partial) <= _MAX_SSE_LINE_BYTES else b""
            for line in lines:
                self._consume(line)
        except Exception:
            # A malformed chunk must never break the stream; the completion checks just see less data.
            self._partial = b""

    def _consume(self, line: bytes) -> None:
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        body = line[len(b"data:") :].strip()
        if not body or body == b"[DONE]":
            return
        try:
            event = json.loads(body)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        choices = event.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str) and content.strip():
                self.visible_answer = True
            reason = choice.get("finish_reason")
            if isinstance(reason, str):
                self.finish_reason = reason
        usage = event.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if isinstance(prompt, int) and not isinstance(prompt, bool):
                self.input_tokens = prompt
            if isinstance(completion, int) and not isinstance(completion, bool):
                self.output_tokens = completion


class SchedulerError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 400, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


class ResultPending(SchedulerError):
    def __init__(self) -> None:
        super().__init__("result_pending", "job result is not available yet", status_code=202, retryable=True)


class ResultExpired(SchedulerError):
    def __init__(self) -> None:
        super().__init__("result_expired", "job result has expired from memory", status_code=410)


# Tokens kept free for the chat template and rounding, on top of each job's prompt and output budget.
CONTEXT_MARGIN = 512


def _content_size(content: Any) -> tuple[int, int]:
    # Text bytes and image parts in one message's content. check_content validates the shape first.
    if isinstance(content, str):
        return len(content.encode("utf-8")), 0
    text_bytes = 0
    images = 0
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                text_bytes += len(part["text"].encode("utf-8"))
            elif part.get("type") == "image_url":
                images += 1
    return text_bytes, images


def _is_inline_image(image_url: Any) -> bool:
    if not isinstance(image_url, dict):
        return False
    url = image_url.get("url")
    return isinstance(url, str) and url.startswith("data:image/") and ";base64," in url


def check_content(content: Any, *, profile: str, supports_vision: bool) -> tuple[int, int]:
    """Validate one message's content and return its (text bytes, image parts).

    Content is a string, or a list of {"type": "text"} parts and {"type": "image_url"} parts that carry an inline
    base64 data URL. The broker never fetches a remote URL.
    """
    if isinstance(content, str):
        return _content_size(content)
    if not isinstance(content, list):
        raise SchedulerError("unsupported_input", "message content must be a string or a list of parts")
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            continue
        if isinstance(part, dict) and part.get("type") == "image_url" and _is_inline_image(part.get("image_url")):
            continue
        raise SchedulerError("unsupported_input", "only text parts and inline base64 image_url parts are supported")
    text_bytes, images = _content_size(content)
    if images and not supports_vision:
        raise SchedulerError("unsupported_input", f"profile '{profile}' does not support vision inputs")
    return text_bytes, images


class _BoundedStreamBuffer:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._chunks: deque[bytes | None] = deque()
        self._bytes = 0
        self._discard = False
        self._condition = asyncio.Condition()

    async def put(self, chunk: bytes | None) -> None:
        if chunk is not None and len(chunk) > self.max_bytes:
            raise SchedulerError("stream_chunk_too_large", "upstream stream chunk exceeds the configured buffer")
        async with self._condition:
            if self._discard:
                return
            if chunk is None:
                self._chunks.append(None)
                self._condition.notify_all()
                return
            while self._bytes + len(chunk) > self.max_bytes and not self._discard:
                await self._condition.wait()
            if self._discard:
                return
            self._chunks.append(chunk)
            self._bytes += len(chunk)
            self._condition.notify_all()

    async def get(self) -> bytes | None:
        async with self._condition:
            while not self._chunks:
                await self._condition.wait()
            chunk = self._chunks.popleft()
            if chunk is not None:
                self._bytes -= len(chunk)
            self._condition.notify_all()
            return chunk

    async def disconnect(self) -> None:
        async with self._condition:
            self._discard = True
            self._chunks.clear()
            self._bytes = 0
            self._condition.notify_all()


@dataclass(slots=True)
class _Record:
    spec: JobSpec
    submitted_at: float
    snapshot: JobSnapshot
    result: dict[str, Any] | None = None
    task: asyncio.Task[None] | None = None
    cancel_requested: bool = False
    stream_queue: _BoundedStreamBuffer | None = None
    # Resolved at submit. tier_auto means the broker chose it, so a larger loaded tier of the same model may run it.
    tier: str | None = None
    tier_auto: bool = False
    # Exact prompt tokens from the loaded model's tokenizer (text) plus the estimated template and image overhead.
    prompt_tokens: int | None = None
    counted_model: str | None = None


class Scheduler:
    def __init__(
        self,
        config: BrokerConfig,
        adapter: BackendAdapter,
        *,
        store: MetadataStore,
        result_ttl_seconds: float | None = None,
        events: EventWriter | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.store = store
        self.result_ttl_seconds = result_ttl_seconds if result_ttl_seconds is not None else config.result_ttl_seconds
        self.events = events
        # Live jobs, and terminal ones whose result is still held. Other terminal jobs are read from the store.
        self._records: dict[str, _Record] = {}
        self._next_prune_at = 0.0  # time.monotonic() value before which no metadata prune runs
        self._queue: deque[str] = deque()
        self._active: dict[str, _Record] = {}
        self._loaded: LoadedModel | None = None
        self._fresh_load = False
        # When the broker last became idle with a model loaded; None while jobs are running or queued.
        self._idle_since: float | None = None
        self._condition = asyncio.Condition()
        self._loop_task: asyncio.Task[None] | None = None
        self._stopped = False
        # Status listeners (GET /status/stream) wait on this. Bumped when a job is queued, starts, ends or is
        # cancelled, when a model loads or unloads, and when the idle-unload timer starts or stops.
        self._change_version = 0
        self._change_event = asyncio.Event()
        # Set when the metadata store or the event sink fails. New submissions are refused until a probe succeeds.
        self.degraded_reason: str | None = None
        self._next_probe_at = 0.0  # time.monotonic() value before which no degraded-state probe runs

    @property
    def change_version(self) -> int:
        return self._change_version

    @property
    def stopped(self) -> bool:
        return self._stopped

    def _notify_change(self) -> None:
        self._change_version += 1
        event, self._change_event = self._change_event, asyncio.Event()
        event.set()

    async def wait_for_change(self, since: int, timeout: float) -> bool:
        """Wait until the status changes after version `since`. False if `timeout` seconds pass first."""
        if self._change_version != since or self._stopped:
            return True
        event = self._change_event
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            return False
        return True

    async def start(self) -> None:
        if self._loop_task is not None:
            return
        # Jobs left unfinished by a crash are marked interrupted and never replayed. Startup does not block;
        # loaded instances are checked when a job needs them.
        self.store.mark_unfinished_interrupted()
        self._stopped = False
        self._loop_task = asyncio.create_task(self._run_loop(), name="local-llm-broker-scheduler")

    async def shutdown(self) -> None:
        self._stopped = True
        self._notify_change()  # wakes status listeners so their streams end
        async with self._condition:
            self._condition.notify_all()
        if self._loop_task is not None:
            await self._loop_task
        active = [record.task for record in self._active.values() if record.task is not None]
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        self._loop_task = None

    async def submit(self, spec: JobSpec, idempotency_key: str | None = None) -> JobSnapshot:
        snapshot, _created = await self.submit_ex(spec, idempotency_key=idempotency_key)
        return snapshot

    async def submit_ex(self, spec: JobSpec, idempotency_key: str | None = None) -> tuple[JobSnapshot, bool]:
        """Submit a job and report whether it was newly created (False means an idempotent replay)."""
        self._validate_spec(spec)
        tier, tier_auto = self.resolve_tier(spec)
        async with self._condition:
            if len(self._queue) >= self.config.queue_limit:
                raise SchedulerError("queue_full", "broker queue is full", status_code=429, retryable=True)
            self._check_admission()
            job_id = str(uuid.uuid4())
            try:
                snapshot = self.store.create_job(job_id, spec, idempotency_key=idempotency_key)
            except IdempotencyConflict as exc:
                raise SchedulerError("idempotency_conflict", str(exc), status_code=409) from exc
            except (sqlite3.Error, OSError) as exc:
                self._set_degraded("storage_unavailable")
                raise SchedulerError(
                    "broker_degraded", "job store is unavailable; new submissions are paused", status_code=503, retryable=True
                ) from exc
            if snapshot.job_id != job_id:
                # Replay of an earlier job. Its record is not created here: a missing record means the job is
                # terminal and its prompt is already gone, so this request's prompt must not be kept either.
                return snapshot, False
            self._records[job_id] = _Record(
                spec,
                time.monotonic(),
                snapshot,
                stream_queue=_BoundedStreamBuffer(self.config.max_stream_buffer_bytes)
                if spec.generation.get("stream")
                else None,
                tier=tier,
                tier_auto=tier_auto,
            )
            self._queue.append(job_id)
            self._emit("queued", snapshot, tier=tier, tier_auto=tier_auto)
            self._condition.notify_all()
            return snapshot, True

    def get(self, job_id: str) -> JobSnapshot | None:
        record = self._records.get(job_id)
        if record is not None:
            return record.snapshot
        return self.store.get_job(job_id)

    async def wait_for_terminal(self, job_id: str, timeout: float | None = None) -> JobSnapshot:
        deadline = time.monotonic() + (timeout or self.config.generation_timeout)
        while True:
            snapshot = self.get(job_id)
            if snapshot is None:
                raise SchedulerError("not_found", "job not found", status_code=404)
            if snapshot.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.INTERRUPTED}:
                return snapshot
            if time.monotonic() >= deadline:
                raise SchedulerError("generation_timeout", "job exceeded the generation deadline", status_code=504, retryable=True)
            await asyncio.sleep(0.01)

    async def stream_events(self, job_id: str) -> AsyncIterator[bytes]:
        record = self._records.get(job_id)
        if record is None or record.stream_queue is None:
            raise SchedulerError("stream_unavailable", "stream is not available for this job", status_code=409)
        while True:
            chunk = await record.stream_queue.get()
            if chunk is None:
                return
            yield chunk

    def result(self, job_id: str) -> dict[str, Any]:
        record = self._records.get(job_id)
        snapshot = self.get(job_id)
        if snapshot is None:
            raise SchedulerError("not_found", "job not found", status_code=404)
        if snapshot.state not in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.INTERRUPTED}:
            raise ResultPending()
        if snapshot.state is not JobState.SUCCEEDED:
            return {"job": snapshot.public_dict(), "result": None}
        if record is None or record.result is None:
            raise ResultExpired()
        if (utc_now() - snapshot.updated_at).total_seconds() > self.result_ttl_seconds:
            record.result = None
            self.store.transition(job_id, snapshot.state, result_expired=True)
            record.snapshot = self.store.get_job(job_id) or snapshot
            raise ResultExpired()
        return {"job": snapshot.public_dict(), "result": record.result}

    async def cancel(self, job_id: str) -> JobSnapshot:
        async with self._condition:
            record = self._records.get(job_id)
            if record is None:
                snapshot = self.store.get_job(job_id)
                if snapshot is None:
                    raise SchedulerError("not_found", "job not found", status_code=404)
                return snapshot
            if record.snapshot.state is JobState.QUEUED:
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                record.snapshot = self.store.transition(job_id, JobState.CANCELLED)
                self._emit("cancellation", record.snapshot, phase="queued")
            elif record.snapshot.state is JobState.RUNNING:
                record.cancel_requested = True
                record.snapshot = self.store.transition(job_id, JobState.CANCELLING, backend_state="in_flight")
                if record.stream_queue is not None:
                    await record.stream_queue.disconnect()
                self._emit("cancellation", record.snapshot, phase="generation")
            elif record.snapshot.state is JobState.PREPARING:
                record.cancel_requested = True
                record.snapshot = self.store.transition(job_id, JobState.CANCELLING, backend_state="loading")
                self._emit("cancellation", record.snapshot, phase="loading")
            self._release_finished_locked()
            self._condition.notify_all()
            return record.snapshot

    def _sink_healthy(self) -> bool:
        return self.events is None or self.events.healthy

    def _set_degraded(self, reason: str) -> None:
        self.degraded_reason = reason
        self._next_probe_at = time.monotonic() + DEGRADED_PROBE_SECONDS
        self._notify_change()

    def _sync_sink_state(self) -> None:
        # The sink's own health is read without I/O; a sink that recovers on its own clears its reason.
        if not self._sink_healthy():
            if self.degraded_reason is None:
                self._set_degraded("event_sink_unavailable")
        elif self.degraded_reason == "event_sink_unavailable":
            self.degraded_reason = None

    def _check_admission(self) -> None:
        """Refuse new work while the store or the event sink is degraded, unless a probe shows both answer again."""
        self._sync_sink_state()
        if self.degraded_reason is None:
            return
        if self._probe_passes():
            self.degraded_reason = None
            return
        raise SchedulerError(
            "broker_degraded",
            f"broker is degraded ({self.degraded_reason}); new submissions are paused",
            status_code=503,
            retryable=True,
        )

    def _probe_passes(self) -> bool:
        # A cheap store read and, if the sink is failing, one sink write; at most once per DEGRADED_PROBE_SECONDS.
        now = time.monotonic()
        if now < self._next_probe_at:
            return False
        self._next_probe_at = now + DEGRADED_PROBE_SECONDS
        try:
            self.store.summary()
        except (sqlite3.Error, OSError):
            self.degraded_reason = "storage_unavailable"
            return False
        if self.events is not None and not self.events.healthy:
            self.events.emit({"event": "sink_probe"})
        if not self._sink_healthy():
            self.degraded_reason = "event_sink_unavailable"
            return False
        return True

    def status(self) -> dict[str, Any]:
        self._sync_sink_state()
        active_profiles = sorted({record.spec.profile for record in self._active.values()})
        return {
            "degraded": self.degraded_reason,
            "queue_count": len(self._queue),
            "active_count": len(self._active),
            "active_profiles": active_profiles,
            "active_profile": self._loaded.profile if self._loaded else None,
            "active_tier": self._loaded.tier if self._loaded else None,
            "idle_unload_in_seconds": self._idle_unload_remaining(),
            "effective_context": self._loaded.context_limit if self._loaded else None,
            "effective_parallel": self._loaded.parallel_limit if self._loaded else None,
            "backend_state": "running" if self._active else "idle",
        }

    async def _verify_resident(self) -> None:
        # Before each batch, the model the broker last loaded must still be the only one loaded in LM Studio.
        # If it was unloaded or replaced outside the broker, the load path below starts from a clean state.
        check = getattr(self.adapter, "still_loaded_alone", None)
        loaded = self._loaded
        if check is None or loaded is None:
            return
        try:
            still_loaded = await check(loaded)
        except Exception:
            # An unreadable listing counts as not loaded; the load path then reports the real error.
            still_loaded = False
        if not still_loaded:
            async with self._condition:
                self._loaded = None
            self._notify_change()

    def _idle_unload_remaining(self) -> float | None:
        if self._idle_since is None or self._loaded is None or self.config.idle_unload_seconds <= 0:
            return None
        return max(0.0, round(self.config.idle_unload_seconds - (time.monotonic() - self._idle_since), 1))

    async def _unload_if_idle(self) -> None:
        # Called each pass of the loop. Counts idle time only while a model is loaded and nothing runs or waits.
        if self._loaded is None or self._active or self._queue or self.config.idle_unload_seconds <= 0:
            if self._idle_since is not None:
                self._idle_since = None
                self._notify_change()
            return
        now = time.monotonic()
        if self._idle_since is None:
            self._idle_since = now
            self._notify_change()
            return
        idle_for = now - self._idle_since
        if idle_for < self.config.idle_unload_seconds:
            return
        loaded = self._loaded
        release = getattr(self.adapter, "release", None)
        try:
            if release is not None:
                await release(loaded)
            else:
                await self.adapter.unload(loaded.instance_id)
        except Exception as exc:
            # Leave it loaded and try again after another idle period; a new job uses it in the meantime.
            self._idle_since = now
            self._emit_backend("idle_unload_failed", profile=loaded.profile, tier=loaded.tier, error_code=self._error_from_exception(exc)["code"])
            return
        async with self._condition:
            # The model is gone even if a job arrived during the unload; that job loads it again.
            if self._loaded is loaded:
                self._loaded = None
            self._idle_since = None
            self._condition.notify_all()
        self._emit_backend("idle_unload", profile=loaded.profile, tier=loaded.tier, idle_seconds=round(idle_for, 1))

    def _reservation(self, record: _Record) -> int:
        """Context a job may use: its prompt (exact once counted) plus its whole output budget and the margin."""
        max_tokens = record.spec.generation.get("max_tokens", 2048)
        prompt = record.prompt_tokens if record.prompt_tokens is not None else self._estimate_prompt_tokens(record.spec)
        return prompt + (max_tokens if isinstance(max_tokens, int) else 2048) + CONTEXT_MARGIN

    async def _count_ready(self) -> None:
        # Exact counts for the queued jobs that would run next on the loaded model, outside the scheduler lock.
        loaded = self._loaded
        counter = getattr(self.adapter, "count_tokens", None)
        if loaded is None or counter is None or not self._queue:
            return
        for job_id in list(self._queue)[: max(1, loaded.parallel_limit)]:
            record = self._records.get(job_id)
            if record is None:
                break
            # A queue head loading its model is PREPARING and still needs its count; a cancelled one is not.
            if record.snapshot.state not in (JobState.QUEUED, JobState.PREPARING):
                continue  # cancelled while an earlier count was awaited; no SDK call for it
            if not self._runs_on_loaded(record):
                break
            if record.counted_model == loaded.model_id:
                continue
            texts = job_texts(record.spec.messages, record.spec.generation)
            try:
                text_tokens = await counter(loaded, "\n".join(texts)) if texts else 0
                record.prompt_tokens = int(text_tokens) + fixed_overhead(record.spec.messages, record.spec.generation)
            except Exception:
                record.prompt_tokens = None  # keep the estimate, which errs high
            record.counted_model = loaded.model_id

    def _larger_tier(self, record: _Record, needed: int) -> str | None:
        tiers = self.config.tiers_for(record.spec.profile)
        for name in TIER_ORDER:
            if name in tiers and tier_rank(name) > tier_rank(record.tier) and tiers[name].context_length >= needed:
                return name
        return None

    async def _run_loop(self) -> None:
        while not self._stopped:
            self._prune_metadata_if_due()
            await self._unload_if_idle()
            await self._count_ready()
            if self._loaded is not None and not self._active:
                if self._fresh_load:
                    # Loaded by this loop on the previous pass; nothing outside the broker can have changed it yet.
                    self._fresh_load = False
                elif self._queue:
                    await self._verify_resident()
            to_load: _Record | None = None
            to_dispatch: list[_Record] = []
            async with self._condition:
                self._expire_results_locked()
                self._release_finished_locked()
                self._expire_queued_locked()
                if not self._queue:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=0.1)
                    except TimeoutError:
                        pass
                    continue
                head = self._records[self._queue[0]]
                if not self._runs_on_loaded(head):
                    if self._active:
                        try:
                            await asyncio.wait_for(self._condition.wait(), timeout=0.1)
                        except TimeoutError:
                            pass
                        continue
                    head.snapshot = self.store.transition(head.snapshot.job_id, JobState.PREPARING, backend_state="loading")
                    self._emit("load_started", head.snapshot, tier=head.tier)
                    to_load = head
                else:
                    assert self._loaded is not None  # _runs_on_loaded is true only with a loaded model
                    available = self._loaded.parallel_limit - len(self._active)
                    if available <= 0:
                        try:
                            await asyncio.wait_for(self._condition.wait(), timeout=0.1)
                        except TimeoutError:
                            pass
                        continue
                    # Parallel jobs share the loaded context (LM Studio's unified KV cache). When it overfills, LM Studio
                    # fails every running job, so the jobs dispatched together must fit it in total.
                    used = sum(self._reservation(record) for record in self._active.values())
                    counting = getattr(self.adapter, "count_tokens", None) is not None
                    while self._queue and len(to_dispatch) < available:
                        candidate = self._records[self._queue[0]]
                        if not self._runs_on_loaded(candidate):
                            break
                        if counting and candidate.counted_model != self._loaded.model_id:
                            break  # counted on the next pass
                        needed = self._reservation(candidate)
                        if needed > self._loaded.context_limit:
                            larger = self._larger_tier(candidate, needed) if candidate.tier_auto else None
                            if larger is not None:
                                # Stays at the head of the queue; the larger tier loads once running jobs finish.
                                candidate.tier = larger
                                self._emit("tier_raised", candidate.snapshot, tier=larger, prompt_tokens=candidate.prompt_tokens)
                                break
                            self._queue.popleft()
                            candidate.snapshot = self.store.transition(
                                candidate.snapshot.job_id,
                                JobState.FAILED,
                                backend_state="idle",
                                error_json=json.dumps(
                                    {
                                        "code": "context_exceeded",
                                        "message": f"prompt and output budget ({needed} tokens) exceed the loaded "
                                        f"{self._loaded.context_limit}-token context",
                                        "retryable": False,
                                    }
                                ),
                            )
                            continue
                        if used + needed > self._loaded.context_limit:
                            break  # waits until running jobs free enough of the shared context
                        self._queue.popleft()
                        used += needed
                        to_dispatch.append(candidate)
                    if not to_dispatch:
                        # Nothing could start (shared context full, or a count or tier change pending). Yield so
                        # running jobs can finish, instead of spinning on the same decision.
                        try:
                            await asyncio.wait_for(self._condition.wait(), timeout=0.1)
                        except TimeoutError:
                            pass
                        continue
                    for record in to_dispatch:
                        self._active[record.snapshot.job_id] = record
                        record.snapshot = self.store.transition(
                            record.snapshot.job_id,
                            JobState.RUNNING,
                            backend_state="generating",
                            load_seconds=self._loaded.load_seconds,
                            queue_seconds=max(0.0, time.monotonic() - record.submitted_at),
                        )
                        record.task = asyncio.create_task(self._run_job(record), name=f"job-{record.snapshot.job_id}")
            if to_load is not None:
                try:
                    if to_load.tier is None:
                        loaded = await self.adapter.ensure_loaded(to_load.spec.profile)
                    else:
                        loaded = await self.adapter.ensure_loaded(
                            to_load.spec.profile, tier=to_load.tier, accept_higher=to_load.tier_auto
                        )
                except Exception as exc:
                    async with self._condition:
                        head = self._records[self._queue.popleft()]
                        error = self._error_from_exception(exc)
                        self._loaded = None
                        # Only this job fails. The next queued job asks the adapter again, so nothing waits on an operator.
                        head.snapshot = self.store.transition(
                            head.snapshot.job_id,
                            JobState.FAILED,
                            backend_state="unknown" if error["code"].endswith("in_flight") else "idle",
                            error_json=json.dumps(error),
                        )
                        self._emit("failure", head.snapshot, error_code=error["code"])
                        self._condition.notify_all()
                else:
                    async with self._condition:
                        self._loaded = loaded
                        self._fresh_load = True
                        self._emit("load_finished", head.snapshot, load_seconds=loaded.load_seconds, tier=loaded.tier)
                        if head.cancel_requested:
                            self._queue.popleft()
                            head.snapshot = self.store.transition(
                                head.snapshot.job_id,
                                JobState.CANCELLED,
                                backend_state="drained",
                            )
                        self._condition.notify_all()

    async def _run_job(self, record: _Record) -> None:
        started = time.monotonic()
        iterator: Any | None = None
        self._emit("inference_started", record.snapshot)
        try:
            stream_mode = bool(record.spec.generation.get("stream"))
            payload = {
                "model": self._loaded.model_id if self._loaded else record.spec.profile,
                "messages": record.spec.messages,
                **record.spec.generation,
                "stream": stream_mode,
                "metadata": {"task_label": record.spec.task_label, "project": record.spec.project},
            }
            reasoning_effort = self.config.profiles[record.spec.profile].reasoning_effort
            if reasoning_effort is not None:
                payload["reasoning_effort"] = reasoning_effort
            if stream_mode:
                if not hasattr(self.adapter, "stream"):
                    raise SchedulerError("stream_unsupported", "backend does not support streaming")
                stream_done = False
                stream_tail = b""
                outcome = _StreamOutcome()
                iterator = self.adapter.stream(payload).__aiter__()
                while True:
                    remaining = self.config.generation_timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError()
                    try:
                        chunk = await asyncio.wait_for(
                            iterator.__anext__(),
                            timeout=min(self.config.stalled_stream_timeout, remaining),
                        )
                    except StopAsyncIteration:
                        break
                    stream_tail = (stream_tail + chunk)[-64:]
                    stream_done = stream_done or b"data: [DONE]" in stream_tail
                    if record.stream_queue is not None:
                        # A stalled downstream consumer must not hold the GPU slot past the same deadlines.
                        remaining = self.config.generation_timeout - (time.monotonic() - started)
                        if remaining <= 0:
                            raise TimeoutError()
                        await asyncio.wait_for(
                            record.stream_queue.put(chunk),
                            timeout=min(self.config.stalled_stream_timeout, remaining),
                        )
                    outcome.feed(chunk)
                    if record.snapshot.time_to_first_token is None:
                        record.snapshot = self.store.transition(
                            record.snapshot.job_id,
                            record.snapshot.state,
                            time_to_first_token=time.monotonic() - started,
                        )
                        self._emit("first_token", record.snapshot)
                if not stream_done:
                    raise SchedulerError("stream_incomplete", "upstream stream ended without [DONE]")
                if record.cancel_requested:
                    record.snapshot = self.store.transition(
                        record.snapshot.job_id,
                        JobState.CANCELLED,
                        backend_state="drained",
                        inference_seconds=time.monotonic() - started,
                    )
                    self._emit("inference_finished", record.snapshot)
                    return
                failure = _completion_failure(outcome.finish_reason, outcome.visible_answer)
                if failure is not None:
                    # Same rule as the non-stream path: a cut-off or empty answer is not a success.
                    code, reason = failure
                    record.snapshot = self.store.transition(
                        record.snapshot.job_id,
                        JobState.FAILED,
                        backend_state="idle",
                        inference_seconds=time.monotonic() - started,
                        input_tokens=outcome.input_tokens,
                        output_tokens=outcome.output_tokens,
                        finish_reason=outcome.finish_reason,
                        error_json=json.dumps({"code": code, "message": reason, "retryable": False}),
                    )
                    self._emit("failure", record.snapshot, error_code=code)
                    return
                record.snapshot = self.store.transition(
                    record.snapshot.job_id,
                    JobState.SUCCEEDED,
                    backend_state="idle",
                    inference_seconds=time.monotonic() - started,
                    input_tokens=outcome.input_tokens,
                    output_tokens=outcome.output_tokens,
                    finish_reason=outcome.finish_reason,
                )
                self._emit("inference_finished", record.snapshot)
                return
            result = await asyncio.wait_for(
                self.adapter.complete(payload),
                timeout=self.config.generation_timeout,
            )
            encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
            if len(encoded) > self.config.max_result_bytes:
                raise SchedulerError("result_too_large", "completion exceeded the configured result limit")
            if record.cancel_requested:
                record.snapshot = self.store.transition(record.snapshot.job_id, JobState.CANCELLED, backend_state="drained", inference_seconds=time.monotonic() - started)
            else:
                raw_usage = result.get("usage")
                usage = raw_usage if isinstance(raw_usage, dict) else {}
                raw_choices = result.get("choices")
                choices = raw_choices if isinstance(raw_choices, list) else []
                finish_reason = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
                message = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
                content = message.get("content") if isinstance(message, dict) else None
                visible_answer = isinstance(content, str) and bool(content.strip())
                failure = _completion_failure(finish_reason, visible_answer)
                if failure is not None:
                    # A cut-off or empty answer is not a success. Keep the usage so the record shows what happened.
                    code, reason = failure
                    record.snapshot = self.store.transition(
                        record.snapshot.job_id,
                        JobState.FAILED,
                        backend_state="idle",
                        inference_seconds=time.monotonic() - started,
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                        finish_reason=finish_reason,
                        error_json=json.dumps({"code": code, "message": reason, "retryable": False}),
                    )
                    self._emit("failure", record.snapshot, error_code=code)
                    return
                record.result = result
                record.snapshot = self.store.transition(
                    record.snapshot.job_id,
                    JobState.SUCCEEDED,
                    backend_state="idle",
                    inference_seconds=time.monotonic() - started,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    finish_reason=finish_reason,
                )
                self._emit("inference_finished", record.snapshot)
        except Exception as exc:
            error = self._error_from_exception(exc)
            backend_unknown = isinstance(exc, asyncio.TimeoutError) or (
                isinstance(exc, BackendFailure) and exc.code.endswith("in_flight")
            ) or (
                stream_mode
                and isinstance(exc, SchedulerError)
                and exc.code in {"stream_chunk_too_large", "stream_incomplete"}
            )
            if iterator is not None and backend_unknown:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        pass
            record.snapshot = self.store.transition(
                record.snapshot.job_id,
                JobState.CANCELLED if record.cancel_requested else JobState.FAILED,
                backend_state="unknown" if backend_unknown else "idle",
                inference_seconds=time.monotonic() - started,
                error_json=json.dumps(error),
            )
            event = "timeout" if error["code"] == "generation_timeout" else "failure"
            self._emit(event, record.snapshot, error_code=error["code"])
        finally:
            if record.stream_queue is not None:
                await record.stream_queue.put(None)
            async with self._condition:
                self._active.pop(record.snapshot.job_id, None)
                self._release_finished_locked()
                self._condition.notify_all()
            self._notify_change()

    def _expire_queued_locked(self) -> None:
        now = time.monotonic()
        retained: deque[str] = deque()
        expired = False
        while self._queue:
            job_id = self._queue.popleft()
            record = self._records[job_id]
            timeout = record.spec.queue_timeout_seconds or self.config.default_queue_timeout
            if now - record.submitted_at > timeout:
                expired = True
                record.snapshot = self.store.transition(
                    job_id,
                    JobState.FAILED,
                    backend_state="idle",
                    error_json=json.dumps({"code": "queue_timeout", "message": "job expired while waiting in the broker queue", "retryable": True}),
                )
            else:
                retained.append(job_id)
        self._queue = retained
        if expired:
            self._notify_change()

    def _expire_results_locked(self) -> None:
        now = utc_now()
        for record in self._records.values():
            snapshot = record.snapshot
            if record.result is not None and snapshot.state is JobState.SUCCEEDED:
                if (now - snapshot.updated_at).total_seconds() > self.result_ttl_seconds:
                    record.result = None
                    record.snapshot = self.store.transition(snapshot.job_id, snapshot.state, result_expired=True)

    def _release_finished_locked(self) -> None:
        # A terminal job never runs again, so its prompt is dropped as soon as it is terminal. Its record stays only
        # while a held result (or a finished stream within the result TTL) can still be read; after that get() reads
        # the store. Called only after every path that uses record.spec has finished with it.
        now = utc_now()
        for job_id in list(self._records):
            record = self._records[job_id]
            if record.snapshot.state not in TERMINAL_STATES:
                continue
            if record.spec.messages:
                record.spec = replace(record.spec, messages=[])
            if record.result is not None:
                continue
            if record.stream_queue is not None and (now - record.snapshot.updated_at).total_seconds() <= self.result_ttl_seconds:
                continue
            del self._records[job_id]

    def _prune_metadata_if_due(self) -> None:
        # Drops old terminal jobs and expired idempotency rows from the metadata store. Skipped when metadata_ttl_days <= 0.
        if self.config.metadata_ttl_days <= 0:
            return
        now = time.monotonic()
        if now < self._next_prune_at:
            return
        self._next_prune_at = now + METADATA_PRUNE_INTERVAL_SECONDS
        before = utc_now() - timedelta(days=self.config.metadata_ttl_days)
        try:
            pruned = self.store.prune(before)
        except (sqlite3.Error, OSError) as exc:
            self._emit_backend("metadata_prune_failed", error_code=type(exc).__name__)
            return
        if pruned:
            self._emit_backend("metadata_pruned", jobs=pruned)

    def _validate_spec(self, spec: JobSpec) -> None:
        if spec.profile not in self.config.profiles:
            raise SchedulerError("unknown_profile", f"unknown profile: {spec.profile}")
        if not spec.client_id or len(spec.client_id) > 128:
            raise SchedulerError("invalid_client_id", "client_id must be non-empty and short")
        if not spec.messages:
            raise SchedulerError("invalid_messages", "messages must not be empty")
        supports_vision = self.config.profiles[spec.profile].supports_vision
        total = 0
        for message in spec.messages:
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise SchedulerError("unsupported_message", "unsupported message role")
            text_bytes, _ = check_content(message.get("content", ""), profile=spec.profile, supports_vision=supports_vision)
            total += text_bytes
        if total > self.config.max_text_bytes:
            raise SchedulerError("input_too_large", "job text exceeds the configured input limit")
        max_tokens = spec.generation.get("max_tokens", 2048)
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= self.config.profiles[spec.profile].context_ceiling:
            raise SchedulerError("invalid_generation", "max_tokens is outside the configured finite range")
        # Reject only what cannot fit even if the estimate's margin was all overshoot; the exact count decides later.
        if self._lowest_prompt_tokens(spec) + max_tokens + CONTEXT_MARGIN > self.config.profiles[spec.profile].context_ceiling:
            raise SchedulerError("context_exceeded", "prompt and output budget exceed the configured context ceiling")

    def resolve_tier(self, spec: JobSpec) -> tuple[str | None, bool]:
        """The tier a job runs at, and whether the broker chose it. Profiles without tiers return (None, False)."""
        tiers = self.config.tiers_for(spec.profile)
        requested = None if spec.tier in (None, "auto") else spec.tier
        if not tiers:
            if requested is not None:
                raise SchedulerError("unknown_tier", f"profile '{spec.profile}' has no load tiers")
            return None, False
        max_tokens = spec.generation.get("max_tokens", 2048)
        assert isinstance(max_tokens, int)  # checked by _validate_spec
        needed = self._estimate_prompt_tokens(spec) + max_tokens + CONTEXT_MARGIN
        lowest = self._lowest_prompt_tokens(spec) + max_tokens + CONTEXT_MARGIN
        if requested is not None:
            tier = tiers.get(requested)
            if tier is None:
                names = ", ".join(name for name in TIER_ORDER if name in tiers)
                raise SchedulerError("unknown_tier", f"tier must be one of: {names}, or omitted")
            if lowest > tier.context_length:
                raise SchedulerError(
                    "context_exceeded",
                    f"prompt and output budget (about {needed} tokens) exceed the '{requested}' tier's "
                    f"{tier.context_length}-token context; use a higher tier or omit tier",
                )
            return requested, False
        for name in TIER_ORDER:
            if name in tiers and needed <= tiers[name].context_length:
                return name, True
        # The estimate errs high: a job that may still fit goes to the largest tier, and its exact count decides.
        largest = [name for name in TIER_ORDER if name in tiers][-1]
        if lowest <= tiers[largest].context_length:
            return largest, True
        raise SchedulerError("context_exceeded", "prompt and output budget exceed the largest tier's context")

    def _runs_on_loaded(self, record: _Record) -> bool:
        loaded = self._loaded
        if loaded is None:
            return False
        # Twin profiles (a thinking profile and its -nothink twin) share one model_id, so they share the loaded model.
        if loaded.profile != record.spec.profile and loaded.model_id != self.config.profiles[record.spec.profile].model_id:
            return False
        # A loaded model without a tier comes from an adapter that does not do tiers: it runs every job of its profile.
        if record.tier is None or loaded.tier is None or loaded.tier == record.tier:
            return True
        # A broker-chosen tier may run on a larger loaded tier of the same model; that avoids a reload.
        return record.tier_auto and tier_rank(loaded.tier) > tier_rank(record.tier)

    @staticmethod
    def _estimate_prompt_tokens(spec: JobSpec) -> int:
        return estimate_prompt_tokens(spec.messages, spec.generation)

    @staticmethod
    def _lowest_prompt_tokens(spec: JobSpec) -> int:
        # The estimate with its safety margin taken off: the smallest count the job plausibly has.
        return int(Scheduler._estimate_prompt_tokens(spec) / SAFETY)

    @staticmethod
    def _error_from_exception(exc: Exception) -> dict[str, Any]:
        if isinstance(exc, SchedulerError):
            return {"code": exc.code, "message": str(exc), "retryable": exc.retryable}
        if isinstance(exc, BackendFailure):
            return {"code": exc.code, "message": str(exc), "retryable": exc.retryable}
        if isinstance(exc, asyncio.TimeoutError):
            return {
                "code": "generation_timeout",
                "message": "generation deadline exceeded; backend termination is unverified",
                "retryable": False,
            }
        return {"code": "inference_failed", "message": "job failed during local inference", "retryable": True}

    def _emit_backend(self, event: str, **fields: Any) -> None:
        self._notify_change()
        if self.events is not None:
            self.events.emit({"event": event, **fields})

    def _emit(self, event: str, snapshot: JobSnapshot, **fields: Any) -> None:
        if event != "first_token":  # the first token changes nothing that status reports
            self._notify_change()
        if self.events is None:
            return
        self.events.emit(
            {
                "event": event,
                "job_id": snapshot.job_id,
                "client_id": snapshot.client_id,
                "profile": snapshot.profile,
                "state": snapshot.state.value,
                **fields,
            }
        )

