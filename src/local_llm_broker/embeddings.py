"""Embedding profiles: resident models served beside the chat scheduler.

An embedding model is small, so it stays loaded next to the chat model instead of taking a turn in the job queue.
Requests go straight to LM Studio, limited per profile to `parallel_ceiling` at a time. Their text is never stored or
logged; only counts and timings are recorded.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Depends, FastAPI, Header, Request

from .contracts import BrokerConfig
from .lm_studio import BackendFailure
from .routes_openai import label
from .scheduler import SchedulerError

MAX_INPUTS = 256
ALLOWED_FIELDS = {"model", "input", "encoding_format", "user"}


class EmbeddingService:
    def __init__(self, config: BrokerConfig, backend: Any, telemetry: Any) -> None:
        self.config = config
        self.backend = backend
        self.telemetry = telemetry
        self._limits = {alias: p.parallel_ceiling for alias, p in config.profiles.items() if p.kind == "embedding"}
        self._gates = {alias: asyncio.Semaphore(limit) for alias, limit in self._limits.items()}
        self._active = dict.fromkeys(self._limits, 0)
        self._requests = dict.fromkeys(self._limits, 0)

    def resolve(self, model: Any) -> str:
        if not isinstance(model, str) or not model:
            raise SchedulerError("invalid_model", "model is required")
        profiles = self.config.profiles
        if model in profiles:
            alias = model
        else:
            matches = [a for a, p in profiles.items() if p.model_id == model and p.kind == "embedding"]
            if len(matches) != 1:
                raise SchedulerError("model_not_found", "model is not a configured broker profile", status_code=404)
            alias = matches[0]
        if profiles[alias].kind != "embedding":
            raise SchedulerError("wrong_profile_kind", f"profile '{alias}' is a chat profile; use it with /v1/chat/completions")
        return alias

    def check_inputs(self, value: Any) -> list[str]:
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list) or not items or not all(isinstance(item, str) and item for item in items):
            raise SchedulerError("invalid_request", "input must be a non-empty string or a non-empty array of non-empty strings")
        if len(items) > MAX_INPUTS:
            raise SchedulerError("invalid_request", f"input has more than {MAX_INPUTS} items")
        if sum(len(item.encode("utf-8")) for item in items) > self.config.max_text_bytes:
            raise SchedulerError("invalid_request", "input is larger than the configured text limit")
        return list(items)

    async def embed(self, alias: str, inputs: list[str], labels: dict[str, str] | None = None) -> dict[str, Any]:
        labels = labels or {}
        gate = self._gates[alias]
        async with gate:
            self._active[alias] += 1
            started = time.monotonic()
            try:
                result: dict[str, Any] = await self.backend.embed(alias, inputs)
            except BackendFailure as exc:
                self._emit("embedding_failed", alias, len(inputs), started, **labels, error_code=exc.code)
                raise SchedulerError(exc.code, str(exc), status_code=502, retryable=exc.retryable) from exc
            finally:
                self._active[alias] -= 1
            self._requests[alias] += 1
            self._emit("embedding_finished", alias, len(inputs), started, **labels)
            return result

    def _emit(self, event: str, alias: str, inputs: int, started: float, **extra: Any) -> None:
        self.telemetry.emit(
            {"event": event, "profile": alias, "inputs": inputs, "seconds": round(time.monotonic() - started, 3), **extra}
        )

    def status(self) -> dict[str, dict[str, int]]:
        return {
            alias: {"active": self._active[alias], "limit": self._limits[alias], "requests": self._requests[alias]}
            for alias in self._limits
        }


def register_embedding_routes(
    app: FastAPI, service: EmbeddingService, auth_dependency: Callable[..., Awaitable[None]]
) -> None:
    @app.post("/v1/embeddings")
    async def embeddings(
        request: Request,
        _: None = Depends(auth_dependency),
        client_id: str = Header(default="openai", alias="X-Client-Id"),
        project: str = Header(default="openai", alias="X-Project"),
        task: str = Header(default="embeddings", alias="X-Task"),
    ) -> Any:
        try:
            payload = await request.json()
        except ValueError as exc:
            raise SchedulerError("invalid_request", "request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise SchedulerError("invalid_request", "request body must be a JSON object")
        unknown = set(payload) - ALLOWED_FIELDS
        if unknown:
            raise SchedulerError("invalid_request", f"unsupported fields: {', '.join(sorted(unknown))}")
        if payload.get("encoding_format", "float") != "float":
            raise SchedulerError("invalid_request", "only encoding_format \"float\" is supported")
        alias = service.resolve(payload.get("model"))
        inputs = service.check_inputs(payload.get("input"))
        labels = {"client_id": label(client_id, "openai"), "project": label(project, "openai"), "task_label": label(task, "embeddings")}
        result = await service.embed(alias, inputs, labels)
        return {**result, "model": alias}
