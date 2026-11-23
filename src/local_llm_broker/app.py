from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import digest_key, ensure_auth_token
from .contracts import BrokerConfig, JobSpec
from .instance_lock import BackendOwnerLock, common_backend_lock_path
from .lm_studio import LMStudioAdapter
from .routes_openai import register_openai_routes
from .scheduler import Scheduler, SchedulerError
from .storage import MetadataStore
from .telemetry import EventWriter


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_id: str = Field(min_length=1, max_length=128)
    project: str = Field(default="", max_length=128)
    task_label: str = Field(default="", max_length=128)
    profile: str
    messages: list[dict[str, Any]]
    generation: dict[str, Any] = Field(default_factory=dict)
    queue_timeout_seconds: float | None = Field(default=None, gt=0, le=86_400)
    # low, medium or high; omitted or "auto" lets the broker pick the smallest tier the job fits.
    tier: str | None = Field(default=None, pattern="^(low|medium|high|auto)$")


class ShutdownRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    force: bool = False


STATUS_KEEPALIVE_SECONDS = 30.0
# Several state changes often come together (a job ends, the next starts); one snapshot covers them.
STATUS_SETTLE_SECONDS = 0.05


def _status_change_key(snapshot: dict[str, Any]) -> str:
    # The idle-unload countdown changes every second; only its starting and stopping count as a change.
    keyed = dict(snapshot)
    keyed["idle_unload_in_seconds"] = snapshot.get("idle_unload_in_seconds") is not None
    return json.dumps(keyed, sort_keys=True, default=str)


def _sse_frame(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


def create_app(
    config: BrokerConfig,
    *,
    adapter: Any | None = None,
    status_keepalive_seconds: float = STATUS_KEEPALIVE_SECONDS,
    status_settle_seconds: float = STATUS_SETTLE_SECONDS,
) -> FastAPI:
    store = MetadataStore(Path(config.data_dir) / "state.sqlite3", digest_key=digest_key(config))
    telemetry = EventWriter(Path(config.data_dir) / "logs" / "events.jsonl")
    backend = adapter or LMStudioAdapter(
        config.backend_url,
        config.profiles,
        load_timeout=config.load_timeout,
        generation_timeout=config.generation_timeout,
        tiers=config.tiers,
        idle_unload_seconds=config.idle_unload_seconds,
        lms_path=config.lms_path,
    )
    scheduler = Scheduler(config, backend, store=store, events=telemetry)
    token = ensure_auth_token(config)
    owner_lock = BackendOwnerLock(
        Path(config.data_dir) / "broker.lock",
        common_path=common_backend_lock_path(config.backend_url),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owner_lock.acquire()
        try:
            await scheduler.start()
            yield
        finally:
            await scheduler.shutdown()
            close = getattr(backend, "close", None)
            if close is not None:
                await close()
            owner_lock.release()

    app = FastAPI(title="Local LLM Broker", version="0.1.0", lifespan=lifespan)
    app.state.scheduler = scheduler
    app.state.auth_token = token
    app.state.telemetry = telemetry
    app.state.owner_lock = owner_lock
    app.state.status_listeners = 0
    # Set by the serving command so that POST /admin/shutdown can ask the server to exit.
    app.state.request_shutdown = None

    @app.middleware("http")
    async def loopback_guard(request: Request, call_next: Any) -> Any:
        host = request.headers.get("host", "").rsplit(":", 1)[0].strip("[]")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return JSONResponse(status_code=400, content={"error": {"code": "loopback_only", "message": "loopback host required"}})
        origin = request.headers.get("origin")
        if origin:
            parsed_origin = urlparse(origin)
            if parsed_origin.hostname not in {"127.0.0.1", "localhost", "::1"}:
                return JSONResponse(status_code=403, content={"error": {"code": "origin_rejected", "message": "origin is not allowed"}})
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = -1
            if declared_length < 0:
                return JSONResponse(status_code=400, content={"error": {"code": "invalid_content_length", "message": "Content-Length must be a non-negative integer"}})
            if declared_length > config.max_body_bytes:
                return JSONResponse(status_code=413, content={"error": {"code": "body_too_large", "message": "request body exceeds configured limit"}})
        if request.method in {"POST", "PUT", "PATCH"} and not content_length:
            received = bytearray()
            more_body = True
            while more_body:
                message = await request._receive()
                body = message.get("body", b"")
                received.extend(body)
                if len(received) > config.max_body_bytes:
                    return JSONResponse(status_code=413, content={"error": {"code": "body_too_large", "message": "request body exceeds configured limit"}})
                more_body = bool(message.get("more_body", False))
            request._body = bytes(received)
        return await call_next(request)

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not secrets.compare_digest((authorization or "").encode("utf-8"), f"Bearer {token}".encode()):
            raise SchedulerError("unauthorized", "bearer authentication required", status_code=401)

    @app.exception_handler(SchedulerError)
    async def scheduler_error_handler(request: Request, exc: SchedulerError) -> JSONResponse:
        if request.url.path.startswith("/v1/"):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
                        "code": exc.code,
                    }
                },
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": str(exc), "retryable": exc.retryable}},
            headers={"Retry-After": "1"} if exc.status_code == 429 else None,
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def status_snapshot() -> dict[str, Any]:
        result = scheduler.status()
        health = getattr(backend, "health", None)
        if health is None:
            result["backend_health"] = {"state": "unknown"}
        else:
            try:
                # Cached by the adapter: LM Studio is asked at most once per 30 seconds, plus after loads and unloads.
                result["backend_health"] = await health()
            except Exception:
                result["backend_health"] = {"state": "unavailable", "code": "backend_health_failed"}
        return result

    @app.get("/status", dependencies=[Depends(require_auth)])
    async def status() -> dict[str, Any]:
        return await status_snapshot()

    @app.get("/status/stream", dependencies=[Depends(require_auth)])
    async def status_stream() -> StreamingResponse:
        """Server-sent events: a `status_snapshot` frame at once, another on each change, and a keep-alive comment
        after each quiet period. Waiting costs nothing; LM Studio is not asked unless the status changed."""

        async def frames() -> AsyncIterator[bytes]:
            app.state.status_listeners += 1
            try:
                version = scheduler.change_version
                snapshot = await status_snapshot()
                last_key = _status_change_key(snapshot)
                yield _sse_frame("status_snapshot", snapshot)
                while not scheduler.stopped:
                    if not await scheduler.wait_for_change(version, status_keepalive_seconds):
                        yield b": keep-alive\n\n"
                        continue
                    if scheduler.stopped:
                        break
                    await asyncio.sleep(status_settle_seconds)
                    version = scheduler.change_version
                    snapshot = await status_snapshot()
                    key = _status_change_key(snapshot)
                    if key != last_key:
                        last_key = key
                        yield _sse_frame("status_snapshot", snapshot)
            finally:
                app.state.status_listeners -= 1

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": profile.alias,
                    "object": "model",
                    "owned_by": "local-llm-broker",
                    "model_id": profile.model_id,
                    "parallel": profile.parallel_ceiling,
                    "thinks": profile.thinks,
                    "vision": profile.supports_vision,
                    "description": profile.description,
                    "tiers": {name: tier.public_dict() for name, tier in config.tiers_for(profile.alias).items()},
                }
                for profile in config.profiles.values()
            ],
        }

    @app.get("/logs", dependencies=[Depends(require_auth)])
    async def logs() -> dict[str, Any]:
        return {"events": telemetry.recent()}

    @app.get("/summary", dependencies=[Depends(require_auth)])
    async def summary() -> dict[str, Any]:
        return store.summary()

    @app.post("/jobs", status_code=202, dependencies=[Depends(require_auth)])
    async def submit_job(payload: JobRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict[str, Any]:
        fields = payload.model_dump()
        if fields.get("tier") == "auto":
            fields["tier"] = None
        spec = JobSpec(**fields)
        snapshot, created = await scheduler.submit_ex(spec, idempotency_key=idempotency_key)
        if created:  # an idempotent replay is not a new acceptance
            telemetry.emit({"event": "accepted", "job_id": snapshot.job_id, "profile": snapshot.profile, "client_id": snapshot.client_id})
        return snapshot.public_dict()

    @app.get("/jobs/{job_id}", dependencies=[Depends(require_auth)])
    async def get_job(job_id: str) -> dict[str, Any]:
        snapshot = scheduler.get(job_id)
        if snapshot is None:
            raise SchedulerError("not_found", "job not found", status_code=404)
        return snapshot.public_dict()

    @app.get("/jobs/{job_id}/result", dependencies=[Depends(require_auth)])
    async def get_result(job_id: str) -> dict[str, Any]:
        return scheduler.result(job_id)

    @app.delete("/jobs/{job_id}", dependencies=[Depends(require_auth)])
    async def cancel_job(job_id: str) -> dict[str, Any]:
        return (await scheduler.cancel(job_id)).public_dict()

    @app.post("/admin/shutdown", status_code=202, dependencies=[Depends(require_auth)], response_model=None)
    async def shutdown(background: BackgroundTasks, payload: ShutdownRequest | None = None) -> Any:
        snapshot = scheduler.status()
        busy = int(snapshot.get("queue_count", 0)) + int(snapshot.get("active_count", 0))
        if busy > 0 and not (payload is not None and payload.force):
            raise SchedulerError(
                "broker_busy",
                f"{busy} job(s) are queued or running; wait for them, or stop with --force",
                status_code=409,
                retryable=True,
            )

        def request_exit() -> None:
            hook = app.state.request_shutdown
            if hook is not None:
                hook()

        # Background tasks run after the response is sent, so the caller always receives the 202.
        background.add_task(request_exit)
        return {"status": "stopping"}

    register_openai_routes(app, config, scheduler, require_auth)

    return app

