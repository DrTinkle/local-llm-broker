from __future__ import annotations

import asyncio
import codecs
import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .contracts import BrokerConfig, JobSpec, JobState
from .scheduler import Scheduler, SchedulerError, check_content

ALLOWED_FIELDS = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "n",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "tier",
}


async def iter_sse_events(chunks: AsyncIterable[bytes], *, max_buffer_bytes: int = 1024 * 1024) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    async for chunk in chunks:
        buffer += decoder.decode(chunk, final=False)
        if len(buffer.encode("utf-8")) > max_buffer_bytes:
            raise ValueError("stream buffer exceeded configured limit")
        while True:
            separator = "\n\n"
            index = buffer.find(separator)
            if index < 0:
                break
            event, buffer = buffer[:index], buffer[index + len(separator) :]
            event = event.strip("\r\n")
            if event:
                yield event
    buffer += decoder.decode(b"", final=True)
    if buffer.strip():
        raise ValueError("incomplete SSE frame")


def register_openai_routes(
    app: FastAPI,
    config: BrokerConfig,
    scheduler: Scheduler,
    auth_dependency: Callable[..., Awaitable[None]],
) -> None:
    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        _: None = Depends(auth_dependency),
        client_id: str = Header(default="openai", alias="X-Client-Id"),
    ) -> Any:
        job_id: str | None = None
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise SchedulerError("invalid_request", "request body must be a JSON object")
            unknown = set(payload) - ALLOWED_FIELDS
            if unknown:
                raise SchedulerError("invalid_request", f"unsupported fields: {', '.join(sorted(unknown))}")
            model, tier = _resolve_profile_and_tier(config, payload.get("model"), payload.get("tier"))
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise SchedulerError("invalid_request", "messages must be a non-empty array")
            _validate_messages(messages, profile=model, supports_vision=config.profiles[model].supports_vision)
            if payload.get("n", 1) != 1:
                raise SchedulerError("unsupported_parameter", "only n=1 is supported")
            stream = payload.get("stream", False)
            if not isinstance(stream, bool):
                raise SchedulerError("invalid_request", "stream must be boolean")
            generation = {
                key: value
                for key, value in payload.items()
                if key not in {"model", "messages", "n", "tier"} and value is not None
            }
            if "stream_options" in generation and not isinstance(generation["stream_options"], dict):
                raise SchedulerError("invalid_request", "stream_options must be an object")
            spec = JobSpec(
                client_id=client_id,
                project="openai",
                task_label="chat-completions",
                profile=model,
                messages=messages,
                generation=generation,
                tier=tier,
            )
            snapshot = await scheduler.submit(spec)
            job_id = snapshot.job_id
            if stream:
                return StreamingResponse(
                    _stream_response(scheduler, snapshot.job_id, config.max_stream_buffer_bytes),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            terminal = await scheduler.wait_for_terminal(snapshot.job_id)
            if terminal.state is not JobState.SUCCEEDED:
                error = terminal.error or {"code": "inference_failed", "message": "local inference failed"}
                raise SchedulerError(
                    str(error.get("code", "inference_failed")),
                    str(error.get("message", "local inference failed")),
                    status_code=502,
                    retryable=bool(error.get("retryable", True)),
                )
            return scheduler.result(snapshot.job_id)["result"]
        except asyncio.CancelledError:
            if job_id is not None:
                try:
                    await scheduler.cancel(job_id)
                except Exception:
                    pass
            raise
        except SchedulerError as exc:
            return _openai_error(exc)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return _openai_error(SchedulerError("invalid_request", str(exc)))


async def _stream_response(scheduler: Scheduler, job_id: str, max_buffer_bytes: int) -> AsyncIterator[bytes]:
    finished = False
    saw_done = False
    try:
        async for event in iter_sse_events(scheduler.stream_events(job_id), max_buffer_bytes=max_buffer_bytes):
            saw_done = saw_done or event.strip() == "data: [DONE]"
            yield (event + "\n\n").encode("utf-8")
        if not saw_done:
            raise ValueError("stream ended without [DONE]")
        finished = True
    except Exception:
        error = {"error": {"message": "stream terminated before completion", "type": "server_error", "code": "stream_failed"}}
        yield ("event: error\ndata: " + json.dumps(error) + "\n\n").encode("utf-8")
    finally:
        try:
            if not finished:
                await scheduler.cancel(job_id)
        except Exception:
            pass


def _resolve_profile_and_tier(config: BrokerConfig, model: Any, tier: Any) -> tuple[str, str | None]:
    # OpenAI clients that cannot add a field can name the tier as a model suffix, for example "my-model:high".
    if isinstance(model, str) and ":" in model and model.rsplit(":", 1)[1] in {"low", "medium", "high", "auto"}:
        base, suffix = model.rsplit(":", 1)
        if tier is not None and tier != suffix:
            raise SchedulerError("invalid_request", "tier field and model suffix disagree")
        model, tier = base, suffix
    if tier is not None and tier not in {"low", "medium", "high", "auto"}:
        raise SchedulerError("invalid_request", "tier must be low, medium, high or auto")
    return _resolve_profile(config, model), (None if tier == "auto" else tier)


def _resolve_profile(config: BrokerConfig, model: Any) -> str:
    if not isinstance(model, str):
        raise SchedulerError("invalid_model", "model is required")
    if model in config.profiles:
        return model
    matches = [alias for alias, profile in config.profiles.items() if profile.model_id == model]
    if len(matches) > 1:
        # A thinking profile and its -nothink twin share one model ID; the canonical ID means the thinking profile.
        matches = [alias for alias in matches if config.profiles[alias].reasoning_effort is None]
    if len(matches) != 1:
        raise SchedulerError("model_not_found", "model is not a configured broker profile", status_code=404)
    return matches[0]


def _validate_messages(messages: list[Any], *, profile: str, supports_vision: bool) -> None:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise SchedulerError("invalid_messages", "messages must contain supported roles")
        check_content(message.get("content", ""), profile=profile, supports_vision=supports_vision)


def _openai_error(exc: SchedulerError) -> JSONResponse:
    kind = "invalid_request_error" if exc.status_code < 500 else "server_error"
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": str(exc), "type": kind, "code": exc.code}},
    )

