"""Shared pieces for the calibration modes: one streamed job, a VRAM reading, and deterministic test inputs."""

from __future__ import annotations

import base64
import json
import shutil
import struct
import subprocess
import time
import zlib
from dataclasses import dataclass
from typing import Any

import httpx

from .calibrate import synthetic_prompt


@dataclass(slots=True)
class JobResult:
    ok: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    started_at: float | None = None
    first_token_at: float | None = None
    ended_at: float | None = None
    finish_reason: str | None = None
    error: str | None = None

    @property
    def time_to_first_token(self) -> float | None:
        if self.started_at is None or self.first_token_at is None:
            return None
        return self.first_token_at - self.started_at

    @property
    def prefill_tps(self) -> float | None:
        ttft = self.time_to_first_token
        return self.prompt_tokens / ttft if self.prompt_tokens and ttft else None

    @property
    def generation_tps(self) -> float | None:
        if self.ok and self.completion_tokens and self.completion_tokens > 1 and self.first_token_at and self.ended_at:
            span = self.ended_at - self.first_token_at
            return (self.completion_tokens - 1) / span if span > 0 else None
        return None


async def stream_job(client: httpx.AsyncClient, body: dict[str, Any]) -> JobResult:
    """Send one streamed chat completion and time it. Never raises for backend or HTTP errors."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    result = JobResult(ok=False, started_at=time.monotonic())
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as response:
            if response.status_code >= 400:
                await response.aread()
                try:
                    error = response.json().get("error", {})
                    result.error = str(error.get("code") or error.get("message") or response.status_code)[:120]
                except ValueError:
                    result.error = f"HTTP {response.status_code}"
                return result
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except ValueError:
                    continue
                if payload.get("error"):
                    error = payload["error"]
                    result.error = str(error.get("code") or error.get("message") or "stream_error")[:120] if isinstance(error, dict) else str(error)[:120]
                    return result
                for choice in payload.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if result.first_token_at is None and (delta.get("content") or delta.get("reasoning_content")):
                        result.first_token_at = time.monotonic()
                    if choice.get("finish_reason"):
                        result.finish_reason = choice["finish_reason"]
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    result.prompt_tokens = usage.get("prompt_tokens")
                    result.completion_tokens = usage.get("completion_tokens")
        result.ended_at = time.monotonic()
        result.ok = result.first_token_at is not None
        if not result.ok:
            result.error = result.error or "empty_output"
    except httpx.HTTPError as exc:
        result.error = type(exc).__name__
    return result


async def count_prompt_tokens(client: httpx.AsyncClient, body: dict[str, Any]) -> int | None:
    """Prompt tokens LM Studio reports for a request, via a short non-streamed completion."""
    try:
        response = await client.post("/v1/chat/completions", json={**body, "max_tokens": 16, "temperature": 0, "stream": False})
        if response.status_code >= 400:
            return None
        usage = response.json().get("usage") or {}
        value = usage.get("prompt_tokens")
        return value if isinstance(value, int) else None
    except (httpx.HTTPError, ValueError):
        return None


def gpu_memory_mib() -> tuple[int, int] | None:
    """(used, total) dedicated GPU memory in MiB from nvidia-smi, or None if it is not available."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        used, total = (int(part.strip()) for part in out.splitlines()[0].split(","))
        return used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def unique(prompt: str, tag: str) -> str:
    # A distinct first line so LM Studio cannot reuse a cached prefix from an earlier request.
    return f"Request {tag} at {time.monotonic_ns()}.\n{prompt}"


# --- deterministic content samples, about `target_bytes` each ----------------------------------------------------


def _repeat(make_line: Any, target_bytes: int) -> str:
    lines: list[str] = []
    size = 0
    index = 0
    while size < target_bytes:
        line = make_line(index)
        lines.append(line)
        size += len(line.encode("utf-8")) + 1
        index += 1
    return "\n".join(lines)


_FI = (
    "rakennustyömaalla aloitettiin aamulla pohjoisseinän runkotyöt ja iltapäivällä tarkastaja pyysi uuden tarkastuksen "
    "ankkuripulteista koska piirustuksissa näkyi eri jako kuin asemapiirroksessa joten työnjohtaja soitti suunnittelijalle"
).split()


def content_samples(target_bytes: int = 16_000) -> dict[str, str]:
    return {
        "English prose": synthetic_prompt(target_bytes // 4)[:target_bytes],
        "Finnish prose": _repeat(lambda i: " ".join(_FI[(i * 5 + k) % len(_FI)] for k in range(14)) + ".", target_bytes),
        "Python code": _repeat(
            lambda i: f"def compute_total_{i}(items: list[dict], rate: float = {i % 7}.5) -> float:\n"
            f"    total = sum(item['price'] * item.get('qty', 1) for item in items if item['id'] != {i})\n"
            f"    return round(total * (1 + rate / 100), 2)\n",
            target_bytes,
        ),
        "JSON": json.dumps(
            [{"id": i, "site": f"site-{i % 37}", "hours": (i * 7) % 40, "status": "open" if i % 3 else "closed", "cost": round(i * 13.37, 2)}
             for i in range(target_bytes // 90)],
            indent=2,
        )[:target_bytes],
        "CSV numbers": _repeat(lambda i: f"{i},{(i * 7919) % 100000},{(i * 31) % 997}.{i % 100:02d},{1700000000 + i * 3600},{(i * 13) % 255}", target_bytes),
        "Log lines": _repeat(
            lambda i: f"2026-11-04T12:{i % 60:02d}:{(i * 7) % 60:02d}.{(i * 137) % 1000:03d}Z WARN worker-{i % 8} "
            f"req={((i * 2654435761) & 0xFFFFFFFF):08x} 10.0.{i % 256}.{(i * 3) % 256} took {(i * 17) % 900}ms status={200 + (i % 5) * 100}",
            target_bytes,
        ),
        "Markdown table": "| id | site | hours | status |\n| --- | --- | ---: | --- |\n"
        + _repeat(lambda i: f"| {i} | site-{i % 37} | {(i * 7) % 40} | {'open' if i % 3 else 'closed'} |", target_bytes - 60),
    }


def png_data_url(width: int, height: int) -> str:
    """A deterministic RGB test image as an inline PNG data URL, built without image libraries."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type: none
        for x in range(width):
            rows += bytes(((x * 255) // max(1, width - 1), (y * 255) // max(1, height - 1), ((x + y) * 127) // max(1, width + height - 2)))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
