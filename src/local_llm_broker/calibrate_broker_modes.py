"""Calibration modes that run through the broker: token counts per content type, and speed by prompt depth."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .calibrate import synthetic_prompt
from .calibrate_probe import (
    JobResult,
    content_samples,
    count_prompt_tokens,
    png_data_url,
    stream_job,
    unique,
)

IMAGE_SIDES = (256, 512, 1024, 1536)
DEPTHS = (2_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000)
DEPTH_OUTPUT_TOKENS = 128
BROKER_BYTES_PER_TOKEN = 4


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(30.0, read=1800.0)


# --- tokens per content type ------------------------------------------------------------------------------------


@dataclass(slots=True)
class TokenReport:
    text: dict[str, dict[str, float | None]] = field(default_factory=dict)  # profile -> content type -> bytes/token
    images: dict[str, dict[int, int | None]] = field(default_factory=dict)  # profile -> side -> tokens
    errors: list[str] = field(default_factory=list)


async def _tokens(client: httpx.AsyncClient, profiles: Iterable[str], image_profiles: Iterable[str]) -> TokenReport:
    report = TokenReport()
    instruction = "Reply with OK."
    samples = content_samples()
    for profile in profiles:
        base = await count_prompt_tokens(client, {"model": profile, "messages": [{"role": "user", "content": instruction}]})
        if base is None:
            report.errors.append(f"{profile}: could not get a baseline token count")
            continue
        row: dict[str, float | None] = {}
        for name, text in samples.items():
            body = {"model": profile, "messages": [{"role": "user", "content": text + "\n\n" + instruction}]}
            count = await count_prompt_tokens(client, body)
            # The separator and template add a few tokens; they are part of the baseline difference.
            row[name] = len(text.encode("utf-8")) / (count - base) if count and count > base else None
        report.text[profile] = row
    for profile in image_profiles:
        base = await count_prompt_tokens(client, {"model": profile, "messages": [{"role": "user", "content": [{"type": "text", "text": instruction}]}]})
        if base is None:
            report.errors.append(f"{profile}: could not get an image baseline")
            continue
        sizes: dict[int, int | None] = {}
        for side in IMAGE_SIDES:
            content = [{"type": "text", "text": instruction}, {"type": "image_url", "image_url": {"url": png_data_url(side, side)}}]
            count = await count_prompt_tokens(client, {"model": profile, "messages": [{"role": "user", "content": content}]})
            sizes[side] = count - base if count is not None else None
        report.images[profile] = sizes
    return report


def run_tokens(base_url: str, headers: dict[str, str], profiles: Iterable[str], image_profiles: Iterable[str]) -> TokenReport:
    async def main() -> TokenReport:
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=_timeout()) as client:
            return await _tokens(client, profiles, image_profiles)

    return asyncio.run(main())


def format_tokens(report: TokenReport) -> str:
    lines: list[str] = []
    profiles = list(report.text)
    if profiles:
        lines.append("bytes per token (lower means more tokens per byte; the broker counts exactly before a job runs)")
        lines.append(f"  {'content':<16}" + "".join(f"{p:>14}" for p in profiles))
        names = list(next(iter(report.text.values())))
        for name in names:
            cells = []
            for p in profiles:
                value = report.text[p].get(name)
                cells.append(f"{value:>14.2f}" if value else f"{'-':>14}")
            lines.append(f"  {name:<16}" + "".join(cells))
        worst = min((v for row in report.text.values() for v in row.values() if v), default=None)
        if worst:
            lines.append(f"  lowest: {worst:.2f} bytes per token, {BROKER_BYTES_PER_TOKEN / worst:.1f}x the tokens of a 4-bytes-per-token guess")
    if report.images:
        lines.append("")
        lines.append("tokens per image (the broker estimates pixels / 1,024 + 2)")
        lines.append(f"  {'side':<8}" + "".join(f"{p:>14}" for p in report.images))
        for side in IMAGE_SIDES:
            lines.append(f"  {str(side) + 'px':<8}" + "".join(
                f"{report.images[p].get(side):>14}" if report.images[p].get(side) is not None else f"{'-':>14}" for p in report.images
            ))
    for error in report.errors:
        lines.append(f"error: {error}")
    return "\n".join(lines)


# --- speed by prompt depth --------------------------------------------------------------------------------------


@dataclass(slots=True)
class DepthResult:
    profile: str
    tier: str
    context_length: int
    load_seconds: float | None
    runs: dict[int, JobResult] = field(default_factory=dict)  # planned depth -> result
    error: str | None = None


def broker_estimate(text: str) -> int:
    # The broker's own prompt estimate (Scheduler._estimate_prompt_tokens) for one text message.
    return max(1, (len(text.encode("utf-8")) + 3) // BROKER_BYTES_PER_TOKEN)


def depths_for(context_length: int, max_depth: int) -> list[int]:
    # Prose tokenizes at about 5 bytes per token, under the broker's 4, so a planned depth always passes its check.
    return [d for d in DEPTHS if d <= max_depth and d + DEPTH_OUTPUT_TOKENS + 512 <= context_length]


async def _depth(client: httpx.AsyncClient, plan: list[tuple[str, str, int]], max_depth: int) -> list[DepthResult]:
    results: list[DepthResult] = []
    for profile, tier, context in plan:
        item = DepthResult(profile, tier, context, None)
        started = time.monotonic()
        warm = await stream_job(client, {"model": profile, "tier": tier, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 8})
        if not warm.ok:
            item.error = warm.error or "warm-up failed"
            results.append(item)
            continue
        # The warm-up includes this tier's load (and the unload of whatever was loaded before).
        item.load_seconds = time.monotonic() - started
        for depth in depths_for(context, max_depth):
            content = unique(synthetic_prompt(depth), f"{profile}-{tier}-{depth}") + "\n\nSummarize the text above in about 80 words."
            if broker_estimate(content) + DEPTH_OUTPUT_TOKENS + 512 > context:
                continue  # the broker would refuse it; leave the depth out rather than report a failure
            body = {
                "model": profile,
                "tier": tier,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": DEPTH_OUTPUT_TOKENS,
                "temperature": 0,
            }
            item.runs[depth] = await stream_job(client, body)
        results.append(item)
    return results


def run_depth(base_url: str, headers: dict[str, str], models: dict[str, Any], profiles: Iterable[str], tiers: Iterable[str], max_depth: int) -> list[DepthResult]:
    by_id = {item.get("id"): item for item in models.get("data", []) if isinstance(item, dict)}
    plan = [
        (profile, tier, int(info["context_length"]))
        for profile in profiles
        for tier in tiers
        if (info := ((by_id.get(profile) or {}).get("tiers") or {}).get(tier))
    ]

    async def main() -> list[DepthResult]:
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=_timeout()) as client:
            return await _depth(client, plan, max_depth)

    return asyncio.run(main())


def format_depth(results: list[DepthResult]) -> str:
    lines: list[str] = []
    for item in results:
        load = f"{item.load_seconds:.1f} s" if item.load_seconds is not None else "-"
        lines.append(f"{item.profile} {item.tier} ({item.context_length:,} context), load and first reply: {load}")
        if item.error:
            lines.append(f"  could not run: {item.error}")
            lines.append("")
            continue
        lines.append("  real prompt tokens   to first token   prompt tok/s   gen tok/s   outcome")
        for _, run in item.runs.items():
            tokens = f"{run.prompt_tokens:,}" if run.prompt_tokens else "-"
            ttft = f"{run.time_to_first_token:.1f} s" if run.time_to_first_token else "-"
            prefill = f"{run.prefill_tps:,.0f}" if run.prefill_tps else "-"
            gen = f"{run.generation_tps:.1f}" if run.generation_tps else "-"
            outcome = "ok" if run.ok else f"failed: {run.error}"
            lines.append(f"  {tokens:>18}   {ttft:>14}   {prefill:>12}   {gen:>9}   {outcome}")
        lines.append("")
    return "\n".join(lines).rstrip()
