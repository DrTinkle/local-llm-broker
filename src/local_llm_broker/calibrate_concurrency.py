"""Measure usable concurrency per model and tier, through the broker.

Throughput sweep: N identical short jobs at once, N = 1 up to the profile's parallel slots. The usable slot count
is the largest N whose combined generation speed is still clearly higher than with fewer jobs.

Capacity sweep: all slots busy, with prompts sized so the jobs together fill 50%, 90%, 120% and 150% of the tier's
context. LM Studio loads with a unified KV cache, so parallel jobs share one context pool; this shows what LM Studio
does when the pool overfills. Profiles with one slot skip it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .calibrate import synthetic_prompt
from .calibrate_probe import JobResult, stream_job

# A step up in parallel jobs counts as useful when the combined generation speed rises at least this much.
USEFUL_GAIN = 1.15
CAPACITY_FILLS = (0.5, 0.9, 1.2, 1.5, 1.8)  # planned at 4 bytes per token; prose is about 5, so 1.8 really overfills
OUTPUT_TOKENS = 200


@dataclass(slots=True)
class Round:
    jobs: list[JobResult]
    started_at: float

    @property
    def failures(self) -> int:
        return sum(1 for job in self.jobs if not job.ok)

    @property
    def combined_tps(self) -> float | None:
        # All generated tokens over the span from the first job's first token to the last job's end.
        done = [
            (job.first_token_at, job.ended_at, job.completion_tokens)
            for job in self.jobs
            if job.ok and job.first_token_at is not None and job.ended_at is not None and job.completion_tokens
        ]
        if not done:
            return None
        span = max(end for _, end, _ in done) - min(first for first, _, _ in done)
        tokens = sum(count for _, _, count in done)
        return tokens / span if span > 0 else None

    @property
    def per_job_tps(self) -> float | None:
        values = [tps for job in self.jobs if (tps := job.generation_tps)]
        return sum(values) / len(values) if values else None

    @property
    def real_tokens(self) -> int:
        return sum((job.prompt_tokens or 0) + (job.completion_tokens or 0) for job in self.jobs)


@dataclass(slots=True)
class TierConcurrency:
    profile: str
    tier: str
    context_length: int
    slots: int
    throughput: dict[int, Round] = field(default_factory=dict)
    capacity: dict[float, Round] = field(default_factory=dict)
    error: str | None = None


def recommend_slots(combined: dict[int, float | None]) -> int:
    """Largest N whose combined speed beats the best smaller N by USEFUL_GAIN; 1 if nothing does."""
    best_n, best = 1, combined.get(1) or 0.0
    for n in sorted(combined):
        value = combined[n] or 0.0
        if n > best_n and value >= best * USEFUL_GAIN:
            best_n, best = n, value
    return best_n


def capacity_prompt_tokens(context_length: int, slots: int, fill: float) -> int:
    """Per-job prompt size so `slots` jobs together use `fill` of the context, after output and the broker's margin."""
    return max(64, int(context_length * fill / slots) - OUTPUT_TOKENS - 512)


async def _one_job(client: httpx.AsyncClient, profile: str, tier: str | None, prompt: str) -> JobResult:
    body: dict[str, Any] = {
        "model": profile,
        "messages": [{"role": "user", "content": prompt + "\n\nSummarize the text above in about 120 words."}],
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
    }
    if tier is not None:  # None when talking to LM Studio directly, which has no tiers
        body["tier"] = tier
    return await stream_job(client, body)


async def _round(client: httpx.AsyncClient, profile: str, tier: str | None, prompts: list[str]) -> Round:
    started = time.monotonic()
    jobs = await asyncio.gather(*(_one_job(client, profile, tier, prompt) for prompt in prompts))
    return Round(list(jobs), started)


def _variant(prompt: str, index: int) -> str:
    # A different first line per job, so LM Studio cannot reuse one job's cached prompt for another.
    return f"Document {index + 1}.\n" + prompt


async def measure_tier(
    client: httpx.AsyncClient, profile: str, tier: str, context_length: int, slots: int, *, capacity: bool = True
) -> TierConcurrency:
    out = TierConcurrency(profile, tier, context_length, slots)
    warm = await _one_job(client, profile, tier, "Reply with OK.")
    if not warm.ok:
        out.error = warm.error or "warm-up failed"
        return out
    short = synthetic_prompt(2000)
    for n in range(1, slots + 1):
        out.throughput[n] = await _round(client, profile, tier, [_variant(short, i) for i in range(n)])
    if capacity and slots > 1:
        for fill in CAPACITY_FILLS:
            text = synthetic_prompt(capacity_prompt_tokens(context_length, slots, fill))
            out.capacity[fill] = await _round(client, profile, tier, [_variant(text, i) for i in range(slots)])
    return out


def run_concurrency(
    base_url: str,
    headers: dict[str, str],
    models: dict[str, Any],
    profiles: Iterable[str],
    tiers: Iterable[str],
    *,
    capacity: bool = True,
) -> list[TierConcurrency]:
    by_id = {item.get("id"): item for item in models.get("data", []) if isinstance(item, dict)}
    plan: list[tuple[str, str, int, int]] = []
    for profile in profiles:
        entry = by_id.get(profile) or {}
        for tier in tiers:
            info = (entry.get("tiers") or {}).get(tier)
            if info:
                plan.append((profile, tier, int(info["context_length"]), int(entry.get("parallel") or 1)))

    async def main() -> list[TierConcurrency]:
        timeout = httpx.Timeout(30.0, read=1800.0)
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout) as client:
            return [await measure_tier(client, *item, capacity=capacity) for item in plan]

    return asyncio.run(main())


def _outcome(round_: Round) -> str:
    if not round_.failures:
        cut = sum(1 for job in round_.jobs if job.finish_reason == "length")
        return "all finished" if not cut else f"all finished ({cut} hit the output limit)"
    codes = sorted({job.error or "failed" for job in round_.jobs if not job.ok})
    return f"{round_.failures} of {len(round_.jobs)} failed: {', '.join(codes)}"


def format_report(results: list[TierConcurrency]) -> str:
    lines: list[str] = []
    for item in results:
        lines.append(f"{item.profile} {item.tier} ({item.context_length:,} context, {item.slots} slots)")
        if item.error:
            lines.append(f"  could not run: {item.error}")
            lines.append("")
            continue
        lines.append("  jobs at once   combined tok/s   per-job tok/s   outcome")
        for n, round_ in item.throughput.items():
            combined = f"{round_.combined_tps:.1f}" if round_.combined_tps else "-"
            per_job = f"{round_.per_job_tps:.1f}" if round_.per_job_tps else "-"
            lines.append(f"  {n:>12}   {combined:>14}   {per_job:>13}   {_outcome(round_)}")
        best = recommend_slots({n: r.combined_tps for n, r in item.throughput.items() if not r.failures})
        lines.append(f"  usable slots: {best} of {item.slots}")
        if item.capacity:
            # Fill is planned from the broker's estimate; the real share depends on how the text tokenizes.
            lines.append("  planned fill   real tokens (all jobs)   real fill   outcome")
            for fill, round_ in item.capacity.items():
                real = f"{round_.real_tokens / item.context_length:.0%}" if round_.real_tokens else "-"
                lines.append(f"  {fill:>12.0%}   {round_.real_tokens:>21,}   {real:>9}   {_outcome(round_)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def to_json(results: list[TierConcurrency]) -> list[dict[str, Any]]:
    def round_dict(round_: Round) -> dict[str, Any]:
        return {
            "combined_tps": round_.combined_tps,
            "per_job_tps": round_.per_job_tps,
            "real_tokens": round_.real_tokens,
            "outcome": _outcome(round_),
            "jobs": [
                {"ok": j.ok, "prompt_tokens": j.prompt_tokens, "completion_tokens": j.completion_tokens,
                 "finish_reason": j.finish_reason, "error": j.error}
                for j in round_.jobs
            ],
        }

    return [
        {
            "profile": item.profile, "tier": item.tier, "context_length": item.context_length, "slots": item.slots,
            "error": item.error,
            "usable_slots": recommend_slots({n: r.combined_tps for n, r in item.throughput.items() if not r.failures}) if item.throughput else None,
            "throughput": {str(n): round_dict(r) for n, r in item.throughput.items()},
            "capacity": {f"{fill:.0%}": round_dict(r) for fill, r in item.capacity.items()},
        }
        for item in results
    ]
