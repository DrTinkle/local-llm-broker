"""Calibration modes that load settings directly in LM Studio, without the broker.

They load configurations that are not in config.toml (larger contexts, more slots, other expert splits), so the
broker must be stopped: the session refuses to start while the broker answers. Every load is unloaded afterwards.

A configuration "fits" when it loads, a probe job succeeds, its prompt processing and generation stay within 15%
of the baseline, and nvidia-smi (when available) shows at least 300 MiB of dedicated VRAM still free.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from .calibrate import synthetic_prompt
from .calibrate_concurrency import Round, _round
from .calibrate_probe import JobResult, gpu_memory_mib, stream_job, unique
from .contracts import TierConfig
from .lms_sdk import sdk_load

NATIVE_MAX_CONTEXT = 262_144
CONTEXT_STEP = 4_096
BASELINE_CONTEXT = 16_384
PROBE_TOKENS = 8_000
SPEED_RATIO = 0.85
VRAM_HEADROOM_MIB = 300


def search_combos(is_moe: bool) -> tuple[tuple[str, float | None], ...]:
    """(kv_cache, cpu_expert_ratio) combinations the context search tries."""
    if is_moe:
        return (("f16", 0.5), ("q8_0", 0.5), ("q8_0", 0.75))
    return (("f16", None), ("q8_0", None))


def slot_candidates(parallel: int) -> tuple[int, ...]:
    """Slot counts to sweep around a model's configured parallel ceiling."""
    if parallel <= 1:
        return (1, 2)
    if parallel <= 3:
        return (parallel - 1, parallel, parallel + 1)
    return (parallel, parallel * 3 // 2, parallel * 2)


EXPERT_RATIOS = tuple(round(0.5 + 0.05 * i, 2) for i in range(11))  # 0.50 .. 1.00

Loader = Callable[[str, TierConfig, int], Awaitable[tuple[str, float]]]


class BrokerRunning(RuntimeError):
    pass


@dataclass(slots=True)
class Probe:
    model: str
    tier: TierConfig
    parallel: int
    loaded: bool = False
    load_seconds: float | None = None
    vram: tuple[int, int] | None = None
    run: JobResult | None = None
    round: Round | None = None
    error: str | None = None

    @property
    def headroom_ok(self) -> bool:
        return self.vram is None or self.vram[1] - self.vram[0] >= VRAM_HEADROOM_MIB

    def fits(self, baseline: Probe | None) -> bool:
        if not self.loaded or self.run is None or not self.run.ok or not self.headroom_ok:
            return False
        if baseline is None or baseline.run is None:
            return True
        for mine, base in ((self.run.prefill_tps, baseline.run.prefill_tps), (self.run.generation_tps, baseline.run.generation_tps)):
            if mine and base and mine < SPEED_RATIO * base:
                return False
        return True


class DirectSession:
    def __init__(self, backend_url: str, broker_url: str, *, loader: Loader | None = None, client: httpx.AsyncClient | None = None,
                 vram_reader: Callable[[], tuple[int, int] | None] = gpu_memory_mib, settle_seconds: float = 2.0) -> None:
        self.backend_url = backend_url
        self.broker_url = broker_url
        self.client = client or httpx.AsyncClient(base_url=backend_url, timeout=httpx.Timeout(30.0, read=1800.0))
        self._loader = loader or (lambda model, tier, parallel: sdk_load(backend_url, model, tier, parallel))
        self._vram = vram_reader
        self.settle_seconds = settle_seconds

    async def close(self) -> None:
        await self.client.aclose()

    async def ensure_broker_stopped(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=2.0) as probe:
                await probe.get(self.broker_url.rstrip("/") + "/health")
        except httpx.HTTPError:
            return
        raise BrokerRunning("the broker is running; stop it before a direct calibration mode, and start it again afterwards")

    async def unload_all(self) -> None:
        response = await self.client.get("/api/v1/models")
        for model in response.json().get("models", []):
            for instance in model.get("loaded_instances") or []:
                await self.client.post("/api/v1/models/unload", json={"instance_id": instance.get("id")})

    async def try_config(self, model: str, tier: TierConfig, parallel: int, *, probe_tokens: int = PROBE_TOKENS, jobs: int = 0) -> Probe:
        """Load, read VRAM, run one probe job (or a round of `jobs` parallel jobs), unload."""
        probe = Probe(model, tier, parallel)
        await self.unload_all()
        started = time.monotonic()
        try:
            instance, _ = await self._loader(model, tier, parallel)
        except Exception as exc:  # a load that does not fit is an answer, not an error
            probe.error = f"load failed: {str(exc)[:100]}"
            return probe
        probe.loaded = True
        probe.load_seconds = time.monotonic() - started
        try:
            await asyncio.sleep(self.settle_seconds)
            probe.vram = await asyncio.to_thread(self._vram)
            if jobs:
                prompts = [unique(synthetic_prompt(probe_tokens), f"slot-{i}") for i in range(jobs)]
                probe.round = await _round(self.client, instance, None, prompts)
                probe.run = next((job for job in probe.round.jobs if job.ok), probe.round.jobs[0])
            else:
                prompt = unique(synthetic_prompt(probe_tokens), f"{model}-{tier.context_length}")
                probe.run = await stream_job(self.client, {
                    "model": instance,
                    "messages": [{"role": "user", "content": prompt + "\n\nSummarize the text above in about 80 words."}],
                    "max_tokens": 128,
                    "temperature": 0,
                })
        finally:
            await self.unload_all()
        return probe


# --- max context search -----------------------------------------------------------------------------------------


@dataclass(slots=True)
class SearchResult:
    model: str
    kv_cache: str
    cpu_expert_ratio: float | None
    baseline: Probe | None = None
    probes: list[Probe] = field(default_factory=list)
    max_context: int | None = None
    error: str | None = None


def _tier(context: int, kv: str, ratio: float | None, model: str) -> TierConfig:
    return TierConfig("calibration", context, kv, cpu_expert_ratio=ratio, try_mmap=False if ratio is not None else None)


async def search_context(session: DirectSession, model: str, kv: str, ratio: float | None, parallel: int, *,
                         baseline_context: int = BASELINE_CONTEXT, top_context: int = NATIVE_MAX_CONTEXT,
                         step: int = CONTEXT_STEP) -> SearchResult:
    result = SearchResult(model, kv, ratio)
    baseline = await session.try_config(model, _tier(baseline_context, kv, ratio, model), parallel)
    result.baseline = baseline
    if not baseline.fits(None):
        result.error = baseline.error or "the baseline context does not fit"
        return result
    lo, hi = baseline_context, top_context
    top = await session.try_config(model, _tier(hi, kv, ratio, model), parallel)
    result.probes.append(top)
    if top.fits(baseline):
        result.max_context = hi
        return result
    while hi - lo > step:
        mid = ((lo + hi) // 2) // step * step
        if mid <= lo:
            break
        probe = await session.try_config(model, _tier(mid, kv, ratio, model), parallel)
        result.probes.append(probe)
        if probe.fits(baseline):
            lo = mid
        else:
            hi = mid
    result.max_context = lo
    return result


# --- slots and expert splits ------------------------------------------------------------------------------------


@dataclass(slots=True)
class SweepResult:
    model: str
    label: str
    probes: dict[float, Probe] = field(default_factory=dict)  # slots or ratio -> probe


async def sweep_slots(session: DirectSession, model: str, tier: TierConfig, candidates: Sequence[int]) -> SweepResult:
    result = SweepResult(model, f"slots at {tier.context_length:,} context, {tier.kv_cache} cache")
    for slots in candidates:
        result.probes[slots] = await session.try_config(model, tier, slots, probe_tokens=1_500, jobs=slots)
    return result


async def sweep_experts(session: DirectSession, model: str, tier: TierConfig, ratios: Sequence[float]) -> SweepResult:
    result = SweepResult(model, f"CPU expert ratio at {tier.context_length:,} context, {tier.kv_cache} cache")
    for ratio in ratios:
        result.probes[ratio] = await session.try_config(model, replace(tier, cpu_expert_ratio=ratio, try_mmap=False), 1)
    return result


def run_direct(backend_url: str, broker_url: str, work: Callable[[DirectSession], Awaitable[Any]]) -> Any:
    async def main() -> Any:
        session = DirectSession(backend_url, broker_url)
        try:
            await session.ensure_broker_stopped()
            return await work(session)
        finally:
            await session.close()

    return asyncio.run(main())


# --- reports ----------------------------------------------------------------------------------------------------


def _vram(probe: Probe) -> str:
    return f"{probe.vram[0]:,}/{probe.vram[1]:,} MiB" if probe.vram else "-"


def _speeds(probe: Probe) -> tuple[str, str]:
    run = probe.run
    prefill = f"{run.prefill_tps:,.0f}" if run and run.prefill_tps else "-"
    gen = f"{run.generation_tps:.1f}" if run and run.generation_tps else "-"
    return prefill, gen


def format_search(results: Sequence[SearchResult]) -> str:
    lines: list[str] = []
    for item in results:
        ratio = f", CPU expert ratio {item.cpu_expert_ratio}" if item.cpu_expert_ratio is not None else ""
        lines.append(f"{item.model}, {item.kv_cache} cache{ratio}")
        if item.error:
            lines.append(f"  could not search: {item.error}")
            lines.append("")
            continue
        lines.append("  context    VRAM used/total   prompt tok/s   gen tok/s   fits")
        for probe in ([item.baseline] if item.baseline else []) + sorted(item.probes, key=lambda p: p.tier.context_length):
            prefill, gen = _speeds(probe)
            fits = "baseline" if probe is item.baseline else ("yes" if probe.fits(item.baseline) else (probe.error or "no"))
            lines.append(f"  {probe.tier.context_length:>7,}   {_vram(probe):>17}   {prefill:>12}   {gen:>9}   {fits}")
        lines.append(f"  largest context that fits: {item.max_context:,}" if item.max_context else "  largest context that fits: -")
        lines.append("")
    return "\n".join(lines).rstrip()


def _prefill_holds(probe: Probe, sweep: SweepResult) -> bool:
    best = max((p.run.prefill_tps or 0.0 for p in sweep.probes.values() if p.run and p.headroom_ok), default=0.0)
    mine = probe.run.prefill_tps if probe.run else None
    return not best or (mine is not None and mine >= 0.5 * best)


def format_sweep(results: Sequence[SweepResult], kind: str) -> str:
    lines: list[str] = []
    for item in results:
        lines.append(f"{item.model}: {item.label}")
        if kind == "slots":
            lines.append("  slots    VRAM used/total   combined tok/s   per-job tok/s   outcome")
        else:
            lines.append("  ratio    VRAM used/total   prompt tok/s   gen tok/s   fits")
        best_key, best_value = None, 0.0
        for key, probe in item.probes.items():
            if kind == "slots":
                round_ = probe.round
                combined = f"{round_.combined_tps:.1f}" if round_ and round_.combined_tps else "-"
                per_job = f"{round_.per_job_tps:.1f}" if round_ and round_.per_job_tps else "-"
                outcome = probe.error or (f"{round_.failures} failed" if round_ and round_.failures else "all finished")
                lines.append(f"  {int(key):>5}   {_vram(probe):>17}   {combined:>14}   {per_job:>13}   {outcome}")
                value = round_.combined_tps if round_ and not round_.failures and probe.headroom_ok else None
            else:
                prefill, gen = _speeds(probe)
                # Free VRAM alone misses an overflow Windows has already moved to shared memory, so a collapse
                # in prompt processing against the sweep's best counts as not fitting too.
                ok = probe.fits(None) and _prefill_holds(probe, item)
                lines.append(f"  {key:>5.2f}   {_vram(probe):>17}   {prefill:>12}   {gen:>9}   {'yes' if ok else (probe.error or 'no')}")
                value = probe.run.generation_tps if ok and probe.run else None
            if value and value > best_value:
                best_key, best_value = key, value
        if best_key is not None:
            label = f"{int(best_key)} slots" if kind == "slots" else f"ratio {best_key:.2f}"
            lines.append(f"  fastest that fits: {label} ({best_value:.1f} tok/s)")
        lines.append("")
    return "\n".join(lines).rstrip()
