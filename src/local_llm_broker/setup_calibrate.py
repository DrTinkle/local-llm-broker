"""Calibration levels for the setup command: a quick pass and a thorough one, a thinking probe, and a resume file.

Both levels load models directly in LM Studio through a DirectSession, so the broker must be stopped. A model that
does not load at the smallest tried context is reported as failed; no tiers are ever made up for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .calibrate_direct import (
    NATIVE_MAX_CONTEXT,
    DirectSession,
    SearchResult,
    search_combos,
    search_context,
    slot_candidates,
    sweep_experts,
    sweep_slots,
)
from .contracts import TierConfig
from .setup_discover import Hardware, ModelInfo

Level = Literal["quick", "thorough"]

STEP = 4_096
MIN_CONTEXT = 4_096
BASELINE = 8_192
QUICK_TOP = 131_072
QUICK_STEP = 16_384
FALLBACK_RATIOS = (0.75, 1.0)
EXPERT_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SLOT_GAIN = 1.15
NOT_LOADED = "did not load at 8,192 context"
THINK_PROMPT = "What is 17 plus 25? Answer with just the number."


@dataclass(slots=True)
class CalibrationResult:
    model_key: str
    tiers: dict[str, TierConfig]
    parallel: int
    thinks: bool | None
    can_disable_thinking: bool | None
    notes: list[str] = field(default_factory=list)
    failed: str | None = None


def floor_step(value: float) -> int:
    """Round down to a multiple of 4096."""
    return int(value // STEP) * STEP


def tiers_from_max(max_f16: int | None, max_q8: int | None, ratio: float | None) -> dict[str, TierConfig]:
    """Turn the largest contexts that fit into low, medium and high tiers with a safety margin."""
    f16, q8 = max_f16 or 0, max_q8 or 0
    best = max(f16, q8)
    if best < MIN_CONTEXT:
        return {}
    high = floor_step(0.85 * best)
    medium = floor_step(0.5 * high)
    low = max(MIN_CONTEXT, min(16_384, floor_step(0.5 * medium)))
    if high <= 8_192:
        planned = [("low", max(MIN_CONTEXT, min(low, high)), "f16")]
    else:
        planned = [
            ("low", low, "f16"),
            ("medium", medium, "f16" if medium <= floor_step(0.85 * f16) else "q8_0"),
            ("high", high, "q8_0" if q8 > f16 else "f16"),
        ]
    tiers: dict[str, TierConfig] = {}
    previous = 0
    for name, context, kv in planned:
        if context <= previous:
            continue
        tiers[name] = TierConfig(name, context, kv, cpu_expert_ratio=ratio, try_mmap=False if ratio is not None else None)
        previous = context
    return tiers


# --- thinking probe ---------------------------------------------------------------------------------------------


def _shows_thinking(body: Any) -> bool:
    message = body["choices"][0]["message"]
    if message.get("reasoning_content") or message.get("reasoning"):
        return True
    content = message.get("content")
    return isinstance(content, str) and "<think>" in content


async def _ask(session: DirectSession, instance: str, extra: dict[str, Any]) -> bool:
    body = {
        "model": instance,
        "messages": [{"role": "user", "content": THINK_PROMPT}],
        "max_tokens": 512,
        "temperature": 0,
        **extra,
    }
    response = await session.client.post("/v1/chat/completions", json=body)
    response.raise_for_status()
    return _shows_thinking(response.json())


async def probe_thinking(session: DirectSession, model_key: str, tier: TierConfig) -> tuple[bool, bool]:
    """Report (thinks, can_disable_thinking). A failed probe answers (False, False) instead of stopping calibration."""
    try:
        instance, _ = await session._loader(model_key, tier, 1)
        thinks = await _ask(session, instance, {})
        if not thinks:
            return False, False
        still_thinks = await _ask(session, instance, {"reasoning_effort": "none"})
        return True, not still_thinks
    except Exception:
        return False, False
    finally:
        with contextlib.suppress(Exception):
            await session.unload_all()


# --- levels -----------------------------------------------------------------------------------------------------


def _fits(result: SearchResult) -> bool:
    return result.max_context is not None


async def _first_fit(session: DirectSession, key: str, ratios: list[float | None], parallel: int, top: int, step: int) -> SearchResult | None:
    for ratio in ratios:
        result = await search_context(session, key, "f16", ratio, parallel, baseline_context=BASELINE, top_context=top, step=step)
        if _fits(result):
            return result
    return None


async def _quick(session: DirectSession, model: ModelInfo, parallel: int, notes: list[str]) -> tuple[int | None, int | None, float | None]:
    ratios: list[float | None] = [0.5 if model.is_moe else None]
    if model.is_moe:
        ratios.extend(FALLBACK_RATIOS)
    top = max(BASELINE, min(model.max_context or QUICK_TOP, QUICK_TOP))
    found = await _first_fit(session, model.key, ratios, parallel, top, QUICK_STEP)
    if found is None:
        return None, None, None
    notes.append(f"quick search: f16 cache fits {found.max_context:,} context")
    return found.max_context, None, found.cpu_expert_ratio


async def _best_expert_ratio(session: DirectSession, key: str, tier: TierConfig, ratio: float, notes: list[str]) -> float:
    # Lower ratios would not hold the contexts found at `ratio`, so only the searched ratio and higher are compared.
    candidates = sorted({r for r in EXPERT_SWEEP if r >= ratio} | {ratio})
    sweep = await sweep_experts(session, key, tier, candidates)
    best, best_speed = ratio, 0.0
    for candidate, probe in sweep.probes.items():
        speed = probe.run.generation_tps if probe.run else None
        if probe.fits(None) and speed and speed > best_speed:
            best, best_speed = candidate, speed
    if best_speed:
        notes.append(f"CPU expert ratio {best:.2f} was the fastest that fits ({best_speed:.1f} tok/s)")
    return best


async def _best_parallel(session: DirectSession, key: str, tier: TierConfig, parallel: int, notes: list[str]) -> int:
    sweep = await sweep_slots(session, key, tier, slot_candidates(parallel))
    speeds: dict[int, float] = {}
    for slots in sorted(sweep.probes):
        probe = sweep.probes[slots]
        round_ = probe.round
        if probe.loaded and round_ is not None and not round_.failures and round_.combined_tps and probe.headroom_ok:
            speeds[int(slots)] = round_.combined_tps
        else:
            break
    if speeds:
        notes.append("slots: " + ", ".join(f"{slots} slot(s) {speed:.1f} tok/s combined" for slots, speed in speeds.items()))
    ordered = sorted(speeds)
    if not ordered:
        return parallel
    chosen = ordered[0]
    for smaller, larger in zip(ordered, ordered[1:], strict=False):
        if speeds[larger] < SLOT_GAIN * speeds[smaller]:
            break
        chosen = larger
    return chosen


async def _thorough(session: DirectSession, model: ModelInfo, parallel: int, notes: list[str]) -> tuple[int | None, int | None, float | None, int]:
    top = min(model.max_context or NATIVE_MAX_CONTEXT, NATIVE_MAX_CONTEXT)
    found: list[SearchResult] = []
    for kv, ratio in search_combos(model.is_moe):
        result = await search_context(session, model.key, kv, ratio, parallel, baseline_context=BASELINE, top_context=top)
        if _fits(result):
            found.append(result)
    if not found and model.is_moe:
        fallback = await _first_fit(session, model.key, list(FALLBACK_RATIOS), parallel, top, STEP)
        if fallback is not None:
            found.append(fallback)
    if not found:
        return None, None, None, parallel
    max_f16 = max((r.max_context or 0 for r in found if r.kv_cache == "f16"), default=0) or None
    max_q8 = max((r.max_context or 0 for r in found if r.kv_cache == "q8_0"), default=0) or None
    leader = max(found, key=lambda r: r.max_context or 0)
    ratio = leader.cpu_expert_ratio if model.is_moe else None
    for r in found:
        notes.append(f"{r.kv_cache} cache" + (f", CPU expert ratio {r.cpu_expert_ratio}" if r.cpu_expert_ratio is not None else "") + f": fits {r.max_context:,} context")
    low = tiers_from_max(max_f16, max_q8, ratio)["low"]
    if model.is_moe and ratio is not None:
        ratio = await _best_expert_ratio(session, model.key, low, ratio, notes)
        low = replace(low, cpu_expert_ratio=ratio, try_mmap=False)
    parallel = await _best_parallel(session, model.key, low, parallel, notes)
    return max_f16, max_q8, ratio, parallel


async def _unload_quietly(session: DirectSession) -> None:
    with contextlib.suppress(Exception):
        await asyncio.shield(session.unload_all())


async def calibrate_model(session: DirectSession, model: ModelInfo, hw: Hardware, level: Level, parallel: int) -> CalibrationResult:
    """Find tiers (and for the thorough level the slot count) for one model. Everything is unloaded afterwards."""
    notes: list[str] = []
    try:
        if level == "quick":
            max_f16, max_q8, ratio = await _quick(session, model, parallel, notes)
            chosen = parallel
        else:
            max_f16, max_q8, ratio, chosen = await _thorough(session, model, parallel, notes)
        tiers = tiers_from_max(max_f16, max_q8, ratio)
        if not tiers:
            return CalibrationResult(model.key, {}, parallel, None, None, notes, failed=NOT_LOADED)
        thinks, can_disable = await probe_thinking(session, model.key, tiers["low"])
        return CalibrationResult(model.key, tiers, chosen, thinks, can_disable, notes)
    except (KeyboardInterrupt, asyncio.CancelledError):
        await _unload_quietly(session)
        raise


# --- resume file ------------------------------------------------------------------------------------------------


class Progress:
    """Finished models, kept in a JSON file so an interrupted setup can skip them on the next run."""

    def __init__(self, path: Path, level: str, entries: dict[str, Any] | None = None) -> None:
        self.path = path
        self.level = level
        self._entries: dict[str, Any] = entries or {}

    @classmethod
    def load(cls, path: Path, level: str) -> Progress:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data["results"]
            if not isinstance(entries, dict):
                raise TypeError("results must be an object")
        except (OSError, ValueError, KeyError, TypeError):
            return cls(path, level)
        return cls(path, level, dict(entries))

    def _key(self, model_key: str) -> str:
        return f"{self.level}:{model_key}"

    def done(self, model_key: str) -> CalibrationResult | None:
        entry = self._entries.get(self._key(model_key))
        if entry is None:
            return None
        try:
            return CalibrationResult(
                model_key=str(entry["model_key"]),
                tiers={str(name): TierConfig(**tier) for name, tier in entry["tiers"].items()},
                parallel=int(entry["parallel"]),
                thinks=entry["thinks"],
                can_disable_thinking=entry["can_disable_thinking"],
                notes=[str(note) for note in entry["notes"]],
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return None

    def save(self, result: CalibrationResult) -> None:
        if result.failed is not None:
            return  # a rerun retries failed models
        self._entries[self._key(result.model_key)] = asdict(result)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, "results": self._entries}, stream, indent=2)
            os.replace(temp_name, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp_name)
            raise
