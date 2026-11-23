"""Profile and tier suggestions that need no model load: aliases, thinking guesses, parallel slots, safe tiers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from local_llm_broker.contracts import ProfileConfig, TierConfig
from local_llm_broker.setup_discover import Hardware, ModelInfo

_ALIAS_MAX = 40
_BAD_ALIAS_CHARS = re.compile(r"[^a-z0-9._-]")
_NOTHINK_SUFFIX = "-nothink"
_THINKING_MARKERS = ("thinking", "-r1", "qwq", "reasoning")
_FIT_RATIO = 0.7


def suggest_alias(model: ModelInfo, taken: set[str]) -> str:
    name = model.key.rsplit("/", 1)[-1].lower()
    name = _BAD_ALIAS_CHARS.sub("-", name)
    name = re.sub(r"^[^a-z0-9]+", "", name)
    base = (name or "model")[:_ALIAS_MAX]
    if base not in taken:
        return base
    number = 2
    while True:
        suffix = f"-{number}"
        candidate = base[: _ALIAS_MAX - len(suffix)] + suffix
        if candidate not in taken:
            return candidate
        number += 1


def guess_thinks(model: ModelInfo) -> bool:
    key = model.key.lower()
    return any(marker in key for marker in _THINKING_MARKERS)


def _size_ratio(model: ModelInfo, hw: Hardware) -> float | None:
    if not hw.vram_mib or hw.vram_mib <= 0:
        return None
    return model.size_bytes / (hw.vram_mib * 1024 * 1024)


def default_parallel(model: ModelInfo, hw: Hardware) -> int:
    ratio = _size_ratio(model, hw)
    if ratio is None:
        return 1
    if ratio < 0.3:
        return 4
    if ratio < 0.7:
        return 2
    return 1


def quick_tiers(model: ModelInfo, hw: Hardware) -> dict[str, TierConfig]:
    """Conservative tiers that need no model load."""
    ratio = _size_ratio(model, hw)
    fits = ratio is not None and ratio < _FIT_RATIO
    limit = model.max_context if model.max_context > 0 else None

    def capped(context: int) -> int:
        return context if limit is None else min(context, limit)

    expert_ratio = 0.5 if model.is_moe else None
    mmap = False if model.is_moe else None

    def tier(name: str, context: int) -> TierConfig:
        return TierConfig(name, context, "f16", cpu_expert_ratio=expert_ratio, try_mmap=mmap)

    if not fits:
        return {"low": tier("low", capped(4096))}
    low = capped(8192)
    medium = capped(16384)
    tiers = {"low": tier("low", low)}
    if medium > low:
        tiers["medium"] = tier("medium", medium)
    return tiers


@dataclass(slots=True)
class ProfileChoice:
    alias: str
    model: ModelInfo
    thinks: bool
    twin: bool
    parallel: int
    description: str = ""


def _twin_alias(alias: str) -> str:
    return alias[: _ALIAS_MAX - len(_NOTHINK_SUFFIX)] + _NOTHINK_SUFFIX if len(alias) + len(_NOTHINK_SUFFIX) > _ALIAS_MAX else alias + _NOTHINK_SUFFIX


def profiles_from_choices(choices: list[ProfileChoice]) -> dict[str, ProfileConfig]:
    profiles: dict[str, ProfileConfig] = {}

    def add(profile: ProfileConfig) -> None:
        if profile.alias in profiles:
            raise ValueError(f"profile alias {profile.alias!r} is used more than once")
        profiles[profile.alias] = profile

    for choice in choices:
        model = choice.model
        add(
            ProfileConfig(
                choice.alias,
                model.key,
                context_ceiling=8192,
                parallel_ceiling=choice.parallel,
                variant=model.variant,
                reasoning_effort=None,
                supports_vision=model.vision,
                thinks=choice.thinks,
                description=choice.description,
            )
        )
        if choice.twin and choice.thinks:
            add(
                ProfileConfig(
                    _twin_alias(choice.alias),
                    model.key,
                    context_ceiling=8192,
                    parallel_ceiling=choice.parallel,
                    variant=model.variant,
                    reasoning_effort="none",
                    supports_vision=model.vision,
                    thinks=False,
                    description=f"Same model as {choice.alias} with thinking off.",
                )
            )
    return profiles


def tiers_for_choices(choices: list[ProfileChoice], hw: Hardware) -> dict[str, dict[str, TierConfig]]:
    tiers: dict[str, dict[str, TierConfig]] = {}
    for choice in choices:
        if choice.model.key not in tiers:
            tiers[choice.model.key] = quick_tiers(choice.model, hw)
    return tiers
