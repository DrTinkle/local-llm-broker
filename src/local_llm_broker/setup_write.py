"""Write config.toml: render it as text, check that it loads, keep a backup of the old file, then replace it.

The text is written by hand because tomllib only reads. Only values that differ from the loader's defaults are written.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_config
from .contracts import TIER_ORDER, ProfileConfig, TierConfig

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_DEFAULT_CONTEXT_CEILING = 8192
_DEFAULT_EVAL_BATCH_SIZE = 2048
_HEADER = (
    "# Written by `llm-broker setup`.",
    "# Edit freely; rerun setup to change profiles.",
)


def _key(name: str) -> str:
    return name if _BARE_KEY.match(name) else json.dumps(name, ensure_ascii=False)


def _value(value: bool | int | float | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _profile_lines(profile: ProfileConfig, tiers: dict[str, dict[str, TierConfig]]) -> list[str]:
    lines = [f"[profiles.{_key(profile.alias)}]", f"model_id = {_value(profile.model_id)}"]
    if not tiers.get(profile.model_id) and profile.context_ceiling != _DEFAULT_CONTEXT_CEILING:
        lines.append(f"context_ceiling = {_value(profile.context_ceiling)}")
    lines.append(f"parallel_ceiling = {_value(profile.parallel_ceiling)}")
    if profile.variant is not None:
        lines.append(f"variant = {_value(profile.variant)}")
    if profile.reasoning_effort is not None:
        lines.append(f"reasoning_effort = {_value(profile.reasoning_effort)}")
    if profile.supports_vision:
        lines.append("supports_vision = true")
    if profile.thinks:
        lines.append("thinks = true")
    if profile.description:
        lines.append(f"description = {_value(profile.description)}")
    return lines


def _tier_lines(model_key: str, tier: TierConfig) -> list[str]:
    lines = [f"[tiers.{_key(model_key)}.{tier.name}]", f"context_length = {_value(tier.context_length)}", f"kv_cache = {_value(tier.kv_cache)}"]
    if not tier.flash_attention:
        lines.append("flash_attention = false")
    if tier.eval_batch_size != _DEFAULT_EVAL_BATCH_SIZE:
        lines.append(f"eval_batch_size = {_value(tier.eval_batch_size)}")
    if tier.cpu_expert_ratio is not None:
        lines.append(f"cpu_expert_ratio = {_value(tier.cpu_expert_ratio)}")
    if tier.try_mmap is not None:
        lines.append(f"try_mmap = {_value(tier.try_mmap)}")
    return lines


def render_config(
    broker: dict[str, Any],
    profiles: dict[str, ProfileConfig],
    tiers: dict[str, dict[str, TierConfig]],
) -> str:
    """TOML text for the given broker settings, profiles and tiers."""
    blocks: list[list[str]] = [list(_HEADER)]
    scalars = [(name, value) for name, value in broker.items() if isinstance(value, (bool, int, float, str))]
    if scalars:
        blocks.append(["[broker]", *(f"{_key(name)} = {_value(value)}" for name, value in scalars)])
    for profile in profiles.values():
        blocks.append(_profile_lines(profile, tiers))
    for model_key, rows in tiers.items():
        for name in TIER_ORDER:
            if name in rows:
                blocks.append(_tier_lines(model_key, rows[name]))
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


def existing_broker_settings(path: Path) -> dict[str, Any]:
    """The plain values of the [broker] table of an existing file; {} when it is missing, unreadable or has none."""
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, ValueError):
        return {}
    table = raw.get("broker")
    if not isinstance(table, dict):
        return {}
    return {name: value for name, value in table.items() if isinstance(value, (bool, int, float, str))}


def _free_backup_name(path: Path, now: datetime) -> Path:
    base = path.with_name(f"{path.name}.bak-{now.strftime('%Y%m%d-%H%M%S')}")
    candidate = base
    number = 0
    while candidate.exists():
        number += 1
        candidate = base.with_name(f"{base.name}-{number}")
    return candidate


def write_config(path: Path, text: str, *, now: datetime) -> Path | None:
    """Check `text` loads, back up the old file, then replace it. Returns the backup path (None without an old file).

    Raises ValueError, leaving the old file untouched, when the text is not a valid config.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(text.encode("utf-8"))
        load_config(temp)
        backup: Path | None = None
        if path.exists():
            backup = _free_backup_name(path, now)
            shutil.copy2(path, backup)
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temp.unlink()
        raise
    return backup


def summarize_changes(old: dict[str, ProfileConfig], new: dict[str, ProfileConfig]) -> list[str]:
    """One line per profile that setup would add, remove or change, sorted by alias."""
    lines: list[str] = []
    for alias in sorted(old.keys() | new.keys()):
        before, after = old.get(alias), new.get(alias)
        if before is None and after is not None:
            lines.append(f"added profile {alias} ({after.model_id})")
        elif after is None:
            lines.append(f"removed profile {alias}")
        elif before != after:
            lines.append(f"changed profile {alias}")
    return lines
