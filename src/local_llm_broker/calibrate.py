"""Measure each load tier on this machine, through the broker.

For every tier: one tiny warm-up request (so the measured request does not include the model load), then one
streamed request with a fixed synthetic prompt. Prompt processing speed comes from time to first token and
generation speed from the tokens after it. A tier that generates much slower than the model's fastest tier has
most likely overflowed VRAM into shared system memory.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from .contracts import BrokerConfig

# A VRAM overflow into shared memory slows prompt processing and generation together (one mid-size model at a very
# large context lost about 60% of prompt speed and 35% of generation speed). A quantized cache slows only
# generation, and on small models, which read the cache in every layer, by up to 40% with a 15K-token prompt; that
# is a slower tier, not an overflow.
OVERFLOW_RATIO = 0.85


def default_profiles(config: BrokerConfig) -> list[str]:
    """One profile per distinct model, in config order, preferring one that does not think.

    A model whose profiles all think is left out when any other profile does not think, since the default is
    non-thinking grunt-work profiles; it is used only when the config has nothing else.
    """
    by_model: dict[str, list[str]] = {}
    for alias, profile in config.profiles.items():
        if profile.kind == "chat":
            by_model.setdefault(profile.model_id, []).append(alias)
    quick = [
        next((alias for alias in aliases if not config.profiles[alias].thinks), None) for aliases in by_model.values()
    ]
    chosen = [alias for alias in quick if alias is not None]
    if chosen:
        return chosen
    return [aliases[0] for aliases in by_model.values()]


def vision_profiles(config: BrokerConfig) -> list[str]:
    """One profile per distinct model among those that accept images, preferring one that does not think."""
    by_model: dict[str, list[str]] = {}
    for alias, profile in config.profiles.items():
        if profile.supports_vision and profile.kind == "chat":
            by_model.setdefault(profile.model_id, []).append(alias)
    return [
        next((alias for alias in aliases if not config.profiles[alias].thinks), aliases[0]) for aliases in by_model.values()
    ]


def resolve_models(config: BrokerConfig, names: Iterable[str] | None) -> list[str]:
    """Model IDs for direct calibration. Each name is a profile alias or a model ID; none means every configured model."""
    if names is None:
        names = [profile.model_id for profile in config.profiles.values() if profile.kind == "chat"]
    resolved: list[str] = []
    for name in names:
        profile = config.profiles.get(name)
        model_id = profile.model_id if profile else name
        if model_id not in resolved:
            resolved.append(model_id)
    return resolved


@dataclass(slots=True)
class TierMeasurement:
    profile: str
    tier: str
    context_length: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    time_to_first_token: float | None
    generation_seconds: float | None
    error: str | None = None

    @property
    def prefill_tps(self) -> float | None:
        if self.prompt_tokens and self.time_to_first_token:
            return self.prompt_tokens / self.time_to_first_token
        return None

    @property
    def generation_tps(self) -> float | None:
        if self.completion_tokens and self.completion_tokens > 1 and self.generation_seconds:
            return (self.completion_tokens - 1) / self.generation_seconds
        return None


_WORDS = (
    "the crew finished framing on the north wall before lunch and moved the scaffold to the east side where the "
    "inspector asked for a second look at the anchor bolts because the drawings showed a different spacing than the "
    "site plan so the foreman called the engineer who confirmed the revised detail and the work continued through "
    "the afternoon while the delivery of roof trusses was pushed to the following morning due to wind"
).split()


def synthetic_prompt(target_tokens: int) -> str:
    """Deterministic English prose, about 4 bytes per token like the broker's own estimate.

    Number-heavy text tokenizes at about 2 bytes per token, so it would be twice the intended length.
    """
    words: list[str] = []
    size = 0
    index = 0
    while size < target_tokens * 4:
        word = _WORDS[(index * 7 + index // len(_WORDS)) % len(_WORDS)] + ("." if index % 17 == 16 else "")
        words.append(word)
        size += len(word) + 1
        index += 1
    return " ".join(words)


def _sse_payloads(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if isinstance(payload, dict):
            yield payload


def _has_output(payload: dict[str, Any]) -> bool:
    for choice in payload.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
            return True
    return False


def measure_stream(lines: Iterable[str], started: float, clock: Any = time.monotonic) -> tuple[float | None, float | None, dict[str, Any]]:
    """(time to first token, seconds after it, usage) from a chat-completions SSE stream."""
    first: float | None = None
    usage: dict[str, Any] = {}
    for payload in _sse_payloads(lines):
        if first is None and _has_output(payload):
            first = clock()
        if isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
    end = clock()
    if first is None:
        return None, None, usage
    return first - started, end - first, usage


def _request(client: httpx.Client, profile: str, tier: str, prompt: str, max_tokens: int, stream: bool) -> httpx.Request:
    body: dict[str, Any] = {
        "model": profile,
        "tier": tier,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return client.build_request("POST", "/v1/chat/completions", json=body)


def run_calibration(
    client: httpx.Client,
    models: dict[str, Any],
    profiles: Iterable[str],
    tiers: Iterable[str],
    *,
    prompt_tokens: int,
    max_tokens: int,
) -> list[TierMeasurement]:
    prompt = synthetic_prompt(prompt_tokens) + "\n\nIn about 150 words, describe the patterns in these records."
    by_id = {item.get("id"): item for item in models.get("data", []) if isinstance(item, dict)}
    results: list[TierMeasurement] = []
    for profile in profiles:
        available = (by_id.get(profile) or {}).get("tiers") or {}
        for tier in tiers:
            if tier not in available:
                continue
            context = available[tier].get("context_length")
            try:
                warm = client.send(_request(client, profile, tier, "Reply with OK.", 8, False))
                if warm.status_code >= 400:
                    raise RuntimeError(_error_text(warm))
                started = time.monotonic()
                response = client.send(_request(client, profile, tier, prompt, max_tokens, True), stream=True)
                try:
                    if response.status_code >= 400:
                        response.read()
                        raise RuntimeError(_error_text(response))
                    ttft, gen_seconds, usage = measure_stream(response.iter_lines(), started)
                finally:
                    response.close()
                results.append(
                    TierMeasurement(
                        profile, tier, context, usage.get("prompt_tokens"), usage.get("completion_tokens"), ttft, gen_seconds
                    )
                )
            except Exception as exc:  # one failed tier must not stop the rest
                results.append(TierMeasurement(profile, tier, context, None, None, None, None, error=str(exc)[:200]))
    return results


def _error_text(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        return f"{error.get('code', response.status_code)}: {error.get('message', '')}"
    except Exception:
        return f"HTTP {response.status_code}"


def verdicts(results: list[TierMeasurement]) -> dict[tuple[str, str], str]:
    """Per (profile, tier): "ok", "slower generation", "likely VRAM overflow", "failed" or "no data"."""
    best: dict[str, float] = {}
    best_prefill: dict[str, float] = {}
    for item in results:
        if item.generation_tps:
            best[item.profile] = max(best.get(item.profile, 0.0), item.generation_tps)
        if item.prefill_tps:
            best_prefill[item.profile] = max(best_prefill.get(item.profile, 0.0), item.prefill_tps)
    out: dict[tuple[str, str], str] = {}
    for item in results:
        if item.error:
            out[(item.profile, item.tier)] = "failed"
        elif not item.generation_tps:
            out[(item.profile, item.tier)] = "no data"
        elif item.generation_tps < OVERFLOW_RATIO * best[item.profile]:
            prefill_dropped = bool(item.prefill_tps) and item.prefill_tps < OVERFLOW_RATIO * best_prefill.get(item.profile, 0.0)  # type: ignore[operator]
            out[(item.profile, item.tier)] = "likely VRAM overflow" if prefill_dropped else "slower generation"
        else:
            out[(item.profile, item.tier)] = "ok"
    return out


def format_table(results: list[TierMeasurement]) -> str:
    marks = verdicts(results)
    rows = ["profile      tier    context   prompt tok/s   gen tok/s   verdict"]
    for item in results:
        prefill = f"{item.prefill_tps:,.0f}" if item.prefill_tps else "-"
        gen = f"{item.generation_tps:.1f}" if item.generation_tps else "-"
        verdict = marks[(item.profile, item.tier)] + (f" ({item.error})" if item.error else "")
        rows.append(f"{item.profile:<12} {item.tier:<7} {item.context_length or '-':>8}   {prefill:>12}   {gen:>9}   {verdict}")
    if any(value == "slower generation" for value in marks.values()):
        rows.append("")
        rows.append(
            "'slower generation' means generation is much slower than the model's fastest tier while prompt processing "
            "is not: usually the cost of a q8_0 cache, not an overflow. The tier works; use kv_cache = \"f16\" with a "
            "smaller context_length if speed matters more than context."
        )
    if any(value == "likely VRAM overflow" for value in marks.values()):
        rows.append("")
        rows.append(
            "A tier marked 'likely VRAM overflow' generates much slower than its model's fastest tier. Lower its "
            "context_length (or raise cpu_expert_ratio on a mixture-of-experts model) in [tiers] and run calibrate again. "
            "Task Manager's 'Shared GPU memory' rising during the load confirms it."
        )
    return "\n".join(rows)
