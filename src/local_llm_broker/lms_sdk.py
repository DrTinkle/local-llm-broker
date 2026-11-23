"""Tier loads through LM Studio's Python SDK.

LM Studio's REST load call accepts only context length and a few flags. The SDK's load channel takes a full
load-config stack, the same `llm.load.*` keys LM Studio writes to its saved per-model presets. The SDK's typed
config (lmstudio 1.5.0) has no field for the CPU expert ratio or parallel slots, so those two are added to the
stack as raw keys. This module relies on SDK internals and is pinned to lmstudio 1.5.0 in pyproject.toml.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urlparse

from .contracts import TierConfig

CPU_EXPERT_RATIO_KEY = "llm.load.numCpuExpertLayersRatio"
PARALLEL_KEY = "llm.load.numParallelSessions"


def sdk_load_config(tier: TierConfig) -> dict[str, Any]:
    """Typed SDK load config for one tier (snake_case keys of lmstudio.LlmLoadModelConfig)."""
    # The Python SDK always sends these as checked, so f16 is sent as the explicit type: the same as LM Studio's default.
    cache = tier.kv_cache
    config: dict[str, Any] = {
        "context_length": tier.context_length,
        "gpu": {"ratio": 1.0},
        "offload_kv_cache_to_gpu": True,
        "flash_attention": tier.flash_attention,
        "eval_batch_size": tier.eval_batch_size,
        "llama_k_cache_quantization_type": cache,
        "llama_v_cache_quantization_type": cache,
    }
    if tier.try_mmap is not None:
        config["try_mmap"] = tier.try_mmap
    return config


def raw_load_fields(tier: TierConfig, parallel: int) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = [{"key": PARALLEL_KEY, "value": parallel}]
    if tier.cpu_expert_ratio is not None:
        fields.append({"key": CPU_EXPERT_RATIO_KEY, "value": tier.cpu_expert_ratio})
    return fields


def load_config_stack(tier: TierConfig, parallel: int) -> dict[str, Any]:
    """The loadConfigStack the broker sends: the SDK's own conversion plus the raw fields."""
    from lmstudio import LlmLoadModelConfig
    from lmstudio._kv_config import load_config_to_kv_config_stack

    stack: dict[str, Any] = dict(load_config_to_kv_config_stack(sdk_load_config(tier), LlmLoadModelConfig).to_dict())
    fields: list[dict[str, Any]] = stack["layers"][0]["config"]["fields"]
    keys = {field["key"] for field in fields}
    fields.extend(field for field in raw_load_fields(tier, parallel) if field["key"] not in keys)
    return stack


def api_host(backend_url: str) -> str:
    parsed = urlparse(backend_url)
    return f"{parsed.hostname}:{parsed.port or 1234}"


async def sdk_load(
    backend_url: str, model_key: str, tier: TierConfig, parallel: int, *, ttl: int | None = None
) -> tuple[str, float]:
    """Load model_key with the tier's values. Returns (instance identifier, load seconds).

    ttl is LM Studio's own idle timeout in seconds for this instance; None loads it without one.
    """
    from lmstudio import AsyncClient, LlmLoadModelConfig
    from lmstudio.json_api import LoadModelEndpoint

    stack = load_config_stack(tier, parallel)

    class _TierLoadEndpoint(LoadModelEndpoint):  # type: ignore[type-arg]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._creation_params["loadConfigStack"] = stack  # type: ignore[arg-type]

    started = time.monotonic()
    async with AsyncClient(api_host(backend_url)) as client:
        session = client.llm
        endpoint = _TierLoadEndpoint(
            model_key,
            None,
            ttl,
            session._API_TYPES.REQUEST_NEW_INSTANCE,
            LlmLoadModelConfig,
            None,
            None,
        )
        async with session._create_channel(endpoint) as channel:
            result = await channel.wait_for_result()
    return str(result.identifier), time.monotonic() - started


class SdkTokenCounter:
    """Exact token counts through one long-lived SDK connection.

    Opening an SDK client costs 0.5 to 1 second (measured 2026-11-04), which staggered parallel jobs when every count
    opened its own. A background task keeps one client open and answers counts in order; if the connection fails,
    the task ends and the next count opens a new one.
    """

    def __init__(self, backend_url: str) -> None:
        self.backend_url = backend_url
        self._requests: asyncio.Queue[tuple[str, str, asyncio.Future[int]]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def count(self, identifier: str, text: str) -> int:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._serve(), name="lmstudio-token-counter")
        future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        await self._requests.put((identifier, text, future))
        return await future

    async def _serve(self) -> None:
        from lmstudio import AsyncClient

        try:
            async with AsyncClient(api_host(self.backend_url)) as client:
                while True:
                    identifier, text, future = await self._requests.get()
                    if future.done():
                        continue  # the caller gave up
                    try:
                        future.set_result(int(await client.llm._count_tokens(identifier, text)))
                    except Exception as exc:
                        future.set_exception(exc)
                        return  # reconnect on the next count
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Could not connect: fail what is waiting so callers fall back to the estimate.
            while not self._requests.empty():
                *_, future = self._requests.get_nowait()
                if not future.done():
                    future.set_exception(exc)

    async def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
