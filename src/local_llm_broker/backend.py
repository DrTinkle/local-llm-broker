from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


class BackendAdapter(Protocol):
    async def list_models(self) -> list[dict[str, Any]]: ...

    # Called with only the profile when it has no tiers, so adapters without tiers keep the one-argument form.
    async def ensure_loaded(self, profile: str, *, tier: str | None = None, accept_higher: bool = False) -> LoadedModel: ...

    async def unload(self, instance_id: str) -> None: ...

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]: ...


@dataclass(frozen=True, slots=True)
class LoadedModel:
    profile: str
    model_id: str
    instance_id: str
    context_limit: int
    parallel_limit: int
    load_seconds: float | None = None
    load_config: dict[str, Any] = field(default_factory=dict)
    tier: str | None = None

