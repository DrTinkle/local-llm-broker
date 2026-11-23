from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class JobState(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCELLING = "cancelling"
    INTERRUPTED = "interrupted"


TERMINAL_STATES = frozenset(
    {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.INTERRUPTED}
)


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    alias: str
    model_id: str
    context_ceiling: int
    parallel_ceiling: int
    variant: str | None = None
    reasoning_effort: str | None = None
    supports_vision: bool = False
    thinks: bool = False
    description: str = ""


# Load tiers, from fastest to largest context. A job names one, or the broker picks the smallest that fits.
TIER_ORDER = ("low", "medium", "high")
KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")


@dataclass(frozen=True, slots=True)
class TierConfig:
    """Load values the broker sends to LM Studio for one tier of one model."""

    name: str
    context_length: int
    kv_cache: str = "f16"
    flash_attention: bool = True
    eval_batch_size: int = 2048
    # Share of MoE expert layers kept in system RAM (LM Studio's numCpuExpertLayersRatio). None leaves it unset.
    cpu_expert_ratio: float | None = None
    # None leaves LM Studio's saved choice. Off is faster when expert layers are on the CPU.
    try_mmap: bool | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "context_length": self.context_length,
            "kv_cache": self.kv_cache,
            "cpu_expert_ratio": self.cpu_expert_ratio,
            "try_mmap": self.try_mmap,
        }


def tier_rank(name: str | None) -> int:
    return TIER_ORDER.index(name) if name in TIER_ORDER else -1


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    service_host: str
    service_port: int
    backend_url: str
    data_dir: Path
    queue_limit: int = 100
    max_body_bytes: int = 10 * 1024 * 1024
    max_text_bytes: int = 256 * 1024
    default_queue_timeout: float = 3600.0
    load_timeout: float = 300.0
    generation_timeout: float = 1800.0
    stalled_stream_timeout: float = 120.0
    max_stream_buffer_bytes: int = 1024 * 1024
    max_result_bytes: int = 2 * 1024 * 1024
    result_ttl_seconds: int = 600
    metadata_ttl_days: int = 30
    # Unload the loaded model after this many seconds with no running or queued jobs. 0 keeps it loaded.
    idle_unload_seconds: float = 600.0
    profiles: Mapping[str, ProfileConfig] = field(default_factory=dict)
    auth_token_path: Path | None = None
    # Full path to LM Studio's `lms` tool, for machines where it is not on PATH.
    lms_path: str | None = None
    # Keyed by LM Studio model ID, then tier name. Profiles that share a model share its tiers.
    tiers: Mapping[str, Mapping[str, TierConfig]] = field(default_factory=dict)

    def tiers_for(self, profile: str) -> Mapping[str, TierConfig]:
        return self.tiers.get(self.profiles[profile].model_id, {})

    @property
    def service_url(self) -> str:
        return f"http://{self.service_host}:{self.service_port}"


@dataclass(frozen=True, slots=True)
class JobSpec:
    client_id: str
    project: str
    task_label: str
    profile: str
    messages: list[dict[str, Any]]
    generation: dict[str, Any]
    queue_timeout_seconds: float | None = None
    # Requested tier; None lets the broker choose.
    tier: str | None = None

    def payload_digest(self, key: bytes | None = None) -> str:
        fields: dict[str, Any] = {
            "client_id": self.client_id,
            "project": self.project,
            "task_label": self.task_label,
            "profile": self.profile,
            "messages": self.messages,
            "generation": self.generation,
            "queue_timeout_seconds": self.queue_timeout_seconds,
        }
        # Only a requested tier is part of the digest, so digests of jobs without one are unchanged.
        if self.tier is not None:
            fields["tier"] = self.tier
        body = json.dumps(
            fields,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if key is None:
            return hashlib.sha256(body).hexdigest()
        import hmac

        return hmac.new(key, body, hashlib.sha256).hexdigest()


@dataclass(slots=True)
class JobSnapshot:
    job_id: str
    client_id: str
    project: str
    task_label: str
    profile: str
    state: JobState
    created_at: datetime
    updated_at: datetime
    backend_state: str = "idle"
    result_expired: bool = False
    queue_seconds: float | None = None
    load_seconds: float | None = None
    inference_seconds: float | None = None
    time_to_first_token: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    error: dict[str, Any] | None = None
    request_id: str = field(default_factory=lambda: secrets.token_hex(8))

    def public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "client_id": self.client_id,
            "project": self.project,
            "task_label": self.task_label,
            "profile": self.profile,
            "state": self.state.value,
            "backend_state": self.backend_state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result_expired": self.result_expired,
            "timings": {
                "queue_seconds": self.queue_seconds,
                "load_seconds": self.load_seconds,
                "inference_seconds": self.inference_seconds,
                "time_to_first_token": self.time_to_first_token,
            },
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "finish_reason": self.finish_reason,
            "error": self.error,
            "request_id": self.request_id,
        }


@dataclass(frozen=True, slots=True)
class BrokerError(Exception):
    code: str
    message: str
    retryable: bool = False
    status_code: int = 400
    request_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "request_id": self.request_id,
        }


def utc_now() -> datetime:
    return datetime.now(UTC)

