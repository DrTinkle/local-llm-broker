from __future__ import annotations

import hashlib
import os
import re
import secrets
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ is required
    tomllib = None  # type: ignore[assignment]

from .contracts import KV_CACHE_TYPES, TIER_ORDER, BrokerConfig, ProfileConfig, TierConfig

# Profiles and tiers come only from config.toml; the code ships with none.
DEFAULT_PROFILES: dict[str, ProfileConfig] = {}
DEFAULT_TIERS: dict[str, dict[str, TierConfig]] = {}


def _parse_tiers(raw: object, base: dict[str, dict[str, TierConfig]]) -> dict[str, dict[str, TierConfig]]:
    if not isinstance(raw, dict):
        raise ValueError("[tiers] must be a table")
    tiers = {model: dict(rows) for model, rows in base.items()}
    for model_id, rows in raw.items():
        if not isinstance(rows, dict):
            raise ValueError(f'[tiers."{model_id}"] must be a table')
        current = tiers.setdefault(str(model_id), {})
        for name, value in rows.items():
            label = f'tiers."{model_id}".{name}'
            if name not in TIER_ORDER:
                raise ValueError(f"{label}: tier names are {', '.join(TIER_ORDER)}")
            if not isinstance(value, dict):
                raise ValueError(f"[{label}] must be a table")
            unknown = set(value) - {"context_length", "kv_cache", "flash_attention", "eval_batch_size", "cpu_expert_ratio", "try_mmap"}
            if unknown:
                raise ValueError(f"{label}: unknown settings {', '.join(sorted(unknown))}")
            old = current.get(name)
            if old is None and "context_length" not in value:
                raise ValueError(f"{label}.context_length is required for a new tier")
            kv_cache = str(value.get("kv_cache", old.kv_cache if old else "f16"))
            if kv_cache not in KV_CACHE_TYPES:
                raise ValueError(f"{label}.kv_cache must be one of {', '.join(KV_CACHE_TYPES)}")
            ratio = value.get("cpu_expert_ratio", old.cpu_expert_ratio if old else None)
            if ratio is not None:
                ratio = _finite_float(ratio, f"{label}.cpu_expert_ratio", 0.0, 1.0)
            flash = value.get("flash_attention", old.flash_attention if old else True)
            mmap = value.get("try_mmap", old.try_mmap if old else None)
            if not isinstance(flash, bool) or (mmap is not None and not isinstance(mmap, bool)):
                raise ValueError(f"{label}: flash_attention and try_mmap must be true or false")
            if kv_cache != "f16" and not flash:
                raise ValueError(f"{label}: a quantized KV cache needs flash_attention = true")
            current[name] = TierConfig(
                name,
                _finite_int(value.get("context_length", old.context_length if old else 0), f"{label}.context_length", 512, 1_048_576),
                kv_cache,
                flash_attention=flash,
                eval_batch_size=_finite_int(value.get("eval_batch_size", old.eval_batch_size if old else 2048), f"{label}.eval_batch_size", 1, 65_536),
                cpu_expert_ratio=ratio,
                try_mmap=mmap,
            )
    for model_id, rows in tiers.items():
        contexts = [rows[name].context_length for name in TIER_ORDER if name in rows]
        if contexts != sorted(set(contexts)):
            raise ValueError(f'tiers for "{model_id}" must have strictly increasing context_length from low to high')
    return tiers


def _default_data_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    return Path(root) / "LocalLLMBroker"


_HOST_NAME = re.compile(r"^[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?$")


def _parse_allowed_hosts(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("allowed_hosts must be a list of host names")
    names: list[str] = []
    for item in value:
        name = item.strip().lower() if isinstance(item, str) else ""
        if not _HOST_NAME.match(name):
            raise ValueError("allowed_hosts entries must be plain host names, such as host.docker.internal (no scheme, port or wildcard)")
        names.append(name)
    return tuple(names)


def _loopback(url: str, label: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"{label} must use a loopback URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{label} must not contain credentials")


def _finite_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _finite_float(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum:
        raise ValueError(f"{name} must be a number between {minimum} and {maximum}")
    return float(value)


ALIAS_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,39}$")
REASONING_EFFORTS = ("none", "low", "medium", "high")
_PROFILE_KEYS = {
    "model_id", "context_ceiling", "parallel_ceiling", "variant", "reasoning_effort", "supports_vision", "thinks", "description", "kind",
}
PROFILE_KINDS = ("chat", "embedding")


def _parse_profile(alias: str, value: object, tiers: dict[str, dict[str, TierConfig]]) -> ProfileConfig:
    if not ALIAS_PATTERN.match(alias):
        raise ValueError(f"profile alias {alias!r} must match {ALIAS_PATTERN.pattern}")
    if not isinstance(value, dict):
        raise ValueError(f"[profiles.{alias}] must be a table")
    unknown = set(value) - _PROFILE_KEYS
    if unknown:
        raise ValueError(f"profiles.{alias}: unknown settings {', '.join(sorted(unknown))}")
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"profiles.{alias}.model_id is required")
    kind = value.get("kind", "chat")
    if kind not in PROFILE_KINDS:
        raise ValueError(f"profiles.{alias}.kind must be one of {', '.join(PROFILE_KINDS)}")
    if kind == "embedding":
        chat_only = [key for key in ("thinks", "supports_vision", "reasoning_effort", "variant") if key in value]
        if chat_only:
            raise ValueError(f"profiles.{alias}: an embedding profile cannot set {', '.join(chat_only)}")
        if tiers.get(model_id):
            raise ValueError(f"profiles.{alias}: an embedding model has no [tiers]; set context_ceiling on the profile")
    if "context_ceiling" in value and tiers.get(model_id):
        raise ValueError(f"[profiles.{alias}] context_ceiling is replaced by the model's [tiers] context_length values; remove it")
    variant = value.get("variant")
    if variant is not None and not isinstance(variant, str):
        raise ValueError(f"profiles.{alias}.variant must be a string")
    effort = value.get("reasoning_effort")
    if effort is not None and effort not in REASONING_EFFORTS:
        raise ValueError(f"profiles.{alias}.reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
    vision = value.get("supports_vision", False)
    thinks = value.get("thinks", False)
    if not isinstance(vision, bool) or not isinstance(thinks, bool):
        raise ValueError(f"profiles.{alias}: supports_vision and thinks must be true or false")
    description = value.get("description", "")
    if not isinstance(description, str) or len(description) > 200:
        raise ValueError(f"profiles.{alias}.description must be a string of at most 200 characters")
    return ProfileConfig(
        alias,
        model_id,
        _finite_int(value.get("context_ceiling", 8192), f"{alias}.context_ceiling", 512, 1_000_000),
        _finite_int(value.get("parallel_ceiling", 1), f"{alias}.parallel_ceiling", 1, 128),
        variant,
        effort,
        supports_vision=vision,
        thinks=thinks,
        description=description,
        kind=str(kind),
    )


def load_config(path: Path) -> BrokerConfig:
    raw: dict[str, object] = {}
    if path.exists():
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    broker = raw.get("broker", {})
    if not isinstance(broker, dict):
        raise ValueError("[broker] must be a table")
    service_host = str(broker.get("service_host", "127.0.0.1"))
    service_port = _finite_int(broker.get("service_port", 1235), "service_port", 1, 65535)
    _loopback(f"http://{service_host}:{service_port}", "service_host")
    backend_url = str(broker.get("backend_url", "http://127.0.0.1:1234"))
    _loopback(backend_url, "backend_url")
    data_dir = Path(str(broker.get("data_dir", _default_data_dir())))
    allowed_hosts = _parse_allowed_hosts(broker.get("allowed_hosts"))
    lms_path = broker.get("lms_path")
    if lms_path is not None and (not isinstance(lms_path, str) or not lms_path.strip()):
        raise ValueError("lms_path must be a non-empty string")
    profiles: dict[str, ProfileConfig] = dict(DEFAULT_PROFILES)
    raw_profiles_value = raw.get("profiles", {})
    if not isinstance(raw_profiles_value, dict):
        raise ValueError("[profiles] must be a table")
    tiers = _parse_tiers(raw.get("tiers", {}), DEFAULT_TIERS)
    for alias, value in raw_profiles_value.items():
        profiles[alias] = _parse_profile(str(alias), value, tiers)
    # A profile whose model has tiers accepts jobs up to its largest tier.
    for alias, profile in list(profiles.items()):
        rows = tiers.get(profile.model_id)
        if rows:
            ceiling = max(tier.context_length for tier in rows.values())
            profiles[alias] = replace(profile, context_ceiling=ceiling)
    auth_token_path = data_dir / "auth.token"
    return BrokerConfig(
        service_host=service_host,
        service_port=service_port,
        backend_url=backend_url.rstrip("/"),
        data_dir=data_dir,
        queue_limit=_finite_int(broker.get("queue_limit", 100), "queue_limit", 1, 10_000),
        max_body_bytes=_finite_int(broker.get("max_body_bytes", 10 * 1024 * 1024), "max_body_bytes", 1024, 50 * 1024 * 1024),
        max_text_bytes=_finite_int(broker.get("max_text_bytes", 256 * 1024), "max_text_bytes", 1024, 10 * 1024 * 1024),
        default_queue_timeout=_finite_float(broker.get("default_queue_timeout", 3600), "default_queue_timeout", 1, 86_400),
        load_timeout=_finite_float(broker.get("load_timeout", 300), "load_timeout", 1, 3600),
        generation_timeout=_finite_float(broker.get("generation_timeout", 1800), "generation_timeout", 1, 86_400),
        stalled_stream_timeout=_finite_float(broker.get("stalled_stream_timeout", 120), "stalled_stream_timeout", 1, 3600),
        max_stream_buffer_bytes=_finite_int(broker.get("max_stream_buffer_bytes", 1024 * 1024), "max_stream_buffer_bytes", 4096, 50 * 1024 * 1024),
        max_result_bytes=_finite_int(broker.get("max_result_bytes", 2 * 1024 * 1024), "max_result_bytes", 4096, 50 * 1024 * 1024),
        result_ttl_seconds=_finite_int(broker.get("result_ttl_seconds", 600), "result_ttl_seconds", 1, 86_400),
        metadata_ttl_days=_finite_int(broker.get("metadata_ttl_days", 30), "metadata_ttl_days", 1, 3650),
        idle_unload_seconds=_finite_float(broker.get("idle_unload_seconds", 600), "idle_unload_seconds", 0, 86_400),
        profiles=profiles,
        auth_token_path=auth_token_path,
        lms_path=lms_path,
        allowed_hosts=allowed_hosts,
        tiers={model: rows for model, rows in tiers.items() if any(p.model_id == model for p in profiles.values())},
    )


def ensure_auth_token(config: BrokerConfig) -> str:
    path = config.auth_token_path or (config.data_dir / "auth.token")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    token = secrets.token_urlsafe(32)
    path.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return token


def digest_key(config: BrokerConfig) -> bytes:
    path = Path(config.data_dir) / "idempotency.key"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_bytes()
    value = hashlib.sha256(secrets.token_bytes(32)).digest()
    path.write_bytes(value)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return value

