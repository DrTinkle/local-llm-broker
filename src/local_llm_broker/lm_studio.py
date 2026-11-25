from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

import httpx

from .backend import LoadedModel
from .contracts import ProfileConfig, TierConfig, tier_rank
from .lms_cli import find_lms
from .lms_sdk import SdkTokenCounter, sdk_load

TierLoader = Callable[[str, TierConfig, int], Awaitable[tuple[str, float]]]
TokenCounter = Callable[[str, str], Awaitable[int]]

# LM Studio's own idle timeout on tier loads, as a backup for the broker's idle unload: the broker unloads
# first, and LM Studio unloads the model itself if the broker has stopped.
_LMS_TTL_MARGIN_SECONDS = 60


class BackendFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


# LM Studio reports loaded contexts slightly above the request (16000 -> 16128, 28000 -> 28160).
_CONTEXT_ROUNDING_SLACK = 1024


def _actual_caps(profile_config: ProfileConfig, config: Mapping[str, Any], ceiling: int | None = None) -> tuple[int, int]:
    ceiling = profile_config.context_ceiling if ceiling is None else ceiling
    context = _positive_int(config.get("context_length"))
    parallel = _positive_int(config.get("parallel"))
    if context is None or parallel is None:
        raise BackendFailure(
            "caps_unverified",
            "LM Studio did not report the loaded context and parallel caps; the instance is unloaded and loaded again",
            retryable=True,
        )
    if context > ceiling + _CONTEXT_ROUNDING_SLACK:
        raise BackendFailure(
            "context_above_preset",
            "LM Studio loaded a context above the configured preset; the instance is unloaded and loaded again at the preset",
            retryable=True,
        )
    return min(ceiling, context), min(profile_config.parallel_ceiling, parallel)


def tier_of_instance(tiers: Mapping[str, TierConfig], config: Any) -> str | None:
    """The tier whose context a loaded instance has, read from LM Studio's listing. None if it matches no tier."""
    context = _positive_int(config.get("context_length")) if isinstance(config, dict) else None
    if context is None:
        return None
    for name, tier in tiers.items():
        if tier.context_length <= context <= tier.context_length + _CONTEXT_ROUNDING_SLACK:
            return name
    return None


def _parse_lms_status(stdout: str, instance_id: str) -> dict[str, Any] | None:
    try:
        rows = json.loads(stdout)
    except ValueError as exc:
        raise BackendFailure("backend_status_unknown", "LM Studio status output was not readable", retryable=True) from exc
    if not isinstance(rows, list):
        raise BackendFailure("backend_status_unknown", "LM Studio status output was not a list", retryable=True)
    for row in rows:
        if isinstance(row, dict) and row.get("identifier") == instance_id:
            return {"status": row.get("status"), "queued": row.get("queued")}
    return None


async def _lms_status(instance_id: str, lms_path: str | None = None) -> dict[str, Any] | None:
    # `lms ps --json` is the only source that reports generating/queued state; the REST listing does not.
    executable = find_lms(lms_path)
    if executable is None:
        raise BackendFailure(
            "backend_status_unknown",
            "the LM Studio CLI was not found; install LM Studio's lms tool or set lms_path in config.toml",
            retryable=True,
        )
    try:
        process = await asyncio.create_subprocess_exec(
            executable, "ps", "--json", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
    except OSError as exc:
        raise BackendFailure("backend_status_unknown", "the LM Studio CLI could not be started", retryable=True) from exc
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    except TimeoutError as exc:
        process.kill()
        raise BackendFailure("backend_status_unknown", "the LM Studio CLI did not answer", retryable=True) from exc
    if process.returncode != 0:
        raise BackendFailure("backend_status_unknown", "the LM Studio CLI reported an error", retryable=True)
    return _parse_lms_status(stdout.decode("utf-8", errors="replace"), instance_id)


class LMStudioAdapter:
    def __init__(
        self,
        base_url: str,
        profiles: Mapping[str, ProfileConfig],
        *,
        client: httpx.AsyncClient | None = None,
        request_timeout: float = 30.0,
        load_timeout: float = 300.0,
        generation_timeout: float = 1800.0,
        status_reader: Callable[[str], Awaitable[dict[str, Any] | None]] | None = None,
        busy_wait_timeout: float = 120.0,
        busy_poll_interval: float = 2.0,
        post_load_attempts: int = 3,
        post_load_retry_seconds: float = 1.0,
        tiers: Mapping[str, Mapping[str, TierConfig]] | None = None,
        tier_loader: TierLoader | None = None,
        idle_unload_seconds: float = 0.0,
        token_counter: TokenCounter | None = None,
        health_ttl_seconds: float = 30.0,
        lms_path: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.profiles = dict(profiles)
        self.request_timeout = request_timeout
        self.load_timeout = load_timeout
        self.generation_timeout = generation_timeout
        self._status_reader = status_reader or (lambda instance_id: _lms_status(instance_id, lms_path))
        self.busy_wait_timeout = busy_wait_timeout
        self.busy_poll_interval = busy_poll_interval
        self.post_load_attempts = post_load_attempts
        self.post_load_retry_seconds = post_load_retry_seconds
        self.client = client or httpx.AsyncClient(base_url=self.base_url, timeout=request_timeout)
        self._owns_client = client is None
        self._lifecycle_lock = asyncio.Lock()
        # health() reuses the newest model listing (or failed check) for this long, so status polls reach
        # LM Studio at most once per period however many clients poll. Lifecycle steps always list afresh,
        # and every listing they make refreshes the cache. Loads and unloads clear it.
        self.health_ttl_seconds = health_ttl_seconds
        self._listing: tuple[float, list[dict[str, Any]]] | None = None
        self._failed_health: tuple[float, dict[str, Any]] | None = None
        self._health_lock = asyncio.Lock()
        # Keyed by model ID. Profiles of a model with tiers load through the SDK; others keep the REST load.
        self.tiers = {model: dict(rows) for model, rows in (tiers or {}).items()}
        # Embedding models stay loaded beside the chat model. Chat loads and checks leave them alone.
        self._embedding_models = {p.model_id for p in self.profiles.values() if p.kind == "embedding"}
        self._resident: dict[str, str] = {}
        self._resident_lock = asyncio.Lock()
        self._sdk_counter: SdkTokenCounter | None = None
        if token_counter is None:
            self._sdk_counter = SdkTokenCounter(self.base_url)
            token_counter = self._sdk_counter.count
        self._token_counter = token_counter
        ttl = int(idle_unload_seconds) + _LMS_TTL_MARGIN_SECONDS if idle_unload_seconds > 0 else None
        self._tier_loader = tier_loader or (
            lambda model_key, tier, parallel: sdk_load(self.base_url, model_key, tier, parallel, ttl=ttl)
        )

    async def close(self) -> None:
        if self._sdk_counter is not None:
            await self._sdk_counter.close()
        if self._owns_client:
            await self.client.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self._request("GET", "/api/v1/models")
        payload = response.json()
        models = payload.get("models")
        if not isinstance(models, list):
            raise BackendFailure("invalid_backend_response", "LM Studio returned no model list", retryable=True)
        listed = [item for item in models if isinstance(item, dict)]
        self._listing = (time.monotonic(), listed)
        self._failed_health = None
        return listed

    def invalidate_health(self) -> None:
        """Forget the cached listing, so the next health() asks LM Studio. Called after loads and unloads."""
        self._listing = None
        self._failed_health = None

    async def inspect_instances(self) -> list[dict[str, Any]]:
        models = await self.list_models()
        instances: list[dict[str, Any]] = []
        for model in models:
            key = model.get("key")
            for instance in model.get("loaded_instances", []) or []:
                if isinstance(instance, dict):
                    instances.append({"model_id": key, **instance})
        return instances

    async def ensure_loaded(self, profile: str, *, tier: str | None = None, accept_higher: bool = False) -> LoadedModel:
        if profile not in self.profiles:
            raise BackendFailure("unknown_profile", f"unknown profile: {profile}")
        profile_config = self.profiles[profile]
        if profile_config.kind == "embedding":
            raise BackendFailure("wrong_profile_kind", f"profile '{profile}' is an embedding profile, not a chat profile")
        model_tiers = self.tiers.get(profile_config.model_id, {})
        tier_config: TierConfig | None = None
        if tier is not None:
            tier_config = model_tiers.get(tier)
            if tier_config is None:
                raise BackendFailure("unknown_tier", f"profile '{profile}' has no tier '{tier}'")
        async with self._lifecycle_lock:
            # Check the request before anything is unloaded, so a bad profile never leaves LM Studio with nothing loaded.
            models = await self.list_models()
            matching = self._model_entry(models, profile_config.model_id)
            if matching is None:
                raise BackendFailure("model_unavailable", f"configured model is not installed: {profile_config.model_id}")
            if profile_config.variant:
                variants = matching.get("variants")
                if not isinstance(variants, list) or profile_config.variant not in variants:
                    raise BackendFailure("variant_unavailable", f"configured model variant is not installed: {profile_config.variant}")
            # LM Studio loads the selected variant by model key; a variant-suffixed name is rejected with model_not_found.
            self._require_selected_variant(profile_config, matching)
            # Only one model may be loaded at a time. Everything else goes first, including models loaded by hand.
            others = [item for item in self._chat_instances(models) if item["model_id"] != profile_config.model_id]
            if others:
                for item in others:
                    await self._unload_when_idle(str(item.get("id", "")))
                models = await self.list_models()
                if any(item["model_id"] != profile_config.model_id for item in self._chat_instances(models)):
                    raise BackendFailure(
                        "unload_unverified",
                        "a previous model did not unload; the switch is retried on the next job",
                        retryable=True,
                    )
            existing = [item for item in self._loaded_instances(models) if item["model_id"] == profile_config.model_id]
            if existing:
                # The target is already loaded. Keep its first instance and remove any duplicates.
                for item in existing[1:]:
                    await self._unload_when_idle(str(item.get("id", "")))
                if tier_config is None:
                    return await self._adopt(profile, profile_config, str(existing[0].get("id")), existing[0].get("config"))
                loaded_tier = tier_of_instance(model_tiers, existing[0].get("config"))
                if loaded_tier == tier or (accept_higher and tier_rank(loaded_tier) > tier_rank(tier)):
                    assert loaded_tier is not None
                    return await self._adopt(
                        profile, profile_config, str(existing[0].get("id")), existing[0].get("config"),
                        tier=loaded_tier, ceiling=model_tiers[loaded_tier].context_length,
                    )
                # Loaded at another tier, or by hand at a context no tier uses: reload at the requested tier.
                await self._unload_when_idle(str(existing[0].get("id", "")))
            if tier_config is not None:
                assert tier is not None
                return await self._load_tier(profile, profile_config, tier, tier_config)
            try:
                response = await self._request(
                    "POST",
                    "/api/v1/models/load",
                    json={"model": profile_config.model_id, "context_length": profile_config.context_ceiling, "echo_load_config": True},
                    timeout=self.load_timeout,
                )
            except (TimeoutError, httpx.TimeoutException) as exc:
                # LM Studio may still finish this load. The next job lists instances again and adopts it if it is there.
                raise BackendFailure("load_in_flight", "LM Studio load timed out; the next job checks LM Studio again", retryable=True) from exc
            except BackendFailure as exc:
                raise BackendFailure("model_load_failed", "LM Studio rejected the configured model load", retryable=True) from exc
            finally:
                self.invalidate_health()
            payload = response.json()
            instance_id = payload.get("instance_id")
            if not isinstance(instance_id, str):
                raise BackendFailure("invalid_backend_response", "LM Studio load returned no instance id", retryable=True)
            listed_config = await self._listed_instance_config_after_load(instance_id)
            load_seconds = float(payload["load_time_seconds"]) if isinstance(payload.get("load_time_seconds"), (int, float)) else None
            return await self._adopt(profile, profile_config, instance_id, listed_config, load_seconds=load_seconds)

    async def _load_tier(self, profile: str, profile_config: ProfileConfig, tier: str, tier_config: TierConfig) -> LoadedModel:
        try:
            instance_id, load_seconds = await asyncio.wait_for(
                self._tier_loader(profile_config.model_id, tier_config, profile_config.parallel_ceiling),
                timeout=self.load_timeout,
            )
        except TimeoutError as exc:
            # LM Studio may still finish this load. The next job lists instances again and adopts it if it fits.
            raise BackendFailure("load_in_flight", "LM Studio load timed out; the next job checks LM Studio again", retryable=True) from exc
        except BackendFailure:
            raise
        except Exception as exc:
            raise BackendFailure("model_load_failed", f"LM Studio rejected the '{tier}' tier load", retryable=True) from exc
        finally:
            self.invalidate_health()
        listed_config = await self._listed_instance_config_after_load(instance_id)
        return await self._adopt(
            profile, profile_config, instance_id, listed_config,
            load_seconds=load_seconds, tier=tier, ceiling=tier_config.context_length,
        )

    async def count_tokens(self, loaded: LoadedModel, text: str) -> int:
        """Exact tokens for `text` with the loaded model's tokenizer. Raises BackendFailure if LM Studio cannot say."""
        try:
            return await asyncio.wait_for(self._token_counter(loaded.instance_id, text), timeout=self.request_timeout)
        except TimeoutError as exc:
            raise BackendFailure("token_count_failed", "LM Studio did not count the prompt in time", retryable=True) from exc
        except BackendFailure:
            raise
        except Exception as exc:
            raise BackendFailure("token_count_failed", "LM Studio could not count the prompt", retryable=True) from exc

    async def release(self, loaded: LoadedModel) -> bool:
        """Unload the broker's model after an idle period. True if it is no longer loaded.

        Unloads only the instance the broker loaded or adopted, and only once LM Studio reports it idle.
        """
        async with self._lifecycle_lock:
            listed = [item for item in self._loaded_instances(await self.list_models()) if item.get("id") == loaded.instance_id]
            if not listed:
                return True  # already gone, for example unloaded in LM Studio's window
            await self._unload_when_idle(loaded.instance_id)
            return True

    async def still_loaded_alone(self, loaded: LoadedModel) -> bool:
        # Checked before each batch of jobs. The model the broker last loaded must still be the only one loaded.
        async with self._lifecycle_lock:
            instances = self._chat_instances(await self.list_models())
            return (
                len(instances) == 1
                and instances[0].get("id") == loaded.instance_id
                and instances[0].get("model_id") == loaded.model_id
            )

    async def _adopt(
        self,
        profile: str,
        profile_config: ProfileConfig,
        instance_id: str,
        config: Any,
        *,
        load_seconds: float | None = None,
        tier: str | None = None,
        ceiling: int | None = None,
    ) -> LoadedModel:
        # Caps come only from LM Studio's listing. Nothing has been sent to this instance yet, so an instance
        # that cannot be verified, or that runs above the preset, is unloaded and the job fails retryably.
        try:
            if not isinstance(config, dict):
                raise BackendFailure(
                    "caps_unverified",
                    "LM Studio did not list the loaded instance; the instance is unloaded and loaded again",
                    retryable=True,
                )
            context_limit, parallel_limit = _actual_caps(profile_config, config, ceiling)
        except BackendFailure:
            await self._unload_rejected(instance_id)
            raise
        return LoadedModel(
            profile=profile,
            model_id=profile_config.model_id,
            instance_id=instance_id,
            context_limit=context_limit,
            parallel_limit=parallel_limit,
            load_seconds=load_seconds,
            load_config=dict(config),
            tier=tier,
        )

    async def _unload_when_idle(self, instance_id: str) -> None:
        await self._wait_until_idle(instance_id)
        await self.unload(instance_id)

    async def _unload_rejected(self, instance_id: str) -> None:
        # Best effort. If this unload fails too, the next job lists the instance again and retries the unload.
        try:
            await self.unload(instance_id)
        except BackendFailure:
            pass

    async def _listed_instance_config_after_load(self, instance_id: str) -> dict[str, Any] | None:
        # LM Studio can drop a connection, or not list the new instance yet, right after it reports a load done.
        # Look again a few times before the job is failed.
        for attempt in range(1, self.post_load_attempts + 1):
            last_attempt = attempt == self.post_load_attempts
            try:
                config = await self._listed_instance_config(instance_id)
            except BackendFailure as exc:
                if exc.code != "backend_unavailable" or last_attempt:
                    raise
            else:
                if config is not None or last_attempt:
                    return config
            await asyncio.sleep(self.post_load_retry_seconds)
        return None

    async def _listed_instance_config(self, instance_id: str) -> dict[str, Any] | None:
        models = await self.list_models()
        for item in self._loaded_instances(models):
            if item.get("id") == instance_id:
                config = item.get("config")
                return dict(config) if isinstance(config, dict) else {}
        return None

    async def _wait_until_idle(self, instance_id: str) -> None:
        # Only a reported "idle" with nothing queued allows an unload. Anything else waits, then fails the job retryably.
        deadline = time.monotonic() + self.busy_wait_timeout
        while True:
            status = await self._status_reader(instance_id)
            if status is None:
                raise BackendFailure(
                    "backend_status_unknown",
                    "LM Studio does not report the model to switch from; switch not started",
                    retryable=True,
                )
            if status.get("status") == "idle" and not status.get("queued"):
                return
            if time.monotonic() >= deadline:
                raise BackendFailure(
                    "backend_busy",
                    "LM Studio is still busy with the loaded model; switch not started",
                    retryable=True,
                )
            await asyncio.sleep(self.busy_poll_interval)

    async def unload(self, instance_id: str) -> None:
        try:
            await self._request("POST", "/api/v1/models/unload", json={"instance_id": instance_id})
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise BackendFailure(
                "unload_in_flight",
                "LM Studio unload timed out; the next job lists the instance again and retries the unload",
                retryable=True,
            ) from exc
        finally:
            self.invalidate_health()

    async def embed(self, profile: str, inputs: list[str]) -> dict[str, Any]:
        """Embeddings for `inputs` from a resident embedding model, loading it first if needed."""
        profile_config = self.profiles.get(profile)
        if profile_config is None:
            raise BackendFailure("unknown_profile", f"unknown profile: {profile}")
        if profile_config.kind != "embedding":
            raise BackendFailure("wrong_profile_kind", f"profile '{profile}' is a chat profile, not an embedding profile")
        for attempt in range(2):
            model_key = await self._ensure_resident(profile, profile_config)
            try:
                response = await self._request(
                    "POST", "/v1/embeddings", json={"model": model_key, "input": inputs}, timeout=self.generation_timeout
                )
            except BackendFailure as exc:
                if attempt == 0 and exc.code == "backend_http_error":
                    # Probably unloaded outside the broker. Forget it, list again, and load it once more.
                    self._resident.pop(profile, None)
                    continue
                raise
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                raise BackendFailure("invalid_backend_response", "LM Studio returned no embeddings", retryable=True)
            return data
        raise BackendFailure("embedding_failed", "LM Studio could not embed the input", retryable=True)

    async def _ensure_resident(self, profile: str, profile_config: ProfileConfig) -> str:
        cached = self._resident.get(profile)
        if cached is not None:
            return cached
        async with self._resident_lock:
            cached = self._resident.get(profile)
            if cached is not None:
                return cached
            models = await self.list_models()
            if self._model_entry(models, profile_config.model_id) is None:
                raise BackendFailure("model_unavailable", f"configured model is not installed: {profile_config.model_id}")
            existing = [item for item in self._loaded_instances(models) if item["model_id"] == profile_config.model_id]
            if existing:
                instance_id = str(existing[0].get("id"))
            else:
                try:
                    response = await self._request(
                        "POST",
                        "/api/v1/models/load",
                        json={"model": profile_config.model_id, "context_length": profile_config.context_ceiling, "echo_load_config": True},
                        timeout=self.load_timeout,
                    )
                except (TimeoutError, httpx.TimeoutException) as exc:
                    raise BackendFailure("load_in_flight", "LM Studio load timed out; the next request checks again", retryable=True) from exc
                except BackendFailure as exc:
                    raise BackendFailure("model_load_failed", "LM Studio rejected the embedding model load", retryable=True) from exc
                finally:
                    self.invalidate_health()
                value = response.json().get("instance_id")
                if not isinstance(value, str):
                    raise BackendFailure("invalid_backend_response", "LM Studio load returned no instance id", retryable=True)
                instance_id = value
            self._resident[profile] = instance_id
            return instance_id

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request("POST", "/v1/chat/completions", json=payload, timeout=self.generation_timeout)
        data = response.json()
        if not isinstance(data, dict):
            raise BackendFailure("invalid_backend_response", "LM Studio returned a non-object completion")
        return data

    async def health(self) -> dict[str, Any]:
        # One check at a time: polls that arrive together share the same LM Studio request.
        async with self._health_lock:
            now = time.monotonic()
            if self._listing is not None and now - self._listing[0] < self.health_ttl_seconds:
                return self._health_from(self._listing[1])
            if self._failed_health is not None and now - self._failed_health[0] < self.health_ttl_seconds:
                return dict(self._failed_health[1])
            try:
                models = await self.list_models()
            except (BackendFailure, TimeoutError, httpx.TimeoutException) as exc:
                code = exc.code if isinstance(exc, BackendFailure) else "backend_unavailable"
                result = {"state": "unavailable", "code": code}
                self._failed_health = (time.monotonic(), result)
                return dict(result)
            return self._health_from(models)

    def _health_from(self, models: list[dict[str, Any]]) -> dict[str, Any]:
        installed = [alias for alias, profile in self.profiles.items() if self._model_entry(models, profile.model_id) is not None]
        return {
            "state": "ready",
            "installed_profiles": installed,
            "loaded_instances": len(self._loaded_instances(models)),
        }

    async def _stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=payload, timeout=None) as response:
                if response.status_code >= 400:
                    raise BackendFailure("inference_failed", "LM Studio rejected the completion request", retryable=True)
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.TimeoutException as exc:
            raise BackendFailure(
                "inference_in_flight",
                "LM Studio streaming timed out; LM Studio may still be generating",
                retryable=True,
            ) from exc

    def stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        return self._stream(payload)

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None, timeout: float | None = None) -> httpx.Response:
        try:
            response = await self.client.request(method, path, json=json, timeout=timeout or self.request_timeout)
        except httpx.TimeoutException:
            raise
        except httpx.HTTPError as exc:
            raise BackendFailure("backend_unavailable", "LM Studio is unavailable", retryable=True) from exc
        if response.status_code >= 400:
            raise BackendFailure("backend_http_error", "LM Studio returned an error", retryable=response.status_code >= 500)
        return response

    @staticmethod
    def _model_entry(models: list[dict[str, Any]], model_id: str) -> dict[str, Any] | None:
        return next((model for model in models if model.get("key") == model_id), None)

    @staticmethod
    def _require_selected_variant(profile_config: ProfileConfig, model: dict[str, Any]) -> None:
        if profile_config.variant and model.get("selected_variant") != profile_config.variant:
            raise BackendFailure(
                "variant_mismatch",
                f"LM Studio's selected variant does not match the configured variant: {profile_config.variant}",
                retryable=True,
            )

    def _chat_instances(self, models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Loaded instances the chat scheduler manages: everything except the resident embedding models."""
        return [item for item in self._loaded_instances(models) if item["model_id"] not in self._embedding_models]

    @staticmethod
    def _loaded_instances(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for model in models:
            for instance in model.get("loaded_instances", []) or []:
                if isinstance(instance, dict):
                    result.append({"model_id": model.get("key"), **instance})
        return result
