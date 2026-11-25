from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx


class BrokerClientError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 0, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


class BrokerClient:
    def __init__(self, base_url: str = "http://127.0.0.1:1235", token: str | None = None, *, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("LOCAL_LLM_BROKER_TOKEN", "")
        self.client = client or httpx.Client(base_url=self.base_url, timeout=30.0)
        self._owns_client = client is None

    @classmethod
    def from_token_file(cls, path: Path, base_url: str = "http://127.0.0.1:1235") -> BrokerClient:
        return cls(base_url, path.read_text(encoding="utf-8").strip())

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def submit(self, payload: dict[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return self._request("POST", "/jobs", json=payload, headers=headers)

    def get(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}")

    def wait(self, job_id: str, *, timeout: float = 3600.0, poll_seconds: float = 0.5) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            snapshot = self.get(job_id)
            if snapshot.get("state") in {"succeeded", "failed", "cancelled", "interrupted"}:
                return snapshot
            if time.monotonic() >= deadline:
                raise BrokerClientError("client_timeout", "timed out waiting for broker job", 408, True)
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

    def result(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}/result")

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/jobs/{job_id}")

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def models(self) -> dict[str, Any]:
        return self._request("GET", "/v1/models")

    def embed(self, model: str, inputs: list[str], *, client_id: str = "cli", project: str = "cli", task: str = "embed") -> dict[str, Any]:
        headers = {"X-Client-Id": client_id, "X-Project": project, "X-Task": task}
        return self._request("POST", "/v1/embeddings", json={"model": model, "input": inputs}, headers=headers)

    def logs(self) -> dict[str, Any]:
        return self._request("GET", "/logs")

    def summary(self) -> dict[str, Any]:
        return self._request("GET", "/summary")

    def shutdown(self, force: bool = False) -> dict[str, Any]:
        return self._request("POST", "/admin/shutdown", json={"force": force})

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = dict(kwargs.pop("headers", {}))
        headers.setdefault("Authorization", f"Bearer {self.token}")
        try:
            response = self.client.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise BrokerClientError("unavailable", "broker is unavailable", 0, True) from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400:
            error = body.get("error") if isinstance(body, dict) else {}
            if not isinstance(error, dict):
                error = {}
            message = "broker request failed" if response.status_code >= 500 else str(error.get("message", "broker request failed"))
            raise BrokerClientError(
                str(error.get("code", "http_error")),
                message,
                response.status_code,
                bool(error.get("retryable", response.status_code >= 500)),
            )
        return body if isinstance(body, dict) else {}

