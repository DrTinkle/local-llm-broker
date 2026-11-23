from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from support import PRESET

from local_llm_broker import cli
from local_llm_broker.app import create_app
from local_llm_broker.backend import LoadedModel
from local_llm_broker.cli import EXIT_OK, EXIT_QUEUE, EXIT_UNAVAILABLE, _parser, main
from local_llm_broker.client import BrokerClientError
from local_llm_broker.config import load_config


class StopBackend:
    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        return LoadedModel(profile, profile, "instance", 32_000, 1)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "completion", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


class Recorder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def _app(tmp_path: Path) -> FastAPI:
    config = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token")
    return create_app(config, adapter=StopBackend())


def _busy(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = app.state.scheduler
    original = scheduler.status

    def status() -> dict[str, Any]:
        return {**original(), "queue_count": 1, "active_count": 1}

    monkeypatch.setattr(scheduler, "status", status)


@pytest.mark.asyncio
async def test_shutdown_requires_auth(tmp_path: Path) -> None:
    app = _app(tmp_path)
    recorder = Recorder()
    app.state.request_shutdown = recorder
    await app.state.scheduler.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            response = await client.post("/admin/shutdown", json={})
        assert response.status_code == 401
        assert recorder.calls == 0
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_shutdown_refuses_while_busy_without_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path)
    recorder = Recorder()
    app.state.request_shutdown = recorder
    _busy(app, monkeypatch)
    await app.state.scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            response = await client.post("/admin/shutdown", headers=headers)
            explicit = await client.post("/admin/shutdown", headers=headers, json={"force": False})
        for item in (response, explicit):
            assert item.status_code == 409
            error = item.json()["error"]
            assert error["code"] == "broker_busy"
            assert error["retryable"] is True
            assert error["message"] == "2 job(s) are queued or running; wait for them, or stop with --force"
        assert recorder.calls == 0
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_shutdown_with_force_while_busy_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path)
    recorder = Recorder()
    app.state.request_shutdown = recorder
    _busy(app, monkeypatch)
    await app.state.scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            response = await client.post("/admin/shutdown", headers=headers, json={"force": True})
        assert response.status_code == 202
        assert response.json() == {"status": "stopping"}
        assert recorder.calls == 1
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_shutdown_when_idle_stops(tmp_path: Path) -> None:
    app = _app(tmp_path)
    recorder = Recorder()
    app.state.request_shutdown = recorder
    await app.state.scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            response = await client.post("/admin/shutdown", headers=headers)
        assert response.status_code == 202
        assert response.json() == {"status": "stopping"}
        assert recorder.calls == 1
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_shutdown_without_hook_just_answers(tmp_path: Path) -> None:
    app = _app(tmp_path)
    assert app.state.request_shutdown is None
    await app.state.scheduler.start()
    try:
        headers = {"Authorization": f"Bearer {app.state.auth_token}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
            response = await client.post("/admin/shutdown", headers=headers)
        assert response.status_code == 202
        assert response.json() == {"status": "stopping"}
    finally:
        await app.state.scheduler.shutdown()


class _StopClient:
    def __init__(self, error: BrokerClientError | None = None) -> None:
        self.error = error
        self.forces: list[bool] = []

    def shutdown(self, force: bool = False) -> dict[str, Any]:
        self.forces.append(force)
        if self.error is not None:
            raise self.error
        return {"status": "stopping"}

    def close(self) -> None:
        pass


def _probe(monkeypatch: pytest.MonkeyPatch, answers: list[bool]) -> list[str]:
    asked: list[str] = []
    remaining = list(answers)

    def probe(base_url: str) -> bool:
        asked.append(base_url)
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    monkeypatch.setattr(cli, "_broker_answers", probe)
    return asked


def _install(monkeypatch: pytest.MonkeyPatch, client: _StopClient) -> None:
    config = load_config(PRESET)
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (client, config))


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    pauses: list[float] = []
    monkeypatch.setattr(cli, "_pause", pauses.append)
    return pauses


def test_stop_when_not_running(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    asked = _probe(monkeypatch, [False])

    def no_client(config_path: Path, base_url: str | None) -> Any:
        raise AssertionError("the authenticated client must not be created")

    monkeypatch.setattr(cli, "_client", no_client)
    assert main(["--config", str(PRESET), "stop"]) == EXIT_OK
    assert "The broker is not running." in capsys.readouterr().out
    assert len(asked) == 1


def test_stop_waits_until_the_broker_is_gone(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    client = _StopClient()
    _install(monkeypatch, client)
    _probe(monkeypatch, [True, True, True, False])  # running, then still answering twice, then gone
    pauses = _no_sleep(monkeypatch)
    assert main(["--config", str(PRESET), "stop"]) == EXIT_OK
    assert "The broker stopped." in capsys.readouterr().out
    assert client.forces == [False]
    assert pauses == [0.5, 0.5]


def test_stop_times_out_when_broker_keeps_answering(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    client = _StopClient()
    _install(monkeypatch, client)
    _probe(monkeypatch, [True])
    pauses = _no_sleep(monkeypatch)
    assert main(["--config", str(PRESET), "stop", "--wait-seconds", "2"]) == EXIT_UNAVAILABLE
    out = capsys.readouterr().out
    assert "The broker is still shutting down after 2 seconds; check again, or end the process by hand." in out
    assert sum(pauses) == pytest.approx(2.0)


def test_stop_busy_broker_asks_for_force(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    message = "3 job(s) are queued or running; wait for them, or stop with --force"
    client = _StopClient(BrokerClientError("broker_busy", message, 409, True))
    _install(monkeypatch, client)
    _probe(monkeypatch, [True])
    _no_sleep(monkeypatch)
    assert main(["--config", str(PRESET), "stop"]) == EXIT_QUEUE
    assert message in capsys.readouterr().out
    assert client.forces == [False]

    forced = _StopClient()
    _install(monkeypatch, forced)
    _probe(monkeypatch, [True, False])
    assert main(["--config", str(PRESET), "stop", "--force"]) == EXIT_OK
    assert forced.forces == [True]


def test_stop_flags_parse() -> None:
    parser = _parser()
    defaults = parser.parse_args(["stop"])
    assert defaults.command == "stop"
    assert defaults.force is False
    assert defaults.wait_seconds == 20
    custom = parser.parse_args(["stop", "--force", "--wait-seconds", "5"])
    assert custom.force is True
    assert custom.wait_seconds == 5


def test_serve_uses_server_object_and_sets_shutdown_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.toml"
    data_dir = (tmp_path / "data").as_posix()
    config_path.write_text(f'[broker]\nservice_port = 12360\ndata_dir = "{data_dir}"\n', encoding="utf-8")
    app = FastAPI()
    monkeypatch.setattr(cli, "create_app", lambda config: app)
    monkeypatch.setattr(cli, "ensure_auth_token", lambda config: "token")
    servers: list[Any] = []

    class FakeServer:
        def __init__(self, config: Any) -> None:
            self.config = config
            self.should_exit = False
            self.ran = False
            self.exit_requested_during_run: bool | None = None
            servers.append(self)

        def run(self) -> None:
            self.ran = True
            app.state.request_shutdown()
            self.exit_requested_during_run = self.should_exit

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    assert main(["--config", str(config_path), "serve"]) == EXIT_OK
    assert len(servers) == 1
    server = servers[0]
    assert server.ran is True
    assert server.exit_requested_during_run is True
    assert server.config.port == 12360
    assert server.config.workers == 1
    assert server.config.access_log is False
    assert server.config.timeout_graceful_shutdown == 5
