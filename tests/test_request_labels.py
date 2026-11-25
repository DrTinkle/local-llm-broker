from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from support import PRESET

from local_llm_broker import cli
from local_llm_broker.app import create_app
from local_llm_broker.backend import LoadedModel
from local_llm_broker.cli import EXIT_INVALID_INPUT, main
from local_llm_broker.config import load_config
from local_llm_broker.contracts import ProfileConfig


class Backend:
    async def ensure_loaded(self, profile: str, **_: object) -> LoadedModel:
        return LoadedModel(profile, profile, "instance", 32_000, 1)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": "c", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    async def embed(self, profile: str, inputs: list[str]) -> dict[str, Any]:
        return {"data": [{"index": i, "embedding": [0.1]} for i, _ in enumerate(inputs)]}


def make_app(tmp_path: Path) -> Any:
    config = load_config(PRESET)
    profiles = dict(config.profiles)
    profiles["embed"] = ProfileConfig("embed", "embed-model", 2048, 2, kind="embedding")
    return create_app(
        replace(config, data_dir=tmp_path, auth_token_path=tmp_path / "auth.token", profiles=profiles), adapter=Backend()
    )


def job_body(**changes: Any) -> dict[str, Any]:
    body = {
        "client_id": "agent-1", "project": "billing", "task_label": "triage-log", "profile": "4b",
        "messages": [{"role": "user", "content": "hello"}], "generation": {"max_tokens": 8},
    }
    body.update(changes)
    return body


async def call(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    headers = {"Authorization": f"Bearer {app.state.auth_token}", **kwargs.pop("headers", {})}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
        return await client.request(method, path, headers=headers, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [{"project": ""}, {"task_label": ""}, {"client_id": ""}])
async def test_native_jobs_need_a_client_project_and_task(tmp_path: Path, changes: dict[str, str]) -> None:
    app = make_app(tmp_path)
    await app.state.scheduler.start()
    try:
        assert (await call(app, "POST", "/jobs", json=job_body(**changes))).status_code == 422
        body = job_body()
        del body["project"]
        assert (await call(app, "POST", "/jobs", json=body)).status_code == 422
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_job_events_name_the_client_project_and_task(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    await app.state.scheduler.start()
    try:
        job = (await call(app, "POST", "/jobs", json=job_body())).json()["job_id"]
        await app.state.scheduler.wait_for_terminal(job)
        events = [e for e in app.state.telemetry.recent(limit=200) if e.get("job_id") == job]
        assert events
        for event in events:
            assert (event["client_id"], event["project"], event["task_label"]) == ("agent-1", "billing", "triage-log")
        finished = next(e for e in events if e["event"] == "inference_finished")
        assert finished["state"] == "succeeded"
        assert finished["inference_seconds"] is not None
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_openai_chat_uses_label_headers_and_has_defaults(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    await app.state.scheduler.start()
    try:
        body = {"model": "4b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        await call(app, "POST", "/v1/chat/completions", json=body,
                   headers={"X-Client-Id": "tool-a", "X-Project": "docs", "X-Task": "summarize"})
        await call(app, "POST", "/v1/chat/completions", json=body)
        queued = [e for e in app.state.telemetry.recent(limit=200) if e["event"] == "queued"]
        assert [(e["client_id"], e["project"], e["task_label"]) for e in queued] == [
            ("tool-a", "docs", "summarize"),
            ("openai", "openai", "chat-completions"),
        ]
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_embedding_events_carry_the_labels(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    await app.state.scheduler.start()
    try:
        body = {"model": "embed", "input": ["x"]}
        await call(app, "POST", "/v1/embeddings", json=body, headers={"X-Client-Id": "idx", "X-Project": "search", "X-Task": "index"})
        await call(app, "POST", "/v1/embeddings", json=body)
        done = [e for e in app.state.telemetry.recent(limit=200) if e["event"] == "embedding_finished"]
        assert [(e["client_id"], e["project"], e["task_label"]) for e in done] == [
            ("idx", "search", "index"),
            ("openai", "openai", "embeddings"),
        ]
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.parametrize("missing", ["--project", "--task"])
def test_cli_submit_requires_project_and_task(missing: str, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--config", str(PRESET), "submit", "--model", "4b", "--prompt", "hi", "--project", "p", "--task", "t"]
    index = args.index(missing)
    del args[index : index + 2]
    with pytest.raises(SystemExit) as raised:
        main(args)
    assert raised.value.code == EXIT_INVALID_INPUT
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize("value", ["", "   "])
def test_cli_rejects_blank_labels(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--config", str(PRESET), "submit", "--model", "4b", "--prompt", "hi", "--project", value, "--task", "t"])
    assert code == EXIT_INVALID_INPUT
    assert "--project" in capsys.readouterr().err


class RecordingClient:
    def __init__(self) -> None:
        self.embed_calls: list[dict[str, Any]] = []

    def embed(self, model: str, inputs: list[str], **labels: str) -> dict[str, Any]:
        self.embed_calls.append(labels)
        return {"data": [{"index": 0, "embedding": [0.5]}]}

    def close(self) -> None:
        pass


def test_cli_embed_sends_its_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RecordingClient()
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (client, load_config(PRESET)))
    code = main(["--config", str(PRESET), "embed", "--model", "embed", "--text", "x",
                 "--client", "agent-1", "--project", "search", "--task", "index"])
    assert code == 0
    assert client.embed_calls == [{"client_id": "agent-1", "project": "search", "task": "index"}]
