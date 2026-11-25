from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from support import PRESET

from local_llm_broker import cli
from local_llm_broker.cli import EXIT_INVALID_INPUT, EXIT_OK, format_profiles, main, run_verify
from local_llm_broker.client import BrokerClientError
from local_llm_broker.config import load_config


class EmbedClient:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.fail = fail

    def embed(self, model: str, inputs: list[str], **labels: str) -> dict[str, Any]:
        self.calls.append((model, inputs))
        if self.fail:
            raise BrokerClientError("model_load_failed", "load failed", 502, True)
        return {"object": "list", "model": model, "data": [{"index": i, "embedding": [0.5, 0.25, 0.125]} for i, _ in enumerate(inputs)]}

    def close(self) -> None:
        pass


def run(monkeypatch: pytest.MonkeyPatch, client: EmbedClient, *args: str) -> int:
    config = load_config(PRESET)
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (client, config))
    return int(main(["--config", str(PRESET), "embed", "--project", "p", "--task", "t", *args]))


def test_embed_text_writes_vectors_to_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    client = EmbedClient()
    out = tmp_path / "vectors.json"
    code = run(monkeypatch, client, "--model", "embed", "--text", "one", "two", "--out", str(out))
    assert code == EXIT_OK
    assert client.calls == [("embed", ["one", "two"])]
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == {"model": "embed", "dimensions": 3, "embeddings": [[0.5, 0.25, 0.125], [0.5, 0.25, 0.125]]}
    printed = capsys.readouterr().out
    assert "2 embeddings" in printed
    assert "0.25" not in printed  # the vectors go to the file, not the terminal


def test_embed_file_sends_one_input_per_non_empty_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "lines.txt"
    source.write_text("first\n\nsecond\n  \nthird\n", encoding="utf-8")
    client = EmbedClient()
    run(monkeypatch, client, "--model", "embed", "--file", str(source), "--out", str(tmp_path / "v.json"))
    assert client.calls == [("embed", ["first", "second", "third"])]


def test_embed_without_out_prints_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(monkeypatch, EmbedClient(), "--model", "embed", "--text", "hello")
    assert code == EXIT_OK
    assert json.loads(capsys.readouterr().out)["dimensions"] == 3


def test_embed_needs_some_input(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(monkeypatch, EmbedClient(), "--model", "embed")
    assert code == EXIT_INVALID_INPUT
    assert "--text or --file" in capsys.readouterr().err


def test_embed_reports_broker_errors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(monkeypatch, EmbedClient(fail=True), "--model", "embed", "--text", "x")
    assert code != EXIT_OK
    assert "model_load_failed" in capsys.readouterr().err


def test_profiles_table_shows_kind_and_context_for_embedding_profiles() -> None:
    models = {
        "data": [
            {"id": "4b", "kind": "chat", "model_id": "m", "thinks": False, "vision": False, "parallel": 4,
             "tiers": {"low": {"context_length": 16384}}, "description": ""},
            {"id": "embed", "kind": "embedding", "model_id": "e", "thinks": False, "vision": False, "parallel": 8,
             "context": 2048, "tiers": {}, "description": "Text embeddings."},
        ]
    }
    text = format_profiles(models)
    header, chat_row, embed_row, description = text.splitlines()
    assert "kind" in header
    assert "chat" in chat_row
    assert "embedding" in embed_row and "2,048" in embed_row
    assert description.strip() == "Text embeddings."


def test_verify_embeds_a_short_text_for_an_embedding_profile(capsys: pytest.CaptureFixture[str]) -> None:
    client = EmbedClient()
    entries = [{"id": "embed", "kind": "embedding", "thinks": False}]
    assert run_verify(client, entries, None) == EXIT_OK  # type: ignore[arg-type]
    assert client.calls and client.calls[0][0] == "embed"
    assert "PASS embed" in capsys.readouterr().out


def test_verify_fails_an_embedding_profile_that_returns_no_vector(capsys: pytest.CaptureFixture[str]) -> None:
    class Empty(EmbedClient):
        def embed(self, model: str, inputs: list[str], **labels: str) -> dict[str, Any]:
            return {"data": []}

    code = run_verify(Empty(), [{"id": "embed", "kind": "embedding"}], None)  # type: ignore[arg-type]
    assert code != EXIT_OK
    assert "FAIL embed" in capsys.readouterr().out
