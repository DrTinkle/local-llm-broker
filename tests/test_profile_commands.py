from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from support import PRESET

from local_llm_broker import cli
from local_llm_broker.cli import EXIT_INFERENCE, EXIT_INVALID_INPUT, EXIT_OK, format_profiles, main
from local_llm_broker.config import load_config


def _entry(alias: str, *, thinks: bool, vision: bool, tiers: dict[str, int], description: str = "", parallel: int = 1) -> dict[str, Any]:
    return {
        "id": alias,
        "object": "model",
        "owned_by": "local-llm-broker",
        "model_id": f"vendor/{alias}-model",
        "parallel": parallel,
        "thinks": thinks,
        "vision": vision,
        "description": description,
        "tiers": {name: {"context_length": length} for name, length in tiers.items()},
    }


def _models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            _entry("big", thinks=True, vision=True, tiers={"low": 32768, "medium": 65536, "high": 163840}, description="Hard tasks.", parallel=2),
            _entry("small", thinks=False, vision=False, tiers={"low": 8192}, description=""),
        ],
    }


class _FakeClient:
    def __init__(self, models: dict[str, Any], failing: set[str] | None = None, empty: set[str] | None = None) -> None:
        self._models = models
        self.failing = failing or set()
        self.empty = empty or set()
        self.payloads: list[dict[str, Any]] = []

    def models(self) -> dict[str, Any]:
        return self._models

    def submit(self, payload: dict[str, Any], *, idempotency_key: str | None = None) -> dict[str, str]:
        self.payloads.append(payload)
        return {"job_id": f"job-{payload['profile']}"}

    def wait(self, job_id: str) -> dict[str, Any]:
        alias = job_id.removeprefix("job-")
        if alias in self.failing:
            return {"job_id": job_id, "state": "failed", "error": {"code": "model_load_failed", "message": "could not load"}}
        return {"job_id": job_id, "state": "succeeded"}

    def result(self, job_id: str) -> dict[str, Any]:
        alias = job_id.removeprefix("job-")
        text = "" if alias in self.empty else "OK"
        return {"result": {"choices": [{"message": {"content": text}}]}}

    def close(self) -> None:
        pass


def _install(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    config = load_config(PRESET)
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (client, config))


def test_format_profiles_lists_every_profile() -> None:
    text = format_profiles(_models())
    lines = text.splitlines()
    header = lines[0].split()
    assert header == ["profile", "kind", "model", "thinks", "vision", "slots", "low", "medium", "high"]
    big = next(line for line in lines if line.startswith("big"))
    small = next(line for line in lines if line.startswith("small"))
    assert big.split() == ["big", "chat", "vendor/big-model", "yes", "yes", "2", "32,768", "65,536", "163,840"]
    assert small.split() == ["small", "chat", "vendor/small-model", "no", "no", "1", "8,192", "-", "-"]
    assert "Hard tasks." in text
    assert sum(1 for line in lines if line.startswith(("big", "small"))) == 2


def test_profiles_command_prints_every_profile(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _install(monkeypatch, _FakeClient(_models()))
    assert main(["--config", str(PRESET), "profiles"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "big" in out and "small" in out

    assert main(["--config", str(PRESET), "profiles", "--json"]) == EXIT_OK
    assert [item["id"] for item in json.loads(capsys.readouterr().out)] == ["big", "small"]

    _install(monkeypatch, _FakeClient({"object": "list", "data": []}))
    assert main(["--config", str(PRESET), "profiles"]) == EXIT_OK
    assert "No profiles are configured. Run `llm-broker setup` first." in capsys.readouterr().out


def test_verify_reports_each_profile(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    client = _FakeClient(_models(), failing={"big"})
    _install(monkeypatch, client)
    assert main(["--config", str(PRESET), "verify"]) == EXIT_INFERENCE
    out = capsys.readouterr().out
    assert "FAIL big: model_load_failed" in out
    assert "PASS small (" in out and " s)" in out
    budgets = {payload["profile"]: payload["generation"]["max_tokens"] for payload in client.payloads}
    assert budgets == {"big": 2048, "small": 64}
    first = client.payloads[0]
    assert first["client_id"] == "verify"
    assert first["task_label"] == "verify"
    assert first["messages"] == [{"role": "user", "content": "Reply with the word OK."}]


def test_verify_override_and_empty_answer(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    client = _FakeClient(_models(), empty={"small"})
    _install(monkeypatch, client)
    assert main(["--config", str(PRESET), "verify", "--max-output-tokens", "99"]) == EXIT_INFERENCE
    out = capsys.readouterr().out
    assert "FAIL small: empty answer" in out
    assert "PASS big (" in out
    assert {payload["generation"]["max_tokens"] for payload in client.payloads} == {99}


def test_verify_all_pass_returns_zero(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    client = _FakeClient(_models())
    _install(monkeypatch, client)
    assert main(["--config", str(PRESET), "verify", "--profiles", "small"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "PASS small (" in out
    assert "big" not in out
    assert [payload["profile"] for payload in client.payloads] == ["small"]


def test_verify_unknown_profile_submits_nothing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    client = _FakeClient(_models())
    _install(monkeypatch, client)
    assert main(["--config", str(PRESET), "verify", "--profiles", "small", "nope"]) == EXIT_INVALID_INPUT
    assert "Unknown profile: nope" in capsys.readouterr().out
    assert client.payloads == []


def test_install_skill_copies_and_refuses_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "skills"
    missing_config = tmp_path / "no-such-config.toml"
    argv = ["--config", str(missing_config), "install-skill", "--target", str(target)]
    assert main(argv) == EXIT_OK
    destination = target / "use-local-models"
    assert (destination / "SKILL.md").is_file()
    assert f"Installed the skill to {destination}" in capsys.readouterr().out

    marker = destination / "SKILL.md"
    marker.write_text("local edit", encoding="utf-8")
    stale = destination / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    assert main(argv) == EXIT_INVALID_INPUT
    assert "already exists; use --force to replace it." in capsys.readouterr().out
    assert marker.read_text(encoding="utf-8") == "local edit"
    assert stale.exists()

    assert main([*argv, "--force"]) == EXIT_OK
    assert not stale.exists()
    assert marker.read_text(encoding="utf-8") != "local edit"
