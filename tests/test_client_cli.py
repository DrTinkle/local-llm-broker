from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from support import PRESET

from local_llm_broker import cli
from local_llm_broker.cli import (
    EXIT_CANCELLED,
    EXIT_INVALID_INPUT,
    EXIT_LOAD,
    EXIT_OK,
    EXIT_UNAVAILABLE,
    _use_utf8_output,
    build_messages,
    main,
    write_atomic,
)
from local_llm_broker.client import BrokerClient, BrokerClientError
from local_llm_broker.config import load_config
from local_llm_broker.inputs import InputError, InputFile
from local_llm_broker.tokens import estimate_prompt_tokens


def test_build_messages_keeps_instructions_separate_from_unicode_source_files(tmp_path: Path) -> None:
    source = tmp_path / "данные.txt"
    source.write_text("source body", encoding="utf-8")

    messages = build_messages("Summarize this", system_text="Be concise", files=[source], max_text_bytes=1000)

    assert messages == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Summarize this\n\n--- File: данные.txt ---\nsource body"},
    ]


def test_build_messages_rejects_oversized_files(tmp_path: Path) -> None:
    source = tmp_path / "large.txt"
    source.write_text("x" * 20, encoding="utf-8")
    with pytest.raises(ValueError, match="input limit"):
        build_messages("prompt", files=[source], max_text_bytes=10)


def test_write_atomic_replaces_output_without_partial_file(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "answer.txt"
    write_atomic(output, "final answer")
    assert output.read_text(encoding="utf-8") == "final answer"
    assert not list(output.parent.glob("*.tmp"))


def test_client_submits_and_reads_structured_errors() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/jobs":
            return httpx.Response(202, json={"job_id": "job-1", "state": "queued"}, request=request)
        if request.url.path == "/jobs/job-1":
            return httpx.Response(200, json={"job_id": "job-1", "state": "succeeded"}, request=request)
        if request.url.path == "/jobs/job-1/result":
            return httpx.Response(200, json={"result": {"choices": []}}, request=request)
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "missing"}}, request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://broker")
    client = BrokerClient("http://broker", token="secret", client=http_client)
    submitted = client.submit({"profile": "4b", "messages": []})
    assert submitted["job_id"] == "job-1"
    assert client.wait("job-1")["state"] == "succeeded"
    assert client.result("job-1")["result"] == {"choices": []}
    assert seen[0].headers["authorization"] == "Bearer secret"


def test_client_error_does_not_include_response_body_in_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "SENTINEL BODY", "code": "backend"}}, request=request)

    client = BrokerClient("http://broker", token="secret", client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://broker"))
    with pytest.raises(BrokerClientError) as error:
        client.status()
    assert "SENTINEL" not in str(error.value)
    assert error.value.code == "backend"


def test_cli_invalid_submit_input_returns_documented_exit_code() -> None:
    assert main(["submit", "--model", "4b", "--project", "p", "--task", "t"]) == EXIT_INVALID_INPUT


def test_cli_output_survives_characters_a_cp1252_console_cannot_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    # A model reply with an emoji used to raise UnicodeEncodeError on a cp1252 stream and exit with code 2.
    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="cp1252"))
    _use_utf8_output()
    print("bird 🐦")
    sys.stdout.flush()
    assert "bird 🐦" in buffer.getvalue().decode("utf-8")


class _FinishedJobClient:
    def __init__(self, snapshot: dict[str, object]) -> None:
        self.snapshot = snapshot

    def submit(self, payload: dict[str, object], *, idempotency_key: str | None = None) -> dict[str, str]:
        return {"job_id": "job-9"}

    def wait(self, job_id: str) -> dict[str, object]:
        return self.snapshot

    def close(self) -> None:
        pass


def _submit_and_finish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, snapshot: dict[str, object]) -> int:
    config = load_config(PRESET)
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (_FinishedJobClient(snapshot), config))
    code: int = main(["--config", str(PRESET), "submit", "--model", "4b", "--prompt", "hi", "--project", "p", "--task", "t"])
    return code


def test_cancelled_job_without_error_detail_exits_as_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # A job cancelled while queued carries no error object, which used to crash submit with AttributeError.
    snapshot: dict[str, object] = {"job_id": "job-9", "state": "cancelled", "error": None}
    assert _submit_and_finish(monkeypatch, tmp_path, snapshot) == EXIT_CANCELLED
    assert "cancelled (retryable: no)" in capsys.readouterr().err


def test_interrupted_job_without_error_detail_exits_as_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    snapshot: dict[str, object] = {"job_id": "job-9", "state": "interrupted", "error": None}
    assert _submit_and_finish(monkeypatch, tmp_path, snapshot) == EXIT_CANCELLED
    assert "interrupted (retryable: no)" in capsys.readouterr().err


def test_cli_prints_status_as_json_that_another_program_can_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class StatusClient:
        def status(self) -> dict[str, object]:
            return {"queue_count": 0, "backend_state": "idle"}

        def close(self) -> None:
            pass

    config = load_config(PRESET)
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (StatusClient(), config))
    assert main(["--config", str(PRESET), "status"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out) == {"queue_count": 0, "backend_state": "idle"}


def test_cli_exits_unavailable_when_the_broker_cannot_be_reached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = BrokerClient("http://broker", token="secret", client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://broker"))
    config = load_config(PRESET)
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (client, config))
    assert main(["--config", str(PRESET), "status"]) == EXIT_UNAVAILABLE
    assert "unavailable (retryable: yes): broker is unavailable" in capsys.readouterr().err


def test_failed_job_reports_its_error_code_and_retry_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    snapshot: dict[str, object] = {
        "job_id": "job-9",
        "state": "failed",
        "error": {"code": "model_load_failed", "message": "LM Studio rejected the configured model load", "retryable": True},
    }
    assert _submit_and_finish(monkeypatch, tmp_path, snapshot) == EXIT_LOAD
    assert "model_load_failed (retryable: yes): LM Studio rejected the configured model load" in capsys.readouterr().err



def test_build_messages_reports_missing_file_with_code(tmp_path: Path) -> None:
    with pytest.raises(InputError) as error:
        build_messages("prompt", files=[tmp_path / "absent.txt"])
    assert error.value.code == "input_not_found"
    assert str(error.value) == "path does not exist: absent.txt"


def test_build_messages_rejects_directory_as_input(tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    with pytest.raises(InputError) as error:
        build_messages("prompt", files=[folder])
    assert error.value.code == "input_is_directory"


def test_build_messages_rejects_nul_byte_file(tmp_path: Path) -> None:
    source = tmp_path / "blob.bin"
    source.write_bytes(b"abc\x00def")
    with pytest.raises(InputError) as error:
        build_messages("prompt", files=[source])
    assert error.value.code == "binary_or_non_utf8_file"


def test_build_messages_rejects_non_utf8_file(tmp_path: Path) -> None:
    source = tmp_path / "latin.txt"
    source.write_bytes(b"\xff\xfe")
    with pytest.raises(InputError) as error:
        build_messages("prompt", files=[source])
    assert error.value.code == "binary_or_non_utf8_file"


def test_build_messages_rejects_file_larger_than_limit_with_code(tmp_path: Path) -> None:
    source = tmp_path / "big.txt"
    source.write_text("y" * 500, encoding="utf-8")
    with pytest.raises(InputError) as error:
        build_messages("prompt", files=[source], max_text_bytes=100)
    assert error.value.code == "input_too_large"
    assert "input limit" in str(error.value)


def test_main_rejects_out_path_that_is_a_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class SubmitForbiddenClient:
        def submit(self, payload: dict[str, object], *, idempotency_key: str | None = None) -> dict[str, str]:
            raise AssertionError("no HTTP submit may happen when --out is a directory")

        def close(self) -> None:
            pass

    out_dir = tmp_path / "answers"
    out_dir.mkdir()
    config = load_config(PRESET)
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (SubmitForbiddenClient(), config))
    code = main(["--config", str(PRESET), "submit", "--model", "4b", "--prompt", "hi", "--project", "p", "--task", "t", "--out", str(out_dir)])
    assert code == EXIT_INVALID_INPUT
    assert "input_is_directory (retryable: no)" in capsys.readouterr().err


def test_build_messages_uses_input_file_label_in_header(tmp_path: Path) -> None:
    source = tmp_path / "a.md"
    source.write_text("alpha", encoding="utf-8")
    messages = build_messages("prompt", files=[InputFile("docs/a.md", source)], max_text_bytes=1000)
    assert messages[-1]["content"] == "prompt\n\n--- File: docs/a.md ---\nalpha"


def test_build_messages_distinguishes_same_basename_in_different_folders(tmp_path: Path) -> None:
    first = tmp_path / "one" / "x.py"
    second = tmp_path / "two" / "x.py"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("print(1)", encoding="utf-8")
    second.write_text("print(2)", encoding="utf-8")
    messages = build_messages(
        "prompt",
        files=[InputFile("a/x.py", first), InputFile("b/x.py", second)],
        max_text_bytes=1000,
    )
    content = messages[-1]["content"]
    assert "--- File: a/x.py ---\nprint(1)" in content
    assert "--- File: b/x.py ---\nprint(2)" in content


class _RecordingClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def submit(self, payload: dict[str, object], *, idempotency_key: str | None = None) -> dict[str, str]:
        self.payloads.append(payload)
        return {"job_id": "job-1"}

    def wait(self, job_id: str) -> dict[str, object]:
        return {"state": "succeeded"}

    def result(self, job_id: str) -> dict[str, object]:
        return {"result": {"choices": [{"message": {"content": "ok"}}]}}

    def close(self) -> None:
        pass


def test_submit_expands_folder_into_sorted_labelled_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "b.md").write_text("bravo", encoding="utf-8")
    (docs / "a.md").write_text("alpha", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    client = _RecordingClient()
    config = load_config(PRESET)
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (client, config))

    code = main(["--config", str(PRESET), "submit", "--model", "4b", "--prompt", "hi", "--project", "p", "--task", "t", "--files", "docs"])

    assert code == EXIT_OK
    content = client.payloads[0]["messages"][-1]["content"]  # type: ignore[index]
    assert "--- File: docs/a.md ---" in content
    assert "--- File: docs/b.md ---" in content
    assert content.index("--- File: docs/a.md ---") < content.index("--- File: docs/b.md ---")
    assert "attached 2 files" in capsys.readouterr().err


def test_submit_missing_files_path_reports_input_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_config(PRESET)
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (_RecordingClient(), config))

    code = main(["--config", str(PRESET), "submit", "--model", "4b", "--prompt", "hi", "--project", "p", "--task", "t", "--files", "missing_dir"])

    assert code == EXIT_INVALID_INPUT
    assert "input_not_found (retryable: no)" in capsys.readouterr().err


def test_submit_parser_accepts_expansion_flags() -> None:
    args = cli._parser().parse_args(
        [
            "submit", "--model", "4b", "--prompt", "hi", "--project", "p", "--task", "t",
            "--files", "src", "--include", "*.py", "--exclude", "tests",
            "--no-recursive", "--no-default-excludes", "--max-files", "5",
        ]
    )
    assert args.files == ["src"]
    assert args.include == ["*.py"]
    assert args.exclude == ["tests"]
    assert args.no_recursive is True
    assert args.no_default_excludes is True
    assert args.max_files == 5


class _BatchClient:
    """Stub broker: job N answers 'answer N'; fail_on makes job N fail with model_load_failed."""

    def __init__(self, fail_on: int | None = None) -> None:
        self.fail_on = fail_on
        self.payloads: list[dict[str, Any]] = []
        self.keys: list[str | None] = []

    def submit(self, payload: dict[str, Any], *, idempotency_key: str | None = None) -> dict[str, str]:
        self.payloads.append(payload)
        self.keys.append(idempotency_key)
        return {"job_id": f"job{len(self.payloads)}"}

    def wait(self, job_id: str) -> dict[str, Any]:
        if int(job_id.removeprefix("job")) == self.fail_on:
            return {"state": "failed", "error": {"code": "model_load_failed", "message": "boom", "retryable": False}}
        return {"state": "succeeded"}

    def result(self, job_id: str) -> dict[str, Any]:
        number = job_id.removeprefix("job")
        return {"result": {"choices": [{"message": {"content": f"answer {number}"}}]}}

    def close(self) -> None:
        pass


def _write_split_files(tmp_path: Path) -> int:
    """Writes three equal files and returns a token budget that fits one file per job but not two."""
    singles = []
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("alpha bravo charlie delta " * 300, encoding="utf-8")
        singles.append(InputFile(name, tmp_path / name))
    generation = {"max_tokens": 2048}
    one = estimate_prompt_tokens(build_messages("p", files=singles[:1], max_text_bytes=10**6), generation)
    two = estimate_prompt_tokens(build_messages("p", files=singles[:2], max_text_bytes=10**6), generation)
    budget: int = (one + two) // 2
    return budget


def _run_split(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: _BatchClient, extra: list[str]) -> int:
    config = load_config(PRESET)
    monkeypatch.setattr(cli, "_client", lambda config_path, base_url: (client, config))
    code: int = main(
        [
            "--config", str(PRESET), "submit", "--model", "4b", "--prompt", "summarize", "--project", "p", "--task", "t",
            "--files", "a.txt", "b.txt", "c.txt", "--split", *extra,
        ]
    )
    return code


def test_split_sends_one_job_per_batch_with_derived_keys_and_writes_combined_output_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    budget = _write_split_files(tmp_path)
    client = _BatchClient()
    writes: list[Path] = []
    real_write = cli.write_atomic

    def spy(path: Path, content: str) -> None:
        writes.append(path)
        real_write(path, content)

    monkeypatch.setattr(cli, "write_atomic", spy)
    out = tmp_path / "answer.md"

    code = _run_split(monkeypatch, tmp_path, client, ["--split-budget-tokens", str(budget), "--idempotency-key", "job1", "--out", str(out)])

    assert code == EXIT_OK
    assert client.keys == ["job1-part-1-of-3", "job1-part-2-of-3", "job1-part-3-of-3"]
    first = client.payloads[0]["messages"][-1]["content"]
    assert first.startswith("summarize\n\n[batch 1 of 3]\n\n--- File: a.txt ---")
    assert "b.txt" not in first
    assert client.payloads[2]["messages"][-1]["content"].startswith("summarize\n\n[batch 3 of 3]")
    assert writes == [out]
    assert out.read_text(encoding="utf-8") == (
        "=== BATCH 1 OF 3 (job job1) ===\nanswer 1\n\n"
        "=== BATCH 2 OF 3 (job job2) ===\nanswer 2\n\n"
        "=== BATCH 3 OF 3 (job job3) ===\nanswer 3\n"
    )
    assert "job1\njob2\njob3\n" in capsys.readouterr().out


def test_split_failed_batch_returns_failure_code_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    budget = _write_split_files(tmp_path)
    client = _BatchClient(fail_on=2)
    out = tmp_path / "answer.md"

    code = _run_split(monkeypatch, tmp_path, client, ["--split-budget-tokens", str(budget), "--out", str(out)])

    assert code == EXIT_LOAD
    assert not out.exists()
    assert len(client.payloads) == 2
    assert "model_load_failed (retryable: no): boom" in capsys.readouterr().err


def test_split_detach_submits_every_batch_and_prints_each_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    budget = _write_split_files(tmp_path)
    client = _BatchClient()

    code = _run_split(monkeypatch, tmp_path, client, ["--split-budget-tokens", str(budget), "--detach"])

    assert code == EXIT_OK
    assert len(client.payloads) == 3
    assert capsys.readouterr().out == "job1\njob2\njob3\n"


def test_split_set_that_fits_one_batch_behaves_like_plain_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_split_files(tmp_path)
    client = _BatchClient()

    code = _run_split(monkeypatch, tmp_path, client, ["--split-budget-tokens", "100000", "--idempotency-key", "job1"])

    assert code == EXIT_OK
    assert client.keys == ["job1"]
    assert "[batch" not in client.payloads[0]["messages"][-1]["content"]
    assert capsys.readouterr().out == "job1\nanswer 1\n"


@pytest.mark.parametrize(
    ("model", "tier", "context"),
    [
        ("4b-nothink", None, 81920),
        ("4b-nothink", "low", 16384),
        ("9b", None, 163840),
        ("9b-nothink", "medium", 131072),
        ("35b-nothink", None, 262144),
        ("35b", "auto", 262144),
    ],
)
def test_split_budget_follows_calibrated_tiers(tmp_path: Path, model: str, tier: str | None, context: int) -> None:
    from local_llm_broker.cli import SPLIT_TOKEN_RESERVE, _split_budget_tokens

    config = load_config(PRESET)
    args = argparse.Namespace(split_budget_tokens=None, model=model, tier=tier, max_output_tokens=2048)

    assert _split_budget_tokens(config, args) == context - 2048 - SPLIT_TOKEN_RESERVE


def test_split_budget_is_none_for_unknown_profile_and_explicit_value_wins(tmp_path: Path) -> None:
    from local_llm_broker.cli import _split_budget_tokens

    config = load_config(PRESET)
    unknown = argparse.Namespace(split_budget_tokens=None, model="nope", tier=None, max_output_tokens=2048)
    explicit = argparse.Namespace(split_budget_tokens=1234, model="4b", tier=None, max_output_tokens=2048)

    assert _split_budget_tokens(config, unknown) is None
    assert _split_budget_tokens(config, explicit) == 1234


def test_degraded_broker_exits_with_the_retry_later_code() -> None:
    assert cli._exit_for_error("broker_degraded", 503) == cli.EXIT_QUEUE
