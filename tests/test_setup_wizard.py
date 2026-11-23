from __future__ import annotations

import argparse
import io
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from support import PRESET

from local_llm_broker import setup_wizard
from local_llm_broker.calibrate_direct import DirectSession
from local_llm_broker.cli import _parser
from local_llm_broker.config import load_config
from local_llm_broker.contracts import BrokerConfig, TierConfig
from local_llm_broker.setup_calibrate import NOT_LOADED, CalibrationResult, Level, Progress
from local_llm_broker.setup_discover import Hardware, ModelInfo, Preflight, discover_models
from local_llm_broker.setup_profiles import quick_tiers, suggest_alias
from local_llm_broker.setup_wizard import SetupDeps, run_setup

HW = Hardware("Test GPU", 12_288, 65_536)
NOW = datetime(2026, 11, 23, 10, 15, 0)
BACKUP_NAME = "config.toml.bak-20261123-101500"


def llm(key: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "llm",
        "key": key,
        "display_name": key,
        "architecture": "qwen3",
        "size_bytes": 2_500_000_000,
        "params_string": "4B",
        "max_context_length": 262144,
        "capabilities": {"vision": False},
        "selected_variant": f"{key}@q4_k_m",
    }
    entry.update(overrides)
    return entry


EMBEDDING: dict[str, Any] = {
    "type": "embedding", "key": "text-embedding-nomic", "display_name": "Nomic Embed", "size_bytes": 84_000_000,
    "max_context_length": 2048,
}


class FakeSession:
    def __init__(self) -> None:
        self.closed = 0
        self.unloads = 0

    async def close(self) -> None:
        self.closed += 1

    async def unload_all(self) -> None:
        self.unloads += 1


class Harness:
    """Fake dependencies plus a record of what the wizard did with them."""

    def __init__(
        self, listing: dict[str, Any], *, lmstudio_ok: bool = True, broker_running: bool = False, problems: list[str] | None = None
    ) -> None:
        self.listing = listing
        self.preflight = Preflight(lmstudio_ok, "C:/lms.exe", broker_running, problems or [])
        self.preflight_calls = 0
        self.listing_calls = 0
        self.sessions: list[FakeSession] = []
        self.calibrated: list[tuple[str, str, int]] = []
        self.out = io.StringIO()

    async def _preflight(self, config: BrokerConfig) -> Preflight:
        self.preflight_calls += 1
        return self.preflight

    async def _listing(self, config: BrokerConfig) -> dict[str, Any]:
        self.listing_calls += 1
        return self.listing

    def _session(self, config: BrokerConfig) -> DirectSession:
        session = FakeSession()
        self.sessions.append(session)
        return session  # type: ignore[return-value]

    def deps(self) -> SetupDeps:
        return SetupDeps(
            preflight=self._preflight,
            fetch_listing=self._listing,
            hardware=lambda: HW,
            session_factory=self._session,
            now=lambda: NOW,
        )

    def patch_calibration(
        self, monkeypatch: pytest.MonkeyPatch, results: Callable[[ModelInfo, int], CalibrationResult] | None = None
    ) -> None:
        async def fake(session: DirectSession, model: ModelInfo, hw: Hardware, level: Level, parallel: int) -> CalibrationResult:
            self.calibrated.append((model.key, level, parallel))
            if results is not None:
                return results(model, parallel)
            return ok_result(model.key, parallel)

        monkeypatch.setattr(setup_wizard, "calibrate_model", fake)

    def run(self, args: argparse.Namespace, answers: list[str] | None = None) -> tuple[int, list[str]]:
        asked: list[str] = []
        scripted = list(answers or [])

        def ask(prompt: str) -> str:
            asked.append(prompt)
            if not scripted:
                raise AssertionError(f"unexpected question: {prompt}")
            return scripted.pop(0)

        code = run_setup(args, ask=ask, out=self.out, deps=self.deps())
        return code, asked

    @property
    def text(self) -> str:
        return self.out.getvalue()


def ok_result(key: str, parallel: int = 1, *, thinks: bool | None = False, can_disable: bool | None = False) -> CalibrationResult:
    tiers = {"low": TierConfig("low", 8192), "medium": TierConfig("medium", 16384)}
    return CalibrationResult(key, tiers, parallel, thinks, can_disable, ["note"])


def make_args(config: Path, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {"config": config, "level": None, "models": None, "preset": None, "yes": False}
    values.update(overrides)
    return argparse.Namespace(**values)


def listing_of(*keys: str) -> dict[str, Any]:
    return {"models": [llm(key) for key in keys]}


def config_with_data_dir(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "config.toml"
    path.write_text(f'[broker]\ndata_dir = "{(tmp_path / "data").as_posix()}"\n{extra}', encoding="utf-8")
    return path


def test_setup_no_chat_models(tmp_path: Path) -> None:
    harness = Harness({"models": [EMBEDDING]})
    path = tmp_path / "config.toml"

    code, asked = harness.run(make_args(path))

    assert code == 3
    assert "No chat models are downloaded in LM Studio" in harness.text
    assert "Discover tab" in harness.text
    assert not path.exists()
    assert asked == []


def test_setup_lmstudio_down(tmp_path: Path) -> None:
    problem = "LM Studio is not answering at http://127.0.0.1:1234; open LM Studio and start the local server"
    harness = Harness(listing_of("org/a"), lmstudio_ok=False, problems=[problem])
    path = tmp_path / "config.toml"

    code, asked = harness.run(make_args(path))

    assert code == 3
    assert "start the local server" in harness.text.lower()
    assert "Start the LM Studio local server, then run setup again." in harness.text
    assert problem in harness.text
    assert asked == []
    assert harness.listing_calls == 0
    assert not path.exists()


def test_setup_interactive_level_none(tmp_path: Path) -> None:
    key = "qwen/qwen3-4b-thinking-2507"
    harness = Harness(listing_of(key), problems=["The LM Studio command-line tool (lms) was not found"])
    path = config_with_data_dir(tmp_path)
    info = discover_models(harness.listing)[0]
    alias = suggest_alias(info, set())

    # model number, alias (accept), thinks, twin, level, confirm (accept)
    code, asked = harness.run(make_args(path), ["1", "", "y", "y", "none", ""])

    assert code == 0, harness.text
    config = load_config(path)
    assert set(config.profiles) == {alias, f"{alias}-nothink"}
    assert config.profiles[alias].thinks is True
    assert config.profiles[f"{alias}-nothink"].reasoning_effort == "none"
    assert dict(config.tiers[key]) == quick_tiers(info, HW)
    assert harness.sessions == []
    assert harness.calibrated == []
    assert len(asked) == 6
    assert "Test GPU" in harness.text
    assert "warning" in harness.text.lower() and "lms" in harness.text
    assert f"Wrote {path}" in harness.text
    assert "llm-broker verify" in harness.text
    assert "install-skill" in harness.text


def test_setup_refuses_calibration_while_broker_runs(tmp_path: Path) -> None:
    harness = Harness(listing_of("org/a"), broker_running=True)
    path = tmp_path / "config.toml"

    code, _ = harness.run(make_args(path, level="quick", yes=True))

    assert code == 3
    assert "Stop the broker, then run setup again" in harness.text
    assert not path.exists()
    assert harness.sessions == []

    code, _ = harness.run(make_args(path, level="none", yes=True))

    assert code == 0
    assert "a" in load_config(path).profiles


def test_setup_from_yes_flag_asks_nothing(tmp_path: Path) -> None:
    harness = Harness(listing_of("org/alpha", "qwen/beta-thinking"))
    path = tmp_path / "config.toml"

    code, asked = harness.run(make_args(path, yes=True))

    assert code == 0, harness.text
    assert asked == []
    config = load_config(path)
    assert set(config.profiles) == {"alpha", "beta-thinking", "beta-thinking-nothink"}
    assert config.profiles["beta-thinking"].thinks is True
    assert config.profiles["alpha"].parallel_ceiling == 4  # small model on 12 GB


def test_setup_models_flag_matches_key_or_alias(tmp_path: Path) -> None:
    harness = Harness(listing_of("org/alpha", "org/beta", "org/gamma"))
    path = tmp_path / "config.toml"

    code, asked = harness.run(make_args(path, models=["org/alpha", "gamma"], level="none"), [""])

    assert code == 0, harness.text
    assert asked == [f"Write {path}? [Y/n] "]
    assert set(load_config(path).profiles) == {"alpha", "gamma"}


def test_setup_unknown_model_is_invalid_input(tmp_path: Path) -> None:
    harness = Harness(listing_of("org/alpha"))
    path = tmp_path / "config.toml"

    code, _ = harness.run(make_args(path, models=["org/missing"]))

    assert code == 2
    assert "Unknown model: org/missing" in harness.text
    assert not path.exists()


def test_setup_preset_copies_preset(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[broker]\nservice_port = 1299\ndata_dir = "X:/mine"\n\n[profiles.old]\nmodel_id = "org/old"\n', encoding="utf-8"
    )
    harness = Harness({"models": []}, lmstudio_ok=False)

    code, asked = harness.run(make_args(path, preset=PRESET, yes=True))

    assert code == 0, harness.text
    assert asked == []
    assert harness.preflight_calls == 0 and harness.listing_calls == 0
    preset = load_config(PRESET)
    config = load_config(path)
    assert dict(config.profiles) == dict(preset.profiles)
    assert {m: dict(r) for m, r in config.tiers.items()} == {m: dict(r) for m, r in preset.tiers.items()}
    assert config.service_port == 1299
    assert config.data_dir == Path("X:/mine")
    assert "removed profile old" in harness.text
    assert (tmp_path / BACKUP_NAME).exists()


def test_setup_preset_that_is_invalid_returns_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[profiles.x]\nmodel_id = 3\n", encoding="utf-8")
    harness = Harness({"models": []})

    code, _ = harness.run(make_args(tmp_path / "config.toml", preset=bad, yes=True))

    assert code == 2
    assert not (tmp_path / "config.toml").exists()


def test_setup_invalid_existing_config_returns_2(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[profiles.x]\nmodel_id = 3\n", encoding="utf-8")
    harness = Harness(listing_of("org/a"))

    code, _ = harness.run(make_args(path, yes=True))

    assert code == 2
    assert path.read_text(encoding="utf-8") == "[profiles.x]\nmodel_id = 3\n"
    assert harness.preflight_calls == 0


def test_setup_skips_finished_models_on_rerun(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = Harness(listing_of("org/model-a", "org/model-b"))
    harness.patch_calibration(monkeypatch)
    path = config_with_data_dir(tmp_path)
    data_dir = tmp_path / "data"
    saved = CalibrationResult("org/model-a", {"low": TierConfig("low", 12288)}, 2, False, False, ["earlier"])
    Progress.load(data_dir / "setup-progress.json", "quick").save(saved)

    code, asked = harness.run(make_args(path, level="quick", yes=True))

    assert code == 0, harness.text
    assert asked == []
    assert [call[0] for call in harness.calibrated] == ["org/model-b"]
    assert "already calibrated, skipping" in harness.text
    config = load_config(path)
    assert config.tiers["org/model-a"]["low"].context_length == 12288
    assert config.profiles["model-a"].parallel_ceiling == 2
    assert "medium" in config.tiers["org/model-b"]
    assert Progress.load(data_dir / "setup-progress.json", "quick").done("org/model-b") is not None
    assert [s.closed for s in harness.sessions] == [1]


def test_setup_keeps_backup_with_yes(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    old = "# my notes\n[broker]\nservice_port = 1299\n"
    path.write_text(old, encoding="utf-8")
    harness = Harness(listing_of("org/a"))

    code, asked = harness.run(make_args(path, yes=True))

    assert code == 0, harness.text
    assert asked == []
    backup = tmp_path / BACKUP_NAME
    assert backup.read_text(encoding="utf-8") == old
    assert "Backed up the old file to" in harness.text
    assert "added profile a (org/a)" in harness.text
    assert load_config(path).service_port == 1299


def test_setup_no_changes_writes_nothing(tmp_path: Path) -> None:
    harness = Harness(listing_of("org/a"))
    path = tmp_path / "config.toml"
    assert harness.run(make_args(path, yes=True))[0] == 0
    before = path.read_text(encoding="utf-8")

    code, _ = harness.run(make_args(path, yes=True))

    assert code == 0
    assert "No changes" in harness.text
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]


def test_setup_declined_confirmation_writes_nothing(tmp_path: Path) -> None:
    harness = Harness(listing_of("org/a"))
    path = tmp_path / "config.toml"

    code, _ = harness.run(make_args(path, models=["org/a"], level="none"), ["n"])

    assert code == 0
    assert "Nothing written." in harness.text
    assert not path.exists()


def test_setup_failed_model_is_not_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = Harness(listing_of("org/big"))
    harness.patch_calibration(monkeypatch, lambda model, parallel: CalibrationResult(model.key, {}, parallel, None, None, failed=NOT_LOADED))
    path = config_with_data_dir(tmp_path)
    before = path.read_text(encoding="utf-8")

    code, _ = harness.run(make_args(path, level="thorough", yes=True))

    assert code == 3
    assert f"org/big: {NOT_LOADED}; no profile written for it" in harness.text
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["config.toml"]
    assert [s.closed for s in harness.sessions] == [1]


def test_setup_failed_model_is_skipped_but_others_are_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = Harness(listing_of("org/big", "org/fine"))

    def results(model: ModelInfo, parallel: int) -> CalibrationResult:
        if model.key == "org/big":
            return CalibrationResult(model.key, {}, parallel, None, None, failed=NOT_LOADED)
        return ok_result(model.key, parallel)

    harness.patch_calibration(monkeypatch, results)
    path = config_with_data_dir(tmp_path)

    code, _ = harness.run(make_args(path, level="quick", yes=True))

    assert code == 0, harness.text
    assert set(load_config(path).profiles) == {"fine"}


def test_setup_calibrated_thinking_overrides_guess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = Harness(listing_of("org/plain", "org/x-thinking"))
    harness.patch_calibration(monkeypatch, lambda model, parallel: ok_result(model.key, 3, thinks=True, can_disable=False))
    path = config_with_data_dir(tmp_path)

    code, _ = harness.run(make_args(path, level="thorough", yes=True))

    assert code == 0, harness.text
    config = load_config(path)
    assert set(config.profiles) == {"plain", "x-thinking"}  # no -nothink twin
    assert config.profiles["plain"].thinks is True
    assert config.profiles["x-thinking"].thinks is True
    assert config.profiles["plain"].parallel_ceiling == 3
    assert "thinking" in harness.text.lower() and "-nothink" in harness.text


def test_setup_calibrated_twin_kept_when_thinking_can_be_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = Harness(listing_of("org/x-thinking"))
    harness.patch_calibration(monkeypatch, lambda model, parallel: ok_result(model.key, 1, thinks=True, can_disable=True))
    path = config_with_data_dir(tmp_path)

    code, _ = harness.run(make_args(path, level="quick", yes=True))

    assert code == 0, harness.text
    assert set(load_config(path).profiles) == {"x-thinking", "x-thinking-nothink"}


def test_setup_interrupt_keeps_finished_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = Harness(listing_of("org/model-a", "org/model-b"))

    def results(model: ModelInfo, parallel: int) -> CalibrationResult:
        if model.key == "org/model-b":
            raise KeyboardInterrupt
        return ok_result(model.key, parallel)

    harness.patch_calibration(monkeypatch, results)
    path = config_with_data_dir(tmp_path)
    before = path.read_text(encoding="utf-8")

    code, _ = harness.run(make_args(path, level="quick", yes=True))

    assert code == 3
    assert "Interrupted. Finished models are saved; run setup again to continue." in harness.text
    assert path.read_text(encoding="utf-8") == before
    progress = Progress.load(tmp_path / "data" / "setup-progress.json", "quick")
    assert progress.done("org/model-a") is not None
    assert progress.done("org/model-b") is None
    assert [s.closed for s in harness.sessions] == [1]


def test_setup_interactive_choices_are_validated(tmp_path: Path) -> None:
    harness = Harness(listing_of("org/alpha", "org/beta"))
    path = tmp_path / "config.toml"

    # model list out of range, then "all"; alias: invalid, then good; thinks no; alias: taken, then good; thinks no;
    # level typo, then none; confirm
    answers = ["9", "all", "Bad Alias", "alpha", "n", "alpha", "beta", "n", "bogus", "none", ""]
    code, asked = harness.run(make_args(path), answers)

    assert code == 0, harness.text
    assert set(load_config(path).profiles) == {"alpha", "beta"}
    assert any("numbers" in prompt for prompt in asked)


def test_setup_gives_up_after_three_bad_model_answers(tmp_path: Path) -> None:
    harness = Harness(listing_of("org/alpha"))

    code, asked = harness.run(make_args(tmp_path / "config.toml"), ["x", "0", "7"])

    assert code == 2
    assert len(asked) == 3
    assert not (tmp_path / "config.toml").exists()


def test_setup_cli_flags_parse() -> None:
    args = _parser().parse_args(["setup", "--level", "quick", "--models", "a/b", "--yes"])

    assert args.command == "setup"
    assert args.level == "quick"
    assert args.models == ["a/b"]
    assert args.yes is True
    assert args.preset is None

    args = _parser().parse_args(["setup", "--profiles-file", "p.toml"])

    assert args.preset == Path("p.toml")
    assert args.level is None
    assert args.models is None
    assert args.yes is False

    assert _parser().parse_args(["setup", "--preset", "q.toml"]).preset == Path("q.toml")


def test_main_runs_setup_before_loading_the_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from local_llm_broker import cli

    seen: list[argparse.Namespace] = []

    def fake_run(args: argparse.Namespace) -> int:
        seen.append(args)
        return 0

    monkeypatch.setattr(setup_wizard, "run_setup", fake_run)
    missing = tmp_path / "config.toml"

    assert cli.main(["--config", str(missing), "setup", "--yes"]) == 0
    assert seen[0].config == missing
    assert seen[0].yes is True
