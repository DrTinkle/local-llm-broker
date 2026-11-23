from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from support import PRESET

from local_llm_broker.config import load_config
from local_llm_broker.contracts import ProfileConfig, TierConfig
from local_llm_broker.setup_write import (
    existing_broker_settings,
    render_config,
    summarize_changes,
    write_config,
)

NOW = datetime(2026, 11, 23, 10, 15, 0)


def test_render_roundtrip(tmp_path: Path) -> None:
    preset = load_config(PRESET)
    tiers = {model: dict(rows) for model, rows in preset.tiers.items()}

    text = render_config({"service_port": 1240, "data_dir": "X:/data"}, dict(preset.profiles), tiers)
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    loaded = load_config(path)

    assert len(loaded.profiles) == 6
    assert dict(loaded.profiles) == dict(preset.profiles)
    assert {model: dict(rows) for model, rows in loaded.tiers.items()} == tiers
    assert loaded.service_port == 1240
    assert loaded.data_dir == Path("X:/data")


def test_render_roundtrip_keeps_every_broker_scalar_type(tmp_path: Path) -> None:
    broker = {"service_port": 1240, "idle_unload_seconds": 90.5, "data_dir": 'C:\\Users\\me "x"\\data', "queue_limit": 7}

    text = render_config(broker, {}, {})
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")

    assert existing_broker_settings(path) == broker
    assert load_config(path).idle_unload_seconds == 90.5


def test_render_quotes_model_keys_and_omits_defaults() -> None:
    profiles = {
        "qwen3.5": ProfileConfig("qwen3.5", "org/model", context_ceiling=8192, parallel_ceiling=2),
        "plain": ProfileConfig("plain", "org/untiered", context_ceiling=8192, parallel_ceiling=1),
        "wide": ProfileConfig("wide", "org/wide", context_ceiling=16384, parallel_ceiling=1, description="Big one"),
    }
    tiers = {
        "org/model": {
            "low": TierConfig("low", 8192),
            "high": TierConfig("high", 32768, "q8_0", flash_attention=True, cpu_expert_ratio=0.5, try_mmap=False),
        }
    }

    text = render_config({}, profiles, tiers)

    assert text.startswith("# Written by `llm-broker setup`.")
    assert "[broker]" not in text
    assert '[profiles."qwen3.5"]' in text
    assert "[profiles.plain]" in text
    assert '[tiers."org/model".low]' in text
    assert '[tiers."org/model".high]' in text
    assert text.index('[tiers."org/model".low]') < text.index('[tiers."org/model".high]')
    assert 'model_id = "org/model"' in text
    assert "parallel_ceiling = 2" in text
    assert 'description = "Big one"' in text
    assert "cpu_expert_ratio = 0.5" in text
    assert "try_mmap = false" in text
    for default in ("flash_attention", "eval_batch_size", "supports_vision", "thinks", "variant", "reasoning_effort"):
        assert default not in text
    # A tiered model never gets context_ceiling; an untiered one only when it is not the default.
    assert text.count("context_ceiling") == 1
    assert "context_ceiling = 16384" in text


def test_render_writes_non_default_values() -> None:
    profiles = {
        "v": ProfileConfig(
            "v", "org/v", 8192, 3, variant="org/v@q4", reasoning_effort="none", supports_vision=True, thinks=True
        )
    }
    tiers = {"org/v": {"low": TierConfig("low", 4096, flash_attention=False, eval_batch_size=512)}}

    text = render_config({"queue_limit": 5, "lms_path": "C:/lms.exe", "flag": True}, profiles, tiers)

    assert "queue_limit = 5" in text
    assert 'lms_path = "C:/lms.exe"' in text
    assert "flag = true" in text
    assert 'variant = "org/v@q4"' in text
    assert 'reasoning_effort = "none"' in text
    assert "supports_vision = true" in text
    assert "thinks = true" in text
    assert "flash_attention = false" in text
    assert "eval_batch_size = 512" in text
    assert text.index("[broker]") < text.index("[profiles.v]") < text.index('[tiers."org/v".low]')


def test_write_config_keeps_backup(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("old\n", encoding="utf-8")
    first = render_config({"service_port": 1240}, {}, {})

    backup = write_config(path, first, now=NOW)

    assert backup == tmp_path / "config.toml.bak-20261123-101500"
    assert backup.read_text(encoding="utf-8") == "old\n"
    assert path.read_text(encoding="utf-8") == first

    second = render_config({"service_port": 1241}, {}, {})
    again = write_config(path, second, now=NOW)

    assert again == tmp_path / "config.toml.bak-20261123-101500-1"
    assert again.read_text(encoding="utf-8") == first
    assert path.read_text(encoding="utf-8") == second
    assert (tmp_path / "config.toml.bak-20261123-101500").read_text(encoding="utf-8") == "old\n"


def test_write_config_creates_missing_file_and_folder(tmp_path: Path) -> None:
    path = tmp_path / "new" / "config.toml"

    backup = write_config(path, render_config({}, {}, {}), now=NOW)

    assert backup is None
    assert path.exists()
    assert [p.name for p in path.parent.iterdir()] == ["config.toml"]


def test_write_config_refuses_invalid_text(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(ValueError):
        write_config(path, "[profiles.x]\nmodel_id = 3\n", now=NOW)

    assert path.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]


def test_existing_broker_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[broker]\nservice_port = 1240\ndata_dir = "X:/d"\nflag = true\nratio = 0.5\nlist = [1, 2]\n'
        "[broker.nested]\nx = 1\n[profiles.a]\nmodel_id = \"m\"\n",
        encoding="utf-8",
    )

    assert existing_broker_settings(path) == {"service_port": 1240, "data_dir": "X:/d", "flag": True, "ratio": 0.5}


def test_existing_broker_settings_when_missing_or_unreadable(tmp_path: Path) -> None:
    assert existing_broker_settings(tmp_path / "nope.toml") == {}

    broken = tmp_path / "broken.toml"
    broken.write_text("[broker\n", encoding="utf-8")
    assert existing_broker_settings(broken) == {}

    no_table = tmp_path / "none.toml"
    no_table.write_text('[profiles.a]\nmodel_id = "m"\n', encoding="utf-8")
    assert existing_broker_settings(no_table) == {}


def test_summarize_changes() -> None:
    kept = ProfileConfig("kept", "org/a", 8192, 1)
    old = {
        "kept": kept,
        "gone": ProfileConfig("gone", "org/b", 8192, 1),
        "changed": ProfileConfig("changed", "org/c", 8192, 1),
    }
    new = {
        "kept": kept,
        "changed": ProfileConfig("changed", "org/c", 8192, 2),
        "added": ProfileConfig("added", "org/d", 8192, 1),
    }

    assert summarize_changes(old, new) == ["added profile added (org/d)", "changed profile changed", "removed profile gone"]
    assert summarize_changes(old, old) == []
    assert summarize_changes({}, {}) == []
