from __future__ import annotations

from pathlib import Path

import pytest

from local_llm_broker.config import load_config

EMBED = """
[profiles.embed]
kind = "embedding"
model_id = "text-embedding-nomic-embed-text-v1.5@f16"
context_ceiling = 2048
parallel_ceiling = 8
description = "Text embeddings."
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_profiles_default_to_chat(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, '[profiles.chat]\nmodel_id = "m"\n'))
    assert config.profiles["chat"].kind == "chat"


def test_embedding_profile_is_read(tmp_path: Path) -> None:
    profile = load_config(write(tmp_path, EMBED)).profiles["embed"]
    assert profile.kind == "embedding"
    assert profile.model_id == "text-embedding-nomic-embed-text-v1.5@f16"
    assert profile.context_ceiling == 2048
    assert profile.parallel_ceiling == 8


def test_kind_must_be_chat_or_embedding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kind"):
        load_config(write(tmp_path, '[profiles.x]\nmodel_id = "m"\nkind = "audio"\n'))


@pytest.mark.parametrize("setting", ["thinks = true", "supports_vision = true", 'reasoning_effort = "none"', 'variant = "m@q4"'])
def test_embedding_profile_rejects_chat_only_settings(tmp_path: Path, setting: str) -> None:
    with pytest.raises(ValueError, match="embedding"):
        load_config(write(tmp_path, EMBED + setting + "\n"))


def test_embedding_profile_ignores_tiers_for_its_model(tmp_path: Path) -> None:
    text = EMBED + '\n[tiers."text-embedding-nomic-embed-text-v1.5@f16".low]\ncontext_length = 1024\n'
    with pytest.raises(ValueError, match="tiers"):
        load_config(write(tmp_path, text))
