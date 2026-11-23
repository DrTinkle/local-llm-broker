from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from local_llm_broker.lms_cli import find_lms


def test_find_lms_prefers_config_then_path_then_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configured = tmp_path / "custom" / "lms.exe"
    configured.parent.mkdir()
    configured.write_text("")
    on_path = tmp_path / "pathdir" / "lms.exe"
    home = tmp_path / "home"
    home_lms = home / ".lmstudio" / "bin" / "lms.exe"
    home_lms.parent.mkdir(parents=True)
    home_lms.write_text("")
    monkeypatch.setattr(shutil, "which", lambda name: str(on_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    assert find_lms(str(configured)) == str(configured)
    # A configured path that does not exist is skipped, not trusted.
    assert find_lms(str(tmp_path / "missing" / "lms.exe")) == str(on_path)
    assert find_lms(None) == str(on_path)

    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert find_lms(None) == str(home_lms)

    home_lms.unlink()
    assert find_lms(None) is None
    bare = home / ".lmstudio" / "bin" / "lms"
    bare.write_text("")
    assert find_lms("") == str(bare)

    bare.unlink()
    assert find_lms(str(tmp_path / "missing" / "lms.exe")) is None
