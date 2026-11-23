from __future__ import annotations

import argparse
import re
from pathlib import Path

from local_llm_broker import cli

SKILLS = Path(__file__).resolve().parents[1] / "skills"

FORBIDDEN = re.compile(
    r"\b(?:4|9|35)b\b|qwen|3080|12 ?gb|g:\\|\\users\\|claude",
    re.IGNORECASE,
)

COMMAND = re.compile(r"(?:python -m local_llm_broker\.cli|llm-broker)[ \t]+([A-Za-z][\w-]*)")


def _skill_files() -> list[Path]:
    files = sorted(path for path in SKILLS.rglob("*") if path.is_file())
    assert files, f"no skill files found under {SKILLS}"
    return files


def test_skill_is_generic() -> None:
    hits: list[str] = []
    for path in _skill_files():
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            for match in FORBIDDEN.finditer(line):
                hits.append(f"{path.relative_to(SKILLS)}:{number}: {match.group(0)!r} in {line.strip()!r}")
    assert not hits, "machine-specific text in the skill:\n" + "\n".join(hits)


def _subcommands() -> set[str]:
    for action in cli._parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("no subcommands found in cli._parser()")


def test_skill_commands_use_real_subcommands() -> None:
    real = _subcommands()
    assert {"submit", "status", "profiles", "verify", "setup", "install-skill"} <= real
    bad: list[str] = []
    seen = 0
    for path in _skill_files():
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            for match in COMMAND.finditer(line):
                seen += 1
                if match.group(1) not in real:
                    bad.append(f"{path.relative_to(SKILLS)}:{number}: unknown subcommand {match.group(1)!r}")
    assert seen, "no commands found in the skill files"
    assert not bad, "\n".join(bad)
