from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from local_llm_broker.inputs import InputError, InputFile, expand_inputs, partition_files


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def _write(path: str, text: str = "x") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _labels(items: list[str], **kwargs: Any) -> list[str]:
    return [entry.label for entry in expand_inputs(items, **kwargs)]


def _code(items: list[str], **kwargs: Any) -> str:
    with pytest.raises(InputError) as exc:
        expand_inputs(items, **kwargs)
    code: str = exc.value.code
    return code


def test_recursive_walk_is_deterministic_with_mixed_case() -> None:
    _write("d/B.py")
    _write("d/a.py")
    _write("d/Sub/c.txt")

    assert _labels(["d"]) == ["d/a.py", "d/B.py", "d/Sub/c.txt"]


def test_non_recursive_reads_only_direct_children() -> None:
    _write("d/a.py")
    _write("d/sub/b.py")

    assert _labels(["d"], recursive=False) == ["d/a.py"]


def test_glob_double_star() -> None:
    _write("pkg/x.py")
    _write("pkg/sub/y.py")
    _write("top.py")
    _write("notes.txt")

    assert _labels(["**/*.py"]) == ["pkg/sub/y.py", "pkg/x.py", "top.py"]


def test_file_and_its_folder_given_is_included_once() -> None:
    _write("d/a.py")
    _write("d/b.py")

    assert _labels(["d/a.py", "d"]) == ["a.py", "d/b.py"]
    assert _labels(["d/b.py", "d/a.py"]) == ["b.py", "a.py"]


def test_default_exclusions_drop_noise_and_secrets() -> None:
    _write("proj/.git/config")
    _write("proj/node_modules/pkg/index.js")
    _write("proj/__pycache__/m.pyc")
    _write("proj/.env")
    _write("proj/.env.local")
    _write("proj/image.png")
    _write("proj/package-lock.json")
    _write("proj/README.md")
    _write("proj/.env.example")

    assert _labels(["proj"]) == ["proj/.env.example", "proj/README.md"]


def test_default_excludes_false_keeps_folders_but_still_drops_secrets() -> None:
    _write("proj/node_modules/pkg/index.js")
    _write("proj/package-lock.json")
    _write("proj/image.png")
    _write("proj/README.md")
    _write("proj/.env")
    _write("proj/.env.local")

    labels = set(_labels(["proj"], default_excludes=False))

    assert {
        "proj/node_modules/pkg/index.js",
        "proj/package-lock.json",
        "proj/image.png",
        "proj/README.md",
    } <= labels
    assert "proj/.env" not in labels
    assert "proj/.env.local" not in labels


def test_include_and_exclude_filters() -> None:
    _write("proj/a.py")
    _write("proj/b.py")
    _write("proj/c.txt")
    _write("proj/sub/d.py")
    _write("proj/tests/t.py")

    assert _labels(["proj"], include=["*.py"]) == [
        "proj/a.py",
        "proj/b.py",
        "proj/sub/d.py",
        "proj/tests/t.py",
    ]
    assert _labels(["proj"], include=["*.py"], exclude=["tests", "b.py"]) == [
        "proj/a.py",
        "proj/sub/d.py",
    ]


def test_symlinked_file_is_skipped_in_walk() -> None:
    _write("proj/real.py")
    link = Path("proj") / "link.py"
    try:
        os.symlink("real.py", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform")

    try:
        assert _labels(["proj"]) == ["proj/real.py"]
    finally:
        link.unlink()


def test_empty_expansion_raises() -> None:
    _write("proj/.env")
    _write("proj/.env.local")
    Path("empty").mkdir()

    assert _code(["proj"]) == "empty_expansion"
    assert _code(["empty"]) == "empty_expansion"


def test_too_many_files_raises() -> None:
    _write("d/1.py")
    _write("d/2.py")
    _write("d/3.py")

    assert _code(["d"], max_files=2) == "too_many_files"
    assert len(expand_inputs(["d"], max_files=3)) == 3


@pytest.mark.parametrize(
    "name", [".env", ".env.local", "auth.token", "server.key", "cert.pem"]
)
def test_explicit_secret_file_is_blocked(name: str) -> None:
    _write(name)

    assert _code([name]) == "secret_file_blocked"


def test_explicit_env_example_is_allowed() -> None:
    _write(".env.example")

    assert _labels([".env.example"]) == [".env.example"]


def test_missing_path_and_empty_glob_raise_not_found() -> None:
    Path("nothing").mkdir()

    assert _code(["nope.txt"]) == "input_not_found"
    assert _code(["nothing/*.py"]) == "input_not_found"


def test_trailing_slash_and_backslash_give_same_labels() -> None:
    _write("d/sub/a.py")
    _write("d/sub/nested/c.py")
    expected = ["sub/a.py", "sub/nested/c.py"]

    assert _labels(["d/sub"]) == expected
    assert _labels(["d/sub/"]) == expected
    if os.name == "nt":
        assert _labels(["d\\sub"]) == expected
        assert _labels(["d\\sub\\"]) == expected


def _input_files(tmp_path: Path, count: int) -> list[InputFile]:
    files = []
    for index in range(count):
        path = tmp_path / f"f{index}.txt"
        path.write_text("x", encoding="utf-8")
        files.append(InputFile(f"f{index}.txt", path))
    return files


def test_partition_files_starts_new_batch_when_next_file_does_not_fit(tmp_path: Path) -> None:
    files = _input_files(tmp_path, 5)
    batches = partition_files(files, lambda batch: len(batch) <= 2)
    assert [[item.label for item in batch] for batch in batches] == [["f0.txt", "f1.txt"], ["f2.txt", "f3.txt"], ["f4.txt"]]


def test_partition_files_keeps_everything_in_one_batch_when_it_fits(tmp_path: Path) -> None:
    files = _input_files(tmp_path, 3)
    assert partition_files(files, lambda batch: True) == [files]
    assert partition_files([], lambda batch: True) == []


def test_partition_files_rejects_single_file_that_does_not_fit_alone(tmp_path: Path) -> None:
    files = _input_files(tmp_path, 3)
    with pytest.raises(InputError) as error:
        partition_files(files, lambda batch: all(item.label != "f1.txt" for item in batch))
    assert error.value.code == "input_too_large"
    assert "f1.txt does not fit in one job on its own" in str(error.value)


def test_partition_files_rejects_more_than_ten_batches(tmp_path: Path) -> None:
    files = _input_files(tmp_path, 11)
    assert len(partition_files(files[:10], lambda batch: len(batch) <= 1)) == 10
    with pytest.raises(InputError) as error:
        partition_files(files, lambda batch: len(batch) <= 1)
    assert error.value.code == "input_too_large"
    assert "would need 11 batches (maximum 10)" in str(error.value)
