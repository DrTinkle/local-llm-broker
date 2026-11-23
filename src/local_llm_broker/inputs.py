from __future__ import annotations

import fnmatch
import glob
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class InputError(ValueError):
    """A caller-supplied input that the broker cannot use. The message never contains absolute paths."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class InputFile:
    label: str  # POSIX label shown to the model
    path: Path


_DEFAULT_EXCLUDED_FOLDERS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".next",
        "coverage",
    }
)
_DEFAULT_EXCLUDED_FILE_PATTERNS = (
    "*.min.js",
    "*.min.css",
    "*.map",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "cargo.lock",
)
_BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".pyd",
        ".pyc",
        ".whl",
        ".gguf",
        ".sqlite3",
        ".db",
        ".bin",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp3",
        ".mp4",
        ".wav",
    }
)
_SECRET_NAME_PATTERNS = (".env", ".env.*", "auth.token", "*.pem", "*.key")
_SECRET_ALLOWED_NAMES = frozenset({".env.example", ".env.template"})
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


@dataclass(frozen=True)
class _Rules:
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    recursive: bool
    default_excludes: bool
    max_files: int


def read_text_file(path: Path, *, label: str, remaining_bytes: int | None = None) -> str:
    if not path.exists():
        raise InputError("input_not_found", f"path does not exist: {label}")
    if path.is_dir():
        raise InputError("input_is_directory", f"expected a regular file, got a directory: {label}")
    if remaining_bytes is not None and path.stat().st_size > remaining_bytes:
        raise InputError(
            "input_too_large",
            f"input limit exceeded: {label} is larger than the remaining budget of {remaining_bytes} bytes "
            "(use --include/--exclude to narrow the input)",
        )
    try:
        data = path.read_bytes()
    except PermissionError as exc:
        raise InputError("permission_denied", f"permission denied reading {label}") from exc
    except OSError as exc:
        raise InputError("input_not_found", f"cannot read {label}: path does not exist or is not readable") from exc
    if b"\x00" in data:
        raise InputError("binary_or_non_utf8_file", f"file is binary or not valid UTF-8: {label} (use --exclude to skip it)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError("binary_or_non_utf8_file", f"file is binary or not valid UTF-8: {label} (use --exclude to skip it)") from exc
    # Same newline handling as Path.read_text, so existing inputs keep their exact content.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def expand_inputs(
    items: Sequence[str],
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    recursive: bool = True,
    default_excludes: bool = True,
    max_files: int = 100,
) -> list[InputFile]:
    rules = _Rules(tuple(include), tuple(exclude), recursive, default_excludes, max_files)
    explicit: list[InputFile] = []
    expanded: list[InputFile] = []
    seen: set[Path] = set()

    for item in items:
        path = Path(item)
        if path.is_file():
            if _is_secret_name(path.name):
                raise InputError("secret_file_blocked", f"refusing to send a secret-looking file: {path.name}")
            if _claim(path, seen):
                explicit.append(InputFile(path.name, path))
        elif path.is_dir():
            _take_expansion(item, _walk_dir(path, rules), expanded, seen, rules)
        elif _has_magic(item):
            _take_expansion(item, _glob_candidates(item, rules), expanded, seen, rules)
        else:
            raise InputError("input_not_found", f"path or glob matched no files: {item}")

    expanded.sort(key=lambda f: (f.label.lower(), f.label))
    return explicit + expanded


def partition_files(
    files: list[InputFile],
    fits: Callable[[list[InputFile]], bool],
    *,
    max_batches: int = 10,
) -> list[list[InputFile]]:
    """Greedily pack files, in order, into batches that `fits` accepts. Raises when a file cannot fit alone or too many batches are needed."""
    batches: list[list[InputFile]] = []
    current: list[InputFile] = []
    for item in files:
        if fits([*current, item]):
            current.append(item)
            continue
        if current:
            batches.append(current)
        if not fits([item]):
            raise InputError("input_too_large", f"input limit exceeded: {item.label} does not fit in one job on its own")
        current = [item]
    if current:
        batches.append(current)
    if len(batches) > max_batches:
        raise InputError(
            "input_too_large",
            f"input limit exceeded: would need {len(batches)} batches (maximum {max_batches}); narrow with --include/--exclude",
        )
    return batches


def _take_expansion(
    item: str,
    candidates: list[tuple[Path, str]],
    expanded: list[InputFile],
    seen: set[Path],
    rules: _Rules,
) -> None:
    if not candidates:
        raise InputError("empty_expansion", f"directory or glob matched 0 readable files: {item} (check --include/--exclude)")
    for path, label in candidates:
        if _claim(path, seen):
            expanded.append(InputFile(label, path))
            if len(expanded) > rules.max_files:
                raise _too_many(rules)


def _too_many(rules: _Rules) -> InputError:
    return InputError(
        "too_many_files",
        f"expansion matched more than {rules.max_files} files (narrow with --include/--exclude or pass --split)",
    )


def _claim(path: Path, seen: set[Path]) -> bool:
    key = path.resolve()
    if key in seen:
        return False
    seen.add(key)
    return True


def _walk_dir(root: Path, rules: _Rules) -> list[tuple[Path, str]]:
    parent = root.parent
    found: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        if not rules.recursive:
            dirnames[:] = []
        else:
            kept: list[str] = []
            for name in dirnames:
                child = current / name
                if _is_link_or_junction(str(child)):
                    continue
                if _folder_excluded(name, _posix_label(child, parent), rules):
                    continue
                kept.append(name)
            dirnames[:] = kept
        for name in filenames:
            file_path = current / name
            if _is_link_or_junction(str(file_path)) or not file_path.is_file():
                continue
            label = _posix_label(file_path, parent)
            if not _file_passes(name, label, rules):
                continue
            found.append((file_path, label))
            if len(found) > rules.max_files:
                raise _too_many(rules)
    return found


def _glob_candidates(item: str, rules: _Rules) -> list[tuple[Path, str]]:
    files = [
        match
        for match in glob.glob(item, recursive=True)
        if os.path.isfile(match) and not _is_link_or_junction(match)
    ]
    if not files:
        raise InputError("input_not_found", f"path or glob matched no files: {item}")

    base_parts: list[str] = []
    for part in Path(item).parts:
        if _has_magic(part):
            break
        base_parts.append(part)
    base = Path(*base_parts) if base_parts else Path(".")
    label_root = base.parent

    found: list[tuple[Path, str]] = []
    for match in sorted(files):
        match_path = Path(match)
        try:
            rel_parts = match_path.relative_to(base).parts
        except ValueError:
            rel_parts = (match_path.name,)
        label = _posix_label(match_path, label_root)
        label_parts = PurePosixPath(label).parts
        folder_depth = len(rel_parts) - 1
        if any(
            _folder_excluded(
                rel_parts[i],
                PurePosixPath(*label_parts[: len(label_parts) - folder_depth + i]).as_posix(),
                rules,
            )
            for i in range(folder_depth)
        ):
            continue
        if not _file_passes(match_path.name, label, rules):
            continue
        found.append((match_path, label))
        if len(found) > rules.max_files:
            raise _too_many(rules)
    return found


def _posix_label(path: Path, parent: Path) -> str:
    try:
        return PurePosixPath(*path.relative_to(parent).parts).as_posix()
    except ValueError:
        return path.name


def _has_magic(text: str) -> bool:
    return any(char in text for char in "*?[")


def _folder_excluded(name: str, label: str, rules: _Rules) -> bool:
    if rules.default_excludes and name.lower() in _DEFAULT_EXCLUDED_FOLDERS:
        return True
    return _matches_any(rules.exclude, name, label)


def _file_passes(name: str, label: str, rules: _Rules) -> bool:
    if _is_secret_name(name):
        return False
    if rules.default_excludes and _is_default_excluded_file(name):
        return False
    if rules.include and not _matches_any(rules.include, name, label):
        return False
    return not _matches_any(rules.exclude, name, label)


def _matches_any(patterns: Sequence[str], name: str, label: str) -> bool:
    lower_name = name.lower()
    lower_label = label.lower()
    for pattern in patterns:
        lower_pattern = pattern.lower()
        if fnmatch.fnmatchcase(lower_name, lower_pattern) or fnmatch.fnmatchcase(lower_label, lower_pattern):
            return True
    return False


def _is_secret_name(name: str) -> bool:
    lower = name.lower()
    if lower in _SECRET_ALLOWED_NAMES:
        return False
    return any(fnmatch.fnmatchcase(lower, pattern) for pattern in _SECRET_NAME_PATTERNS)


def _is_default_excluded_file(name: str) -> bool:
    lower = name.lower()
    if any(fnmatch.fnmatchcase(lower, pattern) for pattern in _DEFAULT_EXCLUDED_FILE_PATTERNS):
        return True
    return Path(lower).suffix in _BINARY_EXTENSIONS


def _is_link_or_junction(path: str) -> bool:
    if os.path.islink(path):
        return True
    if os.name == "nt":
        try:
            st = os.stat(path, follow_symlinks=False)
        except OSError:
            return True
        return bool(getattr(st, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)
    return False
