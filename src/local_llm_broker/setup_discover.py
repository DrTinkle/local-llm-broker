"""Check that LM Studio is reachable, read the hardware, and list the chat models that are downloaded."""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from local_llm_broker.contracts import BrokerConfig
from local_llm_broker.lms_cli import find_lms

_MOE_PARAMS = re.compile(r"^\d+(\.\d+)?B-A\d+(\.\d+)?B$")


@dataclass(frozen=True, slots=True)
class ModelInfo:
    key: str
    display_name: str
    architecture: str | None
    params: str | None
    size_bytes: int
    max_context: int
    vision: bool
    is_moe: bool
    variant: str | None


@dataclass(frozen=True, slots=True)
class Hardware:
    gpu_name: str | None
    vram_mib: int | None
    ram_mib: int


@dataclass(frozen=True, slots=True)
class Preflight:
    lmstudio_ok: bool
    lms_path: str | None
    broker_running: bool
    problems: list[str]


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _is_moe(params: str | None, architecture: str | None) -> bool:
    if params is not None and _MOE_PARAMS.match(params):
        return True
    return architecture is not None and architecture.lower().endswith("moe")


def _model_info(entry: dict[str, Any]) -> ModelInfo | None:
    key = _text(entry.get("key"))
    if key is None:
        return None
    architecture = _text(entry.get("architecture"))
    params = _text(entry.get("params_string"))
    capabilities = entry.get("capabilities")
    vision = isinstance(capabilities, dict) and capabilities.get("vision") is True
    return ModelInfo(
        key=key,
        display_name=_text(entry.get("display_name")) or key,
        architecture=architecture,
        params=params,
        size_bytes=_count(entry.get("size_bytes")),
        max_context=_count(entry.get("max_context_length")),
        vision=vision,
        is_moe=_is_moe(params, architecture),
        variant=_text(entry.get("selected_variant")),
    )


def discover_models(listing: dict[str, Any]) -> list[ModelInfo]:
    """The downloaded chat models (embeddings are skipped), smallest first."""
    entries = listing.get("models")
    if not isinstance(entries, list):
        return []
    found: list[ModelInfo] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "llm":
            continue
        info = _model_info(entry)
        if info is not None:
            found.append(info)
    return sorted(found, key=lambda m: (m.size_bytes, m.key))


def _read_nvidia_smi() -> tuple[str, int] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        name, total = (part.strip() for part in out.splitlines()[0].rsplit(",", 1))
        if not name:
            return None
        return name, int(total)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _read_ram_mib() -> int:
    """Total physical memory in MiB, or 0 when it cannot be read."""
    if not hasattr(ctypes, "windll"):
        return 0
    try:
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 0
        return int(status.ullTotalPhys // (1024 * 1024))
    except (OSError, AttributeError):
        return 0


def detect_hardware(
    nvidia_smi: Callable[[], tuple[str, int] | None] = _read_nvidia_smi,
    ram_reader: Callable[[], int] = _read_ram_mib,
) -> Hardware:
    gpu = nvidia_smi()
    return Hardware(
        gpu_name=gpu[0] if gpu else None,
        vram_mib=gpu[1] if gpu else None,
        ram_mib=ram_reader(),
    )


def _not_answering(config: BrokerConfig) -> str:
    return f"LM Studio is not answering at {config.backend_url}; open LM Studio and start the local server"


async def fetch_listing(config: BrokerConfig, client: httpx.AsyncClient) -> dict[str, Any]:
    """LM Studio's model listing, or RuntimeError when the server does not answer."""
    try:
        response = await client.get(f"{config.backend_url.rstrip('/')}/api/v1/models")
        if response.status_code != 200:
            raise RuntimeError(_not_answering(config))
        listing = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise RuntimeError(_not_answering(config)) from error
    return listing if isinstance(listing, dict) else {}


async def preflight(config: BrokerConfig, client: httpx.AsyncClient) -> Preflight:
    problems: list[str] = []
    try:
        await fetch_listing(config, client)
        lmstudio_ok = True
    except RuntimeError as error:
        lmstudio_ok = False
        problems.append(str(error))

    lms_path = find_lms(config.lms_path)
    if lms_path is None:
        problems.append(
            "The LM Studio command-line tool (lms) was not found; install it from LM Studio or set lms_path in config.toml"
        )

    try:
        await client.get(f"{config.service_url}/health", timeout=2.0)
        broker_running = True
    except httpx.HTTPError:
        broker_running = False

    return Preflight(lmstudio_ok=lmstudio_ok, lms_path=lms_path, broker_running=broker_running, problems=problems)
