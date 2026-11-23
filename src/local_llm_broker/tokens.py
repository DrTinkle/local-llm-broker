"""Token estimates for jobs.

The estimate picks a job's tier at submit. Before a job runs, the broker replaces the text part with an exact count
from the loaded model's tokenizer (see Scheduler); image and template overheads stay estimated.

Rates are tokens per character, fitted on 2026-11-04 to real counts from the Qwen3 4B and Qwen3.5 9B tokenizers for
English and Finnish prose, Python, JSON, CSV numbers, log lines and a Markdown table, taking the higher count of the
two tokenizers. With the 15% margin, every sample is overestimated (by 7 to 23%).
"""

from __future__ import annotations

import base64
import json
import math
import re
import struct
from typing import Any

LETTER = 0.124  # ASCII letters in mostly-English text (about 5.3 bytes per token for prose)
NON_ENGLISH_LETTER = 0.344  # ASCII letters when the text has non-ASCII letters (Finnish prose: about 2.7 bytes/token)
DIGIT = 1.109  # each digit is about one token
SPACE = 0.534
PUNCT = 0.711
NON_ASCII = 1.0
NON_ENGLISH_SHARE = 0.01  # this share of non-ASCII characters among letters switches to NON_ENGLISH_LETTER
SAFETY = 1.15

# Chat-template tokens around each message, and once per request. Tool definitions also add a fixed preamble.
MESSAGE_OVERHEAD = 8
REQUEST_OVERHEAD = 32
TOOLS_OVERHEAD = 200

# Qwen3-VL and Qwen3.5/3.6 vision: one token per 32 x 32 pixels, plus 2. Measured 66 / 258 / 1,026 / 2,306 tokens
# for 256 / 512 / 1,024 / 1,536 pixel squares. An image whose size cannot be read counts as this many.
IMAGE_PATCH = 32
IMAGE_EXTRA = 2
UNKNOWN_IMAGE_TOKENS = 4_096

_LETTERS = re.compile(r"[A-Za-z]")
_DIGITS = re.compile(r"[0-9]")
_SPACES = re.compile(r"[ \t\r\n\f\v]")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    letters = len(_LETTERS.findall(text))
    digits = len(_DIGITS.findall(text))
    spaces = len(_SPACES.findall(text))
    non_ascii = len(_NON_ASCII.findall(text))
    punct = max(0, len(text) - letters - digits - spaces - non_ascii)
    letter_rate = NON_ENGLISH_LETTER if non_ascii >= NON_ENGLISH_SHARE * max(1, letters + non_ascii) else LETTER
    raw = letter_rate * letters + DIGIT * digits + SPACE * spaces + PUNCT * punct + NON_ASCII * non_ascii
    return max(1, math.ceil(SAFETY * raw))


def _image_size(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        return width, height
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and len(data) >= 30:
        kind = data[12:16]
        if kind == b"VP8X":
            return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
        if kind == b"VP8L" and data[20] == 0x2F:
            bits = int.from_bytes(data[21:25], "little")
            return 1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF)
        if kind == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            width, height = struct.unpack("<HH", data[26:30])
            return width & 0x3FFF, height & 0x3FFF
        return None
    if data[:2] == b"\xff\xd8":
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                index += 2
                continue
            length = struct.unpack(">H", data[index + 2 : index + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", data[index + 5 : index + 9])
                return width, height
            index += 2 + length
    return None


def image_tokens(url: str) -> int:
    """Tokens for one inline base64 image, from its pixel size."""
    try:
        _, encoded = url.split(";base64,", 1)
        # The header is near the start; JPEG size markers can come later, so read up to 256 KiB.
        head = base64.b64decode(encoded[: 4 * 87_382], validate=False)
    except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
        return UNKNOWN_IMAGE_TOKENS
    size = _image_size(head)
    if size is None or size[0] <= 0 or size[1] <= 0:
        return UNKNOWN_IMAGE_TOKENS
    return math.ceil(size[0] / IMAGE_PATCH) * math.ceil(size[1] / IMAGE_PATCH) + IMAGE_EXTRA


def job_texts(messages: list[dict[str, Any]], generation: dict[str, Any]) -> list[str]:
    """Every text the model reads for a job: message texts, then tool definitions as JSON."""
    texts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str))
    if generation.get("tools"):
        texts.append(json.dumps(generation["tools"], ensure_ascii=False))
    return texts


def fixed_overhead(messages: list[dict[str, Any]], generation: dict[str, Any]) -> int:
    """Template and image tokens, which the text count does not include."""
    tokens = REQUEST_OVERHEAD + MESSAGE_OVERHEAD * len(messages)
    if generation.get("tools"):
        tokens += TOOLS_OVERHEAD
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    tokens += image_tokens(url) if isinstance(url, str) else UNKNOWN_IMAGE_TOKENS
    return tokens


def estimate_prompt_tokens(messages: list[dict[str, Any]], generation: dict[str, Any]) -> int:
    return sum(estimate_text_tokens(text) for text in job_texts(messages, generation)) + fixed_overhead(messages, generation)
