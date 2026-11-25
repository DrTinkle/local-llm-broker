"""Measure an embedding profile through the broker: input length, batch size and concurrency.

Three sweeps, each over `POST /v1/embeddings`:
- Context: one request with a text of growing length. Models cut off input beyond their window instead of refusing it,
  so the sweep also compares vectors: a longer text with the same vector as the one before it was cut off.
- Batch: one request carrying 1, 4, 16, 64 or 128 short texts, to see how many texts per second a batch reaches.
- Concurrency: 1, 2, 4, 8 or 16 requests at once, each with a small batch. The usable count is the largest one whose
  combined speed is still clearly higher than with fewer requests, the same rule the chat concurrency test uses.

LM Studio reports no token counts for embeddings, so lengths are given in words and an estimated token count
(about 1.3 tokens per English word).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .calibrate_concurrency import USEFUL_GAIN

TOKENS_PER_WORD = 1.3
CONTEXT_TARGETS = (256, 512, 1024, 1536, 2048, 2560, 3072, 4096, 8192)
BATCH_SIZES = (1, 4, 16, 64, 128)
CONCURRENCY_LEVELS = (1, 2, 4, 8, 16)
BATCH_TEXT_WORDS = 100  # about 130 tokens
CONCURRENCY_BATCH = 8
REQUEST_TIMEOUT = 300.0

_WORDS = (
    "river stone market window engine garden paper signal harbor copper winter lantern bridge orchard "
    "thunder pencil meadow compass anchor velvet puzzle ladder crystal falcon ribbon summit tunnel whistle"
).split()


def text_of_words(count: int, offset: int = 0) -> str:
    """Deterministic prose-like text of exactly `count` words; `offset` shifts the starting word."""
    return " ".join(_WORDS[(offset + index) % len(_WORDS)] for index in range(count))


@dataclass(slots=True)
class ContextRow:
    tokens: int
    words: int
    ok: bool
    seconds: float
    error: str | None = None
    cut_off: bool = False  # same vector as the shorter text before it: the extra text was ignored


@dataclass(slots=True)
class SweepRow:
    size: int  # texts per request for the batch sweep, simultaneous requests for the concurrency sweep
    texts: int
    seconds: float
    failures: int

    @property
    def texts_per_second(self) -> float:
        return self.texts / self.seconds if self.seconds > 0 else 0.0


@dataclass(slots=True)
class EmbeddingReport:
    profile: str
    context: list[ContextRow] = field(default_factory=list)
    batches: list[SweepRow] = field(default_factory=list)
    concurrency: list[SweepRow] = field(default_factory=list)
    error: str | None = None

    @property
    def largest_ok_tokens(self) -> int | None:
        ok = [row.tokens for row in self.context if row.ok]
        return max(ok) if ok else None

    @property
    def read_limit(self) -> tuple[int, int] | None:
        """Estimated-token range holding the model's input window, or None if no text was cut off.

        The first cut-off row has the same vector as the row before it, so the window ends at or before that earlier
        row and after the row before that one.
        """
        for index, row in enumerate(self.context):
            if row.cut_off:
                lower = self.context[index - 2].tokens if index >= 2 else 0
                return lower, self.context[index - 1].tokens
        return None

    @property
    def usable(self) -> int | None:
        return usable_concurrency([(row.size, row.texts_per_second) for row in self.concurrency if row.failures == 0])


def usable_concurrency(rows: list[tuple[int, float]]) -> int | None:
    """The largest level reached by steps that each gained at least USEFUL_GAIN over the one before."""
    if not rows:
        return None
    best_level, best_speed = rows[0]
    for level, speed in rows[1:]:
        if speed >= best_speed * USEFUL_GAIN:
            best_level, best_speed = level, speed
        else:
            break
    return best_level


class _Failure(Exception):
    pass


async def _embed(client: httpx.AsyncClient, profile: str, texts: list[str]) -> list[Any]:
    try:
        response = await client.post("/v1/embeddings", json={"model": profile, "input": texts}, timeout=REQUEST_TIMEOUT)
    except httpx.HTTPError as exc:
        raise _Failure("broker_unavailable") from exc
    if response.status_code >= 400:
        try:
            code = str(response.json().get("error", {}).get("code", "http_error"))
        except ValueError:
            code = "http_error"
        raise _Failure(code)
    data = response.json().get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise _Failure("wrong_number_of_vectors")
    return [row.get("embedding") if isinstance(row, dict) else None for row in data]


async def run_embeddings(
    client: httpx.AsyncClient,
    profile: str,
    *,
    max_concurrency: int = CONCURRENCY_LEVELS[-1],
    context_targets: tuple[int, ...] = CONTEXT_TARGETS,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    rounds: int = 3,
) -> EmbeddingReport:
    report = EmbeddingReport(profile)
    try:
        await _embed(client, profile, ["warm-up"])  # loads the model, so no sweep pays for it
        previous: Any = None
        for target in context_targets:
            words = max(1, round(target / TOKENS_PER_WORD))
            started = time.monotonic()
            try:
                vector = (await _embed(client, profile, [text_of_words(words)]))[0]
                report.context.append(
                    ContextRow(target, words, True, time.monotonic() - started, cut_off=previous is not None and vector == previous)
                )
                previous = vector
            except _Failure as exc:
                report.context.append(ContextRow(target, words, False, time.monotonic() - started, str(exc)))
        for size in batch_sizes:
            texts = [text_of_words(BATCH_TEXT_WORDS, offset=i) for i in range(size)]
            started = time.monotonic()
            await _embed(client, profile, texts)
            report.batches.append(SweepRow(size, size, time.monotonic() - started, 0))
        level = 1
        while level <= max_concurrency:
            report.concurrency.append(await _concurrent_round(client, profile, level, rounds))
            level *= 2
    except _Failure as exc:
        report.error = str(exc)
    return report


async def _concurrent_round(client: httpx.AsyncClient, profile: str, level: int, rounds: int) -> SweepRow:
    texts = [text_of_words(BATCH_TEXT_WORDS, offset=i) for i in range(CONCURRENCY_BATCH)]
    failures = 0
    started = time.monotonic()
    for _ in range(rounds):
        results = await asyncio.gather(*[_embed(client, profile, texts) for _ in range(level)], return_exceptions=True)
        failures += sum(1 for result in results if isinstance(result, BaseException))
    seconds = time.monotonic() - started
    return SweepRow(level, level * CONCURRENCY_BATCH * rounds, seconds, failures)


def format_report(report: EmbeddingReport) -> str:
    lines = [f"Embedding profile: {report.profile}"]
    if report.error:
        lines.append(f"  stopped early: {report.error}")
    if report.context:
        lines.append("Input length (one text per request)")
        for probe in report.context:
            result = ("ok, cut off" if probe.cut_off else "ok") if probe.ok else f"refused ({probe.error})"
            lines.append(f"  ~{probe.tokens:>5} tokens ({probe.words:>4} words)  {result}  {probe.seconds:.2f} s")
        largest = report.largest_ok_tokens
        lines.append(f"  longest accepted: {'~' + format(largest, ',') + ' tokens' if largest else 'none'}")
        window = report.read_limit
        if window is not None:
            lines.append(f"  longer text is cut off, not refused: the model reads about {window[0]:,} to {window[1]:,} tokens (estimated)")
    if report.batches:
        lines.append("Batch size (one request, ~130-token texts)")
        for batch in report.batches:
            lines.append(f"  {batch.size:>4} texts  {batch.seconds:.2f} s  {batch.texts_per_second:.1f} texts/s")
    if report.concurrency:
        lines.append(f"Concurrency (requests at once, {CONCURRENCY_BATCH} texts each)")
        for level in report.concurrency:
            failed = f"  {level.failures} failed" if level.failures else ""
            lines.append(f"  {level.size:>4} at once  {level.texts_per_second:.1f} texts/s{failed}")
        lines.append(f"  usable concurrency: {report.usable if report.usable is not None else 'unknown'}")
    return "\n".join(lines)


def to_json(report: EmbeddingReport) -> dict[str, Any]:
    return {
        "profile": report.profile,
        "error": report.error,
        "longest_accepted_tokens": report.largest_ok_tokens,
        "read_limit_tokens": list(report.read_limit) if report.read_limit else None,
        "usable_concurrency": report.usable,
        "context": [
            {"tokens": r.tokens, "words": r.words, "ok": r.ok, "cut_off": r.cut_off, "seconds": round(r.seconds, 3), "error": r.error}
            for r in report.context
        ],
        "batches": [
            {"texts": r.size, "seconds": round(r.seconds, 3), "texts_per_second": round(r.texts_per_second, 1)}
            for r in report.batches
        ],
        "concurrency": [
            {"requests": r.size, "failures": r.failures, "texts_per_second": round(r.texts_per_second, 1)}
            for r in report.concurrency
        ],
    }
