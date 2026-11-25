from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from support import PRESET

from local_llm_broker.calibrate import default_profiles, resolve_models, vision_profiles
from local_llm_broker.calibrate_embeddings import (
    format_report,
    run_embeddings,
    text_of_words,
    to_json,
    usable_concurrency,
)
from local_llm_broker.config import load_config
from local_llm_broker.contracts import ProfileConfig


def server(*, max_words: int = 10_000, delay: float = 0.01, limit: int = 4, read_limit: int = 10_000) -> httpx.MockTransport:
    """A fake broker: embeddings take `delay` seconds, at most `limit` at once, longer texts than `max_words` are refused."""
    state = {"active": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if any(len(text.split()) > max_words for text in body["input"]):
            return httpx.Response(400, json={"error": {"code": "invalid_request", "message": "too long"}})
        state["active"] += 1
        try:
            # More simultaneous requests than `limit` stretch the time, so throughput stops growing.
            await asyncio.sleep(delay * max(1, state["active"] / limit))
            # The vector depends on how many words the model really read, like a model that cuts off long input.
            data = [{"index": i, "embedding": [float(min(len(t.split()), read_limit))] * 8} for i, t in enumerate(body["input"])]
            return httpx.Response(200, json={"object": "list", "model": body["model"], "data": data})
        finally:
            state["active"] -= 1

    return httpx.MockTransport(handler)


def test_text_of_words_has_the_requested_word_count() -> None:
    assert len(text_of_words(37).split()) == 37
    assert text_of_words(5).split() == text_of_words(6).split()[:5]


def test_usable_concurrency_is_the_last_step_that_still_gains() -> None:
    rows = [(1, 100.0), (2, 190.0), (4, 330.0), (8, 340.0), (16, 300.0)]
    assert usable_concurrency(rows) == 4
    assert usable_concurrency([(1, 100.0), (2, 101.0)]) == 1
    assert usable_concurrency([]) is None


@pytest.mark.asyncio
async def test_run_embeddings_reports_context_batches_and_concurrency() -> None:
    async with httpx.AsyncClient(transport=server(max_words=1200), base_url="http://broker") as client:
        report = await run_embeddings(client, "embed", max_concurrency=8, context_targets=(256, 1024, 4096), batch_sizes=(1, 4), rounds=1)
    assert report.profile == "embed"
    assert [row.ok for row in report.context] == [True, True, False]
    assert report.largest_ok_tokens == 1024
    assert [row.size for row in report.batches] == [1, 4]
    assert [row.size for row in report.concurrency] == [1, 2, 4, 8]
    assert all(row.failures == 0 for row in report.concurrency)
    assert report.usable is not None and report.usable in {1, 2, 4, 8}
    assert "embed" in format_report(report)
    assert json.loads(json.dumps(to_json(report)))["profile"] == "embed"


@pytest.mark.asyncio
async def test_run_embeddings_survives_a_failing_broker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": {"code": "model_load_failed", "message": "x"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://broker") as client:
        report = await run_embeddings(client, "embed", max_concurrency=2, context_targets=(256,), batch_sizes=(1,), rounds=1)
    assert report.error is not None and "model_load_failed" in report.error
    assert report.usable is None


def test_chat_calibration_skips_embedding_profiles() -> None:
    config = load_config(PRESET)
    profiles = dict(config.profiles)
    profiles["embed"] = ProfileConfig("embed", "embed-model", 2048, 4, kind="embedding")
    config = replace(config, profiles=profiles)
    assert "embed" not in default_profiles(config)
    assert "embed" not in vision_profiles(config)
    assert "embed-model" not in resolve_models(config, None)


@pytest.mark.asyncio
async def test_run_embeddings_notices_input_the_model_cuts_off() -> None:
    async with httpx.AsyncClient(transport=server(read_limit=700), base_url="http://broker") as client:
        report = await run_embeddings(client, "embed", max_concurrency=1, context_targets=(256, 512, 1024, 2048, 4096), batch_sizes=(1,), rounds=1)
    assert [row.cut_off for row in report.context] == [False, False, False, True, True]
    assert report.read_limit == (512, 1024)
    assert "cut off" in format_report(report)
    assert to_json(report)["read_limit_tokens"] == [512, 1024]


@pytest.mark.asyncio
async def test_run_embeddings_reports_no_cut_off_when_everything_is_read() -> None:
    async with httpx.AsyncClient(transport=server(), base_url="http://broker") as client:
        report = await run_embeddings(client, "embed", max_concurrency=1, context_targets=(256, 1024), batch_sizes=(1,), rounds=1)
    assert report.read_limit is None
