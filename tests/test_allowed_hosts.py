from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from support import PRESET

from local_llm_broker.app import create_app
from local_llm_broker.config import load_config
from local_llm_broker.contracts import BrokerConfig


class Backend:
    async def ensure_loaded(self, profile: str, **_: object) -> Any:
        raise AssertionError("not used")


def make_app(tmp_path: Path, allowed: tuple[str, ...] = ()) -> Any:
    config: BrokerConfig = replace(load_config(PRESET), data_dir=tmp_path, auth_token_path=tmp_path / "auth.token", allowed_hosts=allowed)
    return create_app(config, adapter=Backend())


async def get(app: Any, host: str, *, token: bool = True, path: str = "/status") -> httpx.Response:
    headers = {"Host": host}
    if token:
        headers["Authorization"] = f"Bearer {app.state.auth_token}"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:1235") as client:
        return await client.get(path, headers=headers)


@pytest.mark.asyncio
async def test_a_docker_host_name_is_rejected_by_default(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    await app.state.scheduler.start()
    try:
        response = await get(app, "host.docker.internal:1235")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "loopback_only"
    finally:
        await app.state.scheduler.shutdown()


@pytest.mark.asyncio
async def test_a_listed_host_name_works_with_the_token_only(tmp_path: Path) -> None:
    app = make_app(tmp_path, ("host.docker.internal",))
    await app.state.scheduler.start()
    try:
        assert (await get(app, "host.docker.internal:1235")).status_code == 200
        assert (await get(app, "host.docker.internal", path="/status")).status_code == 200
        assert (await get(app, "host.docker.internal:1235", token=False)).status_code == 401
        assert (await get(app, "evil.example:1235")).status_code == 400
        assert (await get(app, "localhost:1235")).status_code == 200
    finally:
        await app.state.scheduler.shutdown()


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_allowed_hosts_is_empty_by_default_and_read_from_the_broker_section(tmp_path: Path) -> None:
    assert load_config(tmp_path / "missing.toml").allowed_hosts == ()
    config = load_config(write(tmp_path, '[broker]\nallowed_hosts = ["host.docker.internal", "Box.local"]\n'))
    assert config.allowed_hosts == ("host.docker.internal", "box.local")


@pytest.mark.parametrize("value", ['"host.docker.internal"', "[1]", '[""]', '["http://x"]', '["a b"]', '["x:80"]', '["*"]'])
def test_allowed_hosts_must_be_plain_host_names(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="allowed_hosts"):
        load_config(write(tmp_path, f"[broker]\nallowed_hosts = {value}\n"))
