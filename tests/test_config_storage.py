from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

import pytest
from support import PRESET

from local_llm_broker.config import load_config
from local_llm_broker.contracts import JobSpec, JobState
from local_llm_broker.instance_lock import BackendOwnerLock, LockHeldError
from local_llm_broker.storage import IdempotencyConflict, MetadataStore

_T = TypeVar("_T")


def present(value: _T | None) -> _T:
    assert value is not None
    return value


def test_default_config_has_loopback_endpoints_finite_limits_and_known_profiles() -> None:
    config = load_config(PRESET)

    assert config.service_url == "http://127.0.0.1:1235"
    assert config.backend_url == "http://127.0.0.1:1234"
    assert config.queue_limit == 100
    assert config.max_body_bytes == 10 * 1024 * 1024
    assert config.profiles["4b"].parallel_ceiling == 4
    # A tiered profile accepts jobs up to its largest tier.
    assert config.profiles["4b"].context_ceiling == 81_920
    assert config.profiles["9b"].context_ceiling == 163_840
    assert (config.profiles["9b"].parallel_ceiling, config.profiles["35b"].parallel_ceiling) == (3, 1)
    assert config.profiles["35b"].model_id == "qwen/qwen3.6-35b-a3b"
    assert config.profiles["35b"].context_ceiling == 262_144
    assert [tier.context_length for tier in config.tiers_for("35b").values()] == [32_768, 131_072, 262_144]


def test_missing_config_has_no_profiles(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml")

    assert config.profiles == {}
    assert config.tiers == {}


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_lms_path_is_optional_and_read_from_broker_section(tmp_path: Path) -> None:
    assert load_config(tmp_path / "missing.toml").lms_path is None
    path = write_config(tmp_path, '[broker]\nlms_path = "D:/LM Studio/lms.exe"\n')
    assert load_config(path).lms_path == "D:/LM Studio/lms.exe"
    with pytest.raises(ValueError, match="lms_path"):
        load_config(write_config(tmp_path, "[broker]\nlms_path = 3\n"))


def test_config_defines_any_profile_alias(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        '[profiles.my-model]\nmodel_id = "org/m"\nparallel_ceiling = 2\nreasoning_effort = "none"\n'
        'supports_vision = true\nthinks = false\ndescription = "d"\n',
    )

    profile = load_config(path).profiles["my-model"]

    assert (profile.alias, profile.model_id, profile.parallel_ceiling) == ("my-model", "org/m", 2)
    assert profile.reasoning_effort == "none"
    assert profile.supports_vision is True
    assert profile.thinks is False
    assert profile.description == "d"
    assert profile.context_ceiling == 8192
    assert profile.variant is None


def test_new_profile_requires_model_id(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[profiles.x]\nparallel_ceiling = 1\n")

    with pytest.raises(ValueError, match="profiles.x.model_id is required"):
        load_config(path)


def test_profile_alias_must_match_pattern(tmp_path: Path) -> None:
    path = write_config(tmp_path, '[profiles."Bad Name"]\nmodel_id = "org/m"\n')

    with pytest.raises(ValueError, match="alias"):
        load_config(path)


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high"])
def test_reasoning_effort_values(tmp_path: Path, effort: str) -> None:
    path = write_config(tmp_path, f'[profiles.x]\nmodel_id = "org/m"\nreasoning_effort = "{effort}"\n')

    assert load_config(path).profiles["x"].reasoning_effort == effort


def test_reasoning_effort_rejects_other_values(tmp_path: Path) -> None:
    path = write_config(tmp_path, '[profiles.x]\nmodel_id = "org/m"\nreasoning_effort = "off"\n')

    with pytest.raises(ValueError, match="reasoning_effort"):
        load_config(path)


def test_profile_rejects_unknown_settings(tmp_path: Path) -> None:
    path = write_config(tmp_path, '[profiles.x]\nmodel_id = "org/m"\nbogus = 1\n')

    with pytest.raises(ValueError, match="profiles.x: unknown settings bogus"):
        load_config(path)


def test_preset_reproduces_the_old_builtin_profiles() -> None:
    config = load_config(PRESET)
    aliases = ("4b", "4b-nothink", "9b", "9b-nothink", "35b", "35b-nothink")

    assert set(config.profiles) == set(aliases)
    assert config.profiles["9b-nothink"].reasoning_effort == "none"
    assert config.profiles["9b"].supports_vision is True
    assert config.profiles["4b"].thinks is True
    assert config.profiles["4b-nothink"].thinks is False
    assert all(0 < len(profile.description) <= 200 for profile in config.profiles.values())
    high = config.tiers["qwen/qwen3.6-35b-a3b"]["high"]
    assert (high.context_length, high.kv_cache, high.cpu_expert_ratio, high.try_mmap) == (262_144, "q8_0", 0.775, False)
    assert [config.profiles[alias].parallel_ceiling for alias in aliases] == [4, 4, 3, 3, 1, 1]


def test_example_config_has_no_profiles_and_no_machine_specifics() -> None:
    path = Path(__file__).resolve().parents[1] / "config.example.toml"

    assert load_config(path).profiles == {}
    text = path.read_text(encoding="utf-8").lower()
    for word in ("qwen", "3080", "12 gb", "35b", "9b", "4b"):
        assert word not in text


def test_non_thinking_4b_is_its_own_profile_and_leaves_the_thinking_4b_unchanged() -> None:
    config = load_config(PRESET)

    assert config.profiles["4b"].model_id == "qwen/qwen3-4b-thinking-2507"
    assert config.profiles["4b-nothink"].model_id == "qwen/qwen3-4b-2507"
    assert config.profiles["4b-nothink"].variant == "qwen/qwen3-4b-2507@q4_k_m"
    assert config.profiles["4b-nothink"].context_ceiling == config.profiles["4b"].context_ceiling
    assert config.profiles["4b-nothink"].parallel_ceiling == config.profiles["4b"].parallel_ceiling


def test_non_thinking_9b_and_35b_use_the_same_model_and_switch_thinking_off() -> None:
    config = load_config(PRESET)

    for thinking, plain in (("9b", "9b-nothink"), ("35b", "35b-nothink")):
        assert config.profiles[plain].model_id == config.profiles[thinking].model_id
        assert config.profiles[plain].variant == config.profiles[thinking].variant
        assert config.profiles[plain].context_ceiling == config.profiles[thinking].context_ceiling
        assert config.profiles[plain].parallel_ceiling == config.profiles[thinking].parallel_ceiling
        assert config.profiles[plain].reasoning_effort == "none"
        assert config.profiles[thinking].reasoning_effort is None


def test_config_file_override_keeps_the_reasoning_effort_setting(tmp_path: Path) -> None:
    path = write_config(tmp_path, '[profiles.9b-nothink]\nmodel_id = "m"\nparallel_ceiling = 1\nreasoning_effort = "none"\n')

    config = load_config(path)

    assert config.profiles["9b-nothink"].parallel_ceiling == 1
    assert config.profiles["9b-nothink"].reasoning_effort == "none"


def test_vision_is_enabled_only_for_the_9b_and_35b_profiles() -> None:
    config = load_config(PRESET)

    assert {alias for alias, profile in config.profiles.items() if profile.supports_vision} == {
        "9b",
        "9b-nothink",
        "35b",
        "35b-nothink",
    }
    assert config.profiles["4b"].supports_vision is False
    assert config.profiles["4b-nothink"].supports_vision is False


def test_config_file_override_keeps_the_vision_setting(tmp_path: Path) -> None:
    path = write_config(tmp_path, '[profiles.9b]\nmodel_id = "m"\nparallel_ceiling = 1\nsupports_vision = true\n')

    config = load_config(path)

    assert config.profiles["9b"].parallel_ceiling == 1
    assert config.profiles["9b"].supports_vision is True


def test_preset_declares_the_non_thinking_4b_profile() -> None:
    preset = load_config(PRESET)

    assert preset.profiles["4b-nothink"].model_id == "qwen/qwen3-4b-2507"
    assert preset.max_body_bytes == 10 * 1024 * 1024


def test_config_rejects_non_loopback_backend(tmp_path: Path) -> None:
    path = write_config(tmp_path, '[broker]\nbackend_url = "http://example.test:1234"\n')

    with pytest.raises(ValueError, match="loopback"):
        load_config(path)


def test_config_rejects_non_loopback_service_host(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[broker]\nservice_host = "0.0.0.0"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="loopback"):
        load_config(path)


def test_storage_persists_metadata_but_never_payload_or_result(tmp_path: Path) -> None:
    digest_key = b"protected-test-key"
    store = MetadataStore(tmp_path / "state.sqlite3", digest_key=digest_key)
    spec = JobSpec(
        client_id="client-a",
        project="demo",
        task_label="summarize",
        profile="4b",
        messages=[{"role": "user", "content": "SENTINEL PROMPT"}],
        generation={"max_tokens": 32},
    )

    snapshot = store.create_job("job-1", spec, idempotency_key="same-key")
    store.transition("job-1", JobState.RUNNING)
    store.transition("job-1", JobState.SUCCEEDED, result_expired=False)

    assert snapshot.job_id == "job-1"
    raw = sqlite3.connect(store.path).execute("SELECT * FROM jobs").fetchone()
    assert raw is not None
    assert "SENTINEL PROMPT" not in repr(raw)
    assert "ANSWER BODY" not in repr(raw)
    assert present(store.get_job("job-1")).state is JobState.SUCCEEDED
    digest = sqlite3.connect(store.path).execute("SELECT payload_digest FROM idempotency").fetchone()[0]
    assert digest == spec.payload_digest(digest_key)


def test_idempotency_reuse_conflict_and_expiry(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "state.sqlite3", idempotency_ttl=timedelta(hours=24))
    first = JobSpec("client-a", "p", "t", "4b", [{"role": "user", "content": "a"}], {})
    second = JobSpec("client-a", "p", "t", "4b", [{"role": "user", "content": "b"}], {})

    store.create_job("job-1", first, idempotency_key="key")
    assert present(store.get_idempotency("client-a", "key", first)).job_id == "job-1"
    with pytest.raises(IdempotencyConflict):
        store.create_job("job-2", second, idempotency_key="key")

    old = datetime.now(UTC) - timedelta(days=2)
    store.set_idempotency_created_at("client-a", "key", old)
    assert store.get_idempotency("client-a", "key", first) is None


def test_backend_owner_lock_rejects_second_owner_even_for_another_data_dir(tmp_path: Path) -> None:
    common = tmp_path / "backend.lock"
    first = BackendOwnerLock(tmp_path / "one.lock", common_path=common)
    second = BackendOwnerLock(tmp_path / "two.lock", common_path=common)
    first.acquire()
    try:
        with pytest.raises(LockHeldError):
            second.acquire()
    finally:
        first.release()


def test_prune_deletes_old_terminal_job_with_its_rows_and_keeps_newer_jobs(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "prune.sqlite3")
    old_spec = JobSpec("client-a", "p", "t", "4b", [{"role": "user", "content": "old"}], {})
    new_spec = JobSpec("client-a", "p", "t", "4b", [{"role": "user", "content": "new"}], {})
    store.create_job("job-old", old_spec, idempotency_key="old-key")
    store.transition("job-old", JobState.SUCCEEDED)
    store.create_job("job-new", new_spec)
    store.transition("job-new", JobState.SUCCEEDED)

    connection = sqlite3.connect(store.path)
    connection.execute(
        "UPDATE jobs SET updated_at=? WHERE job_id=?",
        ((datetime.now(UTC) - timedelta(days=60)).isoformat(), "job-old"),
    )
    connection.commit()
    connection.close()

    removed = store.prune(datetime.now(UTC) - timedelta(days=30))

    assert removed == 1
    assert store.get_job("job-old") is None
    assert store.get_job("job-new") is not None
    connection = sqlite3.connect(store.path)
    try:
        for table in ("transitions", "idempotency"):
            count = connection.execute(f"SELECT COUNT(*) FROM {table} WHERE job_id=?", ("job-old",)).fetchone()[0]
            assert count == 0
        assert connection.execute("SELECT COUNT(*) FROM transitions WHERE job_id=?", ("job-new",)).fetchone()[0] == 2
    finally:
        connection.close()

