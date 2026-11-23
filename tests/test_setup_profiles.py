from __future__ import annotations

import pytest

from local_llm_broker.config import ALIAS_PATTERN
from local_llm_broker.contracts import KV_CACHE_TYPES, TIER_ORDER
from local_llm_broker.setup_discover import Hardware, ModelInfo
from local_llm_broker.setup_profiles import (
    ProfileChoice,
    default_parallel,
    guess_thinks,
    profiles_from_choices,
    quick_tiers,
    suggest_alias,
    tiers_for_choices,
)

GB = 1_000_000_000


def model(
    key: str = "qwen/qwen3.5-9b",
    size_gb: float = 6.5,
    max_context: int = 262144,
    moe: bool = False,
    vision: bool = False,
    variant: str | None = None,
) -> ModelInfo:
    return ModelInfo(
        key=key,
        display_name=key,
        architecture=None,
        params=None,
        size_bytes=int(size_gb * GB),
        max_context=max_context,
        vision=vision,
        is_moe=moe,
        variant=variant,
    )


def hardware(vram_mib: int | None = 12288) -> Hardware:
    return Hardware(gpu_name="Test GPU", vram_mib=vram_mib, ram_mib=32768)


def test_suggest_alias() -> None:
    assert suggest_alias(model("qwen/qwen3.5-9b"), set()) == "qwen3.5-9b"
    assert suggest_alias(model("qwen/qwen3.5-9b"), {"qwen3.5-9b"}) == "qwen3.5-9b-2"
    assert suggest_alias(model("qwen/qwen3.5-9b"), {"qwen3.5-9b", "qwen3.5-9b-2"}) == "qwen3.5-9b-3"
    assert suggest_alias(model("Org/My Model!!"), set()) == "my-model--"

    nasty = [
        "Org/My Model!!",
        "a" * 100,
        "org/" + "b" * 100,
        "no-slash-key",
        "ends/with/",
        "ends-with-slash/",
        "org/___",
        "org/.hidden",
        "org/-dash",
        "org/Ünïcode näme",
        "",
        "/",
    ]
    for key in nasty:
        alias = suggest_alias(model(key), set())
        assert ALIAS_PATTERN.match(alias), (key, alias)
        taken = {alias}
        second = suggest_alias(model(key), taken)
        assert second not in taken
        assert ALIAS_PATTERN.match(second), (key, second)

    long_taken = {suggest_alias(model("a" * 100), set())}
    assert len(long_taken.copy().pop()) == 40
    again = suggest_alias(model("a" * 100), long_taken)
    assert len(again) <= 40 and again.endswith("-2")
    assert suggest_alias(model("ends/with/"), set()) == "model"


def test_guess_thinks() -> None:
    assert guess_thinks(model("qwen/Qwen3-4B-Thinking-2507"))
    assert guess_thinks(model("deepseek/deepseek-r1-distill"))
    assert guess_thinks(model("qwen/QwQ-32B"))
    assert guess_thinks(model("org/some-reasoning-model"))
    assert not guess_thinks(model("qwen/qwen3.5-9b"))
    assert not guess_thinks(model("meta/llama-3.1-8b-instruct"))


def test_default_parallel() -> None:
    hw = hardware(12288)
    assert default_parallel(model(size_gb=2.5), hw) == 4
    assert default_parallel(model(size_gb=6.5), hw) == 2
    assert default_parallel(model(size_gb=22), hw) == 1
    assert default_parallel(model(size_gb=2.5), hardware(None)) == 1
    assert default_parallel(model(size_gb=2.5), hardware(0)) == 1


def test_quick_tiers_are_conservative() -> None:
    hw = hardware(12288)

    fits = quick_tiers(model(size_gb=6.5), hw)
    assert list(fits) == ["low", "medium"]
    assert fits["low"].context_length == 8192
    assert fits["medium"].context_length == 16384
    assert fits["low"].kv_cache == "f16" and fits["medium"].kv_cache == "f16"
    assert "high" not in fits

    small_context = quick_tiers(model(size_gb=2.5, max_context=4096), hw)
    assert list(small_context) == ["low"]
    assert small_context["low"].context_length == 4096

    equal = quick_tiers(model(size_gb=2.5, max_context=8192), hw)
    assert list(equal) == ["low"]
    assert equal["low"].context_length == 8192

    between = quick_tiers(model(size_gb=2.5, max_context=12000), hw)
    assert [t.context_length for t in between.values()] == [8192, 12000]

    unknown_context = quick_tiers(model(size_gb=2.5, max_context=0), hw)
    assert [t.context_length for t in unknown_context.values()] == [8192, 16384]

    too_big = quick_tiers(model(size_gb=22), hw)
    assert list(too_big) == ["low"]
    assert too_big["low"].context_length == 4096
    assert too_big["low"].cpu_expert_ratio is None

    no_vram = quick_tiers(model(size_gb=2.5), hardware(None))
    assert list(no_vram) == ["low"]
    assert no_vram["low"].context_length == 4096

    no_vram_small_ctx = quick_tiers(model(size_gb=2.5, max_context=2048), hardware(None))
    assert no_vram_small_ctx["low"].context_length == 2048

    no_vram_unknown_ctx = quick_tiers(model(size_gb=2.5, max_context=0), hardware(None))
    assert no_vram_unknown_ctx["low"].context_length == 4096

    moe_big = quick_tiers(model(size_gb=22, moe=True), hw)
    assert list(moe_big) == ["low"]
    assert moe_big["low"].cpu_expert_ratio == 0.5
    assert moe_big["low"].try_mmap is False

    moe_fits = quick_tiers(model(size_gb=2.5, moe=True), hw)
    assert list(moe_fits) == ["low", "medium"]
    for tier in moe_fits.values():
        assert tier.cpu_expert_ratio == 0.5
        assert tier.try_mmap is False

    dense = quick_tiers(model(size_gb=2.5), hw)
    for tier in dense.values():
        assert tier.cpu_expert_ratio is None
        assert tier.try_mmap is None


@pytest.mark.parametrize("max_context", [0, 1024, 4096, 8192, 12000, 16384, 262144])
@pytest.mark.parametrize("size_gb", [2.5, 22])
@pytest.mark.parametrize("vram", [None, 12288])
@pytest.mark.parametrize("moe", [False, True])
def test_quick_tiers_load_through_config_validation(max_context: int, size_gb: float, vram: int | None, moe: bool) -> None:
    tiers = quick_tiers(model(size_gb=size_gb, max_context=max_context, moe=moe), hardware(vram))
    assert tiers
    previous = 0
    for name, tier in tiers.items():
        assert name == tier.name
        assert name in TIER_ORDER
        assert 512 <= tier.context_length <= 1_048_576
        assert tier.kv_cache in KV_CACHE_TYPES
        assert tier.flash_attention
        if max_context > 0:
            assert tier.context_length <= max_context
        assert tier.context_length > previous
        previous = tier.context_length
    ranks = [TIER_ORDER.index(name) for name in tiers]
    assert ranks == sorted(ranks)


def choice(
    alias: str = "m",
    thinks: bool = True,
    twin: bool = True,
    key: str = "org/some-model",
    size_gb: float = 6.5,
) -> ProfileChoice:
    info = model(key, size_gb=size_gb, vision=True, variant="q4")
    return ProfileChoice(alias=alias, model=info, thinks=thinks, twin=twin, parallel=3, description="Mine.")


def test_profiles_from_choices_adds_nothink_twin() -> None:
    profiles = profiles_from_choices([choice()])
    assert list(profiles) == ["m", "m-nothink"]
    base, twin = profiles["m"], profiles["m-nothink"]
    assert base.thinks is True
    assert base.reasoning_effort is None
    assert base.context_ceiling == 8192
    assert base.description == "Mine."
    assert twin.thinks is False
    assert twin.reasoning_effort == "none"
    assert twin.description == "Same model as m with thinking off."
    for field in ("model_id", "variant", "parallel_ceiling", "supports_vision", "context_ceiling"):
        assert getattr(base, field) == getattr(twin, field)
    assert base.model_id == "org/some-model"
    assert base.variant == "q4"
    assert base.parallel_ceiling == 3
    assert base.supports_vision is True
    assert base.alias == "m" and twin.alias == "m-nothink"

    plain = profiles_from_choices([choice(thinks=False, twin=True)])
    assert list(plain) == ["m"]
    assert plain["m"].thinks is False

    no_twin = profiles_from_choices([choice(thinks=True, twin=False)])
    assert list(no_twin) == ["m"]

    long_alias = "x" * 40
    long = profiles_from_choices([choice(alias=long_alias)])
    assert list(long)[0] == long_alias
    twin_alias = list(long)[1]
    assert len(twin_alias) == 40
    assert twin_alias.endswith("-nothink")
    assert ALIAS_PATTERN.match(twin_alias)
    assert long[twin_alias].alias == twin_alias
    assert long[twin_alias].description == f"Same model as {long_alias} with thinking off."


def test_profiles_from_choices_rejects_duplicate_alias() -> None:
    with pytest.raises(ValueError, match="m"):
        profiles_from_choices([choice(alias="m"), choice(alias="m", key="org/other")])
    with pytest.raises(ValueError):
        profiles_from_choices([choice(alias="m"), choice(alias="m-nothink", thinks=False)])


def test_tiers_for_choices_one_entry_per_model() -> None:
    hw = hardware(12288)
    a = choice(alias="a", key="org/a")
    a_twin = choice(alias="b", key="org/a")
    c = choice(alias="c", key="org/c", size_gb=22)
    tiers = tiers_for_choices([a, a_twin, c], hw)
    assert set(tiers) == {"org/a", "org/c"}
    assert tiers["org/a"] == quick_tiers(a.model, hw)
    assert list(tiers["org/c"]) == ["low"]
