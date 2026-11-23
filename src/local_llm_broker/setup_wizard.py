"""The `llm-broker setup` command: check the machine, pick models, optionally calibrate, then write config.toml.

Setup never starts or stops the broker or LM Studio. Calibration loads models directly, so it needs the broker stopped
and says so instead of doing it.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, TextIO, TypeVar

import httpx

from .calibrate_direct import DirectSession
from .config import ALIAS_PATTERN, load_config
from .contracts import BrokerConfig, ProfileConfig, TierConfig
from .setup_calibrate import CalibrationResult, Level, Progress, calibrate_model
from .setup_discover import Hardware, ModelInfo, Preflight, detect_hardware, discover_models
from .setup_discover import fetch_listing as read_listing
from .setup_discover import preflight as run_preflight
from .setup_profiles import (
    ProfileChoice,
    default_parallel,
    guess_thinks,
    profiles_from_choices,
    suggest_alias,
    tiers_for_choices,
)
from .setup_write import existing_broker_settings, render_config, summarize_changes, write_config

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_UNAVAILABLE = 3

_ATTEMPTS = 3
_LEVELS = ("none", "quick", "thorough")
_PROGRESS_FILE = "setup-progress.json"
_T = TypeVar("_T")


async def _default_preflight(config: BrokerConfig) -> Preflight:
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await run_preflight(config, client)


async def _default_listing(config: BrokerConfig) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await read_listing(config, client)


def _default_session(config: BrokerConfig) -> DirectSession:
    return DirectSession(config.backend_url, config.service_url)


@dataclass
class SetupDeps:
    """The pieces of setup that touch the machine or the network, so tests can replace them."""

    preflight: Callable[[BrokerConfig], Awaitable[Preflight]] = field(default=_default_preflight)
    fetch_listing: Callable[[BrokerConfig], Awaitable[dict[str, Any]]] = field(default=_default_listing)
    hardware: Callable[[], Hardware] = field(default=detect_hardware)
    session_factory: Callable[[BrokerConfig], DirectSession] = field(default=_default_session)
    now: Callable[[], datetime] = field(default=datetime.now)


class _Exit(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


async def _await(awaitable: Awaitable[_T]) -> _T:
    return await awaitable


def _gb(size_bytes: int) -> str:
    return f"{size_bytes / 1_000_000_000:.1f}"


class _Setup:
    def __init__(self, args: argparse.Namespace, ask: Callable[[str], str], out: TextIO, deps: SetupDeps) -> None:
        self.args = args
        self._ask = ask
        self.out = out
        self.deps = deps

    # --- console ------------------------------------------------------------------------------------------------

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def ask(self, prompt: str) -> str:
        try:
            return self._ask(prompt).strip()
        except EOFError:
            self.say("Input ended before setup finished; nothing was written.")
            raise _Exit(EXIT_INVALID_INPUT) from None

    def confirm(self, prompt: str, default: bool, fallback: bool | None = None) -> bool:
        for _ in range(_ATTEMPTS):
            answer = self.ask(prompt).lower()
            if not answer:
                return default
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            self.say("Please answer y or n.")
        return default if fallback is None else fallback

    # --- flow ---------------------------------------------------------------------------------------------------

    def run(self) -> int:
        path = self.args.config
        try:
            config = load_config(path)
        except (ValueError, OSError) as error:
            self.say(f"Cannot use {path}: {error}")
            return EXIT_INVALID_INPUT
        if self.args.preset is not None:
            profiles, tiers = self._from_preset()
        else:
            profiles, tiers = self._from_models(config)
        return self._finish(config, profiles, tiers)

    def _from_preset(self) -> tuple[dict[str, ProfileConfig], dict[str, dict[str, TierConfig]]]:
        try:
            preset = load_config(self.args.preset)
        except (ValueError, OSError) as error:
            self.say(f"Cannot use {self.args.preset}: {error}")
            raise _Exit(EXIT_INVALID_INPUT) from None
        return dict(preset.profiles), {model: dict(rows) for model, rows in preset.tiers.items()}

    def _from_models(self, config: BrokerConfig) -> tuple[dict[str, ProfileConfig], dict[str, dict[str, TierConfig]]]:
        checked = asyncio.run(_await(self.deps.preflight(config)))
        if not checked.lmstudio_ok:
            for problem in checked.problems:
                self.say(problem)
            self.say("Start the LM Studio local server, then run setup again.")
            raise _Exit(EXIT_UNAVAILABLE)
        for problem in checked.problems:
            self.say(f"Warning: {problem}")
        try:
            listing = asyncio.run(_await(self.deps.fetch_listing(config)))
        except RuntimeError as error:
            self.say(str(error))
            self.say("Start the LM Studio local server, then run setup again.")
            raise _Exit(EXIT_UNAVAILABLE) from None
        models = discover_models(listing)
        if not models:
            self.say(
                "No chat models are downloaded in LM Studio. Open LM Studio, download a chat model from the "
                "Discover tab, then run setup again."
            )
            raise _Exit(EXIT_UNAVAILABLE)
        hw = self.deps.hardware()
        self._print_machine(hw, models)
        choices = self._choose_profiles(self._choose_models(models), hw)
        level = self._choose_level()
        if level != "none" and checked.broker_running:
            self.say("Stop the broker, then run setup again to calibrate. Level none works while the broker runs.")
            raise _Exit(EXIT_UNAVAILABLE)
        if level == "none":
            tiers = tiers_for_choices(choices, hw)
        else:
            choices, tiers = self._calibrate(config, choices, hw, level)
        try:
            return profiles_from_choices(choices), tiers
        except ValueError as error:
            self.say(str(error))
            raise _Exit(EXIT_INVALID_INPUT) from None

    # --- models -------------------------------------------------------------------------------------------------

    def _print_machine(self, hw: Hardware, models: list[ModelInfo]) -> None:
        gpu = f"GPU: {hw.gpu_name}, {hw.vram_mib:,} MiB memory" if hw.vram_mib else "GPU memory unknown; calibration relies on speed checks"
        ram = f"RAM: {hw.ram_mib / 1024:.0f} GB" if hw.ram_mib else "RAM unknown"
        self.say(f"{gpu}. {ram}.")
        self.say()
        self.say(f"{len(models)} chat model(s) downloaded in LM Studio:")
        for number, model in enumerate(models, start=1):
            flags = ["vision"] if model.vision else []
            if model.is_moe:
                flags.append("MoE")
            context = f"{model.max_context:,}" if model.max_context else "?"
            self.say(
                f"{number:>3}. {model.display_name}  {_gb(model.size_bytes)} GB  {model.params or '?'}  "
                f"max context {context}{'  ' + ', '.join(flags) if flags else ''}  [{model.key}]"
            )
        self.say()

    def _choose_models(self, models: list[ModelInfo]) -> list[ModelInfo]:
        wanted: list[str] | None = self.args.models or None
        if wanted is not None:
            chosen: list[ModelInfo] = []
            for entry in wanted:
                match = next((m for m in models if m.key == entry or suggest_alias(m, set()) == entry), None)
                if match is None:
                    self.say(f"Unknown model: {entry}")
                    raise _Exit(EXIT_INVALID_INPUT)
                if match not in chosen:
                    chosen.append(match)
            return chosen
        if self.args.yes:
            return list(models)
        for _ in range(_ATTEMPTS):
            answer = self.ask("Which models should get profiles? Enter numbers separated by commas (or 'all'): ")
            picked = self._parse_numbers(answer, models)
            if picked:
                return picked
            self.say(f"Enter numbers from 1 to {len(models)} separated by commas, or 'all'.")
        raise _Exit(EXIT_INVALID_INPUT)

    @staticmethod
    def _parse_numbers(answer: str, models: list[ModelInfo]) -> list[ModelInfo]:
        if answer.lower() == "all":
            return list(models)
        picked: list[ModelInfo] = []
        for part in re.split(r"[,\s]+", answer):
            if not part.isdigit() or not 1 <= int(part) <= len(models):
                return []
            model = models[int(part) - 1]
            if model not in picked:
                picked.append(model)
        return picked

    def _choose_profiles(self, models: list[ModelInfo], hw: Hardware) -> list[ProfileChoice]:
        scripted = bool(self.args.yes or self.args.models)
        taken: set[str] = set()
        choices: list[ProfileChoice] = []
        for model in models:
            suggestion = suggest_alias(model, taken)
            guess = guess_thinks(model)
            if scripted:
                choice = ProfileChoice(suggestion, model, guess, guess, default_parallel(model, hw))
            else:
                choice = self._ask_choice(model, suggestion, guess, taken, default_parallel(model, hw))
            names = set(profiles_from_choices([choice]))
            if names & taken:
                self.say(f"{model.display_name}: the '-nothink' profile name is already used, so it was not added.")
                choice = replace(choice, twin=False)
                names = set(profiles_from_choices([choice]))
            taken |= names
            choices.append(choice)
        return choices

    def _ask_choice(self, model: ModelInfo, suggestion: str, guess: bool, taken: set[str], parallel: int) -> ProfileChoice:
        self.say(f"{model.display_name} [{model.key}]")
        alias = self._ask_alias(suggestion, taken)
        thinks = self.confirm(f"Does this model think before answering? {'[Y/n]' if guess else '[y/N]'} ", guess)
        twin = thinks and self.confirm(f"Also add a '{alias}-nothink' profile that turns thinking off? [Y/n] ", True)
        return ProfileChoice(alias, model, thinks, twin, parallel)

    def _ask_alias(self, suggestion: str, taken: set[str]) -> str:
        for _ in range(_ATTEMPTS):
            alias = self.ask(f"Profile name [{suggestion}]: ") or suggestion
            if not ALIAS_PATTERN.match(alias):
                self.say("A profile name uses lowercase letters, digits, '.', '_' and '-', starts with a letter or digit, and has at most 40 characters.")
            elif alias in taken:
                self.say(f"The name {alias} is already used by another profile.")
            else:
                return alias
        raise _Exit(EXIT_INVALID_INPUT)

    def _choose_level(self) -> str:
        if self.args.level:
            return str(self.args.level)
        if self.args.yes:
            return "none"
        prompt = "Setup depth: none (no model loads), quick (about 3 min per model), thorough (about 30 min per model) [none]: "
        for _ in range(_ATTEMPTS):
            answer = self.ask(prompt).lower() or "none"
            if answer in _LEVELS:
                return answer
            self.say("Enter none, quick or thorough.")
        raise _Exit(EXIT_INVALID_INPUT)

    # --- calibration --------------------------------------------------------------------------------------------

    def _calibrate(
        self, config: BrokerConfig, choices: list[ProfileChoice], hw: Hardware, level: str
    ) -> tuple[list[ProfileChoice], dict[str, dict[str, TierConfig]]]:
        mode: Level = "thorough" if level == "thorough" else "quick"
        progress = Progress.load(config.data_dir / _PROGRESS_FILE, mode)
        results: dict[str, CalibrationResult] = {}
        todo: list[ProfileChoice] = []
        for choice in choices:
            finished = progress.done(choice.model.key)
            if finished is None:
                todo.append(choice)
            else:
                self.say(f"{choice.model.display_name}: already calibrated, skipping")
                results[choice.model.key] = finished
        if todo:
            try:
                asyncio.run(self._calibrate_all(config, todo, hw, mode, progress, results))
            except KeyboardInterrupt:
                self.say("Interrupted. Finished models are saved; run setup again to continue.")
                raise _Exit(EXIT_UNAVAILABLE) from None
        return self._apply_results(choices, results)

    async def _calibrate_all(
        self,
        config: BrokerConfig,
        todo: list[ProfileChoice],
        hw: Hardware,
        level: Level,
        progress: Progress,
        results: dict[str, CalibrationResult],
    ) -> None:
        session = self.deps.session_factory(config)
        try:
            for number, choice in enumerate(todo, start=1):
                name = choice.model.display_name
                self.say(f"[{number}/{len(todo)}] Calibrating {name} ({level}); this loads the model several times.")
                result = await calibrate_model(session, choice.model, hw, level, choice.parallel)
                if result.failed is None:
                    progress.save(result)
                    self._print_result(name, result)
                results[choice.model.key] = result
        finally:
            await session.close()

    def _print_result(self, name: str, result: CalibrationResult) -> None:
        sizes = ", ".join(f"{tier.name} {tier.context_length:,}" for tier in result.tiers.values())
        self.say(f"{name}: {sizes}; {result.parallel} slot(s)")
        for note in result.notes:
            self.say(f"  {note}")

    def _apply_results(
        self, choices: list[ProfileChoice], results: dict[str, CalibrationResult]
    ) -> tuple[list[ProfileChoice], dict[str, dict[str, TierConfig]]]:
        kept: list[ProfileChoice] = []
        tiers: dict[str, dict[str, TierConfig]] = {}
        for choice in choices:
            result = results[choice.model.key]
            name = choice.model.display_name
            if result.failed is not None or not result.tiers:
                self.say(f"{name}: {result.failed or 'no tiers found'}; no profile written for it")
                continue
            thinks = choice.thinks if result.thinks is None else result.thinks
            twin = choice.twin
            if twin and not (thinks and result.can_disable_thinking is True):
                twin = False
                reason = "its thinking cannot be turned off" if thinks else "it did not think during the check"
                self.say(f"{name}: {reason}, so no '{choice.alias}-nothink' profile was added.")
            kept.append(replace(choice, thinks=thinks, twin=twin, parallel=result.parallel))
            tiers[choice.model.key] = dict(result.tiers)
        if not kept:
            self.say("No model could be calibrated, so nothing was written.")
            raise _Exit(EXIT_UNAVAILABLE)
        return kept, tiers

    # --- writing ------------------------------------------------------------------------------------------------

    def _finish(
        self, config: BrokerConfig, profiles: dict[str, ProfileConfig], tiers: dict[str, dict[str, TierConfig]]
    ) -> int:
        path = self.args.config
        profiles = _with_ceilings(profiles, tiers)
        lines = summarize_changes(dict(config.profiles), profiles)
        for model in sorted(set(config.tiers) & set(tiers)):
            if dict(config.tiers[model]) != tiers[model]:
                lines.append(f"changed tiers for {model}")
        if not lines and path.exists():
            self.say("No changes")
            return EXIT_OK
        self.say(f"Changes to {path}:")
        for line in lines:
            self.say(f"  {line}")
        if not self.args.yes and not self.confirm(f"Write {path}? [Y/n] ", True, fallback=False):
            self.say("Nothing written.")
            return EXIT_OK
        text = render_config(existing_broker_settings(path), profiles, tiers)
        try:
            backup = write_config(path, text, now=self.deps.now())
        except (ValueError, OSError) as error:
            self.say(f"Could not write {path}: {error}")
            return EXIT_INVALID_INPUT
        self.say(f"Wrote {path}")
        if backup is not None:
            self.say(f"Backed up the old file to {backup}")
        self.say(
            "Next: start the broker (scripts\\start-broker.ps1), then run `llm-broker verify`, then "
            "`llm-broker install-skill --target <folder>` to give agents the skill."
        )
        return EXIT_OK


def _with_ceilings(
    profiles: dict[str, ProfileConfig], tiers: dict[str, dict[str, TierConfig]]
) -> dict[str, ProfileConfig]:
    """Give each profile the context ceiling the config loader derives from its model's tiers."""
    result: dict[str, ProfileConfig] = {}
    for alias, profile in profiles.items():
        rows = tiers.get(profile.model_id)
        if rows:
            profile = replace(profile, context_ceiling=max(tier.context_length for tier in rows.values()))
        result[alias] = profile
    return result


def run_setup(
    args: argparse.Namespace,
    *,
    ask: Callable[[str], str] = input,
    out: TextIO | None = None,
    deps: SetupDeps | None = None,
) -> int:
    setup = _Setup(args, ask, sys.stdout if out is None else out, deps or SetupDeps())
    try:
        return setup.run()
    except _Exit as stop:
        return stop.code
