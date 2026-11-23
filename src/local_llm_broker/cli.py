from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from .app import create_app
from .client import BrokerClient, BrokerClientError
from .config import ensure_auth_token, load_config
from .contracts import BrokerConfig
from .inputs import InputError, InputFile, expand_inputs, partition_files, read_text_file
from .tokens import estimate_prompt_tokens

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_UNAVAILABLE = 3
EXIT_QUEUE = 4
EXIT_LOAD = 5
EXIT_INFERENCE = 6
EXIT_CANCELLED = 7
EXIT_EXPIRED = 8

# Most jobs one --split submit may send; also the width reserved for the "[batch k of B]" marker.
MAX_SPLIT_BATCHES = 10
SPLIT_FILE_CEILING = 500
SPLIT_TOKEN_RESERVE = 512

VERIFY_THINKING_TOKENS = 2048
VERIFY_PLAIN_TOKENS = 64
STOP_WAIT_SECONDS = 20
STOP_POLL_SECONDS = 0.5
STOP_PROBE_TIMEOUT = 2.0
NO_PROFILES_HINT = "No profiles are configured. Run `llm-broker setup` first."


def build_messages(
    prompt: str,
    *,
    system_text: str | None = None,
    files: Iterable[Path | InputFile] = (),
    max_text_bytes: int = 256 * 1024,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    parts = [prompt]
    total = len(prompt.encode("utf-8")) + len(system_text.encode("utf-8")) if system_text else len(prompt.encode("utf-8"))
    for item in files:
        if isinstance(item, InputFile):
            label, path = item.label, item.path
        else:
            label, path = item.name, item
        overhead = len(label.encode("utf-8")) + 32
        remaining = max(0, max_text_bytes - total - overhead)
        content = read_text_file(path, label=label, remaining_bytes=remaining)
        total += len(content.encode("utf-8")) + overhead
        if total > max_text_bytes:
            raise InputError("input_too_large", f"input limit exceeded: adding {label} goes over the {max_text_bytes} byte limit")
        parts.append(f"\n\n--- File: {label} ---\n{content}")
    if total > max_text_bytes:
        raise InputError("input_too_large", "input limit exceeded")
    messages.append({"role": "user", "content": "".join(parts)})
    return messages


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-broker")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--base-url", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve")
    setup = sub.add_parser("setup", help="check LM Studio, choose models and write config.toml")
    setup.add_argument("--level", choices=["none", "quick", "thorough"], default=None, help="how much to calibrate (asked when omitted)")
    setup.add_argument("--models", nargs="*", default=None, help="LM Studio model keys (or suggested profile names) to make profiles for")
    setup.add_argument("--preset", "--profiles-file", dest="preset", type=Path, default=None, help="copy the profiles and tiers of this TOML file; LM Studio is not contacted")
    setup.add_argument("--yes", action="store_true", help="ask nothing: all models, suggested names, level none unless --level is given")
    sub.add_parser("doctor")
    profiles = sub.add_parser("profiles", help="list the profiles this broker serves")
    profiles.add_argument("--json", action="store_true", help="print the raw profile list as JSON")
    verify = sub.add_parser("verify", help="run one tiny job per profile to prove it works")
    verify.add_argument("--profiles", nargs="*", default=None, help="default: every profile")
    verify.add_argument("--max-output-tokens", type=int, default=None, help="override the output budget (default 2048 for thinking profiles, else 64)")
    install = sub.add_parser("install-skill", help="copy the use-local-models skill into an agent's skills folder")
    install.add_argument("--target", type=Path, required=True, help="the skills folder to install into")
    install.add_argument("--force", action="store_true", help="replace an existing copy")
    stop = sub.add_parser("stop", help="ask the running broker to shut down")
    stop.add_argument("--force", action="store_true", help="stop even while jobs are queued or running")
    stop.add_argument("--wait-seconds", type=int, default=STOP_WAIT_SECONDS, help="how long to wait for the broker to go away (default 20)")
    sub.add_parser("models")
    sub.add_parser("status")
    submit = sub.add_parser("submit")
    submit.add_argument("--client", default="cli")
    submit.add_argument("--project", default="")
    submit.add_argument("--task", default="")
    submit.add_argument("--model", required=True)
    submit.add_argument("--prompt")
    submit.add_argument("--prompt-file", type=Path)
    submit.add_argument("--system-file", type=Path)
    submit.add_argument("--files", nargs="*", type=str, default=[], help="files, folders or glob patterns to attach")
    submit.add_argument("--include", nargs="*", default=[], help="keep only expanded files whose name or relative path matches")
    submit.add_argument("--exclude", nargs="*", default=[], help="drop expanded files or folders whose name or relative path matches")
    submit.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders of attached folders")
    submit.add_argument("--no-default-excludes", action="store_true", help="also keep build, VCS, lock and binary-like files")
    submit.add_argument("--max-files", type=int, default=100, help="most files one submit may attach (1 to 500)")
    submit.add_argument("--split", action="store_true", help="send an over-budget file set as several jobs, one batch at a time")
    submit.add_argument(
        "--split-budget-tokens", type=int, default=None,
        help="with --split: prompt tokens per job (default: from the model's tier context)",
    )
    submit.add_argument("--out", type=Path)
    submit.add_argument("--max-output-tokens", type=int, default=2048)
    submit.add_argument("--detach", action="store_true")
    submit.add_argument("--idempotency-key")
    submit.add_argument(
        "--tier", choices=["low", "medium", "high", "auto"], default=None,
        help="load tier; omit (or auto) to let the broker pick the smallest tier the job fits",
    )
    calibrate = sub.add_parser("calibrate", help="measure each load tier's speed on this machine")
    calibrate.add_argument("--profiles", nargs="*", default=None, help="default: one non-thinking profile per model")
    calibrate.add_argument("--tiers", nargs="*", default=["low", "medium", "high"])
    calibrate.add_argument("--prompt-tokens", type=int, default=8000)
    calibrate.add_argument("--max-output-tokens", type=int, default=300)
    calibrate.add_argument("--json", action="store_true")
    calibrate.add_argument("--concurrency", action="store_true", help="measure parallel jobs per tier instead of single-job speed")
    calibrate.add_argument("--no-capacity", action="store_true", help="with --concurrency: skip the shared-context overfill test")
    calibrate.add_argument(
        "--mode",
        choices=["speed", "concurrency", "tokens", "depth", "search", "slots", "experts"],
        default="speed",
        help="speed (default), concurrency, tokens and depth run through the broker; "
        "search, slots and experts load LM Studio directly and need the broker stopped",
    )
    calibrate.add_argument(
        "--models", nargs="*", default=None,
        help="direct modes: profile aliases or model IDs (default: the model of every configured profile)",
    )
    calibrate.add_argument("--max-depth", type=int, default=32_768, help="depth mode: largest prompt to try, in tokens")
    calibrate.add_argument("--ratios", nargs="*", type=float, default=None, help="experts mode: CPU expert ratios (default 0.50 to 1.00)")
    wait = sub.add_parser("wait")
    wait.add_argument("job_id")
    result = sub.add_parser("result")
    result.add_argument("job_id")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("job_id")
    sub.add_parser("logs")
    sub.add_parser("summary")
    return parser


def _calibrate(client: BrokerClient, config: BrokerConfig, args: argparse.Namespace) -> int:
    from .calibrate import default_profiles, format_table, run_calibration, vision_profiles

    models = client.models()
    served = {item.get("id") for item in models.get("data", [])}
    profiles = args.profiles or [name for name in default_profiles(config) if name in served]
    print("Calibrating; each tier loads its model once. This can take several minutes.", file=sys.stderr)
    headers = {"Authorization": f"Bearer {client.token}"}
    if args.mode == "tokens":
        from .calibrate_broker_modes import format_tokens, run_tokens

        report = run_tokens(client.base_url, headers, default_profiles(config), vision_profiles(config))
        if args.json:
            _print_json({"bytes_per_token": report.text, "image_tokens": {p: {str(k): v for k, v in r.items()} for p, r in report.images.items()}, "errors": report.errors})
        else:
            print(format_tokens(report))
        return EXIT_OK if not report.errors else EXIT_UNAVAILABLE
    if args.mode == "depth":
        from .calibrate_broker_modes import format_depth, run_depth

        found_depth = run_depth(client.base_url, headers, models, profiles, args.tiers, args.max_depth)
        print(format_depth(found_depth))
        return EXIT_OK if all(item.error is None for item in found_depth) else EXIT_UNAVAILABLE
    if args.concurrency or args.mode == "concurrency":
        from .calibrate_concurrency import format_report, run_concurrency, to_json

        found = run_concurrency(
            client.base_url, {"Authorization": f"Bearer {client.token}"}, models, profiles, args.tiers, capacity=not args.no_capacity
        )
        if args.json:
            _print_json(to_json(found))
        else:
            print(format_report(found))
        return EXIT_OK if all(item.error is None for item in found) else EXIT_UNAVAILABLE
    with httpx.Client(base_url=client.base_url, headers={"Authorization": f"Bearer {client.token}"}, timeout=httpx.Timeout(30.0, read=1800.0)) as http:
        results = run_calibration(http, models, profiles, args.tiers, prompt_tokens=args.prompt_tokens, max_tokens=args.max_output_tokens)
    if args.json:
        _print_json([{**vars_of(item), "prefill_tps": item.prefill_tps, "generation_tps": item.generation_tps} for item in results])
    else:
        print(format_table(results))
    return EXIT_OK if all(item.error is None for item in results) else EXIT_UNAVAILABLE


def _calibrate_direct(config: BrokerConfig, args: argparse.Namespace) -> int:
    from .calibrate import resolve_models
    from .calibrate_direct import (
        EXPERT_RATIOS,
        BrokerRunning,
        DirectSession,
        format_search,
        format_sweep,
        run_direct,
        search_combos,
        search_context,
        slot_candidates,
        sweep_experts,
        sweep_slots,
    )

    models = resolve_models(config, args.models)

    def is_moe(model: str) -> bool:
        # Stand-in until models are inspected: a model with an expert split in its tiers is a mixture of experts.
        return any(t.cpu_expert_ratio is not None for t in config.tiers.get(model, {}).values())

    if args.mode == "experts":
        models = [m for m in models if is_moe(m)]
    parallel = {p.model_id: p.parallel_ceiling for p in config.profiles.values()}
    print(
        f"Direct calibration ({args.mode}): loads LM Studio without the broker, unloading everything first. "
        "This takes several minutes per model.",
        file=sys.stderr,
    )

    async def work(session: DirectSession) -> str:
        if args.mode == "search":
            found = [
                await search_context(session, model, kv, ratio, parallel.get(model, 1))
                for model in models
                for kv, ratio in search_combos(is_moe(model))
            ]
            return format_search(found)
        low = {model: config.tiers[model]["low"] for model in models if "low" in config.tiers.get(model, {})}
        if args.mode == "slots":
            sweeps = [await sweep_slots(session, model, low[model], slot_candidates(parallel.get(model, 1))) for model in low]
            return format_sweep(sweeps, "slots")
        sweeps = [await sweep_experts(session, model, low[model], args.ratios or EXPERT_RATIOS) for model in low]
        return format_sweep(sweeps, "experts")

    try:
        print(run_direct(config.backend_url, config.service_url, work))
    except BrokerRunning as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_UNAVAILABLE
    return EXIT_OK


def vars_of(item: object) -> dict[str, object]:
    return {name: getattr(item, name) for name in getattr(item, "__slots__", ())}


def _client(config_path: Path, base_url: str | None) -> tuple[BrokerClient, BrokerConfig]:
    config = load_config(config_path)
    token_path = config.auth_token_path or (config.data_dir / "auth.token")
    token = os.environ.get("LOCAL_LLM_BROKER_TOKEN")
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise BrokerClientError("unauthorized", "broker token is missing", 401)
    return BrokerClient(base_url or config.service_url, token), config


def _use_utf8_output() -> None:
    # Model replies can contain characters that a non-UTF-8 console cannot encode. Replace them rather than fail the command.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()
    args = _parser().parse_args(argv)
    if args.command == "setup":
        # Setup reads and reports on the config itself; it must run even when the existing file is invalid.
        from .setup_wizard import run_setup

        return run_setup(args)
    if args.command == "install-skill":
        return _install_skill(args.target, args.force)
    try:
        config = load_config(args.config)
        if args.command == "serve":
            ensure_auth_token(config)
            app = create_app(config)
            # Status streams never end on their own, so shutdown closes open connections after 5 seconds.
            server_config = uvicorn.Config(
                app, host=config.service_host, port=config.service_port, workers=1, access_log=False,
                timeout_graceful_shutdown=5,
            )
            server = uvicorn.Server(server_config)
            # `llm-broker stop` reaches this through POST /admin/shutdown.
            app.state.request_shutdown = lambda: setattr(server, "should_exit", True)
            server.run()
            return EXIT_OK
        if args.command == "stop":
            return _stop(args, config)
        if args.command == "calibrate" and args.mode in {"search", "slots", "experts"}:
            return _calibrate_direct(config, args)
        if args.command == "submit" and not args.prompt and not args.prompt_file:
            print("submit requires --prompt or --prompt-file", file=sys.stderr)
            return EXIT_INVALID_INPUT
        client, config = _client(args.config, args.base_url)
        try:
            if args.command == "doctor":
                _print_json(client.status())
                _print_json(client.models())
            elif args.command == "models":
                _print_json(client.models())
            elif args.command == "profiles":
                return _profiles(client, args.json)
            elif args.command == "verify":
                return _verify(client, args)
            elif args.command == "status":
                _print_json(client.status())
            elif args.command == "submit":
                return _submit(client, config, args)
            elif args.command == "wait":
                _print_json(client.wait(args.job_id))
            elif args.command == "result":
                _print_json(client.result(args.job_id))
            elif args.command == "cancel":
                _print_json(client.cancel(args.job_id))
            elif args.command == "calibrate":
                return _calibrate(client, config, args)
            elif args.command == "logs":
                _print_json(client.logs())
            elif args.command == "summary":
                _print_json(client.summary())
            return EXIT_OK
        finally:
            client.close()
    except InputError as exc:
        hint = " (or pass --split)" if exc.code == "input_too_large" and args.command == "submit" and not args.split else ""
        print(f"{exc.code} (retryable: no): {exc}{hint}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INVALID_INPUT
    except BrokerClientError as exc:
        retryable = "yes" if exc.retryable else "no"
        print(f"{exc.code} (retryable: {retryable}): {exc}", file=sys.stderr)
        return _exit_for_error(exc.code, exc.status_code)


def _submit(client: BrokerClient, config: BrokerConfig, args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt is not None else read_text_file(args.prompt_file, label=args.prompt_file.name)
    system = read_text_file(args.system_file, label=args.system_file.name) if args.system_file else None
    if not 1 <= args.max_files <= 500:
        raise InputError("too_many_files", "--max-files must be between 1 and 500")
    inputs = expand_inputs(
        args.files,
        include=args.include,
        exclude=args.exclude,
        recursive=not args.no_recursive,
        default_excludes=not args.no_default_excludes,
        max_files=SPLIT_FILE_CEILING if args.split else args.max_files,
    )
    if args.out is not None and args.out.is_dir():
        raise InputError("input_is_directory", f"--out path is an existing directory: {args.out.name}")
    batches = _plan_batches(prompt, system, inputs, config, args) if args.split and inputs else [inputs]
    if len(inputs) > 1:
        attached_bytes = sum(item.path.stat().st_size for item in inputs)
        print(f"attached {len(inputs)} files, {attached_bytes} bytes", file=sys.stderr)
    if len(batches) == 1:
        messages = build_messages(prompt, system_text=system, files=batches[0], max_text_bytes=config.max_text_bytes)
        return _submit_single(client, args, messages)
    bodies = [
        build_messages(
            f"{prompt}\n\n[batch {index} of {len(batches)}]",
            system_text=system,
            files=batch,
            max_text_bytes=config.max_text_bytes,
        )
        for index, batch in enumerate(batches, start=1)
    ]
    return _submit_batches(client, args, bodies)


def _plan_batches(
    prompt: str,
    system: str | None,
    inputs: list[InputFile],
    config: BrokerConfig,
    args: argparse.Namespace,
) -> list[list[InputFile]]:
    budget = _split_budget_tokens(config, args)
    reserve = f"\n\n[batch {MAX_SPLIT_BATCHES} of {MAX_SPLIT_BATCHES}]"
    generation = {"max_tokens": args.max_output_tokens}

    def fits(batch: list[InputFile]) -> bool:
        try:
            messages = build_messages(prompt + reserve, system_text=system, files=batch, max_text_bytes=config.max_text_bytes)
        except InputError as exc:
            if exc.code == "input_too_large":
                return False
            raise
        return budget is None or estimate_prompt_tokens(messages, generation) <= budget

    return partition_files(inputs, fits, max_batches=MAX_SPLIT_BATCHES)


def _split_budget_tokens(config: BrokerConfig, args: argparse.Namespace) -> int | None:
    explicit: int | None = args.split_budget_tokens
    if explicit is not None:
        return explicit
    profile = config.profiles.get(args.model)
    if profile is None:
        return None  # the broker rejects an unknown profile; there is no calibrated context to plan against
    tiers = config.tiers.get(profile.model_id, {})
    if not tiers:
        context = profile.context_ceiling
    elif args.tier in tiers:
        context = tiers[args.tier].context_length
    else:
        # No tier named: the broker runs a job on the smallest calibrated tier it fits, so a set that fits a tier
        # stays one job. Split only past the largest tier, and fill each batch to it.
        context = max(tier.context_length for tier in tiers.values())
    budget: int = context - args.max_output_tokens - SPLIT_TOKEN_RESERVE
    return budget


def _job_payload(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "client_id": args.client,
        "project": args.project,
        "task_label": args.task,
        "profile": args.model,
        "messages": messages,
        "generation": {"max_tokens": args.max_output_tokens},
    }
    if args.tier is not None:
        payload["tier"] = args.tier
    return payload


def _submit_single(client: BrokerClient, args: argparse.Namespace, messages: list[dict[str, str]]) -> int:
    accepted = client.submit(_job_payload(args, messages), idempotency_key=args.idempotency_key)
    job_id = accepted["job_id"]
    print(job_id)
    if args.detach:
        return EXIT_OK
    snapshot = client.wait(job_id)
    if snapshot.get("state") != "succeeded":
        return _report_failed_job(snapshot)
    content = _answer_text(client.result(job_id))
    if args.out:
        write_atomic(args.out, content)
    else:
        print(content)
    return EXIT_OK


def _submit_batches(client: BrokerClient, args: argparse.Namespace, bodies: list[list[dict[str, str]]]) -> int:
    total = len(bodies)
    sections: list[str] = []
    for index, messages in enumerate(bodies, start=1):
        key = f"{args.idempotency_key}-part-{index}-of-{total}" if args.idempotency_key else None
        job_id = client.submit(_job_payload(args, messages), idempotency_key=key)["job_id"]
        print(job_id, flush=True)
        if args.detach:
            continue
        # One batch at a time: a failure stops the run before later batches are sent.
        snapshot = client.wait(job_id)
        if snapshot.get("state") != "succeeded":
            return _report_failed_job(snapshot)
        answer = _answer_text(client.result(job_id))
        sections.append(f"=== BATCH {index} OF {total} (job {job_id}) ===\n{answer}\n")
    if args.detach:
        return EXIT_OK
    combined = "\n".join(sections)
    if args.out:
        write_atomic(args.out, combined)
    else:
        print(combined)
    return EXIT_OK


def _tokens(value: int) -> str:
    return f"{value:,}"


def format_profiles(models: dict[str, Any]) -> str:
    entries = models.get("data", [])
    header = ["profile", "model", "thinks", "vision", "slots", "low", "medium", "high"]
    rows: list[list[str]] = []
    for entry in entries:
        tiers = entry.get("tiers", {})
        rows.append(
            [
                str(entry.get("id", "")),
                str(entry.get("model_id", "")),
                "yes" if entry.get("thinks") else "no",
                "yes" if entry.get("vision") else "no",
                str(entry.get("parallel", 1)),
                *[_tokens(tiers[name]["context_length"]) if name in tiers else "-" for name in ("low", "medium", "high")],
            ]
        )
    widths = [max(len(line[column]) for line in [header, *rows]) for column in range(len(header))]

    def render(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()

    lines = [render(header)]
    for entry, row in zip(entries, rows, strict=True):
        lines.append(render(row))
        description = str(entry.get("description") or "").strip()
        if description:
            lines.append(f"  {description}")
    return "\n".join(lines)


def _profiles(client: BrokerClient, as_json: bool) -> int:
    models = client.models()
    entries = models.get("data", [])
    if as_json:
        _print_json(entries)
    elif not entries:
        print(NO_PROFILES_HINT)
    else:
        print(format_profiles(models))
    return EXIT_OK


def run_verify(client: BrokerClient, profile_entries: list[dict[str, Any]], max_output_tokens_override: int | None) -> int:
    passed = True
    for entry in profile_entries:
        alias = str(entry["id"])
        budget = max_output_tokens_override or (VERIFY_THINKING_TOKENS if entry.get("thinks") else VERIFY_PLAIN_TOKENS)
        payload: dict[str, object] = {
            "client_id": "verify",
            "project": "",
            "task_label": "verify",
            "profile": alias,
            "messages": [{"role": "user", "content": "Reply with the word OK."}],
            "generation": {"max_tokens": budget},
        }
        started = time.monotonic()
        failure: str | None = None
        try:
            job_id = client.submit(payload)["job_id"]
            snapshot = client.wait(job_id)
            if snapshot.get("state") != "succeeded":
                error = snapshot.get("error") or {}
                failure = str(error.get("code") or snapshot.get("state"))
                if error.get("message"):
                    failure = f"{failure} ({error['message']})"
            elif not _answer_text(client.result(job_id)).strip():
                failure = "empty answer"
        except BrokerClientError as exc:
            if exc.code == "unavailable":
                raise
            failure = f"{exc.code} ({exc})"
        if failure is None:
            print(f"PASS {alias} ({time.monotonic() - started:.1f} s)", flush=True)
        else:
            passed = False
            print(f"FAIL {alias}: {failure}", flush=True)
    return EXIT_OK if passed else EXIT_INFERENCE


def _verify(client: BrokerClient, args: argparse.Namespace) -> int:
    entries: list[dict[str, Any]] = client.models().get("data", [])
    if not entries:
        print(NO_PROFILES_HINT)
        return EXIT_INVALID_INPUT
    if args.profiles:
        by_alias = {str(entry["id"]): entry for entry in entries}
        unknown = [alias for alias in args.profiles if alias not in by_alias]
        if unknown:
            for alias in unknown:
                print(f"Unknown profile: {alias}")
            return EXIT_INVALID_INPUT
        entries = [by_alias[alias] for alias in args.profiles]
    return run_verify(client, entries, args.max_output_tokens)


def _broker_answers(base_url: str) -> bool:
    try:
        httpx.get(f"{base_url.rstrip('/')}/health", timeout=STOP_PROBE_TIMEOUT)
    except httpx.HTTPError:
        return False
    return True


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def wait_until_gone(probe: Callable[[], bool], sleep: Callable[[float], None], wait_seconds: float) -> bool:
    """Poll until `probe` reports the broker is gone (False); True if it is gone within `wait_seconds`."""
    waited = 0.0
    while probe():
        if waited >= wait_seconds:
            return False
        sleep(STOP_POLL_SECONDS)
        waited += STOP_POLL_SECONDS
    return True


def _stop(args: argparse.Namespace, config: BrokerConfig) -> int:
    base_url: str = args.base_url or config.service_url
    # A broker that is not running needs no token; look before building the authenticated client.
    if not _broker_answers(base_url):
        print("The broker is not running.")
        return EXIT_OK
    client, _ = _client(args.config, args.base_url)
    try:
        client.shutdown(force=args.force)
    except BrokerClientError as exc:
        if exc.code == "broker_busy":
            print(str(exc))
            return EXIT_QUEUE
        raise
    finally:
        client.close()
    if wait_until_gone(lambda: _broker_answers(base_url), _pause, args.wait_seconds):
        print("The broker stopped.")
        return EXIT_OK
    print(f"The broker is still shutting down after {args.wait_seconds} seconds; check again, or end the process by hand.")
    return EXIT_UNAVAILABLE


def _install_skill(target: Path, force: bool) -> int:
    source = Path(__file__).resolve().parents[2] / "skills" / "use-local-models"
    if not source.is_dir():
        print("The skill folder was not found next to the installed package; install from a clone of the repository.")
        return EXIT_INVALID_INPUT
    destination = target / "use-local-models"
    if destination.exists():
        if not force:
            print(f"{destination} already exists; use --force to replace it.")
            return EXIT_INVALID_INPUT
        shutil.rmtree(destination)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"Installed the skill to {destination}")
    return EXIT_OK


def _answer_text(result: dict[str, Any]) -> str:
    content: str = result.get("result", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
    return content


def _print_json(value: dict[str, Any] | list[Any]) -> None:
    # Strict JSON, so other programs and sessions can read the output without parsing Python reprs.
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _report_failed_job(snapshot: dict[str, Any]) -> int:
    # Cancelled and interrupted jobs carry no error object, so their state name stands in for the code.
    error = snapshot.get("error") or {}
    code = error.get("code") or snapshot.get("state")
    message = error.get("message") or f"job ended in state {snapshot.get('state')}"
    retryable = "yes" if error.get("retryable") else "no"
    print(f"{code} (retryable: {retryable}): {message}", file=sys.stderr)
    return _exit_for_error(code)


def _exit_for_error(code: str | None, status_code: int = 0) -> int:
    if code in {"queue_full", "queue_timeout", "broker_degraded"} or status_code in {408, 429}:
        return EXIT_QUEUE
    if code in {"model_load_failed", "model_unavailable"}:
        return EXIT_LOAD
    if code in {"cancelled", "interrupted"}:
        return EXIT_CANCELLED
    if code == "result_expired" or status_code == 410:
        return EXIT_EXPIRED
    if code == "unavailable" or status_code == 401:
        return EXIT_UNAVAILABLE
    return EXIT_INFERENCE


if __name__ == "__main__":
    raise SystemExit(main())

