"""Live pseudo-list tests against a running broker (not collected by pytest).

Each scenario makes two broker calls:
  1. GENERATE  - a model invents a large simulated dataset (one record per line).
  2. PROCESS   - a model reads that dataset (attached with --files) and extracts
                 the records matching a rule.
The script then recomputes the correct answer in Python and scores the model.

Run:  python tests/live/live_list_tests.py [--model PROFILE] [scenario ...]
Without --model it uses the first profile the broker reports as not thinking.
Output: tests/results/<scenario>/{dataset.csv,extracted.txt,report.json}
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "tests" / "results"
MODEL = ""  # set in main(): --model, or the first non-thinking profile


@dataclass
class Scenario:
    name: str
    rows: int
    generate_prompt: str
    process_prompt: str
    row_ok: Callable[[str], bool]
    wanted: Callable[[str], bool]
    key: Callable[[str], str]


def _fields(line: str) -> list[str]:
    return [p.strip() for p in line.split(",")]


def _log_ok(line: str) -> bool:
    f = _fields(line)
    return len(f) >= 4 and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", f[0]) is not None \
        and f[1] in {"INFO", "WARN", "ERROR"}


def _stock_ok(line: str) -> bool:
    f = _fields(line)
    return len(f) == 5 and f[3].isdigit() and re.fullmatch(r"\d+(\.\d+)?", f[4]) is not None


SCENARIOS = [
    Scenario(
        name="server-log",
        rows=150,
        generate_prompt=(
            "Invent a simulated web-service log with exactly 150 lines. One record per line, "
            "no header, no blank lines, no commentary, no code fences. Format: "
            "timestamp,level,service,message where timestamp is ISO like 2026-03-14T09:21:07 "
            "(increasing), level is INFO, WARN or ERROR (about 15% ERROR), service is one of "
            "auth, billing, search, gateway, cache, and message is a short phrase without commas. "
            "Output only the 150 lines."
        ),
        process_prompt=(
            "The attached file is a log, one record per line: timestamp,level,service,message. "
            "List every record whose level is ERROR. Output one line per match in the form "
            "timestamp|service, in file order, nothing else."
        ),
        row_ok=_log_ok,
        wanted=lambda line: _fields(line)[1] == "ERROR",
        key=lambda line: "|".join((_fields(line)[0], _fields(line)[2])),
    ),
    Scenario(
        name="inventory",
        rows=150,
        generate_prompt=(
            "Invent a simulated warehouse inventory with exactly 150 lines. One record per line, "
            "no header, no blank lines, no commentary, no code fences. Format: "
            "sku,name,category,quantity,unit_price where sku is like SKU-1001 (unique, increasing), "
            "name is a short product name without commas, category is one of tools, toys, garden, "
            "kitchen, office, quantity is an integer from 0 to 200 (about 20% below 10), and "
            "unit_price is a decimal. Output only the 150 lines."
        ),
        process_prompt=(
            "The attached file is an inventory, one record per line: "
            "sku,name,category,quantity,unit_price. List every record whose quantity is below 10. "
            "Output one line per match in the form sku|quantity, in file order, nothing else."
        ),
        row_ok=_stock_ok,
        wanted=lambda line: int(_fields(line)[3]) < 10,
        key=lambda line: "|".join((_fields(line)[0], _fields(line)[3])),
    ),
]


def call(args: list[str], out: Path) -> tuple[int, float]:
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "local_llm_broker.cli", "submit", "--client", "live-tests",
         "--project", "local-llm-broker", "--model", MODEL, "--out", str(out), *args],
        cwd=ROOT, capture_output=True, text=True, timeout=1500,
    )
    if proc.returncode:
        print(proc.stdout[-800:], proc.stderr[-800:])
    return proc.returncode, time.monotonic() - start


LOG_SPACED = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)[,\s]+(INFO|WARN|ERROR)[,\s]+(\w+)[,\s]+(.*)$")


RECORD = re.compile(r"[\w:-]+\|[\w.:-]+")


def clean(text: str) -> list[str]:
    """Drop blanks and fences; accept space-separated log rows by turning them into CSV."""
    out = []
    for ln in (raw.strip() for raw in text.splitlines()):
        if not ln or ln.startswith("```"):
            continue
        m = LOG_SPACED.match(ln)
        if m:
            ln = ",".join((m[1], m[2], m[3], m[4].replace(",", ";")))
        out.append(ln)
    return out


def run(s: Scenario) -> dict[str, object]:
    folder = RESULTS / s.name
    folder.mkdir(parents=True, exist_ok=True)
    dataset, extracted = folder / "dataset.csv", folder / "extracted.txt"

    rc1, t1 = call(["--task", f"{s.name}-generate", "--max-output-tokens", "8000",
                    "--prompt", s.generate_prompt], dataset)
    rows = clean(dataset.read_text(encoding="utf-8")) if rc1 == 0 and dataset.exists() else []
    valid = [r for r in rows if s.row_ok(r)]

    rc2 = 1
    t2 = 0.0
    got: list[str] = []
    if valid:
        dataset.write_text("\n".join(valid) + "\n", encoding="utf-8")  # keep only usable rows
        rc2, t2 = call(["--task", f"{s.name}-process", "--max-output-tokens", "8000",
                        "--files", str(dataset), "--prompt", s.process_prompt], extracted)
        if rc2 == 0 and extracted.exists():
            got = clean(extracted.read_text(encoding="utf-8"))

    expected = [s.key(r) for r in valid if s.wanted(r)]
    # Score only lines shaped like the requested answer (a|b); anything else is chatter.
    records = [ln for ln in got if RECORD.fullmatch(ln)]
    chatter = len(got) - len(records)
    got = records
    hit = set(got) & set(expected)
    report: dict[str, object] = {
        "scenario": s.name, "model": MODEL,
        "generate": {"exit": rc1, "seconds": round(t1, 1), "rows_requested": s.rows,
                     "rows_returned": len(rows), "rows_well_formed": len(valid)},
        "process": {"exit": rc2, "seconds": round(t2, 1), "expected": len(expected),
                    "returned": len(got), "chatter_lines": chatter, "correct": len(hit),
                    "missed": len(set(expected) - set(got)),
                    "invented": len(set(got) - set(expected))},
        "clean_output": chatter == 0,
        "passed": rc1 == 0 and rc2 == 0 and len(valid) >= s.rows * 0.8
                  and bool(expected) and len(hit) >= len(expected) * 0.9
                  and len(set(got) - set(expected)) <= max(2, len(expected) * 0.1),
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def pick_model(requested: str | None) -> str:
    """The profile to test: the requested one, else the first non-thinking profile the broker lists."""
    if requested:
        return requested
    proc = subprocess.run(
        [sys.executable, "-m", "local_llm_broker.cli", "profiles", "--json"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    if proc.returncode:
        raise SystemExit(f"could not list profiles; is the broker running? {proc.stderr[-400:]}")
    entries = json.loads(proc.stdout)
    for entry in entries:
        if not entry.get("thinks"):
            return str(entry["id"])
    if entries:
        return str(entries[0]["id"])
    raise SystemExit("no profiles are configured; run `llm-broker setup` first")


def main() -> int:
    global MODEL
    args = sys.argv[1:]
    requested = None
    if "--model" in args:
        index = args.index("--model")
        requested = args[index + 1] if index + 1 < len(args) else None
        args = args[:index] + args[index + 2 :]
    MODEL = pick_model(requested)
    wanted = set(args)
    reports = [run(s) for s in SCENARIOS if not wanted or s.name in wanted]
    print(json.dumps(reports, indent=2))
    return 0 if all(r["passed"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
