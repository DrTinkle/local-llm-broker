# Broker client commands

## Setup

Use this command prefix. It works from any folder once the package is installed in that Python:

```bash
python -m local_llm_broker.cli
```

If `llm-broker` is on PATH, it is the same program. Use the `python -m` form when it is not.

Install once, in the Python you run commands with. Installing does not start the broker.

```bash
python -m pip install -e <path-to-repo>
```

The token is read from `%LOCALAPPDATA%\LocalLLMBroker\auth.token`, or from the environment variable `LOCAL_LLM_BROKER_TOKEN`. Never print it, paste it into a prompt, or write it into a project file.

Global options go before the subcommand:

- `--base-url URL` changes the broker address (default `http://127.0.0.1:1235`).
- `--config PATH` names a config file (default `config.toml` in the current folder). Client commands do not need it. The profiles come from the running broker, which reads the owner's `config.toml`.

## Commands

| Command | Use it to | Output on stdout |
| --- | --- | --- |
| `status` | See the queue and the loaded model | JSON object |
| `doctor` | `status` plus the model list | Two JSON objects |
| `profiles` | List this broker's profiles: profile, model, thinks, vision, slots, low/medium/high context, and a description line each. Run it before choosing `--model`. `--json` prints the raw list | A table, or JSON with `--json` |
| `models` | The same profile data as raw JSON, including each profile's tiers | JSON |
| `submit` | Send a job | Job ID line, then the answer (see below) |
| `wait JOB` | Block until the job ends | JSON job snapshot (`state` is `succeeded`, `failed`, `cancelled`, or `interrupted`) |
| `result JOB` | Get the output of a finished job | JSON; the answer is `result.choices[0].message.content` |
| `cancel JOB` | Cancel a queued or running job | JSON job snapshot |
| `verify` | Run one tiny job per profile to prove each works (`--profiles` limits it). It loads models, so leave it to the owner | A pass or fail row per profile |
| `logs`, `summary` | Operator metadata | JSON; sessions do not need these |
| `setup` | Owner only: check LM Studio, choose models and write `config.toml`. Not for agents mid-task | Prompts and a summary |
| `stop` | Owner only: shut the broker down cleanly (`--force` even with jobs queued or running). Agents must not stop the broker | A short message |
| `install-skill` | Owner only: copy this skill into an agent's skills folder (`--target DIR`, `--force` to replace). Not for agents mid-task | A short message |
| `calibrate` | Owner only: measure each tier's speed and flag VRAM overflow. Loads every model in turn, so do not run it from a session | A table, or JSON with `--json` |

`result` on a job that has not finished returns at once with a `result_pending` error (`retryable: yes`), so it works as a non-blocking check. Call `wait` to block until the job ends. For a failed job, read the error from `wait`; `result` is only for succeeded jobs.

`wait` blocks for up to an hour. If the wait times out (`client_timeout`), the job is still running. Call `wait` again later.

## submit

```text
submit --model PROFILE                 required; a name from `profiles`
       [--prompt TEXT | --prompt-file PATH]
       [--system-file PATH]
       [--files ITEM [ITEM ...]]       files, directories (walked recursively) or quoted globs, appended after the prompt
       [--include GLOB [GLOB ...]]     walks and globs only: keep matching files
       [--exclude GLOB [GLOB ...]]     walks and globs only: drop matching files
       [--no-recursive]                walks only: direct children of each directory
       [--no-default-excludes]         turn off the default folder, generated-file and binary skips (secret and symlink rules stay)
       [--max-files N]                 default 100; 1 to 500
       [--split]                       split input that does not fit one job into up to 10 sequential jobs
       [--split-budget-tokens N]       override the per-job prompt budget used by --split
       [--client NAME]                 default "cli"; name your tool, e.g. my-agent
       [--project NAME]                short generic label
       [--task LABEL]                  short generic label
       [--max-output-tokens N]         default 2048; 1 up to the profile's context ceiling
       [--out PATH]                    write the answer to a file (atomic replace)
       [--detach]                      print the job ID and return at once
       [--idempotency-key KEY]         scoped to --client, kept 24 hours
       [--tier low|medium|high|auto]   load tier; omit (or auto) to let the broker choose
```

- `--files` takes files, directories and quoted glob patterns (for example `"src/**/*.py"`). The CLI expands them, not the broker. PowerShell does not expand globs, so quote them. Put `--files` last, or put another option after it, so the paths are not read as more files.
- Order is deterministic: explicitly named files first in command-line order, then expanded files sorted by relative path. A file from a directory or glob is labeled with its path relative to the directory argument's parent (for example `run/a.py`); an explicitly named file keeps its basename. Each file is framed `--- File: <label> ---`, about 15 tokens of overhead per file.
- Walks and globs skip folders `.git`, `.hg`, `.svn`, `node_modules`, `__pycache__`, `.venv`, `venv`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.tox`, `dist`, `build`, `.next` and `coverage`; generated files (`*.min.js`, `*.min.css`, `*.map`, `package-lock.json`, `pnpm-lock.yaml`, `yarn.lock`, `poetry.lock`, `Cargo.lock`); common binary extensions; and symlinks or junctions, which are never followed. Secret-looking files (`.env`, `.env.*` except `.env.example` and `.env.template`, `auth.token`, `*.pem`, `*.key`) are always skipped in walks and rejected with `secret_file_blocked` when named explicitly.
- `--include` and `--exclude` apply to walks and globs only. `--no-default-excludes` turns off the folder, generated-file and binary lists; the secret and symlink rules always apply.
- Input must be UTF-8 text. Prompt, system text, and files together must be at most 1 MiB (the configured `max_text_bytes`). The CLI rejects a larger input before sending it (exit 2).
- A file that cannot be read, is binary or is not UTF-8 is rejected before sending (exit 2). Use `--exclude` to skip it.
- When more than one file is sent, stderr prints `attached N files, B bytes`.
- `--split`: when the input does not fit one job, the CLI splits it into up to 10 sequential jobs sized to the model's largest tier, or to the `--tier` you name. Each job's idempotency key becomes `KEY-part-K-of-B`. Without `--detach`, the answers are combined into one `--out` file with `=== BATCH K OF B (job ID) ===` headers, written only if every batch succeeds. With `--detach`, one job ID is printed per line. `--split-budget-tokens` overrides the per-job prompt budget. `--split` is for grunt work on a bounded set of files that is a little too big for one job; the local models are not meant to take whole repositories.
- `--out` pointing at an existing directory is rejected before the job is submitted. `--out` always writes one file.
- Without `--out`, stdout is the job ID line followed by the answer. With `--out`, stdout is only the job ID. Use `--out` for answers that you will read from a file.
- With `--detach`, only the job ID is printed, and `--out` is not written. Keep working, then run `result` (it returns at once) within 10 minutes; use `wait` only when there is nothing else to do. If your tool can run a command in the background, a normal `submit --out` run there avoids the polling and the expiry.
- `--tier` is optional. Without it the broker picks the smallest tier the job fits. See "Load tier" in SKILL.md.
- Through the OpenAI-compatible route, send `"tier": "low"` in the request body, or append the tier to the model name (`"model": "<profile>:high"`) for clients that cannot add fields.

Example, waiting for the answer and writing it to a file:

```bash
python -m local_llm_broker.cli submit --client my-agent --project my-project --task summarize-notes \
  --model <profile> --prompt-file ./instructions.txt --files ./notes.md \
  --max-output-tokens 1024 --idempotency-key summarize-notes-v1 --out ./summary.txt
```

Example, detached (PowerShell uses a backtick for line continuation instead of `\`):

```bash
JOB=$(python -m local_llm_broker.cli submit --client my-agent --project my-project --task extract-risks \
  --model <thinking-profile> --prompt-file ./instructions.txt --files ./contract.txt --max-output-tokens 4096 --detach)
python -m local_llm_broker.cli wait "$JOB"
python -m local_llm_broker.cli result "$JOB"
```

## Exit codes

Exit codes are coarse. The stderr line carries the exact code.

| Exit | Meaning |
| --- | --- |
| 0 | Success |
| 2 | Invalid input found locally: bad flags, missing prompt, unreadable, binary, oversized or over-count input (the input error codes below) |
| 3 | Broker not running or not reachable (`broker is unavailable`), or token rejected (401) |
| 4 | Queue full, queue timeout, client wait timeout, or a degraded broker (`queue_full`, `queue_timeout`, `client_timeout`, `broker_degraded`, HTTP 408 or 429) |
| 5 | Model load problem (`model_load_failed`, `model_unavailable`) |
| 6 | Any other broker or inference failure, including switching errors and rejected input or budget |
| 7 | Cancelled or interrupted |
| 8 | Result expired (`result_expired`, HTTP 410) |

## Error codes

When a job fails, `submit` writes one line to stderr:

```text
model_load_failed (retryable: yes): LM Studio rejected the configured model load
```

`wait` and `result` show the same fields as JSON under `error`: `code`, `message`, `retryable`.

Retryable (`retryable: yes`). Resubmit the same job, at most twice, after a short wait:

- Broker load: `queue_full`, `queue_timeout`, `client_timeout`, `broker_degraded`
- Model switching: `model_load_failed`, `load_in_flight`, `unload_in_flight`, `unload_unverified`, `backend_busy`, `backend_status_unknown`, `caps_unverified`, `context_above_preset`, `variant_mismatch`
- LM Studio: `backend_unavailable`, `backend_http_error`, `invalid_backend_response`, `inference_failed`, `inference_in_flight`

Not retryable without a change:

- `model_unavailable`, `variant_unavailable`, `unknown_profile`: the configured model or profile is wrong. For `unknown_profile`, run `profiles` and use a listed name. Otherwise report it.
- `invalid_generation`, `invalid_messages`, `unsupported_message`, `unsupported_input`, `unsupported_parameter`, `input_too_large`, `context_exceeded`: fix the input or the output budget.
- `output_truncated`, `empty_output`: the answer was cut off at the output budget, or had no visible text. Raise the budget (at least 4096 for a thinking profile) or split the work.
- `generation_timeout`: the job ran past its deadline. Split the work.
- `cancelled`, `interrupted`: resubmit if still needed. Interrupted jobs are never replayed.
- `result_expired`, `result_too_large`, `idempotency_conflict`, `invalid_client_id`, `not_found`: fix the call.
- `unauthorized`: the token is missing or rejected. Report it. Do not guess or print a token.

## Input error codes

Client-side input errors exit 2 and print one stderr line, `<code> (retryable: no): <message>`. Fix the input or the flags; do not resubmit unchanged.

| Code | Meaning |
| --- | --- |
| `input_not_found` | A named path does not exist, or a glob or directory matched nothing. |
| `input_is_directory` | A path that must be a file is a directory, or `--out` names an existing directory. |
| `empty_expansion` | A directory or glob matched no readable files after `--include`/`--exclude` and the default skips. |
| `permission_denied` | A file cannot be read. |
| `binary_or_non_utf8_file` | A file is binary or not valid UTF-8. Exclude it with `--exclude`. |
| `secret_file_blocked` | A secret-looking file was named explicitly. Remove it from `--files`. |
| `too_many_files` | The expansion matched more than `--max-files`. Narrow with `--include`/`--exclude`, raise `--max-files` (up to 500), or add `--split`. |
| `input_too_large` | The input exceeds the 1 MiB limit, or would need more than 10 batches under `--split`. Narrow it or split the work. |

## What is logged

The broker logs job metadata only: IDs, profile, timings, token counts, and error codes. It does not log prompts, answers, file paths, or tokens. Do not add any of those to labels.
