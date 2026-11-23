---
name: use-local-models
description: Offload extraction, inventories, and digests over large raw text (15 KiB or more of logs, documents, schemas, or data you have not read) to the local LM Studio models through the Local LLM Broker on this computer, so the raw text stays out of your context. Use when a shell is available and the broker is running. Do not use it for small inputs, summaries of your own work, arithmetic, code edits, secrets, personal records, or answers you cannot check.
---

# Use local models through the broker

The broker runs on this computer (by default at `127.0.0.1:1235`) and sends jobs to the local LM Studio models. Many sessions can submit at once. The broker keeps one model loaded at a time and runs contiguous queued jobs for the same profile in parallel, up to that profile's slot limit (the `slots` column of `profiles`; a model with 1 slot runs one job at a time and later jobs wait). Parallel jobs share the loaded tier's context, so a large job waits until enough of it is free rather than failing. Active jobs finish in FIFO order before the broker switches profiles and unloads the previous model. You never load, unload, or switch models yourself.

A local model is a worker for small text tasks. It cannot see your tools, your files, or this conversation. Everything it needs must be in the input you send, and you are responsible for checking what it returns.

**Profiles belong to this machine.** They are defined only by the owner's `config.toml` and differ from computer to computer. Run `python -m local_llm_broker.cli profiles` to see what exists here. Never assume a profile name from an example.

**If there are no profiles** (the command says there are none, or the broker is not set up), tell the owner to run `llm-broker setup` (or `python -m local_llm_broker.cli setup`) and then `llm-broker verify`. Do not try to set it up yourself, and continue the parent task without the broker.

## 1. Decide whether to use it (Strict ROI Rule)

Every broker call costs the parent session about 5,000 input-token equivalents in tool-call and turn overhead. **Only use the broker when it shields your context window from at least 15 KiB (about 3,500 tokens) of raw text that you do NOT read yourself.**

Use it **only** for these 6 context-shielding tasks (passing raw files via `--files` without reading them in the parent session):
- **Noisy test / build / DOM-dump failure logs (15 KiB or more):** extract failing file paths, line numbers, and error summaries.
- **Multi-file Markdown / content audits (5+ files, 15 KiB or more):** extract titles, headings, internal links, or frontmatter into a compact JSON/Markdown table.
- **Database schema & SQL migration inventories (15 KiB or more):** list tables, columns, and `CHECK` constraints across schema and migration files.
- **Large research / spec / dataset document extraction (15 KiB or more):** extract structured JSON/tables from long docs so you only read the compact extract.
- **Multi-file CSS / TypeScript export inventories (15 KiB or more):** list `@media` rules, exported types/functions, or source URLs across many files.
- **Bounded table conversion & `git log --stat` digests:** convert a bounded Markdown/CSV table to JSON (verified via `json.loads`) or summarize 20+ commits from `git log --stat`.

Use a non-thinking profile for all of these (see section 3).

Do **not** use it for (these usually cost more tokens than they save):
- **Any input under 15 KiB (about 3,500 tokens),** or any file you have already read (or must still read in full) in the parent session.
- **Checkpoints, commit messages, or summaries of your own work:** you already hold the context; delegating costs more than writing it directly.
- **Arithmetic, financial calculations, or test expected values:** small non-thinking models fail multi-step math, and thinking models burn thousands of tokens reasoning and may truncate.
- **Writing or editing application code / TypeScript components.**
- **Secrets, credentials, tokens, or personal records.**
- **Input over 1 MiB of text in one job.** Split the work.

## 2. Check that the broker is up

Run `python -m local_llm_broker.cli status`.

- A JSON object comes back: the broker is running. Note `queue_count` (jobs waiting), `active_profile` (model loaded now) and `active_tier` (its load tier).
- Exit 3 and `broker is unavailable`: the broker is not running. Tell the owner and continue the parent task without it. Do not start the broker or LM Studio unless the owner asked you to.

## 3. Choose a profile

Run `python -m local_llm_broker.cli profiles`. It prints one row per profile: `profile`, `model`, `thinks`, `vision`, `slots`, and the `low`, `medium` and `high` context sizes, with a description line under each profile. (`--json` prints the raw list.)

- For grunt work (extraction, inventories, digests, table conversion), use a profile whose `thinks` column is `no`. It answers without spending output tokens on reasoning.
- Use a thinking profile (`thinks` = yes) only when the task needs reasoning. Its reasoning counts against `--max-output-tokens`, so a small budget can end with `output_truncated`.
- Prefer the profile with the smallest model that does the job. Move to a larger one only when the smaller one's output is wrong.
- Image input needs a profile whose `vision` column is `yes`.
- A profile that shares its model with another (a thinking profile and its non-thinking twin) switches without a reload as long as the tier stays the same; any other switch loads a model. Group jobs by profile where you can.

Details and budgets are in [model-selection.md](references/model-selection.md).

### Load tier (optional)

Each model loads at one of up to three tiers (`low`, `medium`, `high`). The `low`, `medium` and `high` columns of `profiles` give each tier's context size in tokens, and `-` means the model has no such tier. In general `low` is the fastest and has the shortest context, and higher tiers give more context but are slower or use more memory. Prompt + output + 512 must fit the tier's context.

Leave the tier out in almost every case. The broker then picks the smallest tier your job fits, and if a larger tier of the same model is already loaded it uses that instead of reloading. Name a tier (`--tier low|medium|high`) only when you know you need it, for example `--tier low` to keep a short job fast even though a larger tier is loaded. Naming a tier that is too small for the job fails with `context_exceeded`. Changing tier reloads the model, so do not alternate tiers between jobs.

## 4. Submit

Put instructions and source text in separate files so you can see exactly what was sent:

```bash
python -m local_llm_broker.cli submit --client my-agent --project my-project --task summarize-notes \
  --model <profile> --prompt-file ./instructions.txt --files ./notes.md \
  --max-output-tokens 1024 --idempotency-key summarize-notes-v1 --out ./summary.txt
```

To audit a directory, pass the directory and filter it (PowerShell uses a backtick, bash `\`):

```bash
python -m local_llm_broker.cli submit --client my-agent --project my-project --task audit-schema \
  --model <profile> --prompt-file ./instructions.txt --files ./docs ./src/schema --include "*.md" "*.sql" \
  --max-output-tokens 2048 --idempotency-key audit-schema-v1 --out ./audit.md
```

Rules:
- `--client`, `--project`, and `--task` are short, generic labels. Never put file names, prompt text, or file contents in them.
- `--files` accepts files, directories (walked recursively) and quoted globs such as `"src/**/*.py"`. Do NOT list the files yourself first; the CLI expands them. Quote globs, because PowerShell does not expand them. Explicitly named files come first, then expanded files sorted by relative path.
- Walks and globs skip folders such as `.git`, `node_modules`, `__pycache__`, `dist` and `build`, generated files (`*.min.js`, `*.map`, lockfiles), common binary files, and symlinks. Secret-looking files (`.env`, `*.pem`, `*.key`, `auth.token`) are always skipped, and rejected by name if you list them. Adjust with `--include`, `--exclude`, `--no-recursive`, or `--no-default-excludes` (secret and symlink rules stay on).
- Each file adds about 15 tokens of framing. `--max-files N` (default 100, up to 500) caps the file count. If the input does not fit one job, add `--split` to run up to 10 sequential jobs; without `--detach` their answers are combined into one `--out` file, written only if every batch succeeds.
- `--out` always writes one file. Neither the broker nor the CLI creates several files from model output. The model does not write files either: it only returns text.
- Tell the model exactly what to return, and say it ends there. For example: "Output only the result, one item per line, in this format: ... No explanation, no preamble, no repeat of the instructions." Non-thinking profiles tend to ramble or write their reasoning into the answer when the format is loose, and the extra text is a cost you pay in your own context. Give one concrete example line of the format you want, and check the first few lines of the output before trusting the rest.
- Always set `--max-output-tokens` to a finite number: at least 4096 for thinking profiles (their reasoning counts against the budget), and at least 1024 for the others unless the answer is one word. A tiny budget on a thinking profile returns an empty answer, and even a large one can run out before the answer; the broker then reports `output_truncated`.
- Keep the input plus the output within the tier's limit: prompt tokens + `--max-output-tokens` + 512 must fit the tier's context (the `low`, `medium` and `high` columns of `profiles`). Without `--tier`, the largest tier the model has is the limit. The broker counts prompt tokens exactly with the model's own tokenizer before the job starts. Structured text costs far more tokens per byte than prose: English prose about 5 bytes per token, Python code about 3, JSON and Markdown tables about 2, log lines about 1.4, and numeric CSV about 1. A 100 KB CSV is about 100,000 tokens. Structured text uses 2 to 5 times more tokens per byte than prose, so do not judge fit by byte size alone. Images cost about one token per 1,024 pixels (a 1024 x 1024 image is about 1,000 tokens). Long prompts take time before the first token, most of all on large models. A job that does not fit is rejected with `context_exceeded`.
- Use `--out` for the answer. Without it, stdout is the job ID line followed by the answer. With it, stdout is only the job ID, and the answer goes into the file.
- Use `--idempotency-key` with a stable name for each logical job. If the call is lost and you send the same job again, the broker returns the original job instead of running it twice. Reusing a key with different content is rejected with `idempotency_conflict`, so use a new key for new content.
- Do not sit and wait. A local job takes seconds to minutes, so use that time for other work. Unless your very next step needs the answer, run the job without blocking, as described under "Run it without waiting" below. Loading a model takes seconds to a minute or more depending on its size, and the first job after a switch includes that load.

### Run it without waiting

Pick the first option your tool supports:
1. **Background command.** If your tool can run a shell command in the background and tell you when it ends, run the normal `submit ... --out <file>` that way and carry on. The CLI writes the answer to the file, and you read the file after you are notified. There is nothing to poll and the result does not expire.
2. **`--detach`.** Otherwise add `--detach`. It prints the job ID and returns at once. Keep working. When you need the answer, run `result <job-id>`: it returns at once, with the answer if the job is done and a `result_pending` error (`retryable: yes`) if it is not. Call `wait <job-id>` only when you have nothing else to do, then `result <job-id>`.

With `--detach`, `--out` is not written; fetch the answer with `result` within 10 minutes (see section 5). Wait only when you truly cannot proceed without the answer.

## 5. Read the outcome

- Exit 0: the answer is in the `--out` file, or on stdout if you did not use `--out`.
- Any other exit code: stderr has one line such as `model_load_failed (retryable: yes): LM Studio rejected the configured model load`. The code and the retry flag are what matter. Exit codes are coarse; see [client-commands.md](references/client-commands.md).
- A finished job's result is kept in memory for 10 minutes. After that, `result` reports `result_expired`. Resubmit the job. This matters for `--detach`: collect the answer before it expires.

## 6. Handle failures

Retryable (`retryable: yes`): wait briefly and resubmit the same job, at most twice. The broker already retries its own model loads and unloads. Do not try to repair LM Studio.

Not retryable: fix the request, split the work, or do the task in the parent session.

| Code | What to do |
| --- | --- |
| `queue_full`, `queue_timeout`, `client_timeout` | The broker is busy. Resubmit later. |
| `broker_degraded` | The broker's job database or event log is unavailable and it has stopped taking new jobs. It retries on its own every few seconds. Resubmit later, and tell the owner if it persists. |
| `model_load_failed`, `load_in_flight`, `unload_in_flight`, `backend_busy`, `backend_status_unknown`, `unload_unverified`, `caps_unverified`, `context_above_preset`, `variant_mismatch` | Model switching had a transient problem. Resubmit once. |
| `backend_unavailable`, `backend_http_error`, `invalid_backend_response`, `inference_failed`, `inference_in_flight` | LM Studio had a transient problem. Resubmit once; if it repeats, stop and tell the owner. |
| `cancelled`, `interrupted` | The job was cancelled, or the broker restarted during it. Interrupted jobs are never replayed; resubmit if still needed. |
| `result_expired` | Resubmit. |
| `model_unavailable`, `variant_unavailable` | The configured model is not installed. Tell the owner. Do not pick a different model. |
| `unknown_profile`, `invalid_generation`, `invalid_messages`, `unsupported_*`, `input_too_large`, `context_exceeded` | Fix the input or the output budget. For `unknown_profile`, run `profiles` and use a name from it. |
| `output_truncated`, `empty_output` | The answer was cut off at the output budget, or came back with no visible text. Raise the budget (at least 4096 for a thinking profile), or split the task. Resubmitting unchanged fails the same way. |
| `generation_timeout` | The job ran past its deadline. Split it into smaller jobs. |
| `idempotency_conflict`, `invalid_client_id`, `result_too_large`, `not_found` | Fix the call. |
| `input_not_found`, `input_is_directory`, `empty_expansion`, `permission_denied`, `binary_or_non_utf8_file`, `secret_file_blocked`, `too_many_files`, `input_too_large` (exit 2) | Client-side input error (exit 2): read the message; narrow with `--include`/`--exclude` or add `--split`. |
| `unauthorized` (exit 3) | The token is missing or rejected. Do not print, guess, or copy a token. Report it. |

If the same failure keeps coming back, stop using the broker for this task, say so in your final message, and continue the parent task.

## 7. Check the answer

A local answer is not evidence. Verify numbers, names, dates, links, code, and structured fields against the source yourself. Do not accept a model's statement that it checked its own work. Small non-thinking models make mistakes on fine rules, so spot-check their output.

## Limits by design

- The broker does not read your disk. The CLI reads the files and sends their text; a job cannot reach a file you did not attach.
- The model does not write files. It returns text, and `--out` saves that one answer.
- The models are for grunt work on bounded input, not for exploring a whole repository. Do not point them at one.

## Hard rules

- Use only the broker. Do not call LM Studio directly.
- Do not load, unload, or switch models in LM Studio. Do not edit broker configuration or LM Studio presets.
- Do not restart or stop the broker or LM Studio, and do not kill any process.
- Do not send credentials, tokens, personal records, or untrusted secrets.
- Do not write prompt text, answers, file paths, or tokens into labels, logs, or your final message.
- Do not expose the broker beyond this computer.

## Other clients

An app that accepts an OpenAI-compatible base URL can use `http://127.0.0.1:1235/v1` with the broker token. That endpoint is Chat Completions with `n=1`; the model name is a profile name from `profiles`. Messages are text, except that profiles whose `vision` column is `yes` also accept inline base64 `image_url` parts. It does not run tools and does not implement the Responses API. Use it only when the owner asks.

## References

- [client-commands.md](references/client-commands.md): every command, flag, output shape, and exit code.
- [model-selection.md](references/model-selection.md): choosing a profile, and budgets.
- [session-prompt.md](references/session-prompt.md): a short version to paste into a session that cannot load this skill.
