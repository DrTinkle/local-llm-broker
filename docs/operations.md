# Local LLM Broker operations

## Setup

```powershell
cd <path-to-repo>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m local_llm_broker.cli setup
```

`setup` checks LM Studio, lists the chat models you have downloaded, lets you choose which ones get profiles, optionally calibrates them, and writes `config.toml` (default: the current folder). It never starts or stops the broker or LM Studio.

Depth (`--level`): `none` writes cautious tiers and loads nothing; `quick` finds the largest context that fits for each model (about 3 minutes per model); `thorough` also compares memory settings, parallel slot counts and, for mixture-of-experts models, how many expert layers to keep in system RAM (about 30 minutes per model). Both calibration levels load models directly in LM Studio, so the broker must be stopped; setup says so and exits with code 3 if it is running. Progress is saved in `setup-progress.json` in the data folder, so an interrupted run skips finished models next time.

Scripted use: `--yes` accepts the suggested names and writes without asking, `--models <model-key> ...` picks models, and `--preset <file.toml>` copies the profiles and tiers from a ready-made file (see `examples/presets`). Whenever setup replaces an existing config it first copies it to `config.toml.bak-YYYYMMDD-HHMMSS`. Setup exits with code 3 when LM Studio is not answering or has no chat model downloaded, and nothing is written.

After setup, edit `config.toml` only for loopback addresses, model IDs and load tiers. The service generates a bearer token in `%LOCALAPPDATA%\LocalLLMBroker\auth.token` on first start. Keep that file private.

Then start the broker (`scripts\start-broker.ps1`), run `llm-broker verify` to prove every profile answers, and install the skill for your agents with `llm-broker install-skill --target <skills-folder>`.

## Run and inspect

```powershell
llm-broker serve
llm-broker stop
llm-broker doctor
llm-broker models
llm-broker status
llm-broker logs
llm-broker summary
```

If `llm-broker` is not on PATH, use `python -m local_llm_broker.cli` with the same subcommands. Only one broker can run at a time. If `serve` stops with `lock is already held`, another broker is already running; stop that one first with `llm-broker stop`.

Use one foreground service process and one Uvicorn worker. LM Studio must be available at the configured loopback backend URL. Broker liveness at `/health` does not imply backend readiness; `/status` reports queue, active profile, effective context/parallel caps, and backend health.

### Watching status without polling

`/status` answers from the broker's own state, plus `backend_health` from LM Studio's model list. The broker reuses that list for 30 seconds, so however many clients poll `/status`, LM Studio is asked at most once per 30 seconds (each request shows as `GET /api/v1/models` in LM Studio's server log). A model load or unload clears the cache, and the listings the broker makes during a job refresh it. `backend_health` can therefore be up to 30 seconds old; the rest of `/status` is always current. `degraded` is `null` normally. If the job database or the event log cannot be written, it names the reason (`storage_unavailable` or `event_sink_unavailable`), and the broker rejects new submissions with `broker_degraded` (HTTP 503, retryable) until a check every 5 seconds finds it working again. Running jobs are not interrupted.

A dashboard should use `GET /status/stream` instead of polling. It needs the bearer token, and it is a server-sent event stream (`text/event-stream`):

- On connect, one `status_snapshot` event whose `data` is the same JSON as `/status`.
- Another `status_snapshot` each time the status changes: a job is queued, starts, finishes, fails or is cancelled, a model loads or unloads, or the idle-unload timer starts or stops. Changes that arrive together are sent as one snapshot, and a snapshot equal to the last one is not sent. The countdown in `idle_unload_in_seconds` is not a change; count it down on the client.
- A `: keep-alive` comment after each 30 seconds without a snapshot. Missing keep-alives mean the connection is gone.

A quiet stream costs nothing and asks LM Studio nothing. When the broker stops, the stream ends; reconnect with a back-off, checking `/health` (which never touches LM Studio) before reopening the stream. On Ctrl+C the broker waits at most 5 seconds for open connections, including these streams and any synchronous `/v1/chat/completions` request, before it closes them; jobs cut off this way are marked `interrupted`.

## Token counting and shared context

At submit, the broker estimates a job's prompt tokens from its characters (rates per letter, digit, space, punctuation and non-ASCII character, fitted to the 2026-11-04 `calibrate --mode tokens` measurements; the estimate errs high by 7 to 23% on the measured content types). It uses the estimate to pick a tier, and rejects a job only when even the estimate without its margin cannot fit the largest tier. Before the job runs on the loaded model, the broker counts its text exactly with that model's tokenizer through the LM Studio SDK, over one long-lived SDK connection (opening one per count took 0.5 to 1 second and staggered parallel jobs), adds the chat-template overhead and image tokens (pixels / 1,024 + 2), and uses that count from then on. If counting fails, the estimate is used.

Parallel jobs share the loaded tier's context, because LM Studio loads with a unified KV cache, and when that pool overfills LM Studio fails every running job (measured 2026-11-04). The broker therefore reserves each running job's prompt plus its whole output budget plus 512 tokens, and starts the next queued job only when it fits beside them; otherwise the job waits at the head of the queue. A job that does not fit the loaded tier on its own moves to a larger tier if the broker chose its tier (the larger tier loads once running jobs finish), and fails with `context_exceeded` if the job named its tier.

## Load tiers and calibration

Each model has three load tiers, `low`, `medium` and `high`, set in `[tiers."<model id>".<tier>]` in `config.toml`. A tier sets context length, KV cache type (`f16`, `q8_0`, `q4_0`), flash attention, evaluation batch size and, for MoE models, `cpu_expert_ratio` (the share of expert layers kept in system RAM) and `try_mmap`. The thinking and `-nothink` profiles of one model share its tiers. Context is no longer set per profile.

The broker loads tiers through LM Studio's Python SDK (`lmstudio`, pinned to 1.5.0), because the REST load call accepts only context length. Every tier load sends the tier's settings plus the profile's `parallel_ceiling`, and these override LM Studio's saved preset for the same settings. LM Studio's saved presets now matter only for loads made by hand in its window. They are set to each model's `medium` tier.

A job may name a tier. Without one, the broker picks the smallest tier the job fits, and runs it on a larger tier of the same model if that is already loaded. A named tier is honored exactly, so a job that names `low` while `high` is loaded causes a reload.

`setup` writes tiers for your machine. Measure them again after any change to `[tiers]`, a driver update, or a new model:

```powershell
python -m local_llm_broker.cli calibrate
```

It loads each tier of one non-thinking profile per model in turn, sends one warm-up request and one fixed 8,000-token prompt per tier, and prints prompt-processing and generation speed. A tier whose generation and prompt processing are both more than 15% slower than its model's fastest tier is marked `likely VRAM overflow`: Windows has moved part of it into shared system memory. A tier where only generation is slower is marked `slower generation`; that is usually the cost of a q8_0 cache (up to 40% on the 4B models with a long prompt), and the tier works. Lower that tier's `context_length`, or for the 35B raise `cpu_expert_ratio`, restart the broker, and calibrate again. Task Manager's "Shared GPU memory" rising during the load confirms an overflow. Calibration occupies LM Studio for several minutes; run it when no sessions are submitting jobs. Use `--profiles` and `--tiers` to measure fewer.

### Concurrency

```powershell
python -m local_llm_broker.cli calibrate --concurrency
```

For each tier it runs two tests, through the broker the same way sessions submit:

- **Throughput:** 1, 2, up to the profile's slots of identical short jobs at once (about 2,000 prompt tokens, 200 output tokens). It prints combined and per-job generation speed and a "usable slots" count: the largest number of parallel jobs whose combined speed is still at least 15% above fewer jobs.
- **Shared context:** all slots busy, with prompts sized so the jobs together fill 50%, 90%, 120%, 150% and 180% of the tier's context. Each job fits the tier on its own. LM Studio loads with a unified KV cache, so parallel jobs share one context pool, and this test shows what LM Studio does when it overfills. One-slot profiles skip it; `--no-capacity` skips it everywhere.

The test prompt is English prose (about 5 bytes per token). The broker counts every job's prompt exactly before it runs, so the shared-context check holds for any content type.

This test, not the direct `slots` mode below, decides `parallel_ceiling`, because it measures what sessions get through the broker. On 2026-11-04 a second 35B job added no combined speed here although the direct probe showed 40%, so the 35B runs one job at a time (`parallel_ceiling = 1`).

### More calibration modes

`calibrate --mode <mode>`. The first group runs through the running broker:

- `tokens`: real bytes per token for seven content types (English and Finnish prose, Python, JSON, CSV numbers, log lines, a Markdown table) on the 4B and 9B tokenizers (the 35B shares the 9B's), and real tokens per image at 256, 512, 1024 and 1536 pixels on the 9B and 35B. The broker's submit-time estimate (see "Token counting and shared context") is fitted to these numbers; rerun this after a model change to check it still errs high.
- `depth`: prompt processing and generation speed at 2K, 8K, 16K and 32K prompt tokens (`--max-depth` for more) on each tier, plus the time for each tier's load and first reply. Each prompt starts with a unique line so LM Studio cannot reuse a cached prefix.

The second group loads LM Studio directly with settings that are not in `config.toml`. Stop the broker first; these modes refuse to start while it answers. They unload everything before and after each load, and sessions cannot submit jobs meanwhile.

- `search`: the largest context per model and cache type (and CPU expert split on the 35B) that still fits. It binary-searches in 4,096-token steps between 16,384 and 262,144, loading each size and running an 8,000-token probe. A size fits when it loads, the probe keeps prompt processing and generation within 15% of the 16,384 baseline, and `nvidia-smi` shows at least 300 MiB of dedicated VRAM free. Other programs' GPU use counts against that, so close GPU-heavy programs first.
- `slots`: combined generation speed with more parallel slots than configured (4B: 4, 6, 8; 9B: 2, 3, 4; 35B: 1, 2), at the `low` tier's settings.
- `experts`: the 35B at its `low` tier with CPU expert ratios from 0.50 to 1.00 (`--ratios` to choose), and the fastest ratio that leaves VRAM headroom.

Use `--models <profile or model key> ...` to limit the direct modes. Results feed `[tiers]` and `parallel_ceiling` by hand; calibration never changes the config.

## Submit two clients

```powershell
Start-Job { llm-broker submit --client agent-a --project demo --task summarize --model <profile> --prompt "Summarize this sentence" --detach }
Start-Job { llm-broker submit --client agent-b --project demo --task rewrite --model <profile> --prompt "Rewrite this sentence" --detach }
llm-broker status
```

Do not run a second scheduler against the same LM Studio endpoint. Direct LM Studio UI/API inference during a model switch invalidates the broker's ownership guarantees.

## Directory input and --split

`submit --files` takes files, directories (walked recursively) and quoted glob patterns. The CLI expands them before sending, so the broker only sees the resulting text. Walks skip folders such as `.git`, `node_modules` and `dist`, generated files, binaries and symlinks, and always skip secret-looking files. `--include`, `--exclude`, `--no-recursive`, `--no-default-excludes` and `--max-files` (default 100) narrow the set. When the input does not fit one job, `--split` runs up to 10 sequential jobs sized to the largest calibrated tier of the model (or `--tier`), with idempotency keys `KEY-part-K-of-B`. Without `--detach` the answers are combined into one `--out` file, written only if every batch succeeds. Client-side input errors exit 2 with the codes listed in `skills/use-local-models/references/client-commands.md`.

## Restart and failures

The broker starts without a recovery step, even when a model is loaded in LM Studio. Jobs that were unfinished when it stopped are marked `interrupted` and are never replayed. A model that is already loaded is adopted when a job needs it.

No operator step is needed after a failure. A failed job fails alone, and the next queued job runs normally. Failures that involve LM Studio are retried automatically by the next job:

- A load or unload that timed out fails its job (`load_in_flight` or `unload_in_flight`, retryable). The next job lists LM Studio again. It adopts the instance if the load finished, or retries the unload.
- A switch waits up to 120 seconds for LM Studio to report the loaded model idle. If it stays busy, the job fails with `backend_busy` (retryable) and nothing is unloaded.
- An instance whose caps cannot be verified, or whose context is above the preset (profiles without tiers), is unloaded. The job fails retryably, and the next job loads the model again at the preset.
- A loaded instance of the right model at the wrong tier, or at a context no tier uses (for example loaded by hand), is unloaded once idle and reloaded at the job's tier.
- A tier load that LM Studio rejects fails its job as `model_load_failed` (retryable). The message names the tier. A tier that does not fit in memory fails this way, or loads and runs slowly if Windows spills it into shared memory; `calibrate` finds the second case.
- After LM Studio reports a load done, the broker looks for the new instance up to three times, one second apart, before failing the job. A dropped connection or a missing listing fails it as `backend_unavailable` (retryable) only after the third try.
- A job whose answer reaches its output budget, or comes back with no visible text, fails with `output_truncated` or `empty_output`. These are not retryable as sent. A thinking model can use its whole budget on reasoning, so raise the budget or split the work.

Only one model is loaded at a time. Before the broker loads a model, it unloads every other loaded model once that model is idle, including one you loaded by hand in LM Studio. If you unload a model in LM Studio's own window, the broker loads its model again before the next batch of jobs. The broker unloads its model once it has had no running or queued jobs for `idle_unload_seconds` (600 by default; 0 keeps the model loaded). It unloads only the instance it loaded or adopted, and only after LM Studio reports it idle; if the unload fails, it tries again after another idle period. Tier loads also give LM Studio its own idle timeout of `idle_unload_seconds` plus 60 seconds, so the model is still unloaded if the broker has stopped. The next job loads the model again and waits for the load time. `status` shows `idle_unload_in_seconds` while the timer runs.

If LM Studio itself hangs, restart it outside the broker.

## Logs and privacy

Operational events and usage summaries are stored under the configured data directory. Prompts, source files, results, tokens, authorization headers, and full backend error bodies are not broker log fields. LM Studio keeps its own independent diagnostics. If the database or event sink becomes unavailable, readiness must be treated as degraded and new submissions stopped.
