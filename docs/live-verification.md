# Live verification checklist

These notes were recorded on the author's machine (an RTX 3080 Ti with 12 GB, with the Qwen models in [the example preset](../examples/presets/rtx3080ti-12gb.toml)); your numbers will differ. This document is intentionally an operator checklist; deterministic tests cover unsafe failures without provoking real OOM or killing LM Studio.

The first live run (2026-10-25): the two-client, barrier, cancellation, streaming, restart and log-privacy items passed. A variant-identifier bug that the run exposed is fixed. The 35B then loaded automatically with the saved preset.
- Record Python, LM Studio, broker, configured profile IDs, and the active 35B preset.
- Pause old-helper submissions and establish backend idle before the window.
- Start one foreground broker process and confirm `/health`, `/status`, `/v1/models`.
- Submit two short same-model jobs from independent processes; record queue, load, inference, and first-token metrics.
- Queue a different-model job followed by the original model; verify the barrier and one model switch.
- Verify the 35B preset's CPU placement survives the supported load path. Verified 2026-10-25: a broker-shaped load gets the saved forced-CPU layers (engine log: 23). The 35B is managed and loads automatically.
- Exercise one non-streaming request, one SSE request, queued cancellation, and a restart with a model loaded. The broker should start ready without a recovery step. A failed in-run load or unload fails its job, and the next job retries it automatically.
- Inspect event and usage logs for absence of prompts, answers, file paths, credentials, and full backend error bodies.
- Record untested behavior and limitations; do not claim backend idle from an empty instance list alone.

## Recorded presets (2026-10-25)

Recorded from the author's LM Studio load-panel screenshots. The broker must not change these values. They show what the UI has saved. The 2026-10-25 check below shows the 35B's saved preset is applied to a broker-shaped REST load.

| Setting | `qwen3.6-35b-a3b` (`35b`) | `qwen3.5-9b` (`9b`) | `qwen3-4b-thinking-2507` (`4b`) |
| --- | --- | --- | --- |
| Context length | 28000 | 24000 | 16000 |
| GPU offload | 40 | 32 | 36 |
| CPU thread pool size | 8 | 8 | 8 |
| Evaluation / physical batch size | 2048 / 512 | 2048 / 512 | 2048 / 512 |
| Max concurrent predictions | 1 | 2 | 4 |
| Number of experts | 8 | not shown in panel | not shown in panel |
| Number of layers for which to force… (Experimental; label truncated in screenshot) | 22 | not shown in panel | not shown in panel |
| Unified KV cache / offload KV cache to GPU | on / on | on / on | on / on |
| Keep model in memory / try mmap() | on / on | on / on | on / on |
| Flash attention | on | on | on |
| Context checkpoints | 32 | 32 | 32 |
| Speculative decoding | Off | Off | Off |
| K / V cache quantization | off / off | off / off | off / off |
| Seed | random | random | random |

The forced-layer value was 21 when the design was written (2026-10-24) and is 22 on 2026-10-25. LM Studio's engine logs 23 layers for that preset, so the UI number and the engine's count differ by one; the owner should confirm the intended value. The 9B preset was supplied on 2026-10-25 and sets its ceiling to 24000 context / 2 parallel.

## Shared-context check verified (2026-11-04, evening)

`calibrate --concurrency` on all three models with the calibrated tiers (raw output: `docs/measurements-rtx3080ti-12gb/concurrency-checked.txt`). Every shared-context round finished on every tier, including rounds filled to 103% to 135% of the tier's context by real token count, which before the check made LM Studio fail every running job. The broker held back jobs that did not fit beside the running ones and ran them once there was room.

The throughput rounds in the same run were not usable: combined speed was well below the earlier runs (4B 4 jobs 193 against 385 tok/s; 35B 2 jobs 32.5, below 1 job's 38.5). LM Studio's log showed why: each exact count opened and closed its own SDK connection, taking 0.5 to 1 second, so parallel jobs started one after another. The broker now keeps one SDK connection open for counting. Throughput still needs measuring again through the broker.

## Calibrated values applied (2026-11-04)

From the calibration suite below, keeping 15 to 40% under the largest context `search` found, because that limit varied between days:

| Model | low | medium | high | Slots |
| --- | --- | --- | --- | ---: |
| 4B (both) | 16,384 f16 | 49,152 f16 | 81,920 q8_0 | 4 |
| 9B | 32,768 f16 | 131,072 q8_0 | 163,840 q8_0 | 3 (was 2) |
| 35B | 32,768 f16, ratio 0.65 (was 0.625) | 131,072 q8_0, 0.675 | 262,144 q8_0, 0.775 | 1 |

The 9B slot increase (2 to 3) came with the shared-context check (operations.md, "Token counting and shared context"), since more parallel jobs on a shared pool is where one overfill would fail them all. LM Studio's saved presets were set to the new `medium` values and slot counts for hand loads. None of these values has been re-measured as a set yet.

## Calibration suite (2026-11-04, afternoon)

Raw outputs are in `docs/measurements-rtx3080ti-12gb/` (tokens, depth, search, slots, experts).

**Tokens.** Real bytes per token, 4B / 9B tokenizer: English prose 5.33 / 5.33, Finnish prose 2.67 / 3.23, Python 2.83 / 2.72, JSON 2.13 / 1.90, CSV numbers 1.00 / 1.00, log lines 1.41 / 1.41, Markdown table 1.84 / 1.74. The broker's 4-bytes-per-token estimate is therefore up to 4x too low for structured text; no single ratio is safe. Images on the 9B and 35B cost 66, 258, 1,026 and 2,306 tokens at 256, 512, 1,024 and 1,536 pixels square: about pixels / 1,024 + 2, against the broker's flat 1,200.

**Depth (through the broker).** Generation at about 1.5K / 6K / 12K / 24K real prompt tokens:

| Tier | 1.5K | 6K | 12K | 24K |
| --- | ---: | ---: | ---: | ---: |
| 4B medium (f16) | 170 | 152 | 132 | |
| 4B high (q8_0) | 153 | 108 | 82 | 57 |
| 9B low (f16) | 109 | 102 | 102 | |
| 9B high (q8_0) | 101 | 99 | 94 | 82 |
| 35B low (f16, 0.625) | 30 | 31 | 33 | |
| 35B medium (q8_0, 0.675) | 41 | 44 | 43 | 41 |
| 35B high (q8_0, 0.775) | 39 | 38 | 35 | 36 |

A q8_0 cache costs the 4B heavily as the prompt grows; the 9B and 35B barely notice it. The 32K depth on 32,768-context tiers was skipped by a calibrate bug (its prompt was 1% over the broker's estimate), since fixed. The 35B's prompt processing varied between runs (about 370 to 950 tok/s for the same settings) and tended to rise over the first requests after a load.

**Search (direct, 300 MiB VRAM headroom and speed within 15% of a 16K baseline).** Largest context that fits:

| Model | Cache | CPU expert ratio | Largest context |
| --- | --- | --- | ---: |
| 4B | f16 | | 57,344 |
| 4B | q8_0 | | 102,400 |
| 9B | f16 | | 147,456 |
| 9B | q8_0 | | 258,048 |
| 35B | f16 | 0.625 | 98,304 |
| 35B | q8_0 | 0.625 | 106,496 |
| 35B | q8_0 | 0.775 | 262,144 (11,264 of 12,288 MiB used) |

Overflows showed as prompt processing collapsing to a few hundred tok/s or less while VRAM read close to full. These limits are higher than the 2026-11-03 and first 2026-11-04 overflow observations (9B 196,608 q8_0; 4B 98,304 q8_0), which suggests other programs' VRAM use varies; tier values should keep a margin below them.

**Slots (direct, low-tier settings).** Combined generation tok/s: 4B 286 / 304 / 308 at 4 / 6 / 8 slots; 9B 168 / 206 / 221 at 2 / 3 / 4; 35B 37 / 52 at 1 / 2.

**Experts (direct, 35B at 32,768 context, f16).** Generation tok/s by CPU expert ratio: 0.50 overflowed (16.2), 0.55 overflowed (28.4), 0.60 38.3, 0.65 40.8, 0.70 37.3, 0.75 35.0, 0.80 36.1, 0.85 33.7, 0.90 33.2, 0.95 33.0, 1.00 31.0. The sweep's report marked 0.50 as fitting because it checked VRAM headroom only; it now also requires prompt processing to stay above half of the sweep's best.

## Concurrency calibration (2026-11-04)

`calibrate --concurrency` on the 4B and 9B, after which the 35B check below ran.

Throughput, about 2,000-token prompts and 200 output tokens per job (combined generation tok/s at 1, 2, 3, 4 jobs):

| Profile and tier | 1 | 2 | 3 | 4 | Usable slots |
| --- | ---: | ---: | ---: | ---: | --- |
| 4b-nothink low | 159.3 | 275.5 | 344.5 | 384.6 | 3 of 4 (the 4th adds 12%) |
| 4b-nothink medium | 162.5 | 225.3 | 347.6 | 396.6 | 3 of 4 (the 4th adds 14%) |
| 4b-nothink high | 146.1 | 189.2 | 276.4 | 318.7 | 4 of 4 |
| 9b-nothink low | 95.0 | 118.1 | | | 2 of 2 |
| 9b-nothink medium | 98.2 | 123.9 | | | 2 of 2 |
| 9b-nothink high | 99.5 | 119.6 | | | 2 of 2 |

The 4th 4B slot still adds 12 to 14% combined speed at about 100 tok/s per job, so the 4B keeps 4 slots.

**Throughput rerun, 2026-11-04, after the long-lived token-counting connection fix** (`calibrate --concurrency --no-capacity`, 9B at 3 slots and 35B at 2; raw output in `docs/measurements-rtx3080ti-12gb/concurrency-throughput.txt`). Every job finished.

| Profile and tier | 1 | 2 | 3 | 4 | Usable slots |
| --- | ---: | ---: | ---: | ---: | --- |
| 4b-nothink low | 160.6 | 224.8 | 348.8 | 368.6 | 3 of 4 (the 4th adds 6%) |
| 4b-nothink medium | 157.6 | 219.2 | 309.5 | 361.3 | 4 of 4 |
| 4b-nothink high | 148.0 | 189.8 | 237.8 | 306.3 | 4 of 4 |
| 9b-nothink low | 102.3 | 127.1 | 182.8 | | 3 of 3 |
| 9b-nothink medium | 99.7 | 122.6 | 172.3 | | 3 of 3 |
| 9b-nothink high | 97.5 | 122.6 | 179.4 | | 3 of 3 |
| 35b-nothink low | 39.7 | 40.2 | | | 1 of 2 |
| 35b-nothink medium | 37.3 | 24.9 | | | 1 of 2 |
| 35b-nothink high | 30.9 | 31.4 | | | 1 of 2 |

Through the broker a second 35B job adds no combined speed (and lost a third on medium in this single run), unlike the direct `slots` probe. The 35B profiles are now set to 1 slot, which also gives each 35B job the whole context. 4B keeps 4 and 9B keeps 3.

**End-to-end check, 2026-11-04, final settings** (raw output in `docs/measurements-rtx3080ti-12gb/end-to-end-check.txt`). One job ("What is 17 + 25?", 2,048 output budget) through every profile on every tier, via `/v1/chat/completions`. All 18 answered `42`; `status` reported the requested tier, its context and the configured slots (4 / 3 / 1) each time. Seconds include any load:

| Model | non-thinking (load + answer) | thinking, same tier already loaded | thinking output tokens |
| --- | --- | --- | ---: |
| 4B (two models, each loads) | 3.8 to 5.1 | 5.9 to 6.9 (includes its own load) | 394 |
| 9B | 6.8 to 11.5 (11.5 includes unloading the 4B) | 2.0 to 2.3 | 190 to 210 |
| 35B | 22.0 to 30.1 (30.1 includes unloading the 9B) | 4.7 to 5.3 | 159 to 163 |

Two `35b-nothink` jobs sent together at `high`: A finished at 5.9 s, B at 12.5 s; `status` never showed more than one running and showed one queued, so the one-slot limit holds.

Shared context: every round whose jobs together fit the tier's context finished, on every tier. The planned 150% rounds overfilled the real pool (the planned fills come from the broker's 4-bytes-per-token estimate; this prose tokenized at about 5 bytes per token, so the "120%" rounds were 70 to 90% real). In each of them LM Studio logged `failed to find free space in the KV cache`, then `Context size has been exceeded`, and failed every running job at once, not only the one that overflowed. Parallel jobs therefore must not be dispatched beyond the loaded tier's context in total.

35B CPU expert layers: with `cpu_expert_ratio = 1.0` (all 40 layers) on the `low` tier, generation was 29.5 tok/s against 36.9 at 0.625 (25 layers), so the raw `numCpuExpertLayersRatio` key applies. Restored to 0.625.

## Tier calibration (2026-11-04)

First `calibrate` run, through the broker's SDK tier loads, with a 15,137-token synthetic prompt (`-nothink` profiles):

| Profile | Tier | Context | Prompt tok/s | Generation tok/s | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| 4b-nothink | low | 16,384 | 5,004 | 116.7 | ok |
| 4b-nothink | medium | 32,768 | 4,978 | 121.4 | ok |
| 4b-nothink | high | 98,304 | 700 | 43.5 | VRAM overflow |
| 9b-nothink | low | 32,768 | 3,697 | 94.3 | ok |
| 9b-nothink | medium | 65,536 | 3,740 | 85.7 | ok |
| 9b-nothink | high | 131,072 | 3,711 | 88.5 | ok |
| 35b-nothink | low | 32,768 | 243 | 36.9 | ok |
| 35b-nothink | medium | 131,072 | 389 | 37.6 | ok |
| 35b-nothink | high | 262,144 | 442 | 33.6 | ok |

LM Studio's log shows every load at its tier's exact context, and the 35B loads no longer log the "tensor overrides to CPU are used with mmap enabled" warning, so `try_mmap = false` applied. The log does not report the CPU expert layer count, so whether the raw `numCpuExpertLayersRatio` key applies is not yet confirmed. The 4B `high` tier was lowered to 65,536 after this run. Re-measured the same day: `low` 4,950 prompt / 120.2 generation tok/s, `high` (65,536, q8_0) 4,702 / 72.5. Prompt processing held, so this is the q8_0 cache's cost at a 15K-token depth, not an overflow; calibrate now reports it as `slower generation`.

## Load tiers replace the presets (2026-11-04)

Since 2026-11-04 the broker sends each load's settings itself, per tier, through LM Studio's Python SDK. The context, KV cache, CPU expert ratio and mmap values in the presets tables below no longer apply to broker loads; see `[tiers]` in `config.toml` and "Load tiers and calibration" in operations.md. The saved presets were reset to each model's `medium` tier for loads made by hand.

The 2026-11-03 presets below overflowed VRAM on the 9B. LM Studio's log for the 2026-11-03 benchmark run shows, for the same 8,545-token prompt:

| Model and load | Prompt processing | Generation |
| --- | ---: | ---: |
| 9B, 24K context, f16 cache (before) | about 3,500 tok/s | 91 to 92 tok/s |
| 9B, 192K context, q8_0 cache | 1,313 tok/s | 60 tok/s |
| 4B Instruct, 16K, f16 (before) | about 5,600 tok/s | 125.7 tok/s |
| 4B Instruct, 96K, q8_0 | about 5,300 tok/s | 112.7 tok/s |
| 35B, 28K, f16, 24 CPU expert layers (before) | 72 to 168 tok/s | 31 to 32 tok/s |
| 35B, 262K, q8_0, 30 CPU expert layers | 114 tok/s | 25 tok/s |

The 9B drop is the size of a VRAM overflow into shared memory, which is why 196,608 was ruled out for the 9B `high` tier. The 4B drop is about the cost of a q8_0 cache in a model with full attention in every layer, so its `low` and `medium` tiers use f16. The 35B's slow prompt processing appears with mmap on, which LM Studio's engine warns about on every 35B load ("tensor overrides to CPU are used with mmap enabled"); its tiers set `try_mmap = false`. The tiers were calibrated on 2026-11-04 (see `config.example.toml`): the current `high` tiers are 4B 81,920 q8_0, 9B 163,840 q8_0 and 35B 262,144 q8_0 at `cpu_expert_ratio = 0.775`.

## Approved presets (2026-11-03)

Written to LM Studio's saved per-model defaults (`.lmstudio/.internal/user-concrete-model-default-config/qwen/*.json`) on 2026-11-03, replacing the 2026-10-25 values above. Context sizes were sized from each model's KV cache cost (layer and head counts in LM Studio's GGUF index), not measured; confirm each with a live load. The backups made beside them were deleted on 2026-11-04.

| Setting | `qwen3.6-35b-a3b` (`35b`) | `qwen3.5-9b` (`9b`) | `qwen3-4b-thinking-2507` (`4b`) and `qwen3-4b-2507` (`4b-nothink`) |
| --- | --- | --- | --- |
| Context length | 262144 | 196608 | 98304 |
| K / V cache quantization | q8_0 / q8_0 | q8_0 / q8_0 | q8_0 / q8_0 |
| CPU expert layers (ratio) | 0.75 (30 of 40) | n/a | n/a |
| Max concurrent predictions | 1 | 2 | 4 |
| Flash attention | on | on | on |
| Evaluation batch size | 2048 | 2048 | 2048 |
| CPU thread pool size | 8 | 8 | 8 |

Physical batch size was not changed (no documented key found). Broker ceilings in `config.toml` and `config.example.toml` match this table, including `9b-nothink` and `35b-nothink`. `max_text_bytes` is 1 MiB and `stalled_stream_timeout` is 600 s, because LM Studio sends no stream chunk during prefill.

### What the REST load path can and cannot control

Checked against LM Studio's REST docs on 2026-10-25:

- `POST /api/v1/models/load` documents only `model`, `context_length`, `eval_batch_size`, `flash_attention`, `num_experts`, `offload_kv_cache_to_gpu`, and `echo_load_config`. It has no field for forced CPU layers, GPU offload, or max concurrent predictions.
- The docs do not say whether omitted fields fall back to the saved preset. Live check (2026-10-25): a load that sent only `model` got the saved 28000 context, 1 parallel and 23 CPU expert layers. A load that also sent `context_length` (the broker's shape) still got 23 CPU expert layers. The layer count is read from LM Studio's engine log, because the REST listing does not show it.
- Managed loads send `context_length` explicitly. Keep each managed profile's ceiling equal to its saved preset, or the request overrides the preset. The 35B is managed too and sends the same load body. A loaded context more than 1024 tokens above the ceiling is unloaded. Its job fails retryably with `context_above_preset`, and the next job loads the model at the preset.
- The documented load echo does not list `parallel`. The adapter therefore reads `context_length` and `parallel` from the loaded instance in `GET /api/v1/models`, which documents both and reported both on the 2026-10-25 read-only check. If either is missing, the instance is unloaded, nothing is dispatched to it, and the job fails retryably with `caps_unverified`. A changed "Max concurrent predictions" in LM Studio is therefore picked up at load time.
- No documented field reports GPU offload for model layers. `lms ps --json` does not expose the forced-layer setting (checked 2026-10-24), so read it back from LM Studio's engine log (`Resolved GPU config options`, `Num CPU Expert Layers`) during a live window.


## First-run setup check (2026-11-23)

A first-run check of `llm-broker setup` on a machine with four downloaded chat models, against a temporary config file and a temporary broker on a spare port (the regular broker was stopped for the run and the temporary one was stopped afterwards).

- `setup --level quick --models qwen/qwen3-4b-2507 --yes` took about 3 minutes for one model. It found that an f16 cache fits 49,152 context and wrote three tiers (8,192, 20,480 and 40,960; f16) with 4 slots. Nothing stayed loaded in LM Studio afterwards once the temporary instance was unloaded by hand.
- `llm-broker verify` against the temporary broker passed: one job, 5.4 s including the model load.
- `setup --level none --yes` wrote a valid config for all four models in seconds without loading anything, and kept the `[broker]` settings already in the file.
- A fresh clone installed into a new virtual environment ran `setup`, `profiles` and `install-skill` against a mock LM Studio listing.

The quick level uses coarse steps, so its largest context is lower than the thorough search finds for the same model (an earlier full search on the same machine found f16 fitting 57,344 for this model). The tiers are cautious by design.
