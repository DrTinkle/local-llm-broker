# Local LLM Broker

A local service, command-line client and agent skill that turn one LM Studio install into a shared, managed model server for every agent session on your machine.

## The problem

A single GPU can hold one large model at a time, and LM Studio will happily let two tools fight over it: one loads a model, another asks for a different one, and both jobs fail or crawl. Agents also tend to read big logs, schemas and documents themselves, which burns their own context window on text a small local model could digest for free.

## What the broker does

- **One queue, one model at a time.** Sessions submit jobs; the broker runs them in order, loads and unloads models for you, and never lets two clients fight over the GPU. Jobs for the same model run in parallel up to the number of slots that model has, sharing its context safely.
- **Named profiles and load tiers.** A profile is a model plus its limits (context, parallel slots, thinking or not, vision). Each model has low, medium and high load tiers, and the broker picks the smallest one a request fits.
- **Guided setup and calibration.** `llm-broker setup` finds the chat models LM Studio has downloaded, lets you choose which get profiles, and can measure your hardware (how much context fits, how many parallel slots, which memory settings are fastest) so the limits fit your machine instead of guesses.
- **Safe operation.** Jobs can be cancelled, a crash or restart marks unfinished work as interrupted (never replayed), and a lock keeps a second broker from starting. `verify` runs one tiny job per profile to prove everything answers.
- **Two ways in.** A CLI (`submit`, `result`, `status`, `logs` and more) for shell-capable agents, and a bounded OpenAI-compatible endpoint (`/v1`) for any client that lets you set a base URL. Image input works on vision models.
- **An agent skill.** `use-local-models` teaches agents when delegating is worth it (large raw inputs they do not need to read themselves), how to submit files, and how to check the answer.
- **Private by design.** It binds to `127.0.0.1` only and requires a locally generated bearer token. Prompts and results stay in memory; only operational metadata (job states, timings, token counts) goes to SQLite and JSONL files. It does not depend on any other project.

## What you need

- Windows 11.
- Python 3.11 or newer.
- [LM Studio](https://lmstudio.ai) with its local server running (default `http://127.0.0.1:1234`) and at least one downloaded chat model.
- The `lms` command-line tool that ships with LM Studio. The broker finds it automatically, or you can set `lms_path` in `config.toml`.

## Quick start

1. In LM Studio, download at least one chat model and start the local server.
2. Install the broker from a clone of this repository:

   ```powershell
   git clone https://github.com/DrTinkle/local-llm-broker.git local-llm-broker
   cd local-llm-broker
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -e .
   ```

   Run every command below from this folder. Setup writes `config.toml` into the folder you run it from, and `scripts\start-broker.ps1` reads it from the repository root.

3. Create your profiles. Setup lists the models LM Studio has downloaded, asks which ones to use, and writes `config.toml`:

   ```powershell
   .\.venv\Scripts\python.exe -m local_llm_broker.cli setup
   ```

   Pick a depth when it asks (or pass `--level`):

   | Level | What it does | Time |
   | --- | --- | --- |
   | `none` | Writes profiles with cautious context sizes. Loads no model. | seconds |
   | `quick` | Loads each model a few times to find how much context fits. | about 3 minutes per model |
   | `thorough` | Also tests memory settings, parallel slots and thinking behavior. | about 30 minutes per model |

   Both calibration levels load models directly, so stop the broker first (`llm-broker stop`, see "Stopping the broker" below). If a run is interrupted, rerun setup; finished models are skipped. Setup backs up an existing `config.toml` (as `config.toml.bak-<date>-<time>`) before replacing it. If you already know what you want, skip the questions with `--yes` (it picks every downloaded model, the suggested names and level `none` unless you pass `--level`), `--models <key> ...`, or `--preset <file.toml>` (see [examples/presets](./examples/presets)).

4. Start the broker:

   ```powershell
   .\scripts\start-broker.ps1
   ```

   To start it at every logon, run `.\scripts\register-startup.ps1` once.

5. Check that every profile answers:

   ```powershell
   .\.venv\Scripts\python.exe -m local_llm_broker.cli verify
   ```

6. Give your agents the skill. Copy it into the folder your agent tool reads skills from:

   ```powershell
   .\.venv\Scripts\python.exe -m local_llm_broker.cli install-skill --target <your-skills-folder>
   ```

   Agents then run `python -m local_llm_broker.cli profiles` to see which models exist on this machine.

If `llm-broker` is on your PATH you can use it instead of `python -m local_llm_broker.cli`; it is the same program.

## Using the broker from an agent

Agents with the skill installed run `python -m local_llm_broker.cli profiles` to see what exists, then delegate bulk reading to a local model. For example, to extract the failures from a large log without putting the log into the agent's context:

```powershell
python -m local_llm_broker.cli submit --client my-agent --project my-project --task triage-log `
  --model <profile> --prompt "List each failing test with its error." --files .\build.log `
  --max-output-tokens 1024 --out .\failures.txt
```

`--files` accepts files, folders and quoted globs; `--out` saves the answer to one file; `--detach` returns at once so the agent can keep working (collect the answer with `result <job-id>`). The skill explains when delegating is worth it and how to read failures.

## Stopping the broker

```powershell
python -m local_llm_broker.cli stop
```

This shuts the broker down cleanly. If jobs are queued or running it refuses and says so; add `--force` to stop anyway (unfinished jobs are marked interrupted, never replayed). Stop the broker before calibrating, and start it again afterwards. To remove the logon start-up, run `.\scripts\remove-startup.ps1`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `LM Studio is not answering at ...` (setup exits with code 3) | Open LM Studio and start its local server (Developer tab), then rerun. |
| `No chat models are downloaded in LM Studio` | Download a chat model in LM Studio's Discover tab, then rerun setup. |
| `Stop the broker, then run setup again to calibrate` | Run `llm-broker stop`, or use `--level none`, which works while the broker runs. |
| `lock is already held` when starting | Another broker is already running. Stop it with `llm-broker stop` first. |
| `The LM Studio command-line tool (lms) was not found` | Install it from LM Studio, or set `lms_path` under `[broker]` in `config.toml`. |
| `unknown_profile` from `submit` | The name is not in your config. Run `llm-broker profiles` and use a name from it. |
| `broker is unavailable` | The broker is not running. Start it with `.\scripts\start-broker.ps1`. |
| A profile fails `verify` | Check LM Studio still has that model downloaded, then rerun `setup` for it. |

## Everyday commands

| Command | Purpose |
| --- | --- |
| `profiles` | List the configured profiles, their models and load tiers |
| `submit` | Send a job (`--files` accepts files, folders and quoted globs) |
| `result`, `wait`, `cancel` | Collect, wait for, or cancel a job |
| `status`, `doctor`, `logs`, `summary` | Inspect the broker |
| `verify` | Run one tiny job per profile |
| `stop` | Shut the broker down cleanly (`--force` even with jobs running) |
| `setup` | Create or update profiles (see Quick start) |
| `install-skill` | Copy the agent skill to a folder |
| `calibrate` | Measure speed, slots and limits for the configured tiers |

Every command and flag is listed in [the skill's command reference](./skills/use-local-models/references/client-commands.md).

## Documentation

- [Operations](./docs/operations.md): setup details, calibration, restarts and failures, logs and privacy.
- [Client integration](./docs/client-integration.md): the job API and the OpenAI-compatible endpoint.
- [Agent skill](./skills/use-local-models/SKILL.md): how agents decide when and how to delegate.
- [Design and decisions](./docs/local-llm-broker-design.md): why it works the way it does.
- [Live verification notes](./docs/live-verification.md): the checks run against a real LM Studio, recorded on the author's machine.
- [Measurements from one example machine](./docs/measurements-rtx3080ti-12gb) (an RTX 3080 Ti with 12 GB), the data behind the example preset.

## License

MIT. See [LICENSE](./LICENSE).
