# Choosing a profile

Profiles are defined by the owner's `config.toml` and differ on every machine. Run `python -m local_llm_broker.cli profiles` to see this machine's profiles: profile, model, thinks, vision, slots, the `low`, `medium` and `high` context sizes, and a description of each.

## How to choose

- Grunt work (extraction, inventories, digests, table conversion): a profile whose `thinks` column is `no`.
- Work that needs reasoning: a profile whose `thinks` column is `yes`. Keep it to small, bounded tasks.
- Prefer the profile with the smallest model that does the job. Move to a larger one only when the smaller one's output is wrong.
- Image input needs a profile whose `vision` column is `yes`.
- The `slots` column is how many jobs the model serves at once. A model with 1 slot runs one job at a time, and later jobs wait.
- Always name the profile on the command: `--model <profile>`, using a name exactly as `profiles` prints it. Do not guess other names.

## Rules

- The broker loads the profile's model for you and unloads anything else first. Only one model is loaded at a time. Loading a model takes seconds to a minute or more depending on its size, and the first job after a switch includes the load. Batch jobs by profile to avoid switching.
- Profiles that share a model (a thinking profile and its non-thinking twin) do not reload when you switch between them, unless the tier changes.
- Parallel jobs on one model share its context. A job that fits on its own but not beside the running jobs waits for them, so a large job can take longer to start when the model is busy.
- Budget check: prompt tokens + `--max-output-tokens` + 512 must not exceed the tier's context. The broker counts the prompt exactly before the job starts; structured text such as CSV, logs or JSON uses 2 to 5 times more tokens per byte than prose (see section 4 of SKILL.md). Without `--tier` the broker picks a tier that fits, and moves the job to a larger tier if the exact count needs it; a job too large for every tier, or for the tier you named, fails with `context_exceeded`.
- Output budget: at least 4096 for thinking profiles (their reasoning counts against the budget), at least 1024 for the others unless the answer is one word.
- Check small-model output. Small non-thinking models slip on fine rules (for example, which words stay lowercase in a title), so spot-check the result against the source. These are starting points, not guarantees.
- The broker unloads the model after the configured idle time with no jobs (`idle_unload_seconds`), so the GPU is free when nobody is using it. The next job loads it again and waits for the load.
- The broker sets each tier's load settings itself. Never load a model by hand, and never change LM Studio's settings.
- If a profile fails with `model_unavailable` or `variant_unavailable`, the model is not installed under the configured name. Do not substitute another model. Report it to the owner.
- Keep each source file bounded, and state an explicit output size in the instructions.
