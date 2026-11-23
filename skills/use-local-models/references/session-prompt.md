# Short prompt for sessions without skill support

Paste this into a session that cannot load the `use-local-models` skill:

---

You may delegate large, context-shielding extraction jobs (15 KiB or more, about 3,500 tokens, of raw files you have not read) to local models through the Local LLM Broker on this computer. Each broker call costs about 5,000 input-token equivalents in tool-turn overhead, so never use it for small inputs (under 15 KiB), files you already read (or must read in full anyway), checkpoints or commit messages, arithmetic or expected-value math, or code edits.

Command prefix (run from any folder): `python -m local_llm_broker.cli`

1. Check the broker: `python -m local_llm_broker.cli status`. If it says `broker is unavailable` (exit 3), do not use it. Tell the owner and do the task yourself.
2. See which profiles exist on this machine (they differ from computer to computer) and pick one. Use a profile whose `thinks` column is `no` for grunt work, a thinking profile only when the task needs reasoning, and a larger model only when a smaller one's output is wrong.
3. Pass the unseen raw files with `--files` (files, directories or quoted globs; do not list the files yourself), with `--max-output-tokens` of at least 1024 (at least 4096 for a thinking profile):
   `python -m local_llm_broker.cli submit --client <your-tool> --project <generic-name> --task <short-label> --model <profile> --prompt-file <instructions> --files <large-unseen-files...> --max-output-tokens 1536 --idempotency-key <stable-name> --out <extract-file>`
   Do not sit and wait for it: unless your next step needs the answer, run this command in the background if your tool can (it saves the answer to `--out` and you read the file when it ends), or add `--detach`, keep working, and later run `python -m local_llm_broker.cli result <job-id>` within 10 minutes (it returns at once; `--out` is not written with `--detach`).
   Read only the compact `--out` file in your session and spot-check specific lines with `grep` or a script rather than reading the raw input files into your context window.
4. If the exit code is not 0, read the stderr line, for example `model_load_failed (retryable: yes): ...`. If `retryable: yes`, resubmit the same job once after a short wait. Otherwise stop and report the code to the owner.
5. Never call LM Studio directly. Never load, unload, or switch models. Never restart the broker or LM Studio. Never send secrets or personal records. Never put prompt text, file contents, or tokens in labels.
6. Check every answer yourself before you use it. A local answer is not evidence.

The full guide is at `<path-to-repo>\skills\use-local-models\SKILL.md`. Run `python -m local_llm_broker.cli profiles` to see the available models.

---
