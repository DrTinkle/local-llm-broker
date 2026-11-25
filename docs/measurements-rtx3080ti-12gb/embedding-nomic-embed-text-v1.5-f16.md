# Embedding profile measurements: Nomic Embed Text v1.5 (F16)

Measured 2026-11-25 on the example machine (RTX 3080 Ti, 12 GB; Qwen chat models), through the running broker.
The profile is `embed`: `kind = "embedding"`, `model_id = "text-embedding-nomic-embed-text-v1.5@f16"`,
`context_ceiling = 2048`. Your numbers will differ. Reproduce with `llm-broker calibrate --mode embeddings --profiles embed`.

## Calibration sweep

Raw output of `calibrate --mode embeddings` (parallel ceiling raised to 16 for the sweep):

```
Embedding profile: embed
Input length (one text per request)
  ~  256 tokens ( 197 words)  ok  0.03 s
  ~  512 tokens ( 394 words)  ok  0.03 s
  ~ 1024 tokens ( 788 words)  ok  0.05 s
  ~ 1536 tokens (1182 words)  ok  0.05 s
  ~ 2048 tokens (1575 words)  ok  0.06 s
  ~ 2560 tokens (1969 words)  ok  0.11 s
  ~ 3072 tokens (2363 words)  ok  0.09 s
  ~ 4096 tokens (3151 words)  ok, cut off  0.11 s
  ~ 8192 tokens (6302 words)  ok, cut off  0.09 s
  longest accepted: ~8,192 tokens
  longer text is cut off, not refused: the model reads about 2,560 to 3,072 tokens (estimated)
Batch size (one request, ~130-token texts)
     1 texts  0.03 s  32.3 texts/s
     4 texts  0.05 s  85.1 texts/s
    16 texts  0.16 s  102.6 texts/s
    64 texts  0.56 s  113.7 texts/s
   128 texts  1.08 s  118.7 texts/s
Concurrency (requests at once, 8 texts each)
     1 at once  102.6 texts/s
     2 at once  106.0 texts/s
     4 at once  125.3 texts/s
     8 at once  135.0 texts/s
    16 at once  133.6 texts/s
  usable concurrency: 1
```

- The model never refuses long input. Text past its 2,048-token window is cut off silently: vectors for longer
  texts equal the vector of the text cut at the window. The token counts above are estimates (about 1.3 tokens per
  word), and the synthetic words used here take more than that, so the cut-off shows up at 2,560 to 3,072
  estimated tokens. Chunk documents to fit the window.
- Vectors have 768 dimensions.
- One run to the next, throughput moved by up to 40% (the GPU was shared with other work), so read the plateau, not
  single numbers. A larger batch helps up to about 64 texts per request.

## Concurrency with and without a chat job

`mixed_test`: 8 texts of about 130 tokens per request, N requests at once, first on an idle GPU and then while a
chat job (the 4B model, about 600 generated tokens) was generating.

```
idle GPU (no chat job running):
  1 at once: 76.9 texts/s
  2 at once: 170.2 texts/s
  4 at once: 205.1 texts/s
chat model alone, one job:
  582 tokens in 4.3s = 134.4 tok/s
chat job running while embeddings run:
  1 at once: 102.6 texts/s
  2 at once: 120.8 texts/s
  4 at once: 124.0 texts/s
  chat job: 631 tokens in 4.6s = 136.4 tok/s
```

- Throughput stops growing at about 4 requests at once, so the example profile uses `parallel_ceiling = 4`.
- A running chat job costs embeddings about 40% of their throughput, but the chat job's own speed did not change
  (134 against 136 tokens per second).

## Coexistence

The embedding model stays loaded beside the chat model. With it resident, the 4B, 9B and 35B chat profiles all
loaded at their low, medium and high tiers, answered, and embedding requests kept working after each switch.
With the 35B loaded the GPU used about 11.8 of 12.3 GB, so there is little headroom. On a smaller card, check that
a chat tier still loads with the embedding model resident.
