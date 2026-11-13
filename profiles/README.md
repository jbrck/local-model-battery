# Model profiles

One file per model. A profile is an override for a model that neither reader can
describe: not the GGUF header, and not a HuggingFace model directory. Even then
it only supplies the numbers those readers could not. The normal path is no
profile at all: point `scripts/plan_run.py` at the model with `--model-file` or
`--model-dir`, or let it read from a live endpoint, and the shape comes off the
model itself. Nothing here is a benchmark setting: caps, thinking and seeds all
come from `standard.toml`.

## Fields

Copy `example.toml` only when you actually need the override. Every number is
one you read off a source that knows the model, never one you derive from
another number.

| Field | What it is | Where it comes from |
|---|---|---|
| `id` | the name this run uses, defaults to the file stem | you |
| `weights_gb` | size of the weights on disk | `ls -l` the model file, divide bytes by 1073741824 |
| `quant` | the quant label, for the record | the filename, e.g. `Q4_K_M` |
| `[model].n_layers` | transformer block count | the model's own published metadata |
| `[model].n_kv_heads` | KV heads (GQA) | the model's own published metadata; equal to `n_head` only when there is no GQA |
| `[model].head_dim` | attention head dimension | the model's own published metadata. Some architectures have a `head_dim` that differs from `n_embd / n_head`, so never derive it |
| `[model].kv_type` | KV cache type actually served | the serve command, `--cache-type-k`; defaults to `standard.toml`'s `serve.kv_k` |

`weights_gb` and the three shape numbers are required for a real fit answer. Leave them out
and the plan says so and hands the placement question to the engine.

## What the numbers do

The plan computes KV bytes per token, then the window that fits:

    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * bits_per_element / 8

The `2` is keys and values. For a 27B q4_k_m with 64 layers, 8 KV heads, head_dim 128 and a
q8_0 cache:

    2 * 64 * 8 * 128 * 8 / 8 = 131072 bytes per token = 128 KiB

So 32768 tokens of context costs 4 GiB of VRAM. That is the number that decides whether a
model "barely fits" on a 24GB card, and therefore whether the window can stay at the
standard's 131072 or has to come down. When it comes down, every run on this box goes
off-standard and the manifest says so.

## The worked example

`example.toml` describes a 27B q4_k_m on a single 24GB card: about 16.4 GB of weights, a
q8_0 KV cache, and enough VRAM for a long window because the KV is quantised. A 40B q4_k_m
with the same shape numbers but a f16 cache does not fit the same way, which is exactly the
kind of thing the profile is for.
