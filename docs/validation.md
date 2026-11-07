> Historical snapshot before the native MLX adapter/refactoring. Current results
> and remaining upstream-GGUF discrepancies are in [mlx-validation.md](mlx-validation.md).

# Validation snapshot — 2026-11-07

Hardware: MacBook Pro, Apple M4 (10 GPU cores), 16 GB unified memory, macOS 27.0.
Reference: ai-janitor/llama-metal revision
`72ba0238e1a65defbffaa384caadd84121c75f1d`. The reference is a development oracle,
not a runtime dependency. Raw measurements and model hashes are preserved in
[validation-results.json](validation-results.json).

## Correctness

18 Cargo tests pass, including actual Metal execution with both API validation
and GPU shader validation enabled. Clippy passes with warnings denied, and Rust
format checks pass. Tests cover every supported packed weight type, odd matrix
sizes, tiled prefill, GQA, greedy tie breaking, nonfinite logits, recurrent state,
MoE, Next legacy projections, prefix edits, cancellation, context overflow,
corrupt GGUF ranges and stable live GPU allocations across 192 decode steps.

Independent tokenizer comparison passed 20 cases across the TinyStories Llama
and Qwen3.5 vocabularies: Latin/Cyrillic/Chinese text, composed/decomposed accents,
emoji, repeated whitespace, numbers, contractions, CRLF and special-token markers.

Reference final-logit comparisons use the same token IDs and F32 caches:

| Check | Maximum absolute difference |
|---|---:|
| TinyStories 260K Llama real GGUF | 0.0000062 |
| Small F32 Llama fixture | 0.00000012 |
| Small F32 Qwen3.5 dense fixture | 0.0000154 |
| Small F32 Qwen3.5 MoE fixture | 0.0000221 |
| Small F32 Qwen3-Next fixture | 0.0000352 |
| Real Qwen3.5-0.8B Q4_K_M | 0.16325; RMSE 0.03205 |

All cases selected the same top-1 token. The quantized comparison is accepted at
an explicit absolute tolerance of 0.2, not the much tighter F32 fixture tolerance.
Its remaining difference is not explained by these tests; matching top-1 on one
prompt does not establish general generation equivalence. Real Qwen3.5 chat also
answered `What is 2 + 2?` with `The answer is **4**.`

## Performance

Release build; 128 identical prompt IDs, 32 identical teacher-forced decode IDs,
512 context capacity, three measured repetitions after warmup. Greedy selection
is included, and KV/recurrent states are F32 in both engines. These are local
snapshots under ordinary desktop conditions, not cross-device performance claims.

| Engine | Median prefill tokens/s | Median decode tokens/s |
|---|---:|---:|
| Rust/Metal | 378.7 | 62.2 |
| Pinned llama-metal reference | 1004.4 | 72.4 |

The final Rust decode path reaches about 86% of reference throughput on this case.
Packed K-quant dot products improved decode from roughly 25 to 62 tokens/s on the
same teacher-forced workload. Wider SIMD-group matrix tiles improved prefill from
roughly 327 to 379 tokens/s. Prefill remains the larger performance gap.

The session estimate for this workload is 45,574,144 bytes, in addition to
521,555,200 bytes of tensor weights. Benchmark JSON also reports retained Metal
buffer allocation bytes and process RSS. RSS alone does not account reliably for
all GPU-resident unified memory; neither estimate includes every driver allocation.

To reproduce the same-token reference benchmark after building the oracle:

```sh
cargo run --release -p llama-cli -- bench -m models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -c 512 --prompt-tokens 128 --decode-tokens 32 --repetitions 3 --output rust.json
python3 - <<'PY'
import json
from pathlib import Path
j = json.loads(Path('rust.json').read_text())
Path('prompt.tokens').write_text(''.join(f'{t}\n' for t in j['prompt_tokens']))
PY
target/reference_bench models/Qwen3.5-0.8B-Q4_K_M.gguf prompt.tokens 32 3 > reference.json
```

Reference decode inputs cycle through the supplied prompt IDs, matching `bench`.
For tokenizer verification:

```sh
python3 tools/check_tokenizers.py models/stories260K.gguf models/Qwen3.5-0.8B-Q4_K_M.gguf
```

## Verification limits

Small generated fixtures establish operation/layout compatibility, including
Qwen3.5 tiled Q/K head repetition and Next interleaved repetition. They do not
establish full compatibility with real large MoE/Next checkpoints. Those require
a Mac with enough memory. Real billion-parameter Llama 2/3 checkpoints and M1–M3/M5
hardware were not available for verification. Training, multimodal inference,
IQ formats, split-file loading and a server are outside this implementation.

Current remaining optimization work includes parallel DeltaNet chunk prefill,
more shape-specific matrix tiles/fusions, lower-precision activation/cache modes,
and GPU stochastic sampling. There is no claim of reference-wide feature parity.
