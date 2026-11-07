# Native MLX validation — 2026-11-07

M4 MacBook Pro, 10 GPU cores, 16GB unified memory, macOS 27.0. Rust/Metal executes
inference directly; MLX-LM 0.32.2 / MLX 0.32.4 is the independent development oracle.
The version-pinned checkpoints are listed in [models.json](../tools/benchmarks/models.json).

Current tests compare full F32 logit vectors and persistent state, including
prefill partitioning and continuation. Actual timings are collected only after
the numerical gate; final tables and compact raw evidence are generated under
`docs/benchmarks-2026-11-07/`. The complete methodology and remaining limitations
are in [benchmark-protocol.md](benchmark-protocol.md).

## Outcome

Native MLX loading and text generation work for the five real artifacts below.
29 regular Cargo tests pass, plus the opt-in real MLX session test. Metal API and
shader validation are enabled for GPU tests; formatting and Clippy with warnings
denied pass. Three statistical-analysis tests also pass. All five CLI chat smoke
runs answer the arithmetic prompt after correcting the no-tools Jinja context.

The passed gates contain 2,488 full-logit-vector comparisons, with equal argmax
at every checked step, plus persistent-state comparisons. Timings contain 392
measured requests across 108 isolated engine blocks (warmups are stored separately).
One BF16 matched-F32 block pair is excluded for swap growth; every observation
remains in the raw evidence.

At P512/D64, Rust is slower than MLX-LM on all five matched-F32 paths. Prefill
is the larger gap. This engine currently provides an independent Rust/Metal
implementation and direct checkpoint compatibility; these results do not establish
a speed advantage. No ranking is inferred from the native mixed-precision track
or from the failed quantized-Qwen GGUF gate.

[Throughput and 95% intervals](benchmarks-2026-11-07/results.md) ·
[Startup, TTFT, memory and endurance](benchmarks-2026-11-07/details.md) ·
[Raw evidence, gzip JSON](benchmarks-2026-11-07/raw.json.gz)

| Native MLX artifact | Maximum logit absolute error across short/long F32 gates |
|---|---:|
| qwen35-08-4bit | 0.0000787 |
| qwen35-08-8bit | 0.0000443 |
| qwen35-08-bf16 | 0.0000448 |
| llama32-1b-4bit | 0.0000758 |
| qwen35-4b-4bit | 0.0010620 |

## Numerical coverage

Real native checkpoints: Qwen3.5 0.8B affine 4-bit/8-bit/BF16, Qwen3.5 4B affine
4-bit and Llama 3.2 1B Instruct affine 4-bit. Strict comparisons preserve packed
integer codes and cast floating parameters/scales to F32 in both engines.
Token IDs, every logit, token-major K/V, convolution history and F32 recurrent
matrices are checked. Small prompts cover prose/code/Russian at lengths 32/129
and Rust partitions batch/1/17. The long gate covers prose 128/512/2048 followed
by 64 identical teacher-forced IDs. Long tests compare every decode step, not only
the final prediction. Tolerances remain absolute 0.01 plus relative 0.001; exact
argmax equality is also required.

Synthetic MoE/Next fixtures use official MLX-LM model definitions, deterministic
seed 42, tiny dimensions, full-attention and recurrent layers, shared experts,
F32 and mixed affine 4/8-bit modules. Both normalized and unnormalized expert
probabilities are exercised. They establish operator/layout consistency, not
real 80B-checkpoint validation.

Metal tests independently cover every supported affine bits/group/metadata-dtype
combination, odd dimensions and selected experts. Partitioned decode attention
is compared to an F64 scalar causal GQA oracle at contexts up to 8193 and head
dimensions up to 512, including poisoned unused future cache positions. The
opt-in real MLX session test checks prefix partitioning, two independent sessions,
reset, cancellation and fixed retained Metal bytes across 128 decode steps.

## Known GGUF discrepancy

The current quantized Qwen3.5 GGUF exceeds the upstream comparison's existing
absolute 0.2 plus relative 0.01 gate on some longer prompts (maximum about 0.288),
although the tested top-one predictions agree. Correcting the old clamp versus
additive-epsilon convention did not remove the residual difference. Its cause
is not established, and its speed is excluded from comparative conclusions.
The earlier fork snapshot in `validation.md` is historical evidence, not a pass
against the newer upstream revision. The small F32 TinyStories GGUF comparison
checks the upstream integration/dispatch path; it cannot establish large-model
throughput.

## Interpretation

Matched-F32 and native precision tracks are reported separately. Native MLX
usually computes and stores KV in BF16; Rust currently uses F32. The benchmark
uses MLX-LM's optimized generation prefill strategy and synchronous full host
logits for both engines. It does not measure the highest asynchronous token-only
streaming throughput, multi-user serving, energy or every model architecture.
Thermal state, low power mode, swap growth, exact artifacts, raw per-step timing,
allocator metrics and balanced process order are retained. A [separate memory
experiment](memory-method.md) records OS charged footprint and RSS alongside
framework allocation counters. Confounded blocks stay
in the evidence and are excluded from paired ratio inference.
