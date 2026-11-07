# Reproducible inference comparison protocol (v2)

This protocol is written before collecting v2 timing results. Historical results
in `validation-results.json` use an older protocol and must not be pooled with v2.
The objective is a measured comparison of a synchronous single-session text API,
not a universal ranking of frameworks or their asynchronous generation APIs.

## Three separate tracks

1. **matched_f32:** identical MLX checkpoint files, integer codes unchanged.
   Floating parameters, affine scales/biases, activations, KV and recurrent state
   are F32 in both Rust and MLX-LM. MLX `model.set_dtype(mx.float32)` casts floating
   parameters; Rust `--float-weights-f32` does the same. No requantization.
2. **native:** identical files at original floating storage precision. Rust still
   uses F32 activations/KV, whereas MLX usually uses BF16. This is a practical
   configuration comparison, explicitly **not precision matched**.
3. **gguf:** identical GGUF bytes in Rust and pinned upstream llama.cpp, GPU layers
   fully offloaded and F32 caches. Internal operation precision remains each
   implementation's default. GGUF Q4_K_M and MLX affine 4-bit are different weights
   and cannot form a matched pair across these tracks.

## Correctness gate

Before MLX timings: compare token IDs on ten Unicode/whitespace/special-token
cases; compare the entire final-token logit vector after prefill and four fixed
continuation IDs on prose, code and Russian. Prompt lengths 32 and 129 exercise
unaligned tails. Rust partitions are full request, single token, and 17 tokens;
its internal prefill tile is 32. Require finite equal-shaped logits, equal top-1,
and per-element `abs(error) <= 0.01 + 0.001*abs(reference)` in matched_f32.
Report maximum absolute error, RMSE, normalized RMSE, top-five and top-one margin.
No threshold is adjusted after seeing a failure. Failed models are excluded from
performance conclusions. Native precision differences are reported separately;
strict numerical acceptance is never inferred from a native speed measurement.

The separate GGUF comparison uses the pre-existing GGUF tolerance of absolute
0.2 plus relative 0.01 and requires equal top-one. Its internal mixed accumulation
is not a matched-F32 operation path. The first upstream comparison failed this
gate (maximum 0.288); it exposed the older fork's clamp versus additive-epsilon
normalization. The normalization policy was corrected to match upstream, without loosening
that threshold, but a remaining logit difference still fails the quantized Qwen
GGUF gate. Its speed is excluded from performance conclusions. A small F32 Llama
GGUF is used for the upstream smoke comparison instead; it cannot establish
billion-parameter throughput. Failed observations are retained as diagnostics.

State diagnostics compare token-major KV, channel-major convolution history and
`[value_head,value_dim,key_dim]` F32 recurrent matrices where requested. Synthetic
MoE/Next tests do not establish large real-checkpoint compatibility.

## Workloads and timing boundary

All prompt and continuation IDs are persisted. The prompt is drawn from repeated
prose/code/Russian text using the checkpoint's reference tokenizer. Decode is
teacher forced from the next corpus IDs: every engine sees exactly the same
history, independent of a divergent greedy output. No EOS early exit, token text
rendering, sampling RNG, network, speculative decoding, or concurrent requests.
Primary lengths: prefill 128/512/2048, decode 64; context capacity 4096. Smaller
explicit smoke runs are labelled separately. Long-context conclusions require
additional runs, rather than extrapolation from these lengths.

A forward call ends after the full last-token vocabulary logits have been copied
to host memory, checked finite and greedy argmax computed. Both Python MLX and Rust
use that synchronous API boundary. This includes Python overhead for MLX; it is
not `mlx_lm.generate`'s asynchronous token-only streaming throughput. llama.cpp
also copies its exposed logits into a host vector. MLX prefill follows `mlx_lm.generate_step`: evaluate cache state in chunks of
up to 2048 for all but the final prompt token, clear temporary cache, then compute
that token's logits. This lets lazy evaluation prune the unused vocabulary head.
A preliminary whole-prompt forward run was stopped and retained under
`timing-initial-full-forward`; it is excluded from results because it did not
exercise the optimized generation prefill path. Rust evaluates its head only on
the final internal chunk. Chunk sizes are recorded, not forced artificially equal.
Rust additionally executes its
normal GPU argmax. Input token tensor construction is inside the boundary.

Caches are preallocated to the configured capacity before forward measurements.
Reset/zeroing is timed separately. Report forward prefill, reset-inclusive request
TTFT, total decode, each decode-step latency and tokens/second. The first predicted
token belongs to TTFT; D decode steps process D further teacher-forced tokens.
No per-dispatch profiling or Metal validation during performance runs.

## Replication and order

One engine process at a time, including correctness runs, prevents doubled model
residency. Balanced blocks alternate Rust/reference order (AB, BA, AB, BA).
Each process performs two complete workload-specific warmups and two measured
repetitions per cell; four blocks yield eight observations per engine/cell.
Process start/import/load/compile is recorded separately. These are warm filesystem
cache loads, not cold disk load claims. A fresh process per engine/block limits
cross-block allocator/compilation history. All workload cells have fixed order,
recorded with the corpus IDs and hashes; different styles are not silently pooled.

Store every raw sample without trimming outliers. Summarize median throughput and
latencies; p95 is descriptive with this small N. Use per-block median paired log
ratios and 10,000 bootstrap resamples (seed 42) to form a 95% interval for geometric
speed ratio. Four independent blocks are a small sample: report the interval and
avoid fine-grained percentage claims. A CI excluding one is evidence only for
this workload/device/configuration. No hardware-wide claims from an M4 run.

## Environment, memory and invalidation

Record model revision and SHA256 of weights/config/tokenizer/template, executable
and protocol hashes, software/compiler versions, macOS/device/RAM, context,
precision policy, command lines, warmups, repetitions and order. Check thermal
state, low power mode and swap before/after each engine block. Mark samples
confounded if thermal state is serious/critical (>=2), low power mode is enabled,
or allocated swap grows during that block. Preserve them with the reason; do not
quietly replace them. Existing allocated swap is recorded, not treated as proof
of active paging. Background application load remains a limitation.

Rust retained Metal allocations, MLX active/peak/cache allocations, and sampled
process RSS are distinct metrics. RSS alone misses GPU residency; allocator
accounting is not a directly comparable whole-system footprint. Report each with
its label. Loading time includes engine initialization, compilation and session
creation; also record process-start-to-ready. A non-growing retained allocation
count over repeated fixed-capacity requests tests bounded storage, not absence
of every possible driver allocation.

## Reproduction

Use `tools/benchmarks/setup.sh` to install pinned development dependencies and build
the upstream oracle. `verify.py` writes the numerical gate and fixed workloads;
`run.py` writes raw timing blocks and `summarize.py` creates JSON/Markdown tables.
See `--help` for narrowed model/workload smoke runs. Model downloads are explicit,
revision pinned via `models.json`; model licenses remain their owners'. No models,
virtualenvs, upstream checkouts or raw logit traces are committed to source.
