# Contributing

The runtime is a standalone Rust/Metal implementation. Reference C++/MLX programs
are development oracles only. Please keep file formats, numerical semantics and
GPU execution separate; see [architecture](docs/architecture.md).

Use stable Rust and `cargo fmt`. On any development machine, format and CPU
reference tests run with `cargo test -p model-format -p tensor-core`. Native builds
need macOS 14+ and Apple Silicon. `tools/test-metal.sh` runs the complete suite
with Metal API and shader validation. Hosted CI checks compilation; it does not
certify GPU execution. Run all tests after changing a kernel or state layout.

For a new checkpoint family, add an explicit format adapter and numerical policy,
then compare deterministic small fixtures against an independent implementation.
Include malformed/truncated input tests for loaders. Never infer model support
from successful tensor loading alone. Test prefill partitioning, causal attention,
continuation, reset, cancellation, independent sessions and fixed-capacity memory.
Keep additions scoped and document unsupported options with typed errors.

For optimization, follow [the benchmark protocol](docs/benchmark-protocol.md),
keep raw observations and model/precision identity, and run the correctness gate
before interpreting speed. Don't compare different quantizations as equivalent
work. Profiler and Metal validation runs must be separate from timing runs.
Large downloaded models, virtual environments, upstream checkouts and raw logits
belong in ignored `models/`, `target/` and `bench-results/`, not source control.
Small synthetic fixture generators and compact result summaries belong in source.

Changes should explain the behavior they alter, the checks performed and remaining
hardware/checkpoint limitations. Preserve upstream license notices when adapting
algorithms or shaders. No benchmark speedup or full model support should be claimed
without evidence on the documented hardware and exact artifact.
