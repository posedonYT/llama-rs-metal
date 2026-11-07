# llama-rs-metal

Standalone Rust inference for Apple Silicon, with native Metal kernels and direct
**GGUF and MLX checkpoint loading**. No llama.cpp, GGML, Python or MLX runtime
is linked into the engine. Python/MLX and C++ are optional development oracles.

The engine is experimental. Compatibility and performance claims below refer to
tested artifacts, rather than every model published under `mlx-community`.

## Run

Requires an Apple Silicon Mac, macOS 14+, Xcode/Command Line Tools and stable Rust.

```sh
cargo build --release --workspace

# A GGUF file or a local MLX directory is selected automatically.
target/release/llama-cli inspect /path/to/checkpoint --tensors
target/release/llama-cli generate -m /path/to/checkpoint \
  --chat -p 'What is 2 + 2?' -n 64 --temperature 0
target/release/llama-cli chat -m /path/to/checkpoint -c 4096
```

An MLX directory contains `config.json`, `tokenizer.json`, `tokenizer_config.json`,
and `model.safetensors` or indexed SafeTensors shards. `generation_config.json`
and `chat_template.jinja` are read when present. No conversion to GGUF or Python
process is required. The runtime accepts local paths; the optional development
[download tool](tools/benchmarks/download.py) fetches revision-pinned test models.

`-c` sets capacity, default `min(8192, model limit)`. Context overflow is an error.
`--memory-mib` lowers the engine's total model/session budget. `/reset` clears
chat history; `/quit` exits. Checkpoint files must remain unchanged while loaded.
Models and benchmark output are ignored by source control and downloaded separately.

## Compatibility

| Path | Implementation | Validation on this M4 |
|---|---|---|
| MLX Llama 3 text | HF layout, GQA, split-half RoPE, Llama3 scaling | Llama 3.2 1B Instruct affine 4-bit |
| MLX Qwen3.5 dense text | gated attention, convolution, DeltaNet | 0.8B affine 4-bit, 8-bit, BF16; 4B affine 4-bit |
| MLX Qwen3.5 MoE / Qwen3-Next | GPU routing, indexed experts, shared experts, hybrid state | Deterministic small synthetic F32 and mixed 4/8-bit fixtures |
| MLX affine weights | 2/4/8 bits; group sizes 32/64/128; per-module overrides | CPU/Metal operations for all combinations; real checkpoints above |
| GGUF | F32/F16/BF16, Q4_0/Q8_0, Q4_K/Q5_K/Q6_K | Operation tests; historical TinyStories/Qwen tests; see current caveats |

F16/BF16/F32 floating MLX weights and affine scales/biases are supported.
Quantized weights remain packed during inference. Native HF tokenization currently
requires a ByteLevel decoder; SentencePiece Llama 2 remains available through GGUF. **Activations, KV and recurrent
state currently use F32**, including for BF16 checkpoints. This differs from MLX's
usual BF16 compute and must be considered when comparing native configurations.
Q4_K_M/Q5_K_M are GGUF presets; the actual type is read per tensor.

MLX MXFP/NVFP and affine 3/5/6-bit modes, other architectures, unsupported RoPE
schemes and biased/sliding-window variants return explicit errors. Split GGUF
files need merging first. Vision, audio, training, HTTP serving and speculative
decoding are outside this version. Vision tensors bundled in a supported MLX
checkpoint are ignored by the text graph; retained shard pages still count in
the memory budget.

Large real MoE/Next checkpoints and M1–M3/M5 hardware have not been verified here.
Qwen3-Next-80B cannot be certified with synthetic tests or fit this 16GB Mac.
The quantized Qwen GGUF path currently exceeds the numerical gate against the
new upstream llama.cpp revision on some longer prompts; its speed is excluded
from comparative conclusions. See [current validation](docs/mlx-validation.md).

## Rust API

```rust,no_run
use llama_rs::{Engine, GenerationOptions, SessionOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(Default::default())?;
    let model = engine.load("/path/to/mlx-checkpoint-or-model.gguf")?;
    let mut session = model.session(SessionOptions { context: 4096 })?;
    for event in session.generate("The capital of France is", GenerationOptions::default())? {
        println!("{:?}", event?);
    }
    Ok(())
}
```

Models share immutable weights; sessions own independent caches. `prefill`,
`decode`, `reset`, `tokens`, `logits` and diagnostic `state_snapshot` are public.
Handles are thread confined (`Rc`); an atomic cancellation handle can be signaled
from another thread. Prefix edits reset and replay state. Chat templates use
MiniJinja with Python-compatible methods and explicit errors for unsupported
operations. Greedy, temperature, top-k/top-p, repetition penalty and seeded
sampling are available. GPU argmax is used for plain greedy; stochastic sampling
uses the CPU reference. See [architecture](docs/architecture.md) for ownership,
module boundaries, scratch lifetimes and format-specific numerical policies.

## Verify and compare

```sh
cargo test -p model-format -p tensor-core   # CPU-only, also runs on Linux
tools/test-metal.sh                        # physical Apple Silicon GPU required

# Optional, development-only dependencies and pinned upstream reference:
tools/benchmarks/setup.sh
target/mlx-env/bin/python tools/benchmarks/download.py qwen35-08-4bit
target/mlx-env/bin/python tools/benchmarks/verify.py --models qwen35-08-4bit
target/mlx-env/bin/python tools/benchmarks/run.py --models qwen35-08-4bit \
  --styles prose --output bench-results/my-run
target/mlx-env/bin/python tools/benchmarks/summarize.py bench-results/my-run \
  --output bench-results/my-summary
```

For synthetic MoE/Next validation, run `make_fixtures.py`, then `verify.py
--manifest models/mlx/fixtures/manifest.json --output bench-results/fixtures`.
The opt-in real session test runs with `LLAMA_MLX_MODEL=/path/to/checkpoint cargo
test -p llama-rs --test mlx_checkpoint -- --ignored`.

The [benchmark protocol](docs/benchmark-protocol.md) separates matched-F32,
native mixed-precision and same-GGUF tracks, records fixed token histories,
isolates processes, balances order and preserves per-step timings and environment
confounds. It measures synchronous host-visible logits; it does not claim to
measure maximum asynchronous token-only generation throughput. The CLI `bench`
command is a convenient single-engine smoke tool, not the formal comparison.
`LLAMA_METAL_PROFILE=1` synchronizes each operation for diagnosis; unset it and
Metal validation for timing.

[Contributing](CONTRIBUTING.md) · [Current validation](docs/mlx-validation.md) ·
[Historical GGUF snapshot](docs/validation.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

MIT.
