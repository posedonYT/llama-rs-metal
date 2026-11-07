# Architecture

The central rule is that checkpoint conventions end at the file adapter. The GPU
executor accepts semantic tensor roles and explicit numerical policies; it does
not parse GGUF metadata, SafeTensors names or Hugging Face JSON during decoding.

## Crate boundaries

| Crate | Owns | Does not own |
|---|---|---|
| `model-format` | GGUF/SafeTensors validation, mmap ownership, immutable packed tensors, config normalization | Metal objects, session state |
| `tensor-core` | Checked shapes/views, scalar operation oracles | File names, model architecture |
| `metal-backend` | Device/resources, opaque weights, pipelines, dispatch and MSL operations | Tokenizers, checkpoint name conventions |
| `model-runtime` | Per-layer execution plans, bounded workspace, KV/recurrent state | Framework file parsing, chat formatting |
| `llama-rs` | Engine/model/session facade, tokenizer, chat template, generation/sampling | Shader argument layout |
| `llama-cli` | User commands and human/JSON output | Numerical implementations |

`model-format::Checkpoint` is the common contract. GGUF retains its low-level
reader for inspection. `mlx/` separates JSON/config interpretation, shard parsing
and name mapping. Both adapters expose `Tensor` with GGML axis zero contiguous:
SafeTensors `[out,in]` becomes semantic `[in,out]` without copying. Checkpoint
provenance retains the source-to-canonical name map for diagnostics.

`WeightData` distinguishes ordinary/GGML packed bytes from MLX affine integer
codes plus scale/bias arrays. Tensor construction validates shapes, byte lengths,
group alignment and metadata types before upload. Fields that could invalidate
shader bounds are private. Affine 2/4/8-bit codes are never converted to a
persistent full-model F16 matrix; GEMV/GEMM decode in registers/shared tile memory.
Per-module mixed quantization stays per module. `float_weights_f32` is an explicit
comparison option: it casts floating storage and affine metadata, preserving codes.

## Layout and numerical policy

`ModelConfig` carries `RotaryLayout`, `RecurrentHeadLayout`, `L2Normalization`,
normalization epsilon and expert-probability normalization. These are numerical
contracts, not equivalent labels inferred from the architecture name alone.

For example, HF/MLX Llama Q/K use split-half RoPE, whereas converted GGUF Llama
uses adjacent pairs. MLX Qwen3.5 repeats recurrent heads interleaved; the GGUF
reference uses tiled repetition. Current MLX and upstream GGUF Qwen normalize Q/K with additive epsilon;
the older fork used a clamp convention. Both operations exist in the backend,
and model policy selects additive epsilon for Qwen. Llama3 frequency scaling is calculated from
HF JSON or loaded from the GGUF frequency-factor tensor. Unsupported scaling
modes fail explicitly. SSM `A_log` becomes the common negative exponential
coefficient once at load. Convolution axes and raw-HF norm offsets are handled
at the adapter boundary, with no per-token format conditionals.

## Ownership and resource safety

Tensor bytes are owned arrays or retained read-only mappings. Complete aligned
mmap pages are wrapped with `newBufferWithBytesNoCopy`; a shard is wrapped only
once. Misaligned ranges and incomplete page tails are copied once. If a mapping
exceeds the device buffer limit or cannot be wrapped, tensor copies provide the
fallback. Metal buffers retain mapping owners. Do not modify, truncate or replace
checkpoint contents while a model/session is alive.

`apple.rs` holds backend types; `apple/memory.rs`, `weights.rs`, `commands.rs` and
`ops/` separate allocation, upload, synchronization and operation dispatch.
MSL files have the same operation boundaries. Shared wire parameters stay in one
place. Unsafe pointer/Objective-C access is contained in the backend; mmap construction is contained in the format crate.

Each submission exclusively borrows the GPU. CPU reads/clears require that same
exclusive borrow, and command buffers complete before it is released. Dispatch
barriers preserve producer/consumer order. Cross-engine buffers and hazardous
aliases fail before submission. Exact in-place elementwise aliases are allowed.
Mapped storage is never a writable output. Pipeline specialization is cached on
the engine; command submission uses autorelease pools.

`Model` clones share immutable weights. `Session`s have independent state and a
single reusable workspace. Handles use `Rc` and are intentionally thread confined;
an atomic cancellation handle may be signaled from another thread. `engine.rs`,
`model.rs`, `session.rs`, `generation.rs`, and `sampling.rs` keep the public facade
separate from generation policy. HF tokenization uses Rust `tokenizers`, while
GGUF reconstruction remains a separate adapter. Streaming preserves incomplete
UTF-8 bytes for the supported ByteLevel tokenizers.

## Plans and memory

`Workspace` centrally defines named scratch slots and byte estimates. Fourteen
F32 slots sized for 32 tokens, one greedy index and one vocabulary logits buffer
are allocated once. Semantic aliases document which temporary lifetimes overlap.
Views/reshape allocate no tensor storage. Prefill projects up to 32 tokens layer
by layer; the vocabulary head is evaluated only for the final request chunk.
Decode uses specialized packed GEMV. Affine addressing uses power-of-two shifts,
avoiding software 64-bit integer division in the inner loop. Decode GEMV
specializes 2/4/8-bit words and reuses each scale/bias across a packed dot product,
rather than fetching them for every scalar weight. One-token attention beyond
128 cached positions splits context into at most 32 SIMD-group partitions and
merges their online-softmax maxima/masses. It reuses the dead K-projection slot
for partial sums; persistent allocation size is unchanged. Prefill retains the
causal streaming kernel with parallel query heads/tokens.

Full-attention layers own token-major `[token,kv_head,dimension]` K/V buffers.
Linear layers own chronological `[channel,conv_history]` convolution state and
`[value_head,value_dimension,key_dimension]` F32 DeltaNet state. No KV is allocated
for linear layers. Legacy Next grouped Q/K/V/Z and beta/alpha projections are
unpacked by GPU gathers. Routing/top-k/selected-expert GEMM stay on the GPU.

Convolution/DeltaNet keep state in registers across a chunk and directly write
persistent buffers. The recurrence inside each chunk is sequential; a parallel
triangular-solve prefill remains future work. Causal attention uses online softmax
without a score matrix. GEMM uses SIMD-group F32 matrix tiles. CPU logits remain
available for inspection/stochastic sampling; greedy GPU argmax is part of the
last-token command buffer. `state_snapshot()` makes diagnostic CPU copies only
when requested, in documented canonical layouts.

Memory reservations include whole retained shard mappings (including unrelated
vision bytes), copied tails/metadata, auxiliary weights and every live session's
KV/recurrent/scratch. Driver overhead and Rust tokenizer/metadata allocations are
outside this estimate; benchmarking reports them separately where measurable.
The default budget is 80% of Metal's recommended working set, capped at that
recommendation. Allocation failure is typed; context is never silently reduced.

## Failure behavior and limits

Apple7+ unified-memory capability is required. macOS 14+ baseline kernels use no
Metal 4 or native BF16 instructions; BF16 converts by bit manipulation. F32
activations/state currently favor straightforward numerical checking over peak
throughput. No claim of parity with MLX's BF16 compute path is implied.

A prefill range is validated before submission. Cancellation is checked between
chunks; processed chunks leave a coherent state/token prefix. If cancelled before
the final chunk, logits are empty rather than stale. Reset/replay restores a
prediction if needed. GPU failure poisons the session until reset. Context
overflow errors; it never drops history. Prefix edits reset/replay because recurrent
state cannot be restored by trimming KV. State snapshots are diagnostic, not a
portable session serialization format.

No paged KV, server batching, disk expert offload, parallel DeltaNet scan,
device-wide autotuning, training or speculative decoding is implemented. The
format adapter is extensible; it does not make every model on `mlx-community`
compatible. New architecture/quantization modes need explicit kernels and tests.
