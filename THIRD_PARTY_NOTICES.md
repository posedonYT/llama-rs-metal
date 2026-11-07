# Third-party notices

The GGUF/GGML packed quantization layouts, scalar dequantization algorithms,
model tensor naming, rotary conventions, and Gated DeltaNet algorithms were
adapted from or checked against:

- https://github.com/ai-janitor/llama-metal
- https://github.com/ggml-org/llama.cpp

Reference revision: `72ba0238e1a65defbffaa384caadd84121c75f1d` of ai-janitor/llama-metal.
Copyright (c) 2023–2026 The ggml authors. MIT license; the complete notice is in
[docs/LLAMA-METAL-LICENSE](docs/LLAMA-METAL-LICENSE).

The Rust runtime does not link to llama.cpp or GGML. The C++ programs in `tools/`
are optional development oracles linked against a separately built reference.

Downloaded model weights in `models/` are excluded from source control and retain
their respective model licenses. Cargo dependencies retain their upstream licenses.

Native MLX affine layouts, Qwen numerical conventions, and development oracle
behavior were checked against [MLX](https://github.com/ml-explore/mlx) 0.32.4 and
[MLX-LM](https://github.com/ml-explore/mlx-lm) 0.32.2. Copyright Apple Inc., MIT;
see [docs/MLX-LICENSE](docs/MLX-LICENSE). MLX is not a Rust runtime dependency.
SafeTensors and Hugging Face tokenizers are Rust dependencies under their own
Apache-2.0 licenses. See Cargo.lock for exact dependency versions.
