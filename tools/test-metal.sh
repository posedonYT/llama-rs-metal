#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo test --locked --workspace -- --test-threads=1
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
