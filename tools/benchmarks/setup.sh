#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python_bin="${PYTHON:-python3.14}"
"$python_bin" -m venv target/mlx-env
target/mlx-env/bin/pip install -r tools/benchmarks/requirements.lock
python3 -m venv target/build-tools
target/build-tools/bin/pip install 'cmake==4.4.4'
ref="$PWD/target/llama-cpp-reference"
revision=f7591a6a01f9273de03331732bdc0b39721168be
if [[ ! -d "$ref/.git" ]]; then
  git init "$ref"
  git -C "$ref" remote add origin https://github.com/ggml-org/llama.cpp.git
  git -C "$ref" fetch --depth 1 origin "$revision"
  git -C "$ref" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$ref" rev-parse HEAD)" == "$revision" ]] || { echo 'Unexpected reference revision' >&2; exit 1; }
printf '%s\n' "$revision" > target/llama-cpp-revision.txt
target/build-tools/bin/cmake -S "$ref" -B "$ref/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF
target/build-tools/bin/cmake --build "$ref/build" -j 6 --target llama
c++ -O3 -std=c++17 -I "$ref/include" -I "$ref/ggml/include" -I "$ref/vendor" tools/benchmarks/llama_worker.cpp -L "$ref/build/bin" -lllama -Wl,-rpath,"$ref/build/bin" -o target/llama_benchmark_worker
swiftc tools/benchmarks/host_probe.swift -o target/host_probe
cargo build --release --workspace --examples
