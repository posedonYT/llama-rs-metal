#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=72ba0238e1a65defbffaa384caadd84121c75f1d
reference="$PWD/target/llama-metal-reference"
mkdir -p target
if [[ ! -d "$reference/.git" ]]; then
  git clone https://github.com/ai-janitor/llama-metal.git "$reference"
  git -C "$reference" checkout --detach "$revision"
fi
if [[ "$(git -C "$reference" rev-parse HEAD)" != "$revision" ]]; then
  echo "Reference checkout must be at $revision; refusing to change an existing checkout." >&2
  exit 1
fi
if command -v cmake >/dev/null; then
  cmake_bin="$(command -v cmake)"
else
  python3 -m venv target/build-tools
  target/build-tools/bin/pip install cmake
  cmake_bin="$PWD/target/build-tools/bin/cmake"
fi
"$cmake_bin" -S "$reference" -B "$reference/build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF
"$cmake_bin" --build "$reference/build" --config Release -j 6 --target llama
for name in reference reference_bench; do
  c++ -O3 -std=c++17 -I "$reference/include" -I "$reference/ggml/include" \
    "tools/$name.cpp" -L "$reference/build/bin" -lllama \
    -Wl,-rpath,"$reference/build/bin" -o "target/$name"
done
