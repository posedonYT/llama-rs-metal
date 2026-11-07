# Additional measurements

All times are synchronous API times. Startup uses warm filesystem/driver caches; it is not a cold-install or cold-disk claim. TTFT includes preallocated-cache reset. Median values below exclude a pair if either process block was confounded.

| Model | Track | Load Rust / MLX (s) | Process start to ready Rust / MLX (s) | P512 request TTFT Rust / MLX (ms) |
|---|---|---:|---:|---:|
| qwen35-08-4bit | matched_f32 | 0.481 / 0.839 | 0.486 / 1.439 | 1071.8 / 257.1 |
| qwen35-08-4bit | native | 0.446 / 0.880 | 0.450 / 1.601 | 1207.4 / 225.8 |
| llama32-1b-4bit | matched_f32 | 0.406 / 0.441 | 0.410 / 1.258 | 1727.2 / 391.5 |
| llama32-1b-4bit | native | 0.365 / 0.412 | 0.370 / 1.145 | 1836.6 / 320.0 |
| qwen35-08-8bit | matched_f32 | 0.486 / 0.836 | 0.490 / 1.423 | 1106.7 / 265.3 |
| qwen35-08-8bit | native | 0.457 / 0.829 | 0.461 / 1.416 | 1115.8 / 209.4 |
| qwen35-08-bf16 | matched_f32 | 2.174 / 1.004 | 2.180 / 1.580 | 1262.3 / 289.7 |
| qwen35-08-bf16 | native | 0.364 / 0.834 | 0.368 / 1.410 | 1219.5 / 205.4 |
| qwen35-4b-4bit | matched_f32 | 3.104 / 1.608 | 3.109 / 2.576 | 5335.4 / 1542.1 |
| qwen35-4b-4bit | native | 2.112 / 1.221 | 2.117 / 2.145 | 5357.7 / 1200.1 |

## Memory

One separate inference trial per configuration, after readiness: P512/D64, context 4096. OS charge, RSS and Metal counters are different accounts. Do not add them or infer a whole-system RAM winner from one column. [Method](../memory-method.md).

| Model | Track | OS charged peak Rust / MLX GiB | RSS peak Rust / MLX GiB | Rust Metal retained span GiB | MLX active / peak / cache GiB |
|---|---|---:|---:|---:|---:|
| qwen35-08-4bit | matched_f32 | 0.88 / 2.30 | 1.35 / 0.90 | 0.56 | 0.55 / 1.70 / 0.30 |
| qwen35-08-4bit | native | 0.74 / 1.62 | 1.21 / 0.90 | 0.52 | 0.46 / 1.05 / 0.16 |
| llama32-1b-4bit | matched_f32 | 1.32 / 2.12 | 2.06 / 0.97 | 0.98 | 0.97 / 1.89 / 0.08 |
| llama32-1b-4bit | native | 1.10 / 1.49 | 1.83 / 0.98 | 0.91 | 0.77 / 1.26 / 0.05 |
| qwen35-4b-4bit | matched_f32 | 3.57 / 4.97 | 2.02 / 2.59 | 2.77 | 2.75 / 4.16 / 0.46 |
| qwen35-4b-4bit | native | 2.74 / 3.66 | 5.00 / 2.70 | 2.52 | 2.38 / 3.09 / 0.25 |

## 512-step continuation

Qwen3.5 0.8B 4-bit, matched F32, P512 followed by D512. Two balanced blocks, one warmup and two measured requests per process (four measured runs per engine); this is a small endurance experiment. Teacher-forced history and position ranges are identical. Values are median tokens/s within each 64-step window, so increasing context remains visible.

| Consumed positions | Rust tok/s | MLX tok/s |
|---|---:|---:|
| 513–576 | 68.9 | 107.6 |
| 577–640 | 67.7 | 109.4 |
| 641–704 | 68.6 | 109.4 |
| 705–768 | 66.9 | 109.6 |
| 769–832 | 68.5 | 108.5 |
| 833–896 | 67.2 | 106.2 |
| 897–960 | 67.8 | 102.6 |
| 961–1024 | 67.4 | 105.7 |

Rust retained Metal allocation counts remained constant across workload cells in all 11 model/precision/context configurations (each had one distinct retained-byte value). The real MLX session test also checks the count before/after 128 decode steps. These counters establish bounded retained resources, not zero allocator churn or a bound on all driver memory.

## Evidence scope

The primary matrix has four independent balanced blocks and eight measured observations per engine/cell; the endurance subset has two blocks. Only the BF16 matched-F32 block 0 pair was excluded for swap growth. Raw observations are retained, including that pair. No outlier trimming was applied. Other applications and dynamic GPU clocks remain uncontrolled; confidence intervals describe this run/configuration, not hardware-wide performance.

The upstream llama.cpp table uses a 260K-parameter F32-weight GGUF with F32 caches and implementation-default internal precision. Its maximum observed logit difference was about 0.009 and argmax agreed. It is a small integration/dispatch comparison, not evidence for billion-parameter parity. Quantized Qwen GGUF failed its numerical gate and is excluded.

[Throughput and confidence intervals](results.md) · [Compact raw JSON (gzip)](raw.json.gz)
