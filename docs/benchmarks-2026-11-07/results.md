# Benchmark results

Synchronous full-host-logit boundary; teacher-forced decode. Native is not precision matched. Table medians and ratios exclude confounded pairs. Small-N CIs describe this machine/run only.

| Model | Track | Workload | Prefill Rust / ref tok/s | Decode Rust / ref tok/s | Decode ratio [95% CI] | Valid pairs |
|---|---|---|---:|---:|---|---:|
| qwen35-08-4bit | matched_f32 | prose-p128-d64 | 518.1 / 1756.8 | 82.5 / 123.6 | 0.737 [0.636, 0.948] | 4 |
| qwen35-08-4bit | matched_f32 | prose-p512-d64 | 478.6 / 2005.8 | 75.6 / 119.2 | 0.647 [0.588, 0.728] | 4 |
| qwen35-08-4bit | matched_f32 | prose-p2048-d64 | 366.8 / 1853.9 | 72.6 / 117.8 | 0.646 [0.609, 0.705] | 4 |
| qwen35-08-4bit | native | prose-p128-d64 | 460.9 / 1947.8 | 73.4 / 123.9 | 0.600 [0.589, 0.619] | 4 |
| qwen35-08-4bit | native | prose-p512-d64 | 424.5 / 2279.4 | 69.4 / 122.8 | 0.564 [0.546, 0.579] | 4 |
| qwen35-08-4bit | native | prose-p2048-d64 | 331.8 / 2255.6 | 69.7 / 120.6 | 0.564 [0.539, 0.598] | 4 |
| llama32-1b-4bit | matched_f32 | prose-p128-d64 | 331.0 / 1193.3 | 67.8 / 96.9 | 0.725 [0.630, 0.835] | 4 |
| llama32-1b-4bit | matched_f32 | prose-p512-d64 | 297.5 / 1326.8 | 67.3 / 90.0 | 0.723 [0.673, 0.798] | 4 |
| llama32-1b-4bit | matched_f32 | prose-p2048-d64 | 219.2 / 1335.5 | 56.5 / 88.9 | 0.641 [0.581, 0.717] | 4 |
| llama32-1b-4bit | native | prose-p128-d64 | 299.2 / 1422.2 | 67.7 / 112.3 | 0.600 [0.571, 0.619] | 4 |
| llama32-1b-4bit | native | prose-p512-d64 | 279.3 / 1612.0 | 65.4 / 110.0 | 0.590 [0.567, 0.610] | 4 |
| llama32-1b-4bit | native | prose-p2048-d64 | 205.9 / 1603.9 | 55.5 / 102.7 | 0.544 [0.536, 0.551] | 4 |
| qwen35-08-8bit | matched_f32 | prose-p512-d64 | 463.2 / 1942.5 | 58.4 / 82.5 | 0.708 [0.687, 0.730] | 4 |
| qwen35-08-8bit | native | prose-p512-d64 | 459.4 / 2459.8 | 59.2 / 85.8 | 0.681 [0.668, 0.692] | 4 |
| qwen35-08-bf16 | matched_f32 | prose-p512-d64 | 406.0 / 1778.5 | 25.0 / 27.1 | 0.917 [0.906, 0.929] | 3 |
| qwen35-08-bf16 | native | prose-p512-d64 | 420.3 / 2506.1 | 34.9 / 50.6 | 0.691 [0.680, 0.707] | 4 |
| qwen35-4b-4bit | matched_f32 | prose-p512-d64 | 96.0 / 332.9 | 23.1 / 29.3 | 0.824 [0.775, 0.879] | 4 |
| qwen35-4b-4bit | native | prose-p512-d64 | 95.6 / 427.6 | 23.5 / 35.0 | 0.645 [0.615, 0.676] | 4 |
| qwen35-08-4bit | matched_f32 | code-p512-d64 | 364.0 / 1617.0 | 62.0 / 89.4 | 0.720 [0.639, 0.862] | 4 |
| qwen35-08-4bit | matched_f32 | russian-p512-d64 | 367.0 / 1581.5 | 62.9 / 91.0 | 0.678 [0.618, 0.735] | 4 |
| qwen35-08-4bit | native | code-p512-d64 | 435.2 / 2255.9 | 69.0 / 107.9 | 0.600 [0.587, 0.617] | 4 |
| qwen35-08-4bit | native | russian-p512-d64 | 404.5 / 2221.0 | 64.3 / 109.7 | 0.593 [0.566, 0.629] | 4 |
| qwen35-08-4bit | matched_f32 | prose-p512-d512 | 431.9 / 1722.2 | 67.9 / 106.6 | 0.632 [0.629, 0.634] | 2 |
| stories260K | gguf | prose-p128-d64 | 34130.5 / 112443.3 | 1317.2 / 1849.4 | 0.727 [0.714, 0.740] | 4 |
| stories260K | gguf | prose-p512-d64 | 20798.1 / 104782.4 | 1166.8 / 1713.6 | 0.686 [0.667, 0.705] | 4 |

Ratios are geometric means of paired block-median ratios, not ratios of pooled medians. Each primary cell has eight observations per engine before confound exclusion. Raw distributions, all samples, startup, TTFT and memory remain in the JSON evidence.
