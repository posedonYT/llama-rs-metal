# Separate memory experiment

`tools/benchmarks/memory.py` runs one engine process at a time, independently of
the throughput measurements. Context capacity is 4096, with the persisted prose
prompt of 512 IDs and 64 teacher-forced continuation IDs. Each process performs
one warmup and one further request. Both matched-F32 and native precision policies
are labelled. Samples start after the ready handshake, so this is an inference
memory experiment and does not capture the entire loading peak.

Darwin `proc_pid_rusage(..., RUSAGE_INFO_V2, ...)` supplies OS charged physical
footprint, resident size, wired size and page-ins every 100ms. The ctypes layout
matches `sys/resource.h` in the installed macOS SDK. Its resident counter was
cross-checked with psutil on the coordinator process. Raw samples and peak values
are retained. Sampling can miss short-lived peaks; these are observed values,
not a proof of a maximum possible footprint.

Three different accounts are reported:

- OS charged footprint follows task ownership/compression/reclaimability rules.
  It is not the total number of physical bytes reachable through every mapping.
- RSS counts resident process pages and can omit GPU allocations/charges.
- Metal allocator counters describe resources retained by a framework. Rust's
  no-copy buffer can span a whole shard including unused vision tensors; span
  length does not prove every corresponding page is resident. MLX reports active,
  peak and cached allocations separately.

Do not add these numbers (they overlap), equate them, or use one counter alone to
claim a whole-system RAM winner. The experiment makes the accounts visible under
the same workload. Loading time, reset, throughput, scratch reservations and the
real-session fixed-allocation test are complementary evidence. Resource estimates
exclude driver bookkeeping and Rust tokenizer/JSON allocations.
