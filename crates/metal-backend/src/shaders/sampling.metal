// Exact top-k routing on the GPU; selected ids/weights never visit the host.
kernel void route(device const float *x [[buffer(0)]],
                  device uint *ids [[buffer(1)]],
                  device float *weights [[buffer(2)]],
                  constant P &p [[buffer(8)]],
                  uint i [[thread_position_in_grid]]) {
  if (i)
    return;
  float mx = -INFINITY;
  for (uint j = 0; j < p.n; j++)
    mx = max(mx, x[j]);
  float den = 0;
  for (uint j = 0; j < p.n; j++)
    den += exp(x[j] - mx);
  float selected = 0;
  for (uint k = 0; k < p.a; k++) {
    uint best = 0;
    float score = -INFINITY;
    for (uint j = 0; j < p.n; j++) {
      bool used = false;
      for (uint s = 0; s < k; s++)
        if (ids[s] == j)
          used = true;
      if (!used && x[j] > score) {
        score = x[j];
        best = j;
      }
    }
    ids[k] = best;
    weights[k] = exp(score - mx) / den;
    selected += weights[k];
  }
  if (p.b)
    for (uint k = 0; k < p.a; k++)
      weights[k] /= selected;
}
// Deterministic greedy sampling, including lowest-id tie breaking and nonfinite
// detection.
kernel void argmax(device const float *x [[buffer(0)]],
                   device uint *out [[buffer(1)]], constant P &p [[buffer(8)]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint width [[threads_per_simdgroup]],
                   uint3 nt [[threads_per_threadgroup]]) {
  threadgroup float values[32];
  threadgroup uint ids[32], bad[32];
  float best = -INFINITY;
  uint id = 0xffffffff;
  bool invalid = false;
  for (uint i = tid; i < p.n; i += nt.x) {
    float v = x[i];
    invalid |= !isfinite(v);
    if (v > best || (v == best && i < id)) {
      best = v;
      id = i;
    }
  }
  float mx = simd_max(best);
  uint mi = simd_min(best == mx ? id : 0xffffffff);
  bool error = simd_any(invalid);
  if (lane == 0) {
    values[sg] = mx;
    ids[sg] = mi;
    bad[sg] = uint(error);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float maxv = -INFINITY;
    uint selected = 0xffffffff, errors = 0;
    for (uint i = 0; i < nt.x / width; i++) {
      errors |= bad[i];
      if (values[i] > maxv || (values[i] == maxv && ids[i] < selected)) {
        maxv = values[i];
        selected = ids[i];
      }
    }
    out[0] = errors ? 0xffffffff : selected;
  }
}
