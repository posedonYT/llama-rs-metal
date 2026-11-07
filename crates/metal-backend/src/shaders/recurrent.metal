// Fused convolution, SiLU, and persistent state write. State is channel-major
// chronological.
kernel void conv(device const float *x [[buffer(0)]],
                 device const uchar *w [[buffer(1)]],
                 device float *state [[buffer(2)]],
                 device float *out [[buffer(3)]], constant P &p [[buffer(8)]],
                 uint c [[thread_position_in_grid]]) {
  if (c >= p.n)
    return;
  float history[64];
  for (uint j = 0; j + 1 < p.a; j++)
    history[j] = state[c * (p.a - 1) + j];
  for (uint t = 0; t < p.f; t++) {
    float current = x[t * p.n + c];
    float sum = current * val(w, c * p.a + p.a - 1, p.t);
    for (uint j = 0; j + 1 < p.a; j++)
      sum += history[j] * val(w, c * p.a + j, p.t);
    for (uint j = 0; j + 2 < p.a; j++)
      history[j] = history[j + 1];
    if (p.a > 1)
      history[p.a - 2] = current;
    out[t * p.n + c] = sum / (1.0f + exp(-sum));
  }
  for (uint j = 0; j + 1 < p.a; j++)
    state[c * (p.a - 1) + j] = history[j];
}
// Q/K L2 normalization and recurrence are fused. State
// [value-head][value][key].
kernel void delta(device const float *qkv [[buffer(0)]],
                  device const float *alpha [[buffer(1)]],
                  device const float *beta [[buffer(2)]],
                  device const float *aw [[buffer(3)]],
                  device const float *dt [[buffer(4)]],
                  device float *state [[buffer(5)]],
                  device float *out [[buffer(6)]], constant P &p [[buffer(8)]],
                  uint3 g [[threadgroup_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint width [[threads_per_simdgroup]]) {
  uint h = g.x, j = g.y, kh = p.e ? h / p.d : h % p.b;
  uint dk = p.a, dv = p.c;
  ulong base = (ulong(h) * dv + j) * dk;
  float saved[16];
  for (uint i = lane, s = 0; i < dk; i += width, s++)
    saved[s] = state[base + i];
  for (uint t = 0; t < p.f; t++) {
    uint off = t * (2 * p.b * dk + p.g * dv);
    uint qo = off + kh * dk, ko = off + p.b * dk + kh * dk,
         vo = off + 2 * p.b * dk + h * dv;
    float qn = 0, kn = 0;
    for (uint i = lane; i < dk; i += width) {
      qn += qkv[qo + i] * qkv[qo + i];
      kn += qkv[ko + i] * qkv[ko + i];
    }
    qn = rsqrt(p.y > 0.0f ? simd_sum(qn) + p.x : max(simd_sum(qn), p.x));
    kn = rsqrt(p.y > 0.0f ? simd_sum(kn) + p.x : max(simd_sum(kn), p.x));
    float z = alpha[t * p.g + h] + dt[h];
    float decay = exp(aw[h] * (max(z, 0.0f) + log(1.0f + exp(-abs(z)))));
    float be = 1.0f / (1.0f + exp(-beta[t * p.g + h]));
    float pred = 0;
    for (uint i = lane, s = 0; i < dk; i += width, s++) {
      saved[s] *= decay;
      pred += saved[s] * qkv[ko + i] * kn;
    }
    pred = simd_sum(pred);
    float diff = be * (qkv[vo + j] - pred), res = 0;
    for (uint i = lane, s = 0; i < dk; i += width, s++) {
      saved[s] += diff * qkv[ko + i] * kn;
      res += saved[s] * qkv[qo + i] * qn * rsqrt(float(dk));
    }
    res = simd_sum(res);
    if (lane == 0)
      out[(t * p.g + h) * dv + j] = res;
  }
  for (uint i = lane, s = 0; i < dk; i += width, s++)
    state[base + i] = saved[s];
}
