kernel void embedding(device const uchar *w [[buffer(0)]],
                      device float *out [[buffer(1)]],
                      constant P &p [[buffer(8)]],
                      uint i [[thread_position_in_grid]]) {
  if (i < p.n)
    out[i] = val(w, ulong(p.a) * p.n + i, p.t);
}
// One row per SIMD group, four independent rows per threadgroup; no full weight
// expansion.
#define GEMV(NAME, TYPE)                                                       \
  kernel void NAME(device const uchar *w [[buffer(0)]],                        \
                   device const float *x [[buffer(1)]],                        \
                   device float *y [[buffer(2)]], constant P &p [[buffer(8)]], \
                   uint3 g [[threadgroup_position_in_grid]],                   \
                   uint s [[simdgroup_index_in_threadgroup]],                  \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint width [[threads_per_simdgroup]]) {                     \
    uint row = g.x * 16 + s * 4;                                               \
    float4 sum = 0;                                                            \
    for (uint k = lane; k < p.a; k += width) {                                 \
      float v = x[g.y * p.a + k];                                              \
      for (uint r = 0; r < 4; r++)                                             \
        if (row + r < p.b)                                                     \
          sum[r] += val(w, ulong(row + r) * p.a + k, TYPE) * v;                \
    }                                                                          \
    sum = simd_sum(sum);                                                       \
    if (lane == 0)                                                             \
      for (uint r = 0; r < 4; r++)                                             \
        if (row + r < p.b)                                                     \
          y[g.y * p.b + row + r] = sum[r];                                     \
  }
// clang-format off
GEMV(gemv_f32, 0)
GEMV(gemv_f16, 1)
GEMV(gemv_q4, 2)
GEMV(gemv_q8, 8)
GEMV(gemv_bf16, 30)
// clang-format on
kernel void expert_mv(device const uchar *w [[buffer(0)]],
                      device const float *x [[buffer(1)]],
                      device const uint *ids [[buffer(2)]],
                      device float *y [[buffer(3)]],
                      constant P &p [[buffer(8)]],
                      uint3 g [[threadgroup_position_in_grid]],
                      uint s [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint width [[threads_per_simdgroup]]) {
  uint row = g.x * 4 + s, e = g.y;
  if (row >= p.b)
    return;
  uint expert = ids[e];
  if (expert >= p.d) {
    if (lane == 0)
      y[e * p.b + row] = 0;
    return;
  }
  float sum = 0;
  for (uint k = lane; k < p.a; k += width)
    sum += val(w, (ulong(expert) * p.b + row) * p.a + k, p.t) *
           x[(p.c ? e * p.a : 0) + k];
  sum = simd_sum(sum);
  if (lane == 0)
    y[e * p.b + row] = sum;
}
// Shared-memory tiled prefill GEMM. Each packed weight is decoded once per
// token tile.
kernel void gemm(device const uchar *w [[buffer(0)]],
                 device const float *x [[buffer(1)]],
                 device float *y [[buffer(2)]], constant P &p [[buffer(8)]],
                 uint3 g [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float a[16 * 32];
  threadgroup float b[16 * 32];
  uint row = tid / 16, col = tid % 16;
  float sum = 0;
  for (uint base = 0; base < p.a; base += 32) {
    for (uint i = tid; i < 512; i += 256) {
      uint r = i / 32, k = i % 32;
      uint wr = g.x * 16 + r, xt = g.y * 16 + r;
      a[i] = (wr < p.b && base + k < p.a)
                 ? val(w, ulong(wr) * p.a + base + k, p.t)
                 : 0;
      b[i] = (xt < p.c && base + k < p.a) ? x[xt * p.a + base + k] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 32; k++)
      sum = fma(a[row * 32 + k], b[col * 32 + k], sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint r = g.x * 16 + row, t = g.y * 16 + col;
  if (r < p.b && t < p.c)
    y[t * p.b + r] = sum;
}
kernel void embedding_many(device const uchar *w [[buffer(0)]],
                           device const uint *ids [[buffer(1)]],
                           device float *out [[buffer(2)]],
                           constant P &p [[buffer(8)]],
                           uint i [[thread_position_in_grid]]) {
  if (i < p.n) {
    uint token = ids[i / p.a];
    out[i] = token < p.b ? val(w, ulong(token) * p.a + i % p.a, p.t) : 0;
  }
}
// Apple7+ SIMD-group matrix path. F32 staging preserves the reference
// precision.
kernel void gemm_simd(device const uchar *w [[buffer(0)]],
                      device const float *x [[buffer(1)]],
                      device float *y [[buffer(2)]],
                      constant P &p [[buffer(8)]],
                      uint3 g [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float a[16 * 32], b[16 * 32], out[16 * 16];
  simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
  for (uint base = 0; base < p.a; base += 32) {
    for (uint i = tid; i < 512; i += 128) {
      uint r = i / 32, k = i % 32;
      uint wr = g.x * 16 + r, xt = g.y * 16 + r;
      a[i] = (wr < p.b && base + k < p.a)
                 ? val(w, ulong(wr) * p.a + base + k, p.t)
                 : 0;
      b[i] = (xt < p.c && base + k < p.a) ? x[xt * p.a + base + k] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 32; k += 8) {
      simdgroup_float8x8 ma, mb;
      simdgroup_load(ma, a + (sg / 2 * 8) * 32 + k, 32);
      simdgroup_load(mb, b + (sg % 2 * 8) * 32 + k, 32, ulong2(0), true);
      simdgroup_multiply_accumulate(acc, ma, mb, acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_store(acc, out + (sg / 2 * 8) * 16 + sg % 2 * 8, 16);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 256; i += 128) {
    uint r = g.x * 16 + i / 16, t = g.y * 16 + i % 16;
    if (r < p.b && t < p.c)
      y[t * p.b + r] = out[i];
  }
}
// Packed K-quant dot products: decode four quants at a time and apply block
// scales after the local dot. Adapted from the ggml/llama-metal block-dot
// formulation.
inline uchar4 bytes4(device const uchar *p) {
  return uchar4(p[0], p[1], p[2], p[3]);
}
inline uint2 kscale(device const uchar *s, uint j) {
  return j < 4 ? uint2(s[j] & 63, s[j + 4] & 63)
               : uint2((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4),
                       (s[j + 4] >> 4) | ((s[j] >> 6) << 4));
}
template <uint T>
inline float packed_dot(device const uchar *b, thread const float4 *x,
                        uint half_index, uint l) {
  float4 dots, scales, mins;
  if (T == 14) {
    uchar4 lo = bytes4(b + half_index * 64 + l),
           lo1 = bytes4(b + half_index * 64 + l + 32),
           hi = bytes4(b + 128 + half_index * 32 + l);
    dots =
        float4(dot(x[0], float4((lo & 15) | ((hi & 3) << 4)) - 32.0f),
               dot(x[1], float4((lo1 & 15) | (((hi >> 2) & 3) << 4)) - 32.0f),
               dot(x[2], float4((lo >> 4) | (((hi >> 4) & 3) << 4)) - 32.0f),
               dot(x[3], float4((lo1 >> 4) | (((hi >> 6) & 3) << 4)) - 32.0f));
    uint offset = 192 + half_index * 8 + l / 16;
    scales = float4(as_type<char>(b[offset]), as_type<char>(b[offset + 2]),
                    as_type<char>(b[offset + 4]), as_type<char>(b[offset + 6]));
    return hf(b, 208) * dot(dots, scales);
  }
  uchar4 a = bytes4(b + (T == 12 ? 16 : 48) + half_index * 64 + l),
         c = bytes4(b + (T == 12 ? 16 : 48) + half_index * 64 + l + 32);
  uchar4 q0 = a & 15, q1 = a >> 4, q2 = c & 15, q3 = c >> 4;
  if (T == 13) {
    uchar4 high = bytes4(b + 16 + l);
    uint g = half_index * 4;
    q0 |= ((high >> g) & 1) << 4;
    q1 |= ((high >> (g + 1)) & 1) << 4;
    q2 |= ((high >> (g + 2)) & 1) << 4;
    q3 |= ((high >> (g + 3)) & 1) << 4;
  }
  dots = float4(dot(x[0], float4(q0)), dot(x[1], float4(q1)),
                dot(x[2], float4(q2)), dot(x[3], float4(q3)));
  for (uint j = 0; j < 4; j++) {
    uint2 sm = kscale(b + 4, half_index * 4 + j);
    scales[j] = float(sm.x);
    mins[j] = float(sm.y);
  }
  float4 sx = float4(dot(x[0], float4(1)), dot(x[1], float4(1)),
                     dot(x[2], float4(1)), dot(x[3], float4(1)));
  return hf(b, 0) * dot(dots, scales) - hf(b, 2) * dot(sx, mins);
}
#define PACKED_GEMV(NAME, TYPE, SIZE)                                          \
  kernel void NAME(device const uchar *w [[buffer(0)]],                        \
                   device const float *x [[buffer(1)]],                        \
                   device float *y [[buffer(2)]], constant P &p [[buffer(8)]], \
                   uint3 g [[threadgroup_position_in_grid]],                   \
                   uint sg [[simdgroup_index_in_threadgroup]],                 \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint width [[threads_per_simdgroup]]) {                     \
    uint row = g.x * 16 + sg * 4, nb = p.a / 256;                              \
    float4 sum = 0;                                                            \
    if (width == 32) {                                                         \
      uint parity = lane / 16, local = lane % 16, half_index = local / 8,      \
           l = 4 * (local % 8);                                                \
      for (uint ib = parity; ib < nb; ib += 2) {                               \
        device const float *v =                                                \
            x + g.y * p.a + ib * 256 + half_index * 128 + l;                   \
        float4 xv[4] = {*(device const float4 *)(v),                           \
                        *(device const float4 *)(v + 32),                      \
                        *(device const float4 *)(v + 64),                      \
                        *(device const float4 *)(v + 96)};                     \
        for (uint r = 0; r < 4; r++)                                           \
          if (row + r < p.b)                                                   \
            sum[r] += packed_dot<TYPE>(w + (ulong(row + r) * nb + ib) * SIZE,  \
                                       xv, half_index, l);                     \
      }                                                                        \
    } else {                                                                   \
      for (uint k = lane; k < p.a; k += width)                                 \
        for (uint r = 0; r < 4; r++)                                           \
          if (row + r < p.b)                                                   \
            sum[r] +=                                                          \
                val(w, ulong(row + r) * p.a + k, TYPE) * x[g.y * p.a + k];     \
    }                                                                          \
    sum = simd_sum(sum);                                                       \
    if (lane == 0)                                                             \
      for (uint r = 0; r < 4; r++)                                             \
        if (row + r < p.b)                                                     \
          y[g.y * p.b + row + r] = sum[r];                                     \
  }
PACKED_GEMV(packed_q4k, 12, 144)
PACKED_GEMV(packed_q5k, 13, 176)
PACKED_GEMV(packed_q6k, 14, 210)

// Wider prefill tile: each SIMD group owns four 8x8 accumulators. A decoded
// weight tile is reused across 32 prompt tokens rather than 16.
kernel void gemm_simd_wide(device const uchar *w [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant P &p [[buffer(8)]],
                           uint3 g [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float a[32 * 32], b[32 * 32], out[32 * 32];
  uint row = sg / 2 * 16, col = sg % 2 * 16;
  simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8>(0.0f),
                     c01 = make_filled_simdgroup_matrix<float, 8>(0.0f),
                     c10 = make_filled_simdgroup_matrix<float, 8>(0.0f),
                     c11 = make_filled_simdgroup_matrix<float, 8>(0.0f);
  for (uint base = 0; base < p.a; base += 32) {
    for (uint i = tid; i < 1024; i += 128) {
      uint r = i / 32, k = i % 32;
      uint wr = g.x * 32 + r, xt = g.y * 32 + r;
      a[i] = (wr < p.b && base + k < p.a)
                 ? val(w, ulong(wr) * p.a + base + k, p.t)
                 : 0;
      b[i] = (xt < p.c && base + k < p.a) ? x[xt * p.a + base + k] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 32; k += 8) {
      simdgroup_float8x8 a0, a1, b0, b1;
      simdgroup_load(a0, a + row * 32 + k, 32);
      simdgroup_load(a1, a + (row + 8) * 32 + k, 32);
      simdgroup_load(b0, b + col * 32 + k, 32, ulong2(0), true);
      simdgroup_load(b1, b + (col + 8) * 32 + k, 32, ulong2(0), true);
      simdgroup_multiply_accumulate(c00, a0, b0, c00);
      simdgroup_multiply_accumulate(c01, a0, b1, c01);
      simdgroup_multiply_accumulate(c10, a1, b0, c10);
      simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_store(c00, out + row * 32 + col, 32);
  simdgroup_store(c01, out + row * 32 + col + 8, 32);
  simdgroup_store(c10, out + (row + 8) * 32 + col, 32);
  simdgroup_store(c11, out + (row + 8) * 32 + col + 8, 32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 1024; i += 128) {
    uint r = g.x * 32 + i / 32, t = g.y * 32 + i % 32;
    if (r < p.b && t < p.c)
      y[t * p.b + r] = out[i];
  }
}
