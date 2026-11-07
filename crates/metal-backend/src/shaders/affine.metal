// Native MLX affine quantization. The packed integer codes never expand to a
// persistent float matrix: w = code * scale + bias, grouped along the input
// axis.
// Only floating affine metadata enters this helper; GGML block decoding is
// deliberately absent from the packed dot-product's inner loop.
inline float affine_scalar(device const uchar *data, ulong index, uint type) {
  if (type == 0)
    return ((device const float *)data)[index];
  if (type == 30)
    return as_type<float>(uint(((device const ushort *)data)[index]) << 16);
  return hf(data, index * 2);
}
inline float affine_value(device const uint *packed, device const uchar *scales,
                          device const uchar *biases, ulong i, constant P &p) {
  // Supported bit widths and group sizes are powers of two. Keeping these
  // address calculations as shifts avoids software 64-bit GPU division.
  uint word_shift = p.e == 2 ? 4 : (p.e == 4 ? 3 : 2);
  uint lane = uint(i) & ((1u << word_shift) - 1);
  uint code = (packed[i >> word_shift] >> (lane * p.e)) & ((1u << p.e) - 1);
  ulong group = i >> p.h;
  return float(code) * affine_scalar(scales, group, p.t) +
         affine_scalar(biases, group, p.t);
}
kernel void affine_embedding(device const uint *w [[buffer(0)]],
                             device const uchar *s [[buffer(1)]],
                             device const uchar *b [[buffer(2)]],
                             device const uint *ids [[buffer(3)]],
                             device float *out [[buffer(4)]],
                             constant P &p [[buffer(8)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < p.n) {
    uint token = ids[i / p.a];
    out[i] = token < p.b
                 ? affine_value(w, s, b, ulong(token) * p.a + i % p.a, p)
                 : 0;
  }
}
// Each lane decodes one packed word, reusing its scale/bias and input values.
// Static bit widths let Metal unroll extraction and remove integer division.
template <uint Bits>
inline float4 affine_dot(device const uint *w, device const uchar *scales,
                         device const uchar *biases, device const float *x,
                         ulong expert_base, uint row, uint lane, uint width,
                         constant P &p) {
  constexpr uint pack = 32 / Bits;
  constexpr uint word_shift = Bits == 2 ? 4 : (Bits == 4 ? 3 : 2);
  constexpr uint mask = (1u << Bits) - 1;
  float4 sum = 0;
  for (uint k = lane * pack; k < p.a; k += width * pack) {
    float input[pack];
    float input_sum = 0;
    for (uint j = 0; j < pack; j++) {
      input[j] = x[k + j];
      input_sum += input[j];
    }
    for (uint r = 0; r < 4; r++) {
      if (row + r >= p.b)
        continue;
      ulong index = expert_base + ulong(row + r) * p.a + k;
      uint codes = w[index >> word_shift];
      float dot = 0;
      for (uint j = 0; j < pack; j++)
        dot += float((codes >> (j * Bits)) & mask) * input[j];
      ulong group = index >> p.h;
      sum[r] += affine_scalar(scales, group, p.t) * dot +
                affine_scalar(biases, group, p.t) * input_sum;
    }
  }
  return sum;
}
// clang-format off
#define AFFINE_MV(BITS) \
kernel void affine_mv_##BITS(device const uint *w [[buffer(0)]], \
 device const uchar *s [[buffer(1)]], device const uchar *b [[buffer(2)]], \
 device const float *x [[buffer(3)]], device float *y [[buffer(4)]], \
 constant P &p [[buffer(8)]], uint3 g [[threadgroup_position_in_grid]], \
 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
 uint width [[threads_per_simdgroup]]) { \
 uint row = g.x * 16 + sg * 4; \
 float4 sum = simd_sum(affine_dot<BITS>(w,s,b,x+g.y*p.a,0,row,lane,width,p)); \
 if(lane==0) for(uint r=0;r<4;r++) if(row+r<p.b) y[g.y*p.b+row+r]=sum[r]; \
}
#define AFFINE_EXPERT(BITS) \
kernel void affine_expert_##BITS(device const uint *w [[buffer(0)]], \
 device const uchar *s [[buffer(1)]], device const uchar *b [[buffer(2)]], \
 device const float *x [[buffer(3)]], device const uint *ids [[buffer(4)]], \
 device float *y [[buffer(5)]], constant P &p [[buffer(8)]], \
 uint3 g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
 uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) { \
 uint row=g.x*16+sg*4, expert=ids[g.y]; float4 sum=0; \
 if(expert<p.g) sum=affine_dot<BITS>(w,s,b,x+(p.c?g.y*p.a:0),ulong(expert)*p.b*p.a,row,lane,width,p); \
 sum=simd_sum(sum); \
 if(lane==0) for(uint r=0;r<4;r++) if(row+r<p.b) y[g.y*p.b+row+r]=sum[r]; \
}
AFFINE_MV(2)
AFFINE_MV(4)
AFFINE_MV(8)
AFFINE_EXPERT(2)
AFFINE_EXPERT(4)
AFFINE_EXPERT(8)
// clang-format on
kernel void affine_mm(device const uint *w [[buffer(0)]],
                      device const uchar *s [[buffer(1)]],
                      device const uchar *bias [[buffer(2)]],
                      device const float *x [[buffer(3)]],
                      device float *y [[buffer(4)]],
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
      uint r = i / 32, k = i % 32, wr = g.x * 32 + r, xt = g.y * 32 + r;
      a[i] = (wr < p.b && base + k < p.a)
                 ? affine_value(w, s, bias, ulong(wr) * p.a + base + k, p)
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
