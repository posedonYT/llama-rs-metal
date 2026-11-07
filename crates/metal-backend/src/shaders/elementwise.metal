kernel void add(device const float *a [[buffer(0)]],
                device const float *b [[buffer(1)]],
                device float *y [[buffer(2)]], constant P &p [[buffer(8)]],
                uint i [[thread_position_in_grid]]) {
  if (i < p.n)
    y[i] = a[i] + b[i];
}
kernel void activation(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant P &p [[buffer(8)]],
                       uint i [[thread_position_in_grid]]) {
  if (i < p.n) {
    float v = a[i];
    y[i] = v / (1.0f + exp(-v)) * b[i];
  }
}
kernel void gate(device const float *a [[buffer(0)]],
                 device const float *b [[buffer(1)]],
                 device float *y [[buffer(2)]], constant P &p [[buffer(8)]],
                 uint i [[thread_position_in_grid]]) {
  if (i < p.n)
    y[i] = a[i] / (1.0f + exp(-b[i]));
}
kernel void copy_f32(device const float *a [[buffer(0)]],
                     device float *b [[buffer(1)]], constant P &p [[buffer(8)]],
                     uint i [[thread_position_in_grid]]) {
  if (i < p.n)
    b[p.a + i] = a[i];
}
kernel void norm(device const float *x [[buffer(0)]],
                 device const uchar *w [[buffer(1)]],
                 device float *y [[buffer(2)]], constant P &p [[buffer(8)]],
                 uint3 g [[threadgroup_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint width [[threads_per_simdgroup]]) {
  uint head = g.x;
  float sum = 0;
  for (uint i = lane; i < p.n; i += width) {
    float v = x[head * p.a + i];
    sum += v * v;
  }
  float inv = rsqrt(simd_sum(sum) / float(p.n) + p.x);
  for (uint i = lane; i < p.n; i += width)
    y[head * p.n + i] = x[head * p.a + i] * inv * val(w, i, p.t);
}
kernel void mix(device const float *x [[buffer(0)]],
                device const float *w [[buffer(1)]],
                device float *y [[buffer(2)]], constant P &p [[buffer(8)]],
                uint i [[thread_position_in_grid]]) {
  if (i >= p.n)
    return;
  float v = 0;
  for (uint k = 0; k < p.a; k++)
    v += x[k * p.n + i] * w[k];
  y[i] = v;
}

kernel void gather(device const float *x [[buffer(0)]],
                   device float *y [[buffer(1)]], constant P &p [[buffer(8)]],
                   uint i [[thread_position_in_grid]]) {
  if (i < p.n)
    y[i] = x[(i / p.a) * p.b + i % p.a + p.c];
}
kernel void gate_scalar(device const float *x [[buffer(0)]],
                        device const float *g [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant P &p [[buffer(8)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < p.n)
    y[i] = x[i] / (1.0f + exp(-g[0]));
}
