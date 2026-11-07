// Llama uses adjacent rotary pairs; Qwen uses split-half pairs. Text MRoPE
// positions coincide.
kernel void rope(device const float *x [[buffer(0)]],
                 device float *y [[buffer(1)]],
                 device const float *factors [[buffer(2)]],
                 constant P &p [[buffer(8)]],
                 uint i [[thread_position_in_grid]]) {
  if (i >= p.n)
    return;
  uint dim = i % p.a, head = i / p.a;
  if (dim >= p.b) {
    y[i] = x[i];
    return;
  }
  uint pair = p.c ? dim % (p.b / 2) : dim / 2;
  uint first = p.c ? pair : pair * 2;
  uint second = p.c ? pair + p.b / 2 : first + 1;
  float angle = float(p.d + head / max(p.e, 1u)) * p.y *
                pow(p.x, -2.0f * float(pair) / float(p.b));
  angle /= factors[pair];
  float cs = cos(angle), sn = sin(angle);
  y[i] = dim == first
             ? x[head * p.a + first] * cs - x[head * p.a + second] * sn
             : x[head * p.a + first] * sn + x[head * p.a + second] * cs;
}
// Streaming online-softmax attention: a SIMD group owns a query head, no score
// matrix.
kernel void attention(device const float *q [[buffer(0)]],
                      device const float *k [[buffer(1)]],
                      device const float *v [[buffer(2)]],
                      device float *out [[buffer(3)]],
                      constant P &p [[buffer(8)]],
                      uint3 g [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint width [[threads_per_simdgroup]]) {
  uint h = g.x, kh = h / (p.b / p.c), token = g.y;
  uint nt = p.d + token + 1;
  float acc[16];
  for (uint j = 0; j < 16; j++)
    acc[j] = 0;
  float m = -INFINITY, total = 0;
  for (uint t = 0; t < nt; t++) {
    float dot = 0;
    for (uint d = lane; d < p.a; d += width)
      dot += q[(token * p.b + h) * p.a + d] * k[(t * p.c + kh) * p.a + d];
    dot = simd_sum(dot) * p.x;
    float nm = max(m, dot), old = exp(m - nm), prob = exp(dot - nm);
    total = total * old + prob;
    uint j = 0;
    for (uint d = lane; d < p.a; d += width, j++)
      acc[j] = acc[j] * old + prob * v[(t * p.c + kh) * p.a + d];
    m = nm;
  }
  uint j = 0;
  for (uint d = lane; d < p.a; d += width, j++)
    out[(token * p.b + h) * p.a + d] = acc[j] / total;
}

// Decode splits the context across SIMD groups for occupancy on small head
// counts. Each partial stores an unnormalized weighted sum, maximum and mass.
kernel void attention_partial(device const float *q [[buffer(0)]],
                              device const float *k [[buffer(1)]],
                              device const float *v [[buffer(2)]],
                              device float *partial [[buffer(3)]],
                              constant P &p [[buffer(8)]],
                              uint3 g [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint width [[threads_per_simdgroup]]) {
  uint head = g.x, chunk = g.y, kh = head / (p.b / p.c);
  uint begin = chunk * p.f, end = min(begin + p.f, p.d);
  float acc[16];
  for (uint j = 0; j < 16; j++)
    acc[j] = 0;
  float maximum = -INFINITY, mass = 0;
  for (uint token = begin; token < end; token++) {
    float score = 0;
    for (uint d = lane; d < p.a; d += width)
      score += q[head * p.a + d] * k[(token * p.c + kh) * p.a + d];
    score = simd_sum(score) * p.x;
    float next = max(maximum, score), old = exp(maximum - next),
          probability = exp(score - next);
    mass = mass * old + probability;
    uint j = 0;
    for (uint d = lane; d < p.a; d += width, j++)
      acc[j] = acc[j] * old + probability * v[(token * p.c + kh) * p.a + d];
    maximum = next;
  }
  uint base = (head * p.e + chunk) * (p.a + 2), j = 0;
  for (uint d = lane; d < p.a; d += width, j++)
    partial[base + d] = acc[j];
  if (lane == 0) {
    partial[base + p.a] = maximum;
    partial[base + p.a + 1] = mass;
  }
}
kernel void attention_merge(device const float *partial [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            constant P &p [[buffer(8)]],
                            uint3 g [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint width [[threads_per_simdgroup]]) {
  uint head = g.x;
  float maximum = -INFINITY;
  for (uint c = 0; c < p.e; c++)
    maximum = max(maximum, partial[(head * p.e + c) * (p.a + 2) + p.a]);
  float acc[16];
  for (uint j = 0; j < 16; j++)
    acc[j] = 0;
  float mass = 0;
  for (uint c = 0; c < p.e; c++) {
    uint base = (head * p.e + c) * (p.a + 2);
    float scale = exp(partial[base + p.a] - maximum);
    mass += partial[base + p.a + 1] * scale;
    uint j = 0;
    for (uint d = lane; d < p.a; d += width, j++)
      acc[j] += partial[base + d] * scale;
  }
  uint j = 0;
  for (uint d = lane; d < p.a; d += width, j++)
    out[head * p.a + d] = acc[j] / mass;
}
