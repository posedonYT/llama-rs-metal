#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;
// GGML packed block layouts; see THIRD_PARTY_NOTICES.md.
struct P {
  uint n, a, b, c, d, e, f, g, h, i, j, k, l, m, o, t;
  float x, y, z, w;
};
inline float hf(device const uchar *b, ulong p) {
  return float(as_type<half>(ushort(ushort(b[p]) | (ushort(b[p + 1]) << 8))));
}
inline float val(device const uchar *w, ulong i, uint t) {
  if (t == 0)
    return ((device const float *)w)[i];
  if (t == 1)
    return hf(w, i * 2);
  if (t == 30)
    return as_type<float>(uint(((device const ushort *)w)[i]) << 16);
  if (t == 2) {
    device const uchar *b = w + (i / 32) * 18;
    uint j = i % 32;
    return hf(b, 0) * (float((b[2 + j % 16] >> (4 * (j / 16))) & 15) - 8.0f);
  }
  if (t == 8) {
    device const uchar *b = w + (i / 32) * 34;
    return hf(b, 0) * float(as_type<char>(b[2 + i % 32]));
  }
  if (t == 12 || t == 13) {
    uint sz = t == 12 ? 144 : 176;
    device const uchar *b = w + (i / 256) * sz;
    uint j = i % 256, g = j / 32;
    device const uchar *s = b + 4;
    uint sc = g < 4 ? s[g] & 63 : (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
    uint mn = g < 4 ? s[g + 4] & 63 : (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
    uint q =
        (b[(t == 12 ? 16 : 48) + (j / 64) * 32 + j % 32] >> (4 * (g % 2))) & 15;
    if (t == 13)
      q |= ((b[16 + j % 32] >> g) & 1) << 4;
    return hf(b, 0) * float(sc * q) - hf(b, 2) * float(mn);
  }
  device const uchar *b = w + (i / 256) * 210;
  uint j = i % 256, ha = j / 128, k = j % 128, l = k % 32, q = k / 32;
  uint lo = (b[ha * 64 + l + (q % 2) * 32] >> (q < 2 ? 0 : 4)) & 15;
  uint hi = (b[128 + ha * 32 + l] >> (2 * q)) & 3;
  return hf(b, 208) * float(as_type<char>(b[192 + j / 16])) *
         float(int(lo | (hi << 4)) - 32);
}
