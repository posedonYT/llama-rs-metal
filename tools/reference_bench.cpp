// Same-token, F32-cache benchmark oracle. Development only; never linked into
// Rust.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(
        stderr,
        "usage: reference_bench MODEL TOKEN_FILE DECODE_COUNT REPETITIONS\n");
    return 2;
  }
  std::ifstream input(argv[2]);
  std::vector<llama_token> tokens;
  int id;
  while (input >> id)
    tokens.push_back(id);
  if (tokens.empty())
    return 3;
  const int ng = std::atoi(argv[3]), reps = std::atoi(argv[4]);
  if (ng <= 0 || reps <= 0)
    return 4;
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  auto *model = llama_model_load_from_file(argv[1], mp);
  if (!model)
    return 5;
  auto cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_batch = 512;
  cp.n_ubatch = 512;
  cp.type_k = GGML_TYPE_F32;
  cp.type_v = GGML_TYPE_F32;
  auto *ctx = llama_init_from_model(model, cp);
  if (!ctx)
    return 6;
  auto vocab = llama_model_get_vocab(model);
  int nv = llama_vocab_n_tokens(vocab);
  printf("{\"sampling\":\"greedy_teacher_forced\",\"kv_dtype\":\"f32\","
         "\"recurrent_dtype\":\"f32\",\"context\":512,"
         "\"runs\":[");
  for (int r = -1; r < reps; r++) {
    llama_memory_clear(llama_get_memory(ctx), true);
    auto start = Clock::now();
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch))
      return 7;
    auto *logits = llama_get_logits_ith(ctx, -1);
    volatile llama_token prediction =
        std::max_element(logits, logits + nv) - logits;
    double pp = elapsed(start);
    start = Clock::now();
    for (int i = 0; i < ng; i++) {
      llama_token token = tokens[i % tokens.size()];
      auto one = llama_batch_get_one(&token, 1);
      if (llama_decode(ctx, one))
        return 8;
      logits = llama_get_logits_ith(ctx, -1);
      prediction = std::max_element(logits, logits + nv) - logits;
    }
    double tg = elapsed(start);
    (void)prediction;
    if (r >= 0)
      printf("%s{\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,\"prefill_"
             "tokens_per_second\":%.6f,\"decode_tokens_per_second\":%.6f}",
             r ? "," : "", pp, tg, tokens.size() / pp, ng / tg);
  }
  printf("]}\n");
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
}
