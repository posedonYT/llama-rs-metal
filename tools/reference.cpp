// Development-only oracle. Not linked into the Rust runtime.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: reference MODEL PROMPT LOGITS TOKEN_IDS\n");
    return 2;
  }
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  mp.vocab_only = std::getenv("TOKENIZER_ONLY") != nullptr;
  auto *model = llama_model_load_from_file(argv[1], mp);
  if (!model)
    return 3;
  const auto *vocab = llama_model_get_vocab(model);
  int n;
  std::vector<llama_token> tokens;
  if (argv[2][0] == '@') {
    std::ifstream f(argv[2] + 1);
    int id;
    while (f >> id)
      tokens.push_back(id);
    n = tokens.size();
  } else {
    n = -llama_tokenize(vocab, argv[2], strlen(argv[2]), nullptr, 0, true,
                        true);
    tokens.resize(n);
    n = llama_tokenize(vocab, argv[2], strlen(argv[2]), tokens.data(), n, true,
                       true);
  }
  if (n <= 0)
    return 4;
  FILE *tf = fopen(argv[4], "w");
  for (int i = 0; i < n; i++)
    fprintf(tf, "%d\n", tokens[i]);
  fclose(tf);
  if (mp.vocab_only) {
    llama_model_free(model);
    llama_backend_free();
    return 0;
  }
  auto cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_batch = 512;
  cp.n_ubatch = 512;
  cp.type_k = GGML_TYPE_F32;
  cp.type_v = GGML_TYPE_F32;
  auto *ctx = llama_init_from_model(model, cp);
  if (!ctx)
    return 5;
  auto batch = llama_batch_get_one(tokens.data(), n);
  if (llama_decode(ctx, batch))
    return 6;
  FILE *f = fopen(argv[3], "wb");
  fwrite(llama_get_logits_ith(ctx, -1), sizeof(float),
         llama_vocab_n_tokens(vocab), f);
  fclose(f);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
}
