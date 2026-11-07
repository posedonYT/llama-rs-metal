// Development-only upstream llama.cpp oracle. No linkage into the Rust runtime.
#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
class Reference {
  llama_model *model;
  llama_context *ctx;
  const llama_vocab *vocab;
  int nv;
  size_t context;

public:
  Reference(const char *path, size_t capacity) : context(capacity) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    model = llama_model_load_from_file(path, mp);
    if (!model)
      throw std::runtime_error("model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = capacity;
    cp.n_batch = capacity;
    cp.n_ubatch = 512;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    ctx = llama_init_from_model(model, cp);
    if (!ctx)
      throw std::runtime_error("context allocation failed");
    vocab = llama_model_get_vocab(model);
    nv = llama_vocab_n_tokens(vocab);
  }
  ~Reference() {
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
  }
  void reset() {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_synchronize(ctx);
  }
  std::vector<float> forward(std::vector<llama_token> ids) {
    if (ids.empty())
      throw std::runtime_error("empty forward");
    auto batch = llama_batch_get_one(ids.data(), ids.size());
    if (llama_decode(ctx, batch))
      throw std::runtime_error("decode failed");
    const float *logits = llama_get_logits_ith(ctx, -1);
    // A host copy matches both other workers' public API boundary.
    std::vector<float> result(logits, logits + nv);
    if (!std::all_of(result.begin(), result.end(),
                     [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("nonfinite logits");
    return result;
  }
  static int argmax(const std::vector<float> &x) {
    return std::max_element(x.begin(), x.end()) - x.begin();
  }
  json run(const std::vector<llama_token> &prompt,
           const std::vector<llama_token> &decode) {
    auto start = Clock::now();
    reset();
    double reset_seconds = elapsed(start);
    start = Clock::now();
    auto logits = forward(prompt);
    int last = argmax(logits);
    double prefill = elapsed(start);
    std::vector<double> steps;
    start = Clock::now();
    for (auto token : decode) {
      auto step = Clock::now();
      logits = forward({token});
      last = argmax(logits);
      steps.push_back(elapsed(step));
    }
    double duration = elapsed(start);
    return {{"reset_seconds", reset_seconds},
            {"prefill_seconds", prefill},
            {"request_ttft_seconds", reset_seconds + prefill},
            {"decode_seconds", duration},
            {"decode_step_seconds", steps},
            {"prefill_tokens_per_second", prompt.size() / prefill},
            {"decode_tokens_per_second", decode.size() / duration},
            {"last_prediction", last}};
  }
  json handle(const json &request) {
    std::string action = request.at("action");
    if (action == "tokenize") {
      std::string text = request.at("text");
      bool special = request.value("special", true);
      int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0,
                             special, true);
      std::vector<llama_token> ids(-n);
      n = llama_tokenize(vocab, text.data(), text.size(), ids.data(),
                         ids.size(), special, true);
      if (n < 0)
        throw std::runtime_error("tokenization failed");
      ids.resize(n);
      return {{"ids", ids}};
    }
    auto prompt = request.at("prompt_ids").get<std::vector<llama_token>>();
    auto decode = request.at("decode_ids").get<std::vector<llama_token>>();
    if (prompt.empty() || prompt.size() + decode.size() > context)
      throw std::runtime_error("invalid workload/context");
    if (action == "benchmark") {
      for (int i = 0; i < request.value("warmups", 2); i++)
        run(prompt, decode);
      json runs = json::array();
      for (int i = 0; i < request.value("repetitions", 2); i++)
        runs.push_back(run(prompt, decode));
      return {{"runs", runs},
              {"boundary", "full_logits_host_greedy"},
              {"cache_policy", "preallocated_reset_outside_forward"}};
    }
    if (action == "verify") {
      std::filesystem::path dir = request.at("output_dir").get<std::string>();
      std::filesystem::create_directories(dir);
      reset();
      size_t chunk = request.value("chunk_size", prompt.size());
      if (!chunk)
        throw std::runtime_error("zero chunk");
      std::vector<float> logits;
      for (size_t i = 0; i < prompt.size(); i += chunk)
        logits = forward({prompt.begin() + i,
                          prompt.begin() + std::min(i + chunk, prompt.size())});
      json traces = json::array();
      auto save = [&](std::string name) {
        auto path = dir / name;
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(logits.data()),
                   logits.size() * sizeof(float));
        if (!file)
          throw std::runtime_error("trace write failed");
        traces.push_back({{"path", path.string()},
                          {"top1", argmax(logits)},
                          {"count", logits.size()}});
      };
      save("prefill.f32");
      for (size_t i = 0; i < decode.size(); i++) {
        logits = forward({decode[i]});
        save("decode-" + std::to_string(i) + ".f32");
      }
      return {{"traces", traces}};
    }
    throw std::runtime_error("unknown action");
  }
};
int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::runtime_error("model required");
    size_t capacity = 4096;
    for (int i = 2; i + 1 < argc; i++)
      if (std::string(argv[i]) == "--context")
        capacity = std::stoul(argv[++i]);
    auto start = Clock::now();
    Reference ref(argv[1], capacity);
    std::cout << json({{"type", "ready"},
                       {"protocol", 1},
                       {"engine", "llama.cpp"},
                       {"context", capacity},
                       {"kv_dtype", "f32"},
                       {"activation_dtype", "implementation_default"},
                       {"gpu_layers", 999},
                       {"n_ubatch", 512},
                       {"cpu_threads", 4},
                       {"load_seconds", elapsed(start)}})
                     .dump()
              << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
      try {
        std::cout << json({{"ok", true},
                           {"result", ref.handle(json::parse(line))}})
                         .dump()
                  << std::endl;
      } catch (const std::exception &e) {
        std::cout << json({{"ok", false}, {"error", e.what()}}).dump()
                  << std::endl;
      }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
