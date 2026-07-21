// Standalone layer benchmark for nntrainer PR #4095.
#include <activation_layer.h>
#include <layer_context.h>
#include <swiglu.h>
#include <thread_manager.h>
#include <var_grad.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<nntrainer::Var_Grad *>
makeView(std::vector<nntrainer::Var_Grad> &vars) {
  std::vector<nntrainer::Var_Grad *> view;
  view.reserve(vars.size());
  for (auto &var : vars)
    view.push_back(&var);
  return view;
}

nntrainer::RunLayerContext
makeContext(const std::string &name, std::vector<nntrainer::Var_Grad> &inputs,
            std::vector<nntrainer::Var_Grad> &outputs) {
  std::vector<nntrainer::Weight *> weights;
  std::vector<nntrainer::Var_Grad> tensors;
  return nntrainer::RunLayerContext(name, true, 0.0f, false, 1.0f, nullptr,
                                    false, weights, makeView(inputs),
                                    makeView(outputs), makeView(tensors));
}

template <typename F>
std::vector<double> measure(F &&fn, unsigned int warmup,
                            unsigned int iterations, unsigned int samples) {
  for (unsigned int i = 0; i < warmup; ++i)
    fn();

  std::vector<double> times;
  times.reserve(samples);
  for (unsigned int sample = 0; sample < samples; ++sample) {
    const auto begin = Clock::now();
    for (unsigned int i = 0; i < iterations; ++i)
      fn();
    const auto end = Clock::now();
    const double elapsed_us =
      std::chrono::duration<double, std::micro>(end - begin).count();
    times.push_back(elapsed_us / iterations);
  }
  return times;
}

void printResult(const std::string &layer, const std::vector<double> &times,
                 volatile float checksum) {
  std::vector<double> sorted = times;
  std::sort(sorted.begin(), sorted.end());
  const double median = sorted[sorted.size() / 2];
  const double mean =
    std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  const double min = sorted.front();
  const double max = sorted.back();
  std::cout << layer << ',' << std::fixed << std::setprecision(3) << median
            << ',' << mean << ',' << min << ',' << max << ',' << checksum
            << '\n';
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: layer_benchmark THREADS HEIGHT WIDTH ITERATIONS "
                 "SAMPLES\n";
    return 2;
  }

  const unsigned int threads = std::stoul(argv[1]);
  const unsigned int height = std::stoul(argv[2]);
  const unsigned int width = std::stoul(argv[3]);
  const unsigned int iterations = std::stoul(argv[4]);
  const unsigned int samples = std::stoul(argv[5]);

  nntrainer::ThreadManagerConfig config;
  config.compute_threads = threads;
  config.enable_affinity = true;
  nntrainer::ThreadManager::setConfig(config);
  auto &thread_manager = nntrainer::ThreadManager::Global();

  const nntrainer::TensorDim dim(1, 1, height, width);
  constexpr unsigned int warmup = 10;

  nntrainer::ActivationLayer silu;
  silu.setProperty({"activation=swish"});
  nntrainer::InitLayerContext silu_init({dim}, {true}, false, "silu");
  silu.finalize(silu_init);
  std::vector<nntrainer::Var_Grad> silu_inputs;
  std::vector<nntrainer::Var_Grad> silu_outputs;
  silu_inputs.emplace_back(dim, nntrainer::Initializer::NONE, false, true,
                           "silu_input");
  silu_outputs.emplace_back(dim, nntrainer::Initializer::NONE, false, true,
                            "silu_output");
  auto silu_context = makeContext("silu", silu_inputs, silu_outputs);

  float *silu_input = silu_context.getInput(0).getData<float>();
  for (size_t i = 0; i < dim.getDataLen(); ++i)
    silu_input[i] = static_cast<float>(static_cast<int>(i % 101) - 50) / 10.0f;

  const auto silu_times = measure([&] { silu.forwarding(silu_context, false); },
                                  warmup, iterations, samples);
  volatile float silu_checksum = silu_context.getOutput(0).getData<float>()[0];
  printResult("SiLU", silu_times, silu_checksum);

  causallm::SwiGLULayer swiglu;
  std::vector<nntrainer::Var_Grad> swiglu_inputs;
  std::vector<nntrainer::Var_Grad> swiglu_outputs;
  swiglu_inputs.emplace_back(dim, nntrainer::Initializer::NONE, false, true,
                             "swiglu_gate");
  swiglu_inputs.emplace_back(dim, nntrainer::Initializer::NONE, false, true,
                             "swiglu_up");
  swiglu_outputs.emplace_back(dim, nntrainer::Initializer::NONE, false, true,
                              "swiglu_output");
  auto swiglu_context = makeContext("swiglu", swiglu_inputs, swiglu_outputs);

  float *gate = swiglu_context.getInput(0).getData<float>();
  float *up = swiglu_context.getInput(1).getData<float>();
  for (size_t i = 0; i < dim.getDataLen(); ++i) {
    gate[i] = static_cast<float>(static_cast<int>(i % 101) - 50) / 10.0f;
    up[i] = static_cast<float>(static_cast<int>(i % 67) - 33) / 8.0f;
  }

  const auto swiglu_times = measure(
    [&] { swiglu.incremental_forwarding(swiglu_context, 0, height, false); },
    warmup, iterations, samples);
  volatile float swiglu_checksum =
    swiglu_context.getOutput(0).getData<float>()[0];
  printResult("SwiGLU", swiglu_times, swiglu_checksum);

  std::cerr << "effective_threads=" << thread_manager.getComputeThreadCount()
            << " shape=1x1x" << height << 'x' << width << '\n';
  return 0;
}
