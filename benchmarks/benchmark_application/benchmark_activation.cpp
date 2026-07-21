// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   benchmark_activation.cpp
 * @date   21 July 2026
 * @brief  Compare sequential and parallel SiLU/SwiGLU implementations
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs
 */

#include <cmath>
#include <cstdint>
#include <vector>

#include <acti_func.h>
#include <tensor.h>
#include <thread_manager.h>
#include <util_simd.h>

#include "benchmark/benchmark.h"

namespace {

constexpr size_t SWISH_PARALLEL_THRESHOLD = 4096;

void fillInput(float *data, size_t size, unsigned int period) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<float>(i % period) / 10.0f - 5.0f;
  }
}

bool validateOutput(const float *reference, const float *output, size_t size,
                    float tolerance = 1.0e-6f) {
  for (size_t i = 0; i < size; ++i) {
    if (std::abs(reference[i] - output[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

void swishBefore(nntrainer::Tensor const &input, nntrainer::Tensor &output) {
  input.apply<float>(
    [](float value) { return nntrainer::ActiFunc::sigmoid(value); }, output);
  output.multiply_i(input);
}

void swishAfter(nntrainer::Tensor const &input, nntrainer::Tensor &output) {
  const size_t size = input.size();
  const float *input_data = input.getData<float>();
  float *output_data = output.getData<float>();

  auto swish_chunk = [=](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      const float value = input_data[i];
      output_data[i] = value / (1.0f + std::exp(-value));
    }
  };

  auto &thread_manager = nntrainer::ThreadManager::Global();
  const unsigned int thread_count = thread_manager.getComputeThreadCount();
  if (thread_count <= 1 || size < SWISH_PARALLEL_THRESHOLD) {
    swish_chunk(0, size);
    return;
  }

  thread_manager.parallel_for(
    0, static_cast<size_t>(thread_count), [=](size_t thread_index) {
      swish_chunk(size * thread_index / thread_count,
                  size * (thread_index + 1) / thread_count);
    });
}

void swigluBefore(size_t rows, size_t width, float *input1, float *input2,
                  float *output) {
  for (size_t row = 0; row < rows; ++row) {
    nntrainer::swiglu_util(width, output + row * width, input1 + row * width,
                           input2 + row * width);
  }
}

void swigluAfter(size_t rows, size_t width, float *input1, float *input2,
                 float *output) {
  auto &thread_manager = nntrainer::ThreadManager::Global();
  thread_manager.parallel_for(0, rows, [&](size_t row) {
    nntrainer::swiglu_util(width, output + row * width, input1 + row * width,
                           input2 + row * width);
  });
}

template <typename Function>
void runSwishBenchmark(benchmark::State &state, Function &&function) {
  const size_t size = state.range(0);
  nntrainer::Tensor input(1, 1, 1, size);
  nntrainer::Tensor output(1, 1, 1, size);
  nntrainer::Tensor reference(1, 1, 1, size);
  fillInput(input.getData<float>(), size, 101);
  swishBefore(input, reference);
  function(input, output);
  if (!validateOutput(reference.getData<float>(), output.getData<float>(),
                      size)) {
    state.SkipWithError("Swish output differs from the sequential reference");
    return;
  }

  for (auto _ : state) {
    function(input, output);
    benchmark::DoNotOptimize(output.getData<float>());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * size);
  state.counters["threads"] =
    nntrainer::ThreadManager::Global().getComputeThreadCount();
}

template <typename Function>
void runSwiGLUBenchmark(benchmark::State &state, Function &&function) {
  const size_t rows = state.range(0);
  const size_t width = state.range(1);
  const size_t size = rows * width;
  std::vector<float> input1(size);
  std::vector<float> input2(size);
  std::vector<float> output(size);
  std::vector<float> reference(size);
  fillInput(input1.data(), size, 101);
  fillInput(input2.data(), size, 67);
  swigluBefore(rows, width, input1.data(), input2.data(), reference.data());
  function(rows, width, input1.data(), input2.data(), output.data());
  if (!validateOutput(reference.data(), output.data(), size)) {
    state.SkipWithError("SwiGLU output differs from the sequential reference");
    return;
  }

  for (auto _ : state) {
    function(rows, width, input1.data(), input2.data(), output.data());
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * size);
  state.counters["threads"] =
    nntrainer::ThreadManager::Global().getComputeThreadCount();
}

void BM_SwishBefore(benchmark::State &state) {
  runSwishBenchmark(state, swishBefore);
}

void BM_SwishAfter(benchmark::State &state) {
  runSwishBenchmark(state, swishAfter);
}

void BM_SwiGLUBefore(benchmark::State &state) {
  runSwiGLUBenchmark(state, swigluBefore);
}

void BM_SwiGLUAfter(benchmark::State &state) {
  runSwiGLUBenchmark(state, swigluAfter);
}

void registerSwishArguments(benchmark::internal::Benchmark *benchmark) {
  benchmark->Arg(4096)->Arg(32768)->Arg(524288)->Arg(2097152)->UseRealTime();
}

void registerSwiGLUArguments(benchmark::internal::Benchmark *benchmark) {
  for (const int64_t width : {3072, 11008}) {
    for (const int64_t rows : {1, 32, 128, 512}) {
      benchmark->Args({rows, width});
    }
  }
  benchmark->UseRealTime();
}

BENCHMARK(BM_SwishBefore)->Apply(registerSwishArguments);
BENCHMARK(BM_SwishAfter)->Apply(registerSwishArguments);
BENCHMARK(BM_SwiGLUBefore)->Apply(registerSwiGLUArguments);
BENCHMARK(BM_SwiGLUAfter)->Apply(registerSwiGLUArguments);

} // namespace

BENCHMARK_MAIN();
