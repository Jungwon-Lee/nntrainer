/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   smallthinker_moe_layer_cached_slim.cpp
 * @date   26 June 2026
 * @brief  SmallThinker MoE layer with on-demand expert loading and LRU cache.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#ifdef DEBUG
#include <iostream>
#endif
#include <node_exporter.h>
#include <smallthinker_moe_layer_cached_slim.h>
#include <stdexcept>
#include <thread_manager.h>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

namespace causallm {

namespace {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void normalize_router_weights(nntrainer::Tensor &topk_values,
                              unsigned int total_tokens, unsigned int topk,
                              bool router_apply_softmax) {
  if (router_apply_softmax) {
    topk_values.apply(nntrainer::ActiFunc::softmax<float>, topk_values);
    return;
  }

  float *values_data = topk_values.getData<float>();
  for (unsigned int i = 0; i < total_tokens; ++i) {
    float sum = 0.0f;
    for (unsigned int k = 0; k < topk; ++k) {
      float &value = values_data[i * topk + k];
      value = 1.0f / (1.0f + std::exp(-value));
      sum += value;
    }
    for (unsigned int k = 0; k < topk; ++k) {
      values_data[i * topk + k] /= sum;
    }
  }
}

// Collect the top (topk + extra) expert indices across all tokens, used as the
// LRU prediction set. Indices only; routing weights are computed separately.
std::vector<unsigned int> collect_predicted(nntrainer::Tensor &router_logits,
                                            unsigned int total_tokens,
                                            unsigned int topk,
                                            unsigned int num_experts) {
  const unsigned int extra =
    std::min(topk + 5u, num_experts); // a few near-misses kept warm
  auto extra_topk_result = router_logits.topK(extra);
  auto extra_topk_indices = std::get<1>(extra_topk_result);
  const uint32_t *idx = extra_topk_indices.getData<uint32_t>();

  std::vector<unsigned int> predicted;
  predicted.reserve(total_tokens * extra);
  for (unsigned int i = 0; i < total_tokens; ++i)
    for (unsigned int k = 0; k < extra; ++k)
      predicted.push_back(idx[i * extra + k]);
  return predicted;
}

} // namespace

SmallThinkerCachedSlimMoELayer::SmallThinkerCachedSlimMoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  router_apply_softmax(true),
  cache_size(32),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation(),
            props::MoERouterApplySoftmax(), props::MoECacheSize()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {}

void SmallThinkerCachedSlimMoELayer::finalize(
  nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "SmallThinker cached-slim MoE layer requires expert input and router "
       "input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  router_apply_softmax =
    std::get<props::MoERouterApplySoftmax>(moe_props).get();
  cache_size = std::get<props::MoECacheSize>(moe_props).get();
  // Optional runtime override for tuning / A-B measurement. cache_size==0
  // reproduces the old behavior (evict every expert after each token).
  if (const char *env = std::getenv("NNTR_MOE_CACHE_SIZE"))
    cache_size = static_cast<unsigned int>(std::stoul(env));
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width();

  if (std::get<props::MoEActivation>(moe_props).empty()) {
    throw std::runtime_error("Activation type is not set for MoE layer");
  }
  switch (context.getActivationDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    acti_func.setActiFunc<float>(
      std::get<props::MoEActivation>(moe_props).get());
    break;
  default:
    throw std::runtime_error("Unsupported activation data type for MoE layer");
  }

  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  expert_gate_proj_indices.reserve(num_experts);
  expert_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);

  nntrainer::TensorDim expert_gate_dim(
    1, is_nchw ? 1 : intermediate_size, is_nchw ? hidden_size : 1,
    is_nchw ? intermediate_size : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  nntrainer::TensorDim expert_down_dim(
    1, is_nchw ? 1 : hidden_size, is_nchw ? intermediate_size : 1,
    is_nchw ? hidden_size : intermediate_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  for (unsigned int i = 0; i < num_experts; ++i) {
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_up_" + std::to_string(i), false, true));

    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, true));

    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));
  }

  // All experts start unloaded; the LRU cache populates them on first use.
  need_load.assign(num_experts, true);

  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  expert_mask_idx =
    context.requestTensor({num_experts, 1, topk, total_tokens}, "expert_mask",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void SmallThinkerCachedSlimMoELayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &router_input = context.getInput(1);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits = context.getTensor(router_logits_idx);
  nntrainer::Tensor &expert_mask = context.getTensor(expert_mask_idx);

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  input.reshape({total_tokens, 1, 1, hidden_size});
  router_input.reshape({total_tokens, 1, 1, hidden_size});
  output.reshape({total_tokens, 1, 1, hidden_size});
  output.setZero();

  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  router_input.dot(gate_weights, router_logits);
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  normalize_router_weights(topk_values, total_tokens, topk,
                           router_apply_softmax);

  expert_mask.setZero();
  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(0, total_tokens * topk, [&](size_t idx) {
    const size_t i = idx / topk;
    const size_t k = idx % topk;
    expert_mask.setValue(indices_data[i * topk + k], 0, k, i, 1.0f);
  });

  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      unsigned expert_idx = indices_data[i * topk + k];
      float weight = topk_values.getValue<float>(i, 0, 0, k);
      expert_assignments[expert_idx].emplace_back(i, weight);
    }
  }

  std::vector<unsigned int> predicted =
    collect_predicted(router_logits, total_tokens, topk, num_experts);

  long long activate_ns = 0, compute_ns = 0;
  for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const auto &assignments = expert_assignments[expert_idx];
    if (assignments.empty())
      continue;

    run_active_expert(context, input, output, assignments, expert_idx,
                      hidden_size, activate_ns, compute_ns);
  }
  (void)activate_ns;
  (void)compute_ns;

  touch_predicted(predicted);
  evict_experts(context);

  output.reshape({batch_size, 1, seq_len, hidden_size});
  input.reshape({batch_size, 1, seq_len, hidden_size});
  router_input.reshape({batch_size, 1, seq_len, hidden_size});
}

inline void SmallThinkerCachedSlimMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  nntrainer::Tensor expert_output(output.batch(), output.channel(),
                                  output.height(), output.width(),
                                  output.getTensorType());
  expert_output.setZero();

  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;

    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);

    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);

    token_input.dot(gate_proj, gate_out);
    acti_func.run_fn(gate_out, acti_out);
    token_input.dot(up_proj, up_out);
    acti_out.multiply_i(up_out);

    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);

    token_expert_output.multiply_i(weight);
    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      expert_output.getSharedDataTensor(token_output_dim, output_offset, true);

    token_output.add_i(token_expert_output);
  }

  output.add_i(expert_output);
}

bool SmallThinkerCachedSlimMoELayer::run_active_expert(
  nntrainer::RunLayerContext &context, const nntrainer::Tensor &input,
  nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &assignments,
  unsigned int expert_idx, unsigned int hidden_size, long long &activate_ns,
  long long &compute_ns) {

  bool is_miss;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    is_miss = need_load[expert_idx];
  }

  // On a miss, page the expert in and record it as the most-recently-used.
  // activate()/deactivate() must operate on the persistent context weight so
  // that eviction can later munmap the same mapping. We never deactivate here;
  // eviction is deferred to evict_experts().
  if (is_miss) {
    auto ta0 = high_resolution_clock::now();
    context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
    context.getWeight(expert_up_proj_indices[expert_idx]).activate();
    context.getWeight(expert_down_proj_indices[expert_idx]).activate();
    auto ta1 = high_resolution_clock::now();
    activate_ns += duration_cast<nanoseconds>(ta1 - ta0).count();

    std::lock_guard<std::mutex> lock(cache_mutex);
    loaded_expert_deque.push_back(expert_idx);
    iteration_map[expert_idx] = --loaded_expert_deque.end();
    need_load[expert_idx] = false;
  }

  auto tc0 = high_resolution_clock::now();
  compute_expert_forward(
    input, output, assignments,
    context.getWeight(expert_gate_proj_indices[expert_idx]),
    context.getWeight(expert_up_proj_indices[expert_idx]),
    context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
  auto tc1 = high_resolution_clock::now();
  compute_ns += duration_cast<nanoseconds>(tc1 - tc0).count();

  return is_miss;
}

void SmallThinkerCachedSlimMoELayer::touch_predicted(
  const std::vector<unsigned int> &predicted) {
  // Move already-resident predicted experts (this token's top-k plus a few
  // near-misses) to the MRU end so eviction prefers genuinely cold experts.
  std::lock_guard<std::mutex> lock(cache_mutex);
  for (auto it = predicted.rbegin(); it != predicted.rend(); ++it) {
    auto found = iteration_map.find(static_cast<int>(*it));
    if (found != iteration_map.end()) {
      loaded_expert_deque.erase(found->second);
      loaded_expert_deque.push_back(static_cast<int>(*it));
      iteration_map[static_cast<int>(*it)] = --loaded_expert_deque.end();
    }
  }
}

void SmallThinkerCachedSlimMoELayer::evict_experts(
  nntrainer::RunLayerContext &context) {
  while (true) {
    int target_idx;
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      if (loaded_expert_deque.size() <= cache_size)
        break;
      target_idx = loaded_expert_deque.front();
      loaded_expert_deque.pop_front();
      iteration_map.erase(target_idx);
      need_load[target_idx] = true;
    }
    context.getWeight(expert_gate_proj_indices[target_idx]).deactivate();
    context.getWeight(expert_up_proj_indices[target_idx]).deactivate();
    context.getWeight(expert_down_proj_indices[target_idx]).deactivate();
  }
}

void SmallThinkerCachedSlimMoELayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &router_input_ = context.getInput(1);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);
  nntrainer::Tensor &expert_mask = context.getTensor(expert_mask_idx);

  nntrainer::TensorDim input_step_dim = input_.getDim();
  nntrainer::TensorDim output_step_dim = output_.getDim();
  nntrainer::TensorDim router_logits_step_dim = router_logits_.getDim();

  input_step_dim.batch(1);
  output_step_dim.batch(1);
  router_logits_step_dim.batch(to - from);

  input_step_dim.height(to - from);
  output_step_dim.height(to - from);

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    auto input = input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto router_input = router_input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto output = output_.getSharedDataTensor(
      output_step_dim, b * output_step_dim.getFeatureLen(), true);
    auto router_logits =
      router_logits_.getSharedDataTensor(router_logits_step_dim, 0, true);

    const unsigned batch_size = input.batch();
    const unsigned seq_len = input.height();
    const unsigned hidden_size = input.width();
    const unsigned total_tokens = batch_size * seq_len;

    input.reshape({total_tokens, 1, 1, hidden_size});
    router_input.reshape({total_tokens, 1, 1, hidden_size});
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();
    expert_mask.setZero();

    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    router_input.dot(gate_weights, router_logits);
    auto topk_result = router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);

    normalize_router_weights(topk_values, total_tokens, topk,
                             router_apply_softmax);

    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        expert_mask.setValue(indices_data[i * topk + k], 0, k, i, 1.0f);
      }
    }

    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = topk_values.getValue<float>(i, 0, 0, k);
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    // LRU prediction set (this token's top-k plus a few near-misses), indices
    // only; routing weights above are unchanged so output is bit-identical.
    std::vector<unsigned int> predicted =
      collect_predicted(router_logits, total_tokens, topk, num_experts);

    long long activate_ns = 0, compute_ns = 0;
#ifdef DEBUG
    int hit_count = 0, miss_count = 0;
    auto t_loop0 = high_resolution_clock::now();
#endif

    // Serial outer loop: the Q4_0 expert GEMV parallelizes internally via
    // ThreadManager, and nesting parallel_for deadlocks.
    for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      const auto &assignments = expert_assignments[expert_idx];
      if (assignments.empty())
        continue;

      bool is_miss =
        run_active_expert(context, input, output, assignments, expert_idx,
                          hidden_size, activate_ns, compute_ns);
#ifdef DEBUG
      is_miss ? ++miss_count : ++hit_count;
#else
      (void)is_miss;
#endif
    }

    // Keep predicted experts warm, then evict the coldest beyond cache_size.
    touch_predicted(predicted);
    evict_experts(context);

#ifdef DEBUG
    auto t_loop1 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t_loop1 - t_loop0);
    std::cout << context.getName() << " \t| " << dt.count() / 1'000'000 << " ms"
              << "\t| hit=" << hit_count << " miss=" << miss_count
              << "\t| activate=" << activate_ns / 1'000'000 << "ms"
              << " compute=" << compute_ns / 1'000'000 << "ms"
              << " resident=" << loaded_expert_deque.size() << std::endl;
#else
    (void)activate_ns;
    (void)compute_ns;
#endif

    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
}

void SmallThinkerCachedSlimMoELayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void SmallThinkerCachedSlimMoELayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void SmallThinkerCachedSlimMoELayer::calcGradient(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void SmallThinkerCachedSlimMoELayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

} // namespace causallm
