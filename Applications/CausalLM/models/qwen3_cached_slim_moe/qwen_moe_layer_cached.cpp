/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 *
 * @file	qwen_moe_layer_fsu.cpp
 * @date	09 June 2025
 * @brief	This is a Mixture of Expert Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 * @note    MoE layer with on-the-fly expert FSU
 *
 */

#include <acti_func.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <node_exporter.h>
#include <qwen_moe_layer_cached.h>
#include <stdexcept>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;
static constexpr unsigned int DEFAULT_EXPERT_CACHE_CAPACITY = 32;

static unsigned int getExpertCacheCapacity(unsigned int num_experts) {
  const char *value = std::getenv("NNTR_MOE_CACHE_EXPERTS");
  if (value == nullptr)
    return std::min(num_experts, DEFAULT_EXPERT_CACHE_CAPACITY);

  if (*value == '\0' || *value == '-')
    return std::min(num_experts, DEFAULT_EXPERT_CACHE_CAPACITY);

  errno = 0;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno == ERANGE || end == value || *end != '\0')
    return std::min(num_experts, DEFAULT_EXPERT_CACHE_CAPACITY);

  return std::min(static_cast<unsigned long>(num_experts), parsed);
}

CachedSlimMoELayer::CachedSlimMoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  loaded_expert_deque({}),
  cache_positions({}),
  need_load({}),
  cache_capacity(0),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()) {}

void CachedSlimMoELayer::finalize(nntrainer::InitLayerContext &context) {
  profiler.setName(context.getName());

  // 1. Validate input/output dimensions
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "MoE layer only supports single input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  // 2. Set output dimensions (same as input)
  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  // 3. Get MoE properties
  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width(); // Feature dimension

  // activation function
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

  // 4. Initialie gate layer (router)
  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  // 5. Initializer expert weights
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
    // Up projection
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, // Same dimensions as gate projection
      weight_initializer, weight_regularizer, weight_regularizer_constant,
      weight_decay, "expert_up_" + std::to_string(i), false, true));

    // Gate projection
    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, true));

    // Down projection
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));
  }
  cache_positions.resize(num_experts);
  need_load.assign(num_experts, 1);
  cache_capacity = getExpertCacheCapacity(num_experts);

  // 6. Request intermediate tensors
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  // Router logits :  [batch * seq, num_experts]
  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void CachedSlimMoELayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {}

inline void CachedSlimMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::pair<unsigned, float> *token_assignments, unsigned int num_tokens,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size,
  nntrainer::Tensor &token_input_workspace,
  nntrainer::Tensor &gate_out_workspace, nntrainer::Tensor &acti_out_workspace,
  nntrainer::Tensor &up_out_workspace) {

  const unsigned intermediate_size = gate_proj.width();

  if (num_tokens == 0)
    return;

  // Create tensor dimensions for single token processing
  nntrainer::TensorDim token_input_dim({1, 1, num_tokens, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, num_tokens, intermediate_size},
                                        input.getTensorType());
  nntrainer::Tensor gate_out =
    gate_out_workspace.getSharedDataTensor(intermediate_dim, 0, true);
  nntrainer::Tensor acti_out =
    acti_out_workspace.getSharedDataTensor(intermediate_dim, 0, true);
  nntrainer::Tensor up_out =
    up_out_workspace.getSharedDataTensor(intermediate_dim, 0, true);
  nntrainer::Tensor token_input;

  const unsigned token_idx = token_assignments[0].first;

  const auto gather_start = profiler.start();
  if (num_tokens > 1) {
    /** if prefill, copy data to make a batch */
    token_input =
      token_input_workspace.getSharedDataTensor(token_input_dim, 0, true);
    {
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(num_tokens), [&](size_t i) {
        const unsigned token_idx = token_assignments[i].first;
        // Use tensor's optimized copy operation
        nntrainer::Tensor src_view = input.getSharedDataTensor(
          {1, 1, 1, hidden_size}, token_idx * hidden_size, true);
        nntrainer::Tensor dst_view = token_input.getSharedDataTensor(
          {1, 1, 1, hidden_size}, i * hidden_size, true);
        dst_view.copyData(src_view);
      });
    }
  } else {
    /** if token generation, do not copy but get the shared tensor */
    // Create shared tensor for input token (no memory copy)
    size_t token_offset = token_idx * hidden_size;
    token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);
  }
  profiler.record(MoEProfiler::Phase::DISPATCH, gather_start);

  // Gate projection using optimized dot operation
  const auto gate_up_start = profiler.start();
  token_input.dot(gate_proj, gate_out);

  // Up projection using optimized dot operation
  token_input.dot(up_proj, up_out);
  profiler.record(MoEProfiler::Phase::GATE_UP, gate_up_start);

  const auto activation_start = profiler.start();
  if (num_tokens == 1) {
    nntrainer::swiglu(acti_out.width(), acti_out.getData<float>(),
                      gate_out.getData<float>(), up_out.getData<float>());
  } else {
    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, static_cast<size_t>(num_tokens), [&](size_t i) {
      const unsigned offset = acti_out.getIndex(0, 0, i, 0);
      nntrainer::swiglu(acti_out.width(), acti_out.getData<float>() + offset,
                        gate_out.getData<float>() + offset,
                        up_out.getData<float>() + offset);
    });
  }
  profiler.record(MoEProfiler::Phase::ACTIVATION, activation_start);

  const auto down_start = profiler.start();
  acti_out.dot(down_proj, output);
  profiler.record(MoEProfiler::Phase::DOWN, down_start);
}

void CachedSlimMoELayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  const auto total_start = profiler.start();
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);
  const unsigned int profile_token_count = input_.batch() * (to - from);

  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);

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
    auto output = output_.getSharedDataTensor(
      output_step_dim, b * output_step_dim.getFeatureLen(), true);
    auto router_logits =
      router_logits_.getSharedDataTensor(router_logits_step_dim, 0, true);

    const unsigned batch_size = input.batch();
    const unsigned seq_len = input.height();
    const unsigned hidden_size = input.width();
    const unsigned total_tokens = batch_size * seq_len;

    // reshape input: [B,1,S,H] -> [B*S,1,1,H]
    input.reshape({total_tokens, 1, 1, hidden_size});

    // reshape output: [B,1,S,H] -> [B*S,1,1,H]
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();

    // routing
    const auto router_start = profiler.start();
    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    input.dot(gate_weights, router_logits);

    // Softmax preserves the logit ordering. Select cache candidates once from
    // the raw logits, then normalize only the experts used for routing.
    const unsigned int candidate_count = std::min(num_experts, topk + 5U);
    auto candidate_result = router_logits.topK(candidate_count);
    auto candidate_values = std::get<0>(candidate_result);
    auto candidate_indices = std::get<1>(candidate_result);
    float *candidate_values_data = candidate_values.getData<float>();
    const uint32_t *candidate_indices_data =
      candidate_indices.getData<uint32_t>();

    for (unsigned int i = 0; i < total_tokens; ++i) {
      const size_t candidate_offset = static_cast<size_t>(i) * candidate_count;
      const float max_logit = candidate_values_data[candidate_offset];
      float weight_sum = 0.0f;
      for (unsigned int k = 0; k < topk; ++k) {
        const float weight =
          std::exp(candidate_values_data[candidate_offset + k] - max_logit);
        candidate_values_data[candidate_offset + k] = weight;
        weight_sum += weight;
      }

      const float inverse_weight_sum = 1.0f / weight_sum;
      for (unsigned int k = 0; k < topk; ++k)
        candidate_values_data[candidate_offset + k] *= inverse_weight_sum;
    }
    profiler.record(MoEProfiler::Phase::ROUTER, router_start);

    const auto dispatch_start = profiler.start();
    std::vector<int> extra_top_k;
    extra_top_k.reserve(
      std::min(static_cast<size_t>(num_experts),
               static_cast<size_t>(total_tokens) * candidate_count));
    std::vector<uint8_t> candidate_seen(num_experts, 0);
    for (int i = static_cast<int>(total_tokens) - 1; i >= 0; --i) {
      const size_t candidate_offset = static_cast<size_t>(i) * candidate_count;
      for (unsigned int k = 0; k < candidate_count; ++k) {
        const unsigned int expert_idx =
          candidate_indices_data[candidate_offset + k];
        if (candidate_seen[expert_idx])
          continue;

        candidate_seen[expert_idx] = 1;
        extra_top_k.push_back(expert_idx);
      }
    }

    std::vector<size_t> expert_offsets(num_experts + 1, 0);
    for (unsigned int i = 0; i < total_tokens; ++i) {
      const size_t candidate_offset = static_cast<size_t>(i) * candidate_count;
      for (unsigned int k = 0; k < topk; ++k) {
        const unsigned int expert_idx =
          candidate_indices_data[candidate_offset + k];
        ++expert_offsets[expert_idx + 1];
      }
    }
    for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx)
      expert_offsets[expert_idx + 1] += expert_offsets[expert_idx];

    std::vector<std::pair<unsigned, float>> expert_assignments(
      expert_offsets.back());
    std::vector<size_t> write_offsets(expert_offsets);
    for (unsigned int i = 0; i < total_tokens; ++i) {
      const size_t candidate_offset = static_cast<size_t>(i) * candidate_count;
      for (unsigned int k = 0; k < topk; ++k) {
        const unsigned int expert_idx =
          candidate_indices_data[candidate_offset + k];
        const float weight = candidate_values_data[candidate_offset + k];
        expert_assignments[write_offsets[expert_idx]++] = {i, weight};
      }
    }

    std::vector<int> target_idx_vector;
    target_idx_vector.reserve(num_experts);
    unsigned int max_assignment_count = 0;

    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      const size_t assignment_count =
        expert_offsets[expert_idx + 1] - expert_offsets[expert_idx];
      if (assignment_count == 0)
        continue;

      target_idx_vector.push_back(expert_idx);
      max_assignment_count = std::max(
        max_assignment_count, static_cast<unsigned int>(assignment_count));
    }
    nntrainer::Tensor expert_output_workspace(
      max_assignment_count, 1, 1, hidden_size, output.getTensorType());
    const unsigned int intermediate_size =
      context.getWeight(expert_gate_proj_indices[target_idx_vector.front()])
        .width();
    nntrainer::Tensor token_input_workspace;
    if (max_assignment_count > 1) {
      token_input_workspace = nntrainer::Tensor(
        1, 1, max_assignment_count, hidden_size, input.getTensorType());
    }
    nntrainer::Tensor gate_out_workspace(
      1, 1, max_assignment_count, intermediate_size, input.getTensorType());
    nntrainer::Tensor acti_out_workspace(
      1, 1, max_assignment_count, intermediate_size, input.getTensorType());
    nntrainer::Tensor up_out_workspace(
      1, 1, max_assignment_count, intermediate_size, input.getTensorType());
    profiler.record(MoEProfiler::Phase::DISPATCH, dispatch_start);

    nntrainer::TensorDim token_step_dim({1, 1, 1, hidden_size},
                                        output.getTensorType());
    uint64_t cache_miss_count = 0;
    const auto mmap_activate_start = profiler.start();
    for (int expert_idx : target_idx_vector) {
      const bool cache_miss = need_load[expert_idx];
      candidate_seen[expert_idx] = cache_miss;
      if (!cache_miss)
        continue;

      context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
      context.getWeight(expert_up_proj_indices[expert_idx]).activate();
      context.getWeight(expert_down_proj_indices[expert_idx]).activate();

      std::lock_guard<std::mutex> lock(cache_mutex);
      loaded_expert_deque.push_back(expert_idx);
      cache_positions[expert_idx] = --loaded_expert_deque.end();
      need_load[expert_idx] = 0;
      ++cache_miss_count;
    }
    profiler.record(MoEProfiler::Phase::MMAP, mmap_activate_start);
    profiler.recordCache(target_idx_vector.size() - cache_miss_count,
                         cache_miss_count);

    // Serial outer loop: the expert GEMV/GEMM parallelizes internally via
    // ThreadManager (dot() calls parallel_for), and nesting parallel_for
    // deadlocks because ThreadManager::parallelize() uses a non-recursive
    // execution_mutex_.
    // Resident experts run in pass 0 so their compute overlaps asynchronous
    // page-in requested for all cache misses above.
    for (uint8_t miss_pass = 0; miss_pass < 2; ++miss_pass) {
      for (int expert_idx : target_idx_vector) {
        if (candidate_seen[expert_idx] != miss_pass)
          continue;

        const size_t assignment_offset = expert_offsets[expert_idx];
        const unsigned int assignment_count = static_cast<unsigned int>(
          expert_offsets[expert_idx + 1] - assignment_offset);
        const auto *assignments = expert_assignments.data() + assignment_offset;
        nntrainer::Tensor expert_output =
          expert_output_workspace.getSharedDataTensor(
            {assignment_count, 1, 1, hidden_size}, 0, true);
        const auto expert_start = profiler.start();
        compute_expert_forward(
          input, expert_output, assignments, assignment_count,
          context.getWeight(expert_gate_proj_indices[expert_idx]),
          context.getWeight(expert_up_proj_indices[expert_idx]),
          context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size,
          token_input_workspace, gate_out_workspace, acti_out_workspace,
          up_out_workspace);
        profiler.record(MoEProfiler::Phase::EXPERT, expert_start);

        const auto reduce_start = profiler.start();
        for (unsigned int i = 0; i < assignment_count; ++i) {
          nntrainer::Tensor token_output = output.getSharedDataTensor(
            token_step_dim, assignments[i].first * hidden_size, true);
          nntrainer::Tensor expert_token_output =
            expert_output.getSharedDataTensor(token_step_dim, i * hidden_size,
                                              true);
          token_output.add_i(expert_token_output, assignments[i].second);
        }
        profiler.record(MoEProfiler::Phase::REDUCE, reduce_start);
      }
    }

    for (auto candidate = extra_top_k.rbegin(); candidate != extra_top_k.rend();
         ++candidate) {
      if (need_load[*candidate])
        continue;

      loaded_expert_deque.erase(cache_positions[*candidate]);
      loaded_expert_deque.push_back(*candidate);
      cache_positions[*candidate] = --loaded_expert_deque.end();
    }

    // Evict experts
    /// @todo apply multi thread loop
    uint64_t eviction_count = 0;
    while (loaded_expert_deque.size() > cache_capacity) {
      int target_idx;
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        target_idx = loaded_expert_deque.front();
        loaded_expert_deque.pop_front();
        need_load[target_idx] = 1;
      }

      const auto mmap_deactivate_start = profiler.start();
      context.getWeight(expert_gate_proj_indices[target_idx]).deactivate();
      context.getWeight(expert_up_proj_indices[target_idx]).deactivate();
      context.getWeight(expert_down_proj_indices[target_idx]).deactivate();
      profiler.record(MoEProfiler::Phase::MMAP, mmap_deactivate_start);
      ++eviction_count;
    }
    profiler.recordCache(0, 0, eviction_count);

    // reshape output: [B*S,1,1,H] -> [B,1,S,H]
    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
  profiler.record(MoEProfiler::Phase::TOTAL, total_start);
  profiler.finish(profile_token_count);
}

void CachedSlimMoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void CachedSlimMoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // MoE layer does not support derivative calculation
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void CachedSlimMoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  // MoE layer does not support gradient calculation
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void CachedSlimMoELayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this); // Save MoE specific properties
}

void CachedSlimMoELayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim output_dim =
    context.getOutput(SINGLE_INOUT_IDX).getDim();

  input_dim.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

} // namespace causallm
