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
#include <cmath>
#include <cstdlib>
#include <exception>
#include <node_exporter.h>
#include <qwen_moe_layer_fsu.h>
#include <stdexcept>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {

struct ExpertWeights {
  unsigned int index;
  nntrainer::Tensor *gate;
  nntrainer::Tensor *up;
  nntrainer::Tensor *down;
};

void deactivateExpertWeightsNoThrow(const ExpertWeights &weights) noexcept {
  try {
    weights.down->deactivate();
  } catch (...) {
  }

  try {
    weights.up->deactivate();
  } catch (...) {
  }

  try {
    weights.gate->deactivate();
  } catch (...) {
  }
}

void activateExpertWeights(const ExpertWeights &weights) {
  unsigned int activated = 0;
  try {
    weights.gate->activate();
    activated = 1;
    weights.up->activate();
    activated = 2;
    weights.down->activate();
  } catch (...) {
    if (activated >= 2) {
      try {
        weights.up->deactivate();
      } catch (...) {
      }
    }
    if (activated >= 1) {
      try {
        weights.gate->deactivate();
      } catch (...) {
      }
    }
    throw;
  }
}

void deactivateExpertWeights(const ExpertWeights &weights) {
  std::exception_ptr first_error;
  auto deactivate = [&first_error](nntrainer::Tensor *weight) {
    try {
      weight->deactivate();
    } catch (...) {
      if (!first_error) {
        first_error = std::current_exception();
      }
    }
  };

  deactivate(weights.down);
  deactivate(weights.up);
  deactivate(weights.gate);

  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

void deactivateExpertBatch(const std::vector<ExpertWeights> &experts) {
  std::exception_ptr first_error;
  for (auto iter = experts.rbegin(); iter != experts.rend(); ++iter) {
    try {
      deactivateExpertWeights(*iter);
    } catch (...) {
      if (!first_error) {
        first_error = std::current_exception();
      }
    }
  }

  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

bool weightPrefetchEnabled() {
  static const bool enabled = []() {
    const char *value = std::getenv("NNTR_WEIGHT_PREFETCH");
    return value == nullptr || value[0] != '0' || value[1] != '\0';
  }();

  return enabled;
}

template <typename Compute>
void processExpertWeights(const std::vector<ExpertWeights> &experts,
                          Compute compute) {
  if (!weightPrefetchEnabled()) {
    for (const auto &expert : experts) {
      activateExpertWeights(expert);
      try {
        compute(expert);
      } catch (...) {
        deactivateExpertWeightsNoThrow(expert);
        throw;
      }
      deactivateExpertWeights(expert);
    }
    return;
  }

  size_t activated_count = 0;
  try {
    // mmap() all active experts first. On Android and Linux, activate() also
    // submits MADV_WILLNEED so later expert reads can overlap earlier compute.
    for (const auto &expert : experts) {
      activateExpertWeights(expert);
      ++activated_count;
    }

    // Keep the outer loop serial because dot() already uses ThreadManager.
    for (const auto &expert : experts) {
      compute(expert);
    }
  } catch (...) {
    while (activated_count > 0) {
      deactivateExpertWeightsNoThrow(experts[--activated_count]);
    }
    throw;
  }

  deactivateExpertBatch(experts);
}

} // namespace

SlimMoELayer::SlimMoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {}

void SlimMoELayer::finalize(nntrainer::InitLayerContext &context) {

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

  // 6. Request intermediate tensors
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  // Router logits :  [batch * seq, num_experts]
  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // Expert mask: [num_experts, batch*seq]
  expert_mask_idx =
    context.requestTensor({num_experts, 1, topk, total_tokens}, "expert_mask",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void SlimMoELayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits = context.getTensor(router_logits_idx);
  nntrainer::Tensor &expert_mask = context.getTensor(expert_mask_idx);

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
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  input.dot(gate_weights, router_logits);
  router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
  {
    auto &tm = nntrainer::ThreadManager::Global();
    size_t total_iters =
      static_cast<size_t>(total_tokens) * static_cast<size_t>(topk);
    tm.parallel_for(0, static_cast<size_t>(total_iters), [&](size_t idx) {
      int k = idx % topk;
      int i = idx / topk;
      expert_mask.setValue(indices_data[idx], 0, k, i, 1.0f);
    });
  }

  // Pre-compute expert token assignments for better cache locality
  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      unsigned expert_idx = indices_data[i * topk + k];
      float weight = topk_values.getValue<float>(i, 0, 0, k);
      expert_assignments[expert_idx].emplace_back(i, weight);
    }
  }

  std::vector<ExpertWeights> active_experts;
  active_experts.reserve(num_experts);
  for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    if (expert_assignments[expert_idx].empty()) {
      continue;
    }

    active_experts.push_back(
      {expert_idx, &context.getWeight(expert_gate_proj_indices[expert_idx]),
       &context.getWeight(expert_up_proj_indices[expert_idx]),
       &context.getWeight(expert_down_proj_indices[expert_idx])});
  }

  processExpertWeights(active_experts, [&](const ExpertWeights &expert) {
    compute_expert_forward(input, output, expert_assignments[expert.index],
                           *expert.gate, *expert.up, *expert.down, hidden_size);
  });

  // reshape output: [B*S,1,1,H] -> [B,1,S,H]
  output.reshape({batch_size, 1, seq_len, hidden_size});
}

inline void SlimMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Create tensor dimensions for single token processing
  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  // Create a temporary output tensor for this expert to avoid critical section
  nntrainer::Tensor expert_output(output.batch(), output.channel(),
                                  output.height(), output.width(),
                                  output.getTensorType());
  expert_output.setZero();

  // Process each token individually to avoid memory copies
  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;

    // Create shared tensor for input token (no memory copy)
    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);

    // Create intermediate tensors for this token
    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);

    // Gate projection using optimized dot operation
    token_input.dot(gate_proj, gate_out);

    // Up projection using optimized dot operation
    token_input.dot(up_proj, up_out);

    nntrainer::swiglu(acti_out.width(), acti_out.getData<float>(),
                      gate_out.getData<float>(), up_out.getData<float>());

    // Down projection using optimized dot operation
    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);

    // Apply weight and accumulate to expert's temporary output
    token_expert_output.multiply_i(weight);
    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      expert_output.getSharedDataTensor(token_output_dim, output_offset, true);

    token_output.add_i(token_expert_output);
  }

  // Add expert's result to final output (no critical section in sequential
  // mode)
  output.add_i(expert_output);
}

inline void SlimMoELayer::compute_expert_forward_no_critical(
  const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Create tensor dimensions for single token processing
  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  // Process each token individually to avoid memory copies
  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;

    // Create shared tensor for input token (no memory copy)
    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);

    // Create intermediate tensors for this token
    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);

    // Gate projection using optimized dot operation
    token_input.dot(gate_proj, gate_out);

    // Up projection using optimized dot operation
    token_input.dot(up_proj, up_out);

    nntrainer::swiglu(acti_out.width(), acti_out.getData<float>(),
                      gate_out.getData<float>(), up_out.getData<float>());

    // Down projection using optimized dot operation
    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);

    // Apply weight and accumulate to expert's output (no critical section
    // needed)
    token_expert_output.multiply_i(weight);
    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      expert_output.getSharedDataTensor(token_output_dim, output_offset, true);

    token_output.add_i(token_expert_output);
  }
}

void SlimMoELayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

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
    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    input.dot(gate_weights, router_logits);
    router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
    auto topk_result = router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);

    // norm_topk_prob
    topk_values.divide_i(topk_values.sum(3));

    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    // Set expert mask
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = topk_values.getValue<float>(i, 0, 0, k);
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    // Parallel processing for multiple tokens with many active experts
    std::vector<nntrainer::Tensor> expert_outputs(num_experts);
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty()) {
        expert_outputs[expert_idx] = nntrainer::Tensor(
          total_tokens, 1, 1, hidden_size, output.getTensorType());
      }
    }

    std::vector<ExpertWeights> active_experts;
    active_experts.reserve(num_experts);
    for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      if (expert_assignments[expert_idx].empty()) {
        continue;
      }

      active_experts.push_back(
        {expert_idx, &context.getWeight(expert_gate_proj_indices[expert_idx]),
         &context.getWeight(expert_up_proj_indices[expert_idx]),
         &context.getWeight(expert_down_proj_indices[expert_idx])});
    }

    processExpertWeights(active_experts, [&](const ExpertWeights &expert) {
      compute_expert_forward_no_critical(
        input, expert_outputs[expert.index], expert_assignments[expert.index],
        *expert.gate, *expert.up, *expert.down, hidden_size);
    });

    // Combine expert outputs
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty()) {
        output.add_i(expert_outputs[expert_idx]);
      }
    }

    // reshape output: [B*S,1,1,H] -> [B,1,S,H]
    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
}

void SlimMoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void SlimMoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // MoE layer does not support derivative calculation
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void SlimMoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  // MoE layer does not support gradient calculation
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void SlimMoELayer::exportTo(nntrainer::Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this); // Save MoE specific properties
}

} // namespace causallm
