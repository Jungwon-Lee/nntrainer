// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_5_moe_layer.cpp
 * @date   19 August 2026
 * @brief  Routed and shared-expert MoE layer for Qwen3.5/Qwen3.6.
 */

#include "qwen3_5_moe_layer.h"

#include <algorithm>
#include <cmath>
#include <cpu_backend.h>
#include <node_exporter.h>
#include <q4_0_utils.h>

namespace causallm {

namespace {

constexpr unsigned int SINGLE_INOUT_IDX = 0;

float sigmoid(float value) {
  if (value >= 0.0f) {
    const float exp_neg = std::exp(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = std::exp(value);
  return exp_pos / (1.0f + exp_pos);
}

} // namespace

Qwen3_5MoeLayer::Qwen3_5MoeLayer() :
  LayerImpl(),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(),
            qwen3_5_props::SharedExpertIntermediateSize()) {}

void Qwen3_5MoeLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Qwen3_5MoeLayer requires one input";

  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  const unsigned int hidden_size = in_dim.width();
  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  intermediate_size = std::get<nntrainer::props::Unit>(moe_props).get();
  shared_intermediate_size =
    std::get<qwen3_5_props::SharedExpertIntermediateSize>(moe_props).get();

  NNTR_THROW_IF(topk > num_experts, std::invalid_argument)
    << "num_experts_per_token cannot exceed num_experts";

  context.setOutputDimensions({in_dim});

  const auto &regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  const auto &regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  const auto &initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  const auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  const auto layer_weight_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getWeightDataType());
  const auto fp32_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), nntrainer::TensorDim::DataType::FP32);

  nntrainer::TensorDim gate_dim(1, is_nchw ? 1 : num_experts,
                                is_nchw ? hidden_size : 1,
                                is_nchw ? num_experts : hidden_size,
                                layer_weight_type, is_nchw ? 0b0011 : 0b0101);
  gate_idx =
    context.requestWeight(gate_dim, initializer, regularizer,
                          regularizer_constant, weight_decay, "gate", true);

  nntrainer::TensorDim expert_gate_up_dim(
    1, is_nchw ? 1 : 2 * intermediate_size, is_nchw ? hidden_size : 1,
    is_nchw ? 2 * intermediate_size : hidden_size, layer_weight_type,
    is_nchw ? 0b0011 : 0b0101);
  nntrainer::TensorDim expert_down_dim(
    1, is_nchw ? 1 : hidden_size, is_nchw ? intermediate_size : 1,
    is_nchw ? hidden_size : intermediate_size, layer_weight_type,
    is_nchw ? 0b0011 : 0b0101);

  expert_gate_up_indices.reserve(num_experts);
  expert_down_indices.reserve(num_experts);
  for (unsigned int expert = 0; expert < num_experts; ++expert) {
    expert_gate_up_indices.push_back(context.requestWeight(
      expert_gate_up_dim, initializer, regularizer, regularizer_constant,
      weight_decay, "expert_gate_up_" + std::to_string(expert), false));
    expert_down_indices.push_back(context.requestWeight(
      expert_down_dim, initializer, regularizer, regularizer_constant,
      weight_decay, "expert_down_" + std::to_string(expert), false));
  }

  nntrainer::TensorDim shared_gate_up_dim(
    1, is_nchw ? 1 : 2 * shared_intermediate_size, is_nchw ? hidden_size : 1,
    is_nchw ? 2 * shared_intermediate_size : hidden_size, layer_weight_type,
    is_nchw ? 0b0011 : 0b0101);
  nntrainer::TensorDim shared_down_dim(
    1, is_nchw ? 1 : hidden_size, is_nchw ? shared_intermediate_size : 1,
    is_nchw ? hidden_size : shared_intermediate_size, layer_weight_type,
    is_nchw ? 0b0011 : 0b0101);
  nntrainer::TensorDim shared_gate_dim(
    1, is_nchw ? 1 : 1, is_nchw ? hidden_size : 1, is_nchw ? 1 : hidden_size,
    fp32_type, is_nchw ? 0b0011 : 0b0101);

  shared_gate_up_idx = context.requestWeight(
    shared_gate_up_dim, initializer, regularizer, regularizer_constant,
    weight_decay, "shared_gate_up", false);
  shared_down_idx = context.requestWeight(shared_down_dim, initializer,
                                          regularizer, regularizer_constant,
                                          weight_decay, "shared_down", false);
  shared_expert_gate_idx = context.requestWeight(
    shared_gate_dim, initializer, regularizer, regularizer_constant,
    weight_decay, "shared_expert_gate", false);

  const unsigned int total_tokens = in_dim.batch() * in_dim.height();
  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void Qwen3_5MoeLayer::buildAssignments(
  const nntrainer::Tensor &router_logits, unsigned int total_tokens,
  std::vector<std::vector<std::pair<unsigned int, float>>> &assignments) const {
  const float *logits = router_logits.getData<float>();
  std::vector<std::pair<float, unsigned int>> scores(num_experts);

  for (unsigned int token = 0; token < total_tokens; ++token) {
    const float *row = logits + static_cast<size_t>(token) * num_experts;
    const float row_max = *std::max_element(row, row + num_experts);
    float denominator = 0.0f;
    for (unsigned int expert = 0; expert < num_experts; ++expert) {
      const float probability = std::exp(row[expert] - row_max);
      scores[expert] = {probability, expert};
      denominator += probability;
    }
    for (auto &score : scores)
      score.first /= denominator;

    std::partial_sort(
      scores.begin(), scores.begin() + topk, scores.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });

    float topk_sum = 0.0f;
    for (unsigned int rank = 0; rank < topk; ++rank)
      topk_sum += scores[rank].first;
    const float inv_topk_sum = 1.0f / topk_sum;
    for (unsigned int rank = 0; rank < topk; ++rank)
      assignments[scores[rank].second].emplace_back(token, scores[rank].first *
                                                             inv_topk_sum);
  }
}

void Qwen3_5MoeLayer::runRoutedExperts(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  nntrainer::RunLayerContext &context,
  const std::vector<std::vector<std::pair<unsigned int, float>>> &assignments,
  unsigned int hidden_size) const {
  const auto tensor_type = input.getTensorType();
  const nntrainer::TensorDim single_token_dim({1, 1, 1, hidden_size},
                                              tensor_type);

  for (unsigned int expert = 0; expert < num_experts; ++expert) {
    const auto &expert_assignments = assignments[expert];
    if (expert_assignments.empty())
      continue;

    const unsigned int token_count = expert_assignments.size();
    nntrainer::Tensor expert_input(
      nntrainer::TensorDim({1, 1, token_count, hidden_size}, tensor_type));
    for (unsigned int i = 0; i < token_count; ++i) {
      auto source = input.getSharedDataTensor(
        single_token_dim,
        static_cast<size_t>(expert_assignments[i].first) * hidden_size, true);
      auto target = expert_input.getSharedDataTensor(
        single_token_dim, static_cast<size_t>(i) * hidden_size, true);
      target.copyData(source);
    }

    nntrainer::Tensor gate_up(nntrainer::TensorDim(
      {1, 1, token_count, 2 * intermediate_size}, tensor_type));
    nntrainer::Tensor activated(nntrainer::TensorDim(
      {1, 1, token_count, intermediate_size}, tensor_type));
    nntrainer::Tensor expert_output(
      nntrainer::TensorDim({1, 1, token_count, hidden_size}, tensor_type));

    expert_input.dot(context.getWeight(expert_gate_up_indices[expert]),
                     gate_up);
    for (unsigned int token = 0; token < token_count; ++token) {
      const size_t gate_offset =
        static_cast<size_t>(token) * 2 * intermediate_size;
      const size_t act_offset = static_cast<size_t>(token) * intermediate_size;
      nntrainer::swiglu(
        intermediate_size, activated.getData<float>() + act_offset,
        gate_up.getData<float>() + gate_offset,
        gate_up.getData<float>() + gate_offset + intermediate_size);
    }
    activated.dot(context.getWeight(expert_down_indices[expert]),
                  expert_output);

    for (unsigned int i = 0; i < token_count; ++i) {
      auto destination = output.getSharedDataTensor(
        single_token_dim,
        static_cast<size_t>(expert_assignments[i].first) * hidden_size, true);
      auto source = expert_output.getSharedDataTensor(
        single_token_dim, static_cast<size_t>(i) * hidden_size, true);
      source.multiply_i(expert_assignments[i].second);
      destination.add_i(source);
    }
  }
}

void Qwen3_5MoeLayer::runSharedExpert(const nntrainer::Tensor &input,
                                      nntrainer::Tensor &output,
                                      nntrainer::RunLayerContext &context,
                                      unsigned int total_tokens,
                                      unsigned int hidden_size) const {
  const auto tensor_type = input.getTensorType();
  nntrainer::Tensor gate_up(nntrainer::TensorDim(
    {total_tokens, 1, 1, 2 * shared_intermediate_size}, tensor_type));
  nntrainer::Tensor activated(nntrainer::TensorDim(
    {total_tokens, 1, 1, shared_intermediate_size}, tensor_type));
  nntrainer::Tensor shared_output(
    nntrainer::TensorDim({total_tokens, 1, 1, hidden_size}, tensor_type));
  nntrainer::Tensor shared_gate(
    nntrainer::TensorDim({total_tokens, 1, 1, 1}, tensor_type));

  input.dot(context.getWeight(shared_gate_up_idx), gate_up);
  for (unsigned int token = 0; token < total_tokens; ++token) {
    const size_t gate_offset =
      static_cast<size_t>(token) * 2 * shared_intermediate_size;
    const size_t act_offset =
      static_cast<size_t>(token) * shared_intermediate_size;
    nntrainer::swiglu(
      shared_intermediate_size, activated.getData<float>() + act_offset,
      gate_up.getData<float>() + gate_offset,
      gate_up.getData<float>() + gate_offset + shared_intermediate_size);
  }
  activated.dot(context.getWeight(shared_down_idx), shared_output);
  input.dot(context.getWeight(shared_expert_gate_idx), shared_gate);

  const nntrainer::TensorDim token_dim({1, 1, 1, hidden_size}, tensor_type);
  const float *gate_data = shared_gate.getData<float>();
  for (unsigned int token = 0; token < total_tokens; ++token) {
    auto shared_token = shared_output.getSharedDataTensor(
      token_dim, static_cast<size_t>(token) * hidden_size, true);
    auto output_token = output.getSharedDataTensor(
      token_dim, static_cast<size_t>(token) * hidden_size, true);
    shared_token.multiply_i(sigmoid(gate_data[token]));
    output_token.add_i(shared_token);
  }
}

void Qwen3_5MoeLayer::runStep(nntrainer::RunLayerContext &context,
                              unsigned int step_size) {
  auto &input_full = context.getInput(SINGLE_INOUT_IDX);
  auto &output_full = context.getOutput(SINGLE_INOUT_IDX);
  NNTR_THROW_IF(input_full.getDataType() !=
                  nntrainer::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "Qwen3_5MoeLayer currently requires FP32 activations";

  const unsigned int batch_size = input_full.batch();
  const unsigned int hidden_size = input_full.width();
  const unsigned int total_tokens = batch_size * step_size;
  const auto tensor_type = input_full.getTensorType();

  nntrainer::Tensor input(
    nntrainer::TensorDim({total_tokens, 1, 1, hidden_size}, tensor_type));
  nntrainer::Tensor output(
    nntrainer::TensorDim({total_tokens, 1, 1, hidden_size}, tensor_type));
  const nntrainer::TensorDim token_dim({1, 1, 1, hidden_size}, tensor_type);
  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    for (unsigned int token = 0; token < step_size; ++token) {
      const size_t source_offset =
        (static_cast<size_t>(batch) * input_full.height() + token) *
        hidden_size;
      const size_t target_offset =
        (static_cast<size_t>(batch) * step_size + token) * hidden_size;
      auto source =
        input_full.getSharedDataTensor(token_dim, source_offset, true);
      auto target = input.getSharedDataTensor(token_dim, target_offset, true);
      target.copyData(source);
    }
  }
  output.setZero();

  auto &router_storage = context.getTensor(router_logits_idx);
  nntrainer::TensorDim router_dim = router_storage.getDim();
  router_dim.batch(total_tokens);
  auto router_logits = router_storage.getSharedDataTensor(router_dim, 0, true);
  input.dot(context.getWeight(gate_idx), router_logits);

  std::vector<std::vector<std::pair<unsigned int, float>>> assignments(
    num_experts);
  buildAssignments(router_logits, total_tokens, assignments);
  runRoutedExperts(input, output, context, assignments, hidden_size);
  runSharedExpert(input, output, context, total_tokens, hidden_size);

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    for (unsigned int token = 0; token < step_size; ++token) {
      const size_t source_offset =
        (static_cast<size_t>(batch) * step_size + token) * hidden_size;
      const size_t target_offset =
        (static_cast<size_t>(batch) * output_full.height() + token) *
        hidden_size;
      auto source = output.getSharedDataTensor(token_dim, source_offset, true);
      auto target =
        output_full.getSharedDataTensor(token_dim, target_offset, true);
      target.copyData(source);
    }
  }
}

void Qwen3_5MoeLayer::forwarding(nntrainer::RunLayerContext &context,
                                 bool training) {
  NNTR_THROW_IF(training, std::invalid_argument)
    << "Qwen3_5MoeLayer supports inference only";
  runStep(context, context.getInput(SINGLE_INOUT_IDX).height());
}

void Qwen3_5MoeLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  NNTR_THROW_IF(training || to <= from, std::invalid_argument)
    << "Qwen3_5MoeLayer received an invalid inference range";
  runStep(context, to - from);
}

void Qwen3_5MoeLayer::save(std::ofstream &file,
                           nntrainer::RunLayerContext &run_context,
                           bool opt_var, ml::train::ExecutionMode mode,
                           bool trainable, ml::train::TensorDim::DataType dtype,
                           ml::train::ISA target_isa) const {
  if (opt_var) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      if (!run_context.isGradientFirstAccess(i) || !trainable ||
          !run_context.weightHasGradient(i))
        continue;
      for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i); ++j)
        run_context.getWeightOptVar(i, j).save(file);
    }
    return;
  }

  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (!run_context.isGradientFirstAccess(i))
      continue;
    auto &weight = run_context.getWeight(i);
    const auto effective_dtype = i == shared_expert_gate_idx
                                   ? ml::train::TensorDim::DataType::NONE
                                   : dtype;
    if (effective_dtype == ml::train::TensorDim::DataType::NONE ||
        weight.getDataType() == effective_dtype) {
      weight.save(file);
      continue;
    }

    NNTR_THROW_IF(effective_dtype != ml::train::TensorDim::DataType::Q4_0 ||
                    weight.getDataType() !=
                      ml::train::TensorDim::DataType::FP32,
                  std::runtime_error)
      << "Qwen3_5MoeLayer save supports FP32 to Q4_0 only";

    const auto dim = weight.getDim();
    const unsigned int k = dim.height();
    const unsigned int n = dim.width();
    NNTR_THROW_IF(k % 32 != 0 || n % 32 != 0, std::invalid_argument)
      << "Q4_0 MoE weights require height and width divisible by 32, got " << k
      << "x" << n;

    auto transposed = weight.transpose("0:2:1");
    nntrainer::Tensor quantized(dim.batch(), dim.channel(), k, n,
                                {nntrainer::Tformat::NCHW, effective_dtype});
    std::vector<char> canonical(quantized.size());
    nntrainer::quantize_q4_0(transposed.getData<float>(), canonical.data(), n,
                             k, nullptr);
    nntrainer::repack_q4_0(quantized.getData<uint8_t>(), canonical.data(),
                           quantized.size(), n, k, target_isa);
    quantized.save(file);
  }
}

void Qwen3_5MoeLayer::setProperty(const std::vector<std::string> &values) {
  auto remain = loadProperties(values, moe_props);
  LayerImpl::setProperty(remain);
}

void Qwen3_5MoeLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("Qwen3_5MoeLayer does not support backwarding");
}

void Qwen3_5MoeLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("Qwen3_5MoeLayer does not support gradients");
}

void Qwen3_5MoeLayer::exportTo(nntrainer::Exporter &exporter,
                               const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

} // namespace causallm
