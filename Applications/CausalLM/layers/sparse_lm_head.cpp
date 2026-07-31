// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   sparse_lm_head.cpp
 * @date   31 July 2026
 * @brief  Predictor-guided sparse Q4_0 LM head.
 */

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>

#include <cpu_backend.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <sparse_lm_head.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace causallm {

namespace {

constexpr size_t SINGLE_INOUT_IDX = 0;
constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

} // namespace

SparseLmHeadLayer::SparseLmHeadLayer() :
  LayerImpl(),
  sparse_lmhead_props(nntrainer::props::Unit(), props::PredictorUnit(),
                      props::PredictorThreshold(),
                      props::PredictorTopKFloor()) {}

SparseLmHeadLayer::~SparseLmHeadLayer() {
  if (!sparsity_log || log_calls == 0)
    return;

  const double active_fraction =
    log_examined == 0
      ? 0.0
      : static_cast<double>(log_active) / static_cast<double>(log_examined);
  const double miss_rate =
    static_cast<double>(log_argmax_miss) / static_cast<double>(log_calls);
  std::cerr << "[sparse_lm_head] calls=" << log_calls
            << " vocab_active_fraction=" << active_fraction
            << " dense_argmax_miss_rate=" << miss_rate << std::endl;
}

void SparseLmHeadLayer::finalize(nntrainer::InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  auto &disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props);
  const auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;

  vocab_size = std::get<nntrainer::props::Unit>(sparse_lmhead_props).get();
  predictor_unit = std::get<props::PredictorUnit>(sparse_lmhead_props).get();
  predictor_threshold =
    std::get<props::PredictorThreshold>(sparse_lmhead_props).get();
  predictor_topk_floor =
    std::get<props::PredictorTopKFloor>(sparse_lmhead_props).get();
  if (!std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "sparse lm head layer takes only one input";
  NNTR_THROW_IF(context.getActivationDataType() !=
                  nntrainer::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "sparse lm head requires FP32 activations";
  q4_weights =
    context.getWeightDataType() == nntrainer::TensorDim::DataType::Q4_0;
  NNTR_THROW_IF(!q4_weights && context.getWeightDataType() !=
                                 nntrainer::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "sparse lm head requires FP32 or Q4_0 weights";
  NNTR_THROW_IF(disable_bias.empty() || !disable_bias.get(),
                std::invalid_argument)
    << "sparse lm head does not support bias";

  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  const auto &input_dim = context.getInputDimensions()[0];
  hidden_size = is_nchw ? input_dim.width() : input_dim.channel();
  NNTR_THROW_IF(
    predictor_unit == 0 ||
      (q4_weights && (hidden_size % 32 != 0 || vocab_size % 32 != 0 ||
                      predictor_unit % 32 != 0)),
    std::invalid_argument)
    << "sparse lm head has invalid predictor or Q4_0 dimensions";

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  std::vector<ml::train::TensorDim> output_dims(1, input_dim);
  if (is_nchw)
    output_dims[0].width(vocab_size);
  else
    output_dims[0].channel(vocab_size);
  output_dims[0].height(1);
  output_dims[0].setTensorType(
    {context.getFormat(), nntrainer::TensorDim::DataType::FP32});
  context.setOutputDimensions(output_dims);

  const auto weight_type = ml::train::TensorDim::TensorType(
    context.getFormat(), context.getWeightDataType());
  auto make_dim = [&](unsigned int input, unsigned int output) {
    return ml::train::TensorDim(1, is_nchw ? 1 : output, is_nchw ? input : 1,
                                is_nchw ? output : input, weight_type,
                                is_nchw ? 0b0011 : 0b0101);
  };

  head_idx = context.requestWeight(
    make_dim(hidden_size, vocab_size), weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);
  predictor_w1_idx = context.requestWeight(
    make_dim(hidden_size, predictor_unit), weight_initializer,
    weight_regularizer, weight_regularizer_constant, weight_decay,
    "profiler_w1", true);
  predictor_w2_idx = context.requestWeight(
    make_dim(predictor_unit, vocab_size), weight_initializer,
    weight_regularizer, weight_regularizer_constant, weight_decay,
    "profiler_w2", true);

  const auto activation_type = ml::train::TensorDim::TensorType(
    context.getFormat(), context.getActivationDataType());
  predictor_mid_idx = context.requestTensor(
    nntrainer::TensorDim({1, 1, 1, predictor_unit}, activation_type),
    "predictor_mid");
  predictor_score_idx = context.requestTensor(
    nntrainer::TensorDim({1, 1, 1, vocab_size}, activation_type),
    "predictor_score");

  active_indices.reserve(vocab_size / 4 + 1);
  active_mask.reserve(vocab_size);
  score_scratch.reserve(vocab_size);
}

void SparseLmHeadLayer::resolveRuntimeConfig() {
  if (runtime_resolved)
    return;

  if (const char *env = std::getenv("NNTR_SPARSE_LMHEAD"))
    predictor_active = std::atoi(env) != 0;
  if (const char *env = std::getenv("NNTR_LMHEAD_THRESHOLD"))
    predictor_threshold = static_cast<float>(std::atof(env));
  if (const char *env = std::getenv("NNTR_LMHEAD_TOPK_FLOOR"))
    predictor_topk_floor =
      static_cast<unsigned int>(std::strtoul(env, nullptr, 10));
  sparsity_log = std::getenv("NNTR_LMHEAD_SPARSITY_LOG") != nullptr;
  runtime_resolved = true;
}

void SparseLmHeadLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, sparse_lmhead_props);
  LayerImpl::setProperty(remain_props);
}

bool SparseLmHeadLayer::computeSparseLogits(nntrainer::RunLayerContext &context,
                                            const nntrainer::Tensor &input_step,
                                            nntrainer::Tensor &hidden_step) {
  nntrainer::Tensor &mid = context.getTensor(predictor_mid_idx);
  input_step.dot(context.getWeight(predictor_w1_idx), mid, false, false);

  nntrainer::Tensor &score = context.getTensor(predictor_score_idx);
  mid.dot(context.getWeight(predictor_w2_idx), score, false, false);
  const float *score_data = score.getData<float>();

  active_indices.clear();
  for (unsigned int token = 0; token < vocab_size; ++token) {
    if (score_data[token] > predictor_threshold)
      active_indices.push_back(token);
  }

  if (predictor_topk_floor > 0 &&
      active_indices.size() < predictor_topk_floor &&
      predictor_topk_floor < vocab_size) {
    score_scratch.assign(score_data, score_data + vocab_size);
    std::nth_element(score_scratch.begin(),
                     score_scratch.begin() + predictor_topk_floor - 1,
                     score_scratch.end(), std::greater<float>());
    const float kth = score_scratch[predictor_topk_floor - 1];
    active_indices.clear();
    for (unsigned int token = 0; token < vocab_size; ++token) {
      if (score_data[token] >= kth)
        active_indices.push_back(token);
    }
  }

  if (active_indices.empty()) {
    active_indices.push_back(static_cast<unsigned int>(
      std::max_element(score_data, score_data + vocab_size) - score_data));
  }

  active_mask.assign(vocab_size, 0);
  for (const unsigned int token : active_indices)
    active_mask[token] = 1;

  const nntrainer::Tensor &head = context.getWeight(head_idx);
  float *output = hidden_step.getData<float>();
  const bool used_sparse = nntrainer::gemv_q4_0_masked(
    vocab_size, hidden_size, input_step.getData<float>(),
    head.getData<uint8_t>(), output, active_mask.data());
  if (!used_sparse)
    computeDenseLogits(context, input_step, hidden_step);

  if (sparsity_log) {
    const float *dense_output = output;
    nntrainer::Tensor dense_logits(hidden_step.getDim());
    if (used_sparse) {
      computeDenseLogits(context, input_step, dense_logits);
      dense_output = dense_logits.getData<float>();
    }
    const unsigned int dense_argmax = static_cast<unsigned int>(
      std::max_element(dense_output, dense_output + vocab_size) - dense_output);
    ++log_calls;
    log_active += static_cast<long long>(active_indices.size());
    log_examined += static_cast<long long>(vocab_size);
    log_argmax_miss += active_mask[dense_argmax] == 0 ? 1 : 0;
  }

  if (used_sparse) {
    for (unsigned int token = 0; token < vocab_size; ++token) {
      if (active_mask[token] == 0)
        output[token] = NEG_INF;
    }
  }
  return used_sparse;
}

void SparseLmHeadLayer::computeDenseLogits(nntrainer::RunLayerContext &context,
                                           const nntrainer::Tensor &input_step,
                                           nntrainer::Tensor &hidden_step) {
  input_step.dot(context.getWeight(head_idx), hidden_step, false, false);
}

void SparseLmHeadLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  throw nntrainer::exception::not_supported(
    "Forwarding for SparseLmHead layer is not supported");
}

void SparseLmHeadLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  if (skip_prefill && from == 0)
    return;

  resolveRuntimeConfig();
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::TensorDim input_step_dim = input.getDim();
  nntrainer::TensorDim output_step_dim = output.getDim();
  input_step_dim.batch(1);
  input_step_dim.height(1);
  output_step_dim.batch(1);

  const bool use_predictor = q4_weights && predictor_active && to - from == 1;
  for (unsigned int batch = 0; batch < input.batch(); ++batch) {
    nntrainer::Tensor input_step = input.getSharedDataTensor(
      input_step_dim,
      batch * input.getDim().getFeatureLen() + (to - from - 1) * input.width(),
      true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output.getDim().getFeatureLen(), true);

    if (use_predictor)
      computeSparseLogits(context, input_step, output_step);
    else
      computeDenseLogits(context, input_step, output_step);
  }
}

void SparseLmHeadLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for SparseLmHead layer is not supported");
}

void SparseLmHeadLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for SparseLmHead layer is not supported");
}

void SparseLmHeadLayer::exportTo(nntrainer::Exporter &exporter,
                                 const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(sparse_lmhead_props, method, this);
}

void SparseLmHeadLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  input_dim.height(input_dimensions[0].height());
  context.updateInput(SINGLE_INOUT_IDX, input_dim);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_sparse_lm_head() { return new SparseLmHeadLayer(); }
void destroy_sparse_lm_head(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_sparse_lm_head,
                                                   destroy_sparse_lm_head};
}

#endif

} // namespace causallm
