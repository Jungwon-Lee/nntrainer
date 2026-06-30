// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_prefetch_layer.cpp
 * @date   26 June 2026
 * @brief  SmallThinker MoE prefetch layer — overlaps expert page-in with
 *         attention by firing background activate() tasks at block entry.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#include <cstdlib>
#include <node_exporter.h>
#include <smallthinker_moe_layer_cached_slim.h>
#include <smallthinker_moe_prefetch_layer.h>
#include <stdexcept>

namespace causallm {

// Access the global registry and get the paired compute-node pointer.
// Declared in smallthinker_moe_layer_cached_slim.cpp translation unit —
// expose via a free function to avoid re-declaring global statics here.
SmallThinkerCachedSlimMoELayer *get_cached_slim_layer_for_prefetch(int id);

SmallThinkerMoEPrefetchLayer::SmallThinkerMoEPrefetchLayer() :
  LayerImpl(),
  num_experts_(0),
  topk_(0),
  router_apply_softmax_(true),
  moe_layer_id_(-1),
  prefetch_props_(props::NumExperts(), props::NumExpertsPerToken(),
                  props::MoERouterApplySoftmax(), props::MoELayerId()),
  router_logits_idx_(std::numeric_limits<unsigned>::max()) {}

void SmallThinkerMoEPrefetchLayer::finalize(
  nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "SmallThinkerMoEPrefetchLayer: expected 1 input (router input), got "
    << context.getNumInputs();

  num_experts_ = std::get<props::NumExperts>(prefetch_props_).get();
  topk_        = std::get<props::NumExpertsPerToken>(prefetch_props_).get();
  router_apply_softmax_ =
    std::get<props::MoERouterApplySoftmax>(prefetch_props_).get();
  moe_layer_id_ =
    static_cast<int>(std::get<props::MoELayerId>(prefetch_props_).get());

  const auto &in_dim = context.getInputDimensions()[0];
  // Pass-through: output shape == input shape
  context.setOutputDimensions({in_dim});

  const unsigned batch_size  = in_dim.batch();
  const unsigned seq_len     = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  // Internal tensor for router logits (FP32 regardless of weight dtype).
  nntrainer::TensorDim logits_dim(
    {total_tokens, 1, 1, num_experts_},
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  router_logits_idx_ =
    context.requestTensor(logits_dim, "prefetch_router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void SmallThinkerMoEPrefetchLayer::firePrefetch(
  nntrainer::Tensor &router_input, unsigned int total_tokens) const {

  auto *compute = get_cached_slim_layer_for_prefetch(moe_layer_id_);
  if (!compute)
    return;

  nntrainer::Tensor *gate = compute->getGateTensor();
  if (!gate)
    return; // cold start: tensor pointers not yet populated

  // Compute router logits using the compute node's gate weight (always FP32
  // resident — no virtual tensor, no mmap cost).
  nntrainer::TensorDim logits_dim(
    {total_tokens, 1, 1, num_experts_},
    nntrainer::TensorDim::TensorType(
      router_input.getFormat(),
      nntrainer::TensorDim::DataType::FP32));
  nntrainer::Tensor logits(logits_dim);
  router_input.dot(*gate, logits);

  // Collect topk+5 predicted expert indices and fire background activate()
  // tasks for any that are not yet resident.
  const unsigned int extra = std::min(topk_ + 5u, num_experts_);
  auto topk_result         = logits.topK(extra);
  const auto &indices      = std::get<1>(topk_result);
  const uint32_t *idata    = indices.getData<uint32_t>();

  std::vector<unsigned int> predicted;
  predicted.reserve(total_tokens * extra);
  for (unsigned int t = 0; t < total_tokens; ++t)
    for (unsigned int k = 0; k < extra; ++k)
      predicted.push_back(idata[t * extra + k]);

  // Deduplicate (prefetchExperts handles this under the lock, but avoiding
  // duplicate entries in the vector reduces lock contention).
  std::sort(predicted.begin(), predicted.end());
  predicted.erase(std::unique(predicted.begin(), predicted.end()),
                  predicted.end());

  compute->prefetchExperts(predicted);
}

void SmallThinkerMoEPrefetchLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {

  nntrainer::Tensor &input  = context.getInput(0);
  nntrainer::Tensor &output = context.getOutput(0);

  const unsigned batch_size  = input.batch();
  const unsigned seq_len     = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  // Reshape for dot product, fire prefetch, then restore.
  input.reshape({total_tokens, 1, 1, hidden_size});
  firePrefetch(input, total_tokens);
  input.reshape({batch_size, 1, seq_len, hidden_size});

  // Pass through: output carries the same data as input.
  output.copy(input);
}

void SmallThinkerMoEPrefetchLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &input_  = context.getInput(0);
  nntrainer::Tensor &output_ = context.getOutput(0);

  const unsigned batch_size = input_.batch();
  const unsigned seq_len    = to - from;
  const unsigned hidden_size = input_.width();
  const unsigned total_tokens = batch_size * seq_len;

  nntrainer::TensorDim step_dim = input_.getDim();
  step_dim.batch(1);
  step_dim.height(seq_len);

  for (unsigned int b = 0; b < batch_size; ++b) {
    auto ri = input_.getSharedDataTensor(
      step_dim, b * step_dim.getFeatureLen(), true);
    ri.reshape({seq_len, 1, 1, hidden_size});
    firePrefetch(ri, seq_len);
  }

  // Pass through
  output_.copy(input_);
}

void SmallThinkerMoEPrefetchLayer::calcDerivative(
  nntrainer::RunLayerContext &) {
  throw std::runtime_error(
    "SmallThinkerMoEPrefetchLayer: derivative not supported");
}

void SmallThinkerMoEPrefetchLayer::calcGradient(
  nntrainer::RunLayerContext &) {
  throw std::runtime_error(
    "SmallThinkerMoEPrefetchLayer: gradient not supported");
}

void SmallThinkerMoEPrefetchLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain = loadProperties(values, prefetch_props_);
  nntrainer::LayerImpl::setProperty(remain);
}

void SmallThinkerMoEPrefetchLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(prefetch_props_, method, this);
}

} // namespace causallm
