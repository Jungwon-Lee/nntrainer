// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   sparse_lm_head.cpp
 * @date   30 June 2026
 * @brief  Sparse LM-head layer with activation predictor (paper §6.2).
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>

#include <cpu_backend.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <thread_manager.h>
#include <util_func.h>

#include <q4_0_row_kernels.h>
#include <sparse_lm_head.h>

namespace causallm {

using namespace q4_0_row;

static constexpr size_t SINGLE_INOUT_IDX = 0;
static const float NEG_INF = -std::numeric_limits<float>::infinity();

SparseLmHeadLayer::SparseLmHeadLayer() :
  LayerImpl(),
  sparse_lmhead_props(nntrainer::props::Unit(), props::PredictorUnit()) {}

SparseLmHeadLayer::~SparseLmHeadLayer() {
  if (sparsity_log && log_calls > 0) {
    const double active_frac =
      log_examined ? (double)log_active / (double)log_examined : 0.0;
    const double miss_rate = (double)log_argmax_miss / (double)log_calls;
    std::cerr << "[sparse_lm_head] calls=" << log_calls
              << " vocab_active_fraction=" << active_frac
              << " (avg " << (log_calls ? log_active / log_calls : 0)
              << "/" << vocab_size << " rows)"
              << " dense_argmax_miss_rate=" << miss_rate << std::endl;
  }
}

void SparseLmHeadLayer::finalize(nntrainer::InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  auto unit = std::get<nntrainer::props::Unit>(sparse_lmhead_props).get();
  predictor_unit =
    std::get<props::PredictorUnit>(sparse_lmhead_props).get();
  if (!std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "sparse lm head layer takes only one input";

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  auto const &in_dim = context.getInputDimensions()[0];
  vocab_size = unit;
  hidden_size = is_nchw ? in_dim.width() : in_dim.channel();

  /** output dim: height is always 1 (next-token logits) */
  std::vector<ml::train::TensorDim> output_dims(1);
  output_dims[0] = in_dim;
  if (is_nchw)
    output_dims[0].width(unit);
  else
    output_dims[0].channel(unit);
  output_dims[0].height(1);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);

  const auto wtype = context.getWeightDataType();
  // [in_d, out_d] weight, matching the projection convention used by lm_head.
  auto make_dim = [&](unsigned in_d, unsigned out_d) {
    return ml::train::TensorDim(
      1, is_nchw ? 1 : out_d, is_nchw ? in_d : 1, is_nchw ? out_d : in_d,
      ml::train::TensorDim::TensorType(context.getFormat(), wtype),
      is_nchw ? 0b0011 : 0b0101);
  };

  // Weight request ORDER == quantize_stream emit order (binding is by order):
  //   output_of_causallm (plain row-major Q4_0 [vocab,hidden]) — serves the
  //                       sparse gather AND the dense fallback
  //   output_profiler_w1 (repacked [hidden,H])
  //   output_profiler_w2 (repacked [H,vocab])
  plain_idx = context.requestWeight(
    make_dim(hidden_size, vocab_size), weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);
  w1_idx = context.requestWeight(
    make_dim(hidden_size, predictor_unit), weight_initializer,
    weight_regularizer, weight_regularizer_constant, weight_decay,
    "profiler_w1", true);
  w2_idx = context.requestWeight(
    make_dim(predictor_unit, vocab_size), weight_initializer,
    weight_regularizer, weight_regularizer_constant, weight_decay,
    "profiler_w2", true);
}

void SparseLmHeadLayer::resolveRuntimeConfig() {
  if (runtime_resolved)
    return;
  if (const char *e = std::getenv("NNTR_SPARSE_LMHEAD"))
    predictor_active = std::atoi(e) != 0;
  if (const char *e = std::getenv("NNTR_LMHEAD_THRESHOLD"))
    predictor_threshold = (float)std::atof(e);
  if (const char *e = std::getenv("NNTR_LMHEAD_TOPK_FLOOR"))
    predictor_topk_floor = (unsigned)std::strtoul(e, nullptr, 10);
  sparsity_log = std::getenv("NNTR_LMHEAD_SPARSITY_LOG") != nullptr;
  runtime_resolved = true;
}

void SparseLmHeadLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, sparse_lmhead_props);
  LayerImpl::setProperty(remain_props);
}

void SparseLmHeadLayer::computeSparseLogits(
  nntrainer::RunLayerContext &context, const nntrainer::Tensor &input_step,
  nntrainer::Tensor &hidden_step) {

  const unsigned V = vocab_size;
  const unsigned H = predictor_unit;
  const unsigned nb = (hidden_size + 31) / 32;
  const size_t row_stride = kQ4_0_BLOCK * nb;
  const auto ttype = input_step.getTensorType();

  // --- Predictor: mid = h * W1 ; score = mid * W2 (both repacked GEMVs). ---
  nntrainer::Tensor mid(nntrainer::TensorDim({1, 1, 1, H}, ttype));
  input_step.dot(context.getWeight(w1_idx), mid, false, false);
  nntrainer::Tensor score(nntrainer::TensorDim({1, 1, 1, V}, ttype));
  mid.dot(context.getWeight(w2_idx), score, false, false);
  const float *s = score.getData<float>();

  // --- Active vocabulary set: score > threshold, with a top-K safety floor. ---
  std::vector<unsigned> active;
  active.reserve(V / 4 + 1);
  for (unsigned v = 0; v < V; ++v)
    if (s[v] > predictor_threshold)
      active.push_back(v);

  if (predictor_topk_floor > 0 && active.size() < predictor_topk_floor &&
      predictor_topk_floor < V) {
    score_scratch.assign(s, s + V);
    std::nth_element(score_scratch.begin(),
                     score_scratch.begin() + (predictor_topk_floor - 1),
                     score_scratch.end(), std::greater<float>());
    const float kth = score_scratch[predictor_topk_floor - 1];
    active.clear();
    for (unsigned v = 0; v < V; ++v)
      if (s[v] >= kth)
        active.push_back(v);
  }
  if (active.empty()) // never drop everything: keep the single best score
    active.push_back(
      (unsigned)(std::max_element(s, s + V) - s));

  // --- Quantize hidden once; gather logits for active rows on plain weight. ---
  hidden_q8.resize(kQ8_0_BLOCK * nb);
  quantize_row_q8_0_local(input_step.getData<float>(), hidden_q8.data(),
                          hidden_size);
  const uint8_t *plain = context.getWeight(plain_idx).getData<uint8_t>();
  const uint8_t *xq = hidden_q8.data();
  float *out = hidden_step.getData<float>();

  auto &tm = nntrainer::ThreadManager::Global();

  if (sparsity_log) {
    // Measurement: compute ALL rows to get the dense argmax, then mask.
    tm.parallel_for(0, (size_t)V, [&](size_t v) {
      out[v] = q4_0_row_dot_q8(plain + row_stride * v, xq, nb);
    });
    const unsigned dense_arg =
      (unsigned)(std::max_element(out, out + V) - out);
    active_mask.assign(V, 0);
    for (unsigned idx : active)
      active_mask[idx] = 1;
    for (unsigned v = 0; v < V; ++v)
      if (!active_mask[v])
        out[v] = NEG_INF;
    const unsigned sparse_arg =
      (unsigned)(std::max_element(out, out + V) - out);
    log_calls += 1;
    log_active += (long long)active.size();
    log_examined += (long long)V;
    log_argmax_miss += (dense_arg != sparse_arg) ? 1 : 0;
  } else {
    std::fill(out, out + V, NEG_INF);
    const unsigned n_active = (unsigned)active.size();
    tm.parallel_for(0, (size_t)n_active, [&](size_t i) {
      const unsigned v = active[i];
      out[v] = q4_0_row_dot_q8(plain + row_stride * v, xq, nb);
    });
  }
}

void SparseLmHeadLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  throw nntrainer::exception::not_supported(
    "Forwarding for SparseLmHead layer is not supported");
}

void SparseLmHeadLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  resolveRuntimeConfig();

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim input_dim = input_.getDim();
  ml::train::TensorDim hidden_dim = hidden_.getDim();
  ml::train::TensorDim input_step_dim = input_dim;
  ml::train::TensorDim hidden_step_dim = hidden_dim;
  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);

  // Predictor is a single-token (decode) optimization, matching PowerInfer
  // (ne[1]==1). Multi-token / prefill steps fall back to the dense GEMV.
  const bool use_predictor = predictor_active && ((to - from) == 1);

  unsigned int b_size = input_dim.batch();
  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor input_step = input_.getSharedDataTensor(
      input_step_dim,
      b * input_dim.getFeatureLen() + (to - from - 1) * input_.width(), true);
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    if (use_predictor) {
      computeSparseLogits(context, input_step, hidden_step);
    } else {
      computeDenseLogits(context, input_step, hidden_step);
    }
  }
}

// Dense fallback: all V logits via the plain per-row Q4_0 kernel (same kernel
// the sparse path uses, with no mask). lm_head only ever computes one position
// per step, so this is one GEMV's worth of work.
void SparseLmHeadLayer::computeDenseLogits(
  nntrainer::RunLayerContext &context, const nntrainer::Tensor &input_step,
  nntrainer::Tensor &hidden_step) {
  const unsigned V = vocab_size;
  const unsigned nb = (hidden_size + 31) / 32;
  const size_t row_stride = kQ4_0_BLOCK * nb;
  hidden_q8.resize(kQ8_0_BLOCK * nb);
  quantize_row_q8_0_local(input_step.getData<float>(), hidden_q8.data(),
                          hidden_size);
  const uint8_t *plain = context.getWeight(plain_idx).getData<uint8_t>();
  const uint8_t *xq = hidden_q8.data();
  float *out = hidden_step.getData<float>();
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(0, (size_t)V, [&](size_t v) {
    out[v] = q4_0_row_dot_q8(plain + row_stride * v, xq, nb);
  });
}

void SparseLmHeadLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for SparseLmHead layer is not supported");
}

void SparseLmHeadLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for SparseLmHead layer is not supported");
}

void SparseLmHeadLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(sparse_lmhead_props, method, this);
}

void SparseLmHeadLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  unsigned int height = input_dimensions[0].height();
  in_dim.height(height);
  context.updateInput(SINGLE_INOUT_IDX, in_dim);
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
