// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   sparse_lm_head.h
 * @date   30 June 2026
 * @brief  Sparse LM-head layer with an activation predictor (paper §6.2 /
 *         PowerInfer). On single-token decode steps it runs a tiny 2-layer
 *         linear predictor over the vocabulary, then computes exact logits
 *         only for the predicted-active rows (others set to -inf). Prefill /
 *         multi-token steps and the NNTR_SPARSE_LMHEAD=0 toggle fall back to
 *         the dense repacked GEMV.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#ifndef __SPARSE_LM_HEAD_H__
#define __SPARSE_LM_HEAD_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <array>
#include <common_properties.h>
#include <cstdint>
#include <layer_devel.h>
#include <layer_impl.h>
#include <limits>
#include <tuple>
#include <vector>

namespace causallm {

namespace props {

/**
 * @brief Predictor intermediate dimension H (fc1 out / fc2 in). SmallThinker
 * ships H = 128 for both 4B and 21B.
 */
class PredictorUnit : public nntrainer::Property<unsigned int> {
public:
  PredictorUnit(unsigned int value = 128) { set(value); }
  static constexpr const char *key = "predictor_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @class   SparseLmHeadLayer
 * @brief   LM-head with a sparse vocabulary-activation predictor.
 */
WIN_EXPORT class SparseLmHeadLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT SparseLmHeadLayer();
  WIN_EXPORT ~SparseLmHeadLayer();
  WIN_EXPORT SparseLmHeadLayer(SparseLmHeadLayer &&rhs) noexcept = default;
  WIN_EXPORT SparseLmHeadLayer &operator=(SparseLmHeadLayer &&rhs) = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;
  WIN_EXPORT const std::string getType() const override {
    return SparseLmHeadLayer::type;
  };
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  using Layer::setProperty;
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "sparse_lm_head";

private:
  std::tuple<nntrainer::props::Unit, props::PredictorUnit> sparse_lmhead_props;

  /** weight indices (requested in this exact order; weights bind by ORDER):
   *  lmhead_plain (plain row-major Q4_0 [vocab,hidden]), profiler_w1, w2. The
   *  plain weight serves BOTH the sparse gather and the dense fallback (lm_head
   *  only ever computes one position per step), so no repacked copy is kept. */
  unsigned int plain_idx = std::numeric_limits<unsigned int>::max();
  unsigned int w1_idx = std::numeric_limits<unsigned int>::max();
  unsigned int w2_idx = std::numeric_limits<unsigned int>::max();

  bool skip_prefill = false;
  unsigned int vocab_size = 0;
  unsigned int hidden_size = 0;
  unsigned int predictor_unit = 0; /**< H */

  /** runtime knobs (env, resolved once) */
  bool runtime_resolved = false;
  bool predictor_active = true;  /**< NNTR_SPARSE_LMHEAD != 0 (default on) */
  float predictor_threshold = 0.0f;
  unsigned int predictor_topk_floor = 0;
  bool sparsity_log = false;

  /** instrumentation (NNTR_LMHEAD_SPARSITY_LOG) */
  long long log_calls = 0;
  long long log_active = 0;
  long long log_examined = 0;
  long long log_argmax_miss = 0;

  /** scratch */
  std::vector<float> score_scratch;
  std::vector<uint8_t> hidden_q8;
  std::vector<char> active_mask;

  void resolveRuntimeConfig();
  void computeSparseLogits(nntrainer::RunLayerContext &context,
                           const nntrainer::Tensor &input_step,
                           nntrainer::Tensor &hidden_step);
  void computeDenseLogits(nntrainer::RunLayerContext &context,
                          const nntrainer::Tensor &input_step,
                          nntrainer::Tensor &hidden_step);
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __SPARSE_LM_HEAD_H__ */
