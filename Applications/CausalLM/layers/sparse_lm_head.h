// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   sparse_lm_head.h
 * @date   31 July 2026
 * @brief  Predictor-guided sparse Q4_0 LM head.
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

#include <common_properties.h>
#include <cstdint>
#include <layer_devel.h>
#include <layer_impl.h>
#include <limits>
#include <tuple>
#include <vector>

namespace causallm {

namespace props {

class PredictorUnit : public nntrainer::Property<unsigned int> {
public:
  PredictorUnit(unsigned int value = 128) { set(value); }
  static constexpr const char *key = "predictor_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class PredictorThreshold : public nntrainer::Property<float> {
public:
  PredictorThreshold(float value = -2.0f) { set(value); }
  static constexpr const char *key = "predictor_threshold";
  using prop_tag = nntrainer::float_prop_tag;
};

class PredictorTopKFloor : public nntrainer::Property<unsigned int> {
public:
  PredictorTopKFloor(unsigned int value = 0) { set(value); }
  static constexpr const char *key = "predictor_topk_floor";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @class SparseLmHeadLayer
 * @brief Predict vocabulary candidates and evaluate only selected Q4_0 rows.
 */
WIN_EXPORT class SparseLmHeadLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT SparseLmHeadLayer();
  WIN_EXPORT ~SparseLmHeadLayer();
  WIN_EXPORT SparseLmHeadLayer(SparseLmHeadLayer &&rhs) noexcept = default;
  WIN_EXPORT SparseLmHeadLayer &
  operator=(SparseLmHeadLayer &&rhs) noexcept = default;

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
  }
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  using Layer::setProperty;
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "sparse_lm_head";

private:
  std::tuple<nntrainer::props::Unit, props::PredictorUnit,
             props::PredictorThreshold, props::PredictorTopKFloor>
    sparse_lmhead_props;

  unsigned int head_idx = std::numeric_limits<unsigned int>::max();
  unsigned int predictor_w1_idx = std::numeric_limits<unsigned int>::max();
  unsigned int predictor_w2_idx = std::numeric_limits<unsigned int>::max();
  unsigned int predictor_mid_idx = std::numeric_limits<unsigned int>::max();
  unsigned int predictor_score_idx = std::numeric_limits<unsigned int>::max();

  bool skip_prefill = false;
  bool runtime_resolved = false;
  bool predictor_active = true;
  bool sparsity_log = false;
  bool q4_weights = false;
  unsigned int vocab_size = 0;
  unsigned int hidden_size = 0;
  unsigned int predictor_unit = 0;
  unsigned int predictor_topk_floor = 0;
  float predictor_threshold = -2.0f;

  long long log_calls = 0;
  long long log_active = 0;
  long long log_examined = 0;
  long long log_argmax_miss = 0;

  std::vector<float> score_scratch;
  std::vector<unsigned int> active_indices;
  std::vector<uint8_t> active_mask;

  void resolveRuntimeConfig();
  bool computeSparseLogits(nntrainer::RunLayerContext &context,
                           const nntrainer::Tensor &input_step,
                           nntrainer::Tensor &hidden_step);
  void computeDenseLogits(nntrainer::RunLayerContext &context,
                          const nntrainer::Tensor &input_step,
                          nntrainer::Tensor &hidden_step);
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __SPARSE_LM_HEAD_H__ */
