// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_5_delta_layer.h
 * @date   19 August 2026
 * @brief  Recurrent Gated DeltaNet core used by Qwen3.5/Qwen3.6.
 */

#ifndef __QWEN3_5_DELTA_LAYER_H__
#define __QWEN3_5_DELTA_LAYER_H__

#include <base_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace qwen3_5_props {

class NumKeyHeads : public nntrainer::PositiveIntegerProperty {
public:
  NumKeyHeads(unsigned int value = 1) { set(value); }
  static constexpr const char *key = "num_key_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};

class NumValueHeads : public nntrainer::PositiveIntegerProperty {
public:
  NumValueHeads(unsigned int value = 1) { set(value); }
  static constexpr const char *key = "num_value_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};

class KeyHeadDim : public nntrainer::PositiveIntegerProperty {
public:
  KeyHeadDim(unsigned int value = 1) { set(value); }
  static constexpr const char *key = "key_head_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

class ValueHeadDim : public nntrainer::PositiveIntegerProperty {
public:
  ValueHeadDim(unsigned int value = 1) { set(value); }
  static constexpr const char *key = "value_head_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

class ConvKernelSize : public nntrainer::PositiveIntegerProperty {
public:
  ConvKernelSize(unsigned int value = 4) { set(value); }
  static constexpr const char *key = "conv_kernel_size";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace qwen3_5_props

/**
 * @brief Gated DeltaNet recurrent core.
 *
 * The large input/output projections remain ordinary fully-connected layers so
 * they can use NNTrainer's Q4_0 kernels. This layer owns only the depthwise
 * convolution, recurrent-state parameters and per-value-head RMSNorm scale;
 * those tensors intentionally remain FP32.
 */
class Qwen3_5DeltaLayer final : public nntrainer::LayerImpl {
public:
  Qwen3_5DeltaLayer();
  ~Qwen3_5DeltaLayer() = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;
  void setProperty(const std::vector<std::string> &values) override;
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  const std::string getType() const override { return type; }
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "qwen3_5_delta";

private:
  enum InputIndex { MIXED_QKV = 0, Z_GATE, BETA, DECAY, NUM_INPUTS };
  enum WeightIndex {
    CONV_WEIGHT = 0,
    DT_BIAS,
    A_LOG,
    NORM_WEIGHT,
    NUM_WEIGHTS
  };
  enum TensorIndex { CONV_STATE = 0, RECURRENT_STATE, NUM_TENSORS };

  std::tuple<qwen3_5_props::NumKeyHeads, qwen3_5_props::NumValueHeads,
             qwen3_5_props::KeyHeadDim, qwen3_5_props::ValueHeadDim,
             qwen3_5_props::ConvKernelSize, nntrainer::props::Epsilon>
    delta_props;

  std::array<unsigned int, NUM_WEIGHTS> weight_idx;
  std::array<unsigned int, NUM_TENSORS> tensor_idx;

  unsigned int num_key_heads = 0;
  unsigned int num_value_heads = 0;
  unsigned int key_head_dim = 0;
  unsigned int value_head_dim = 0;
  unsigned int conv_kernel_size = 0;
  unsigned int key_dim = 0;
  unsigned int value_dim = 0;
  unsigned int conv_dim = 0;
  float epsilon = 1e-6f;

  void runStep(nntrainer::RunLayerContext &context, unsigned int step_size,
               bool reset_state);
};

} // namespace causallm

#endif // __QWEN3_5_DELTA_LAYER_H__
