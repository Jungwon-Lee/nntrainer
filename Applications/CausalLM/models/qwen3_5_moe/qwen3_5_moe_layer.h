// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_5_moe_layer.h
 * @date   19 August 2026
 * @brief  Routed and shared-expert MoE layer for Qwen3.5/Qwen3.6.
 */

#ifndef __QWEN3_5_MOE_LAYER_H__
#define __QWEN3_5_MOE_LAYER_H__

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace qwen3_5_props {

class SharedExpertIntermediateSize : public nntrainer::PositiveIntegerProperty {
public:
  SharedExpertIntermediateSize(unsigned int value = 1) { set(value); }
  static constexpr const char *key = "shared_expert_intermediate_size";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace qwen3_5_props

/**
 * @brief Qwen3.5/Qwen3.6 sparse MoE with a sigmoid-gated shared expert.
 */
class Qwen3_5MoeLayer final : public nntrainer::LayerImpl {
public:
  Qwen3_5MoeLayer();
  ~Qwen3_5MoeLayer() = default;

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
  void save(
    std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    ml::train::TensorDim::DataType dtype = ml::train::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  const std::string getType() const override { return type; }
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "qwen3_5_moe";

private:
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit,
             qwen3_5_props::SharedExpertIntermediateSize>
    moe_props;

  unsigned int num_experts = 0;
  unsigned int topk = 0;
  unsigned int intermediate_size = 0;
  unsigned int shared_intermediate_size = 0;

  unsigned int gate_idx = 0;
  std::vector<unsigned int> expert_gate_up_indices;
  std::vector<unsigned int> expert_down_indices;
  unsigned int shared_gate_up_idx = 0;
  unsigned int shared_down_idx = 0;
  unsigned int shared_expert_gate_idx = 0;
  unsigned int router_logits_idx = 0;

  void runStep(nntrainer::RunLayerContext &context, unsigned int step_size);
  void buildAssignments(const nntrainer::Tensor &router_logits,
                        unsigned int total_tokens,
                        std::vector<std::vector<std::pair<unsigned int, float>>>
                          &assignments) const;
  void runRoutedExperts(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    nntrainer::RunLayerContext &context,
    const std::vector<std::vector<std::pair<unsigned int, float>>> &assignments,
    unsigned int hidden_size) const;
  void runSharedExpert(const nntrainer::Tensor &input,
                       nntrainer::Tensor &output,
                       nntrainer::RunLayerContext &context,
                       unsigned int total_tokens,
                       unsigned int hidden_size) const;
};

} // namespace causallm

#endif // __QWEN3_5_MOE_LAYER_H__
