// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_layer.h
 * @date   28 April 2026
 * @brief  This is SmallThinker Mixture of Expert Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This file is part of the SmallThinker Mixture of Expert Layer
 *         implementation.
 *         It does not support shared experts.
 *         This layer is implemented based on the LLama-MoE.
 *         For more information, please refer to the following link:
 *         https://arxiv.org/pdf/2406.16554
 * @todo   This layer does not support backwarding yet.
 */

#ifndef __SMALLTHINKER_MOE_LAYER_H__
#define __SMALLTHINKER_MOE_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {

/**
 * @brief MoERouterApplySoftmax, whether router logits use softmax or sigmoid
 */
class MoERouterApplySoftmax : public nntrainer::Property<bool> {
public:
  MoERouterApplySoftmax(bool value = true) { set(value); }
  static constexpr const char *key =
    "moe_router_apply_softmax";              /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag; /**< property type */
};

/**
 * @brief MoECacheSize, max number of experts kept resident (mmap'd) by the
 *        slim MoE layer's LRU cache. Only used by the on-demand slim path.
 */
class MoECacheSize : public nntrainer::Property<unsigned int> {
public:
  MoECacheSize(unsigned int value = 32) { set(value); }
  static constexpr const char *key = "moe_cache_size"; /**< unique key */
  using prop_tag = nntrainer::uint_prop_tag;           /**< property type */
};

} // namespace props

/**
 * @class   SmallThinkerMoELayer
 * @brief   SmallThinker Mixture of Expert Layer
 */
class SmallThinkerMoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of SmallThinker Mixture of Expert Layer
   */
  SmallThinkerMoELayer();

  /**
   * @brief     Destructor of SmallThinker Mixture of Expert Layer
   */
  ~SmallThinkerMoELayer() = default;

  /**
   * @brief  Move constructor.
   *  @param[in] SmallThinkerMoELayer &&
   */
  SmallThinkerMoELayer(SmallThinkerMoELayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs SmallThinkerMoELayer to be moved.
   */
  SmallThinkerMoELayer &operator=(SmallThinkerMoELayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return SmallThinkerMoELayer::type;
  };

  /**
   * @copydoc Layer::save()
   * @note Overridden so the router (gate) weight is always written as FP32,
   *       while expert weights honor the requested quantization dtype. A 4-bit
   *       router corrupts top-k expert selection and yields garbage output.
   */
  void save(
    std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    nntrainer::TensorDim::DataType dtype = nntrainer::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type =
    "smallthinker_moe"; /**< type of the layer */

private:
  unsigned int num_experts;      /**< number of experts */
  unsigned int topk;             /**< number of experts per token, i.e., topk */
  bool router_apply_softmax;     /**< whether router uses softmax or sigmoid */
  nntrainer::ActiFunc acti_func; /**< activation function for the expert */
  std::tuple<causallm::props::NumExperts, causallm::props::NumExpertsPerToken,
             nntrainer::props::Unit, causallm::props::MoEActivation,
             props::MoERouterApplySoftmax>
    moe_props;

  // weight indices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int gate_idx;

  // Intermediate tensor indices
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;

  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);

  inline void compute_expert_forward_no_critical(
    const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);

  // Batched GEMM variant: gathers all assigned tokens into a contiguous matrix
  // and performs 3 GEMMs (gate, up, down) instead of N separate GEMVs.
  // Significantly faster during prefill (M = num_tokens > 1).
  inline void compute_expert_forward_batched(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_MOE_LAYER_H__ */
