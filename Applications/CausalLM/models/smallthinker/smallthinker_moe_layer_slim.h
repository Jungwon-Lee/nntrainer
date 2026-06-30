// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_layer_slim.h
 * @date   12 May 2026
 * @brief  SmallThinker MoE layer with on-demand expert loading (no cache).
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#ifndef __SMALLTHINKER_MOE_LAYER_SLIM_H__
#define __SMALLTHINKER_MOE_LAYER_SLIM_H__
#ifdef __cplusplus

#include <smallthinker_moe_layer.h>
#include <vector>

namespace causallm {

/**
 * @class   SmallThinkerSlimMoELayer
 * @brief   SmallThinker Mixture of Expert Layer with virtual expert weights.
 *          Each expert is activated (mmap'd), computed, and immediately
 *          deactivated (munmap'd) — no persistent cache across tokens.
 */
class SmallThinkerSlimMoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of SmallThinker slim Mixture of Expert Layer
   */
  SmallThinkerSlimMoELayer();

  /**
   * @brief     Destructor of SmallThinker slim Mixture of Expert Layer
   */
  ~SmallThinkerSlimMoELayer() = default;

  /**
   * @brief  Move constructor.
   * @param[in] SmallThinkerSlimMoELayer &&
   */
  SmallThinkerSlimMoELayer(SmallThinkerSlimMoELayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs SmallThinkerSlimMoELayer to be moved.
   */
  SmallThinkerSlimMoELayer &operator=(SmallThinkerSlimMoELayer &&rhs) = default;

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
    return SmallThinkerSlimMoELayer::type;
  };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type =
    "smallthinker_moe_slim"; /**< type of the layer */

  // protected (not private) so the sparse subclass
  // (SmallThinkerSparseSlimMoELayer) can override the expert FFN and reuse the
  // on-demand activate/compute/deactivate machinery.
protected:
  unsigned int num_experts;      /**< number of experts */
  unsigned int topk;             /**< number of experts per token */
  bool router_apply_softmax;     /**< whether router uses softmax or sigmoid */
  nntrainer::ActiFunc acti_func; /**< activation function for the expert */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation,
             props::MoERouterApplySoftmax>
    moe_props;

  // weight indices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int gate_idx;

  // intermediate tensor indices
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;

  virtual void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_MOE_LAYER_SLIM_H__ */
