// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_layer_sparse_slim.h
 * @date   30 June 2026
 * @brief  Slim (no-cache) SmallThinker MoE layer with ReLU (ReGLU) sparsity.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @note   Subclass of SmallThinkerSlimMoELayer (on-demand activate/compute/
 *         deactivate per expert, no cache) that overrides only the per-expert
 *         FFN with the shared sparse ReGLU compute (B3 hybrid). Uses the same
 *         "smallthinker_sparse" .bin layout (gate repacked, up/down plain).
 */

#ifndef __SMALLTHINKER_MOE_LAYER_SPARSE_SLIM_H__
#define __SMALLTHINKER_MOE_LAYER_SPARSE_SLIM_H__
#ifdef __cplusplus

#include <smallthinker_moe_layer_slim.h>

namespace causallm {

/**
 * @class   SmallThinkerSparseSlimMoELayer
 * @brief   Slim (no-cache) MoE layer with a sparse ReGLU expert FFN.
 */
class SmallThinkerSparseSlimMoELayer : public SmallThinkerSlimMoELayer {
public:
  SmallThinkerSparseSlimMoELayer() = default;
  ~SmallThinkerSparseSlimMoELayer() = default;
  SmallThinkerSparseSlimMoELayer(
    SmallThinkerSparseSlimMoELayer &&rhs) noexcept = default;
  SmallThinkerSparseSlimMoELayer &
  operator=(SmallThinkerSparseSlimMoELayer &&rhs) = default;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return SmallThinkerSparseSlimMoELayer::type;
  };

  static constexpr const char *type = "smallthinker_moe_sparse_slim";

protected:
  /**
   * @copydoc SmallThinkerSlimMoELayer::compute_expert_forward
   * @note Sparse ReGLU FFN for the (already-activated) expert.
   */
  void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size) override;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_MOE_LAYER_SPARSE_SLIM_H__ */
