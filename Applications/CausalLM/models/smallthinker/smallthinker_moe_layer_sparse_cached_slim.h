// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_layer_sparse_cached_slim.h
 * @date   30 June 2026
 * @brief  Cached-slim SmallThinker MoE layer with ReLU (ReGLU) sparsity.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @note   Subclass of SmallThinkerCachedSlimMoELayer that reuses ALL of the
 *         LRU-cache / prefetch / eviction machinery and overrides only the
 *         per-expert FFN with the shared sparse ReGLU compute (B3 hybrid:
 *         repacked gate GEMV -> ReLU mask -> sparse up/down). On a
 *         memory-constrained device the plain per-neuron up/down layout means
 *         only the ~20% active neuron rows fault in from the mmap'd expert,
 *         cutting expert I/O — the paper's §6.1+§6.2 combination. Uses the same
 *         "smallthinker_sparse" .bin layout as the resident base-sparse model
 *         (gate repacked, up/down plain), so no finalize/weight change.
 */

#ifndef __SMALLTHINKER_MOE_LAYER_SPARSE_CACHED_SLIM_H__
#define __SMALLTHINKER_MOE_LAYER_SPARSE_CACHED_SLIM_H__
#ifdef __cplusplus

#include <smallthinker_moe_layer_cached_slim.h>

namespace causallm {

/**
 * @class   SmallThinkerSparseCachedSlimMoELayer
 * @brief   Cached-slim MoE layer with a sparse ReGLU expert FFN.
 */
class SmallThinkerSparseCachedSlimMoELayer
  : public SmallThinkerCachedSlimMoELayer {
public:
  SmallThinkerSparseCachedSlimMoELayer() = default;
  ~SmallThinkerSparseCachedSlimMoELayer() = default;
  SmallThinkerSparseCachedSlimMoELayer(
    SmallThinkerSparseCachedSlimMoELayer &&rhs) noexcept = default;
  SmallThinkerSparseCachedSlimMoELayer &
  operator=(SmallThinkerSparseCachedSlimMoELayer &&rhs) = default;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return SmallThinkerSparseCachedSlimMoELayer::type;
  };

  static constexpr const char *type = "smallthinker_moe_sparse_cached_slim";

protected:
  /**
   * @copydoc SmallThinkerCachedSlimMoELayer::compute_expert_forward
   * @note Sparse ReGLU FFN for the (already-activated) expert.
   */
  void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size,
    long long &sparsity_nz, long long &sparsity_total) override;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_MOE_LAYER_SPARSE_CACHED_SLIM_H__ */
