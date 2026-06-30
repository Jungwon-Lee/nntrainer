// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_layer_base_sparse.h
 * @date   30 June 2026
 * @brief  SmallThinker BASE MoE layer with ReLU (ReGLU) activation sparsity.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @note   This is the resident (all-experts-in-RAM) SmallThinker MoE layer with
 *         a sparse ReGLU FFN, ported from PowerInfer's `fused_sparse_moe`
 *         (Tiiny-AI/PowerInfer, smallthinker/powerinfer/fused_sparse_moe). The
 *         expert FFN is `out += w * down( ReLU(gate*x) (.) (up*x) )`; SmallThinker
 *         is natively trained with ReGLU, so the gate is computed in full and the
 *         up/down projections are computed only for neurons whose ReLU(gate) > 0.
 *
 *         Requires a sparse-layout .bin (quantize_stream recipe
 *         "smallthinker_sparse") where the expert gate/up/down weights are stored
 *         as PLAIN per-neuron-row Q4_0 laid out [intermediate, hidden]
 *         (neuron-major): row j = neuron j's weights, contiguous over hidden.
 *         Byte count is identical to the dense (repacked) layout, so finalize()
 *         and weight requests are inherited unchanged from SmallThinkerMoELayer.
 */

#ifndef __SMALLTHINKER_MOE_LAYER_BASE_SPARSE_H__
#define __SMALLTHINKER_MOE_LAYER_BASE_SPARSE_H__
#ifdef __cplusplus

#include <smallthinker_moe_layer.h>

namespace causallm {

/**
 * @class   SmallThinkerSparseMoELayer
 * @brief   Resident SmallThinker MoE layer with a sparse ReGLU expert FFN.
 */
class SmallThinkerSparseMoELayer : public SmallThinkerMoELayer {
public:
  SmallThinkerSparseMoELayer() = default;
  ~SmallThinkerSparseMoELayer() = default;
  SmallThinkerSparseMoELayer(SmallThinkerSparseMoELayer &&rhs) noexcept = default;
  SmallThinkerSparseMoELayer &
  operator=(SmallThinkerSparseMoELayer &&rhs) = default;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return SmallThinkerSparseMoELayer::type;
  };

  static constexpr const char *type = "smallthinker_moe_sparse";

protected:
  /**
   * @copydoc SmallThinkerMoELayer::compute_expert_forward_no_critical
   * @note Decode (M small): sparse ReGLU FFN over the assigned tokens.
   */
  void compute_expert_forward_no_critical(
    const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size) override;

  /**
   * @copydoc SmallThinkerMoELayer::compute_expert_forward_batched
   * @note Prefill (M>1): the plain per-neuron layout cannot use the repacked
   *       batched GEMM, so this runs the same per-token sparse FFN as decode,
   *       accumulating directly into the shared output (caller is serial).
   */
  void compute_expert_forward_batched(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size) override;

private:
  /**
   * @brief Sparse ReGLU FFN: for each assigned token, compute gate (full) ->
   *        ReLU mask -> up/down only for active neurons, accumulating
   *        `routing_w * down(ReLU(gate*x) (.) (up*x))` into `out` at the token's
   *        offset. Mirrors PowerInfer fused_sparse_moe: one parallel_for over
   *        neuron ranges per token, per-thread output buffers, single reduce.
   */
  void compute_sparse_ffn(
    const nntrainer::Tensor &input, nntrainer::Tensor &out,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_MOE_LAYER_BASE_SPARSE_H__ */
