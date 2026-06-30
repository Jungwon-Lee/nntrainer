// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_sparse_ffn.h
 * @date   30 June 2026
 * @brief  Shared sparse ReGLU FFN compute for SmallThinker MoE variants.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @note   The B3 hybrid sparse expert FFN, shared by the resident base-sparse
 *         layer and the cached-slim / slim sparse variants. The expert weights
 *         must use the "smallthinker_sparse" layout: gate REPACKED (q4_0x8/x4,
 *         computed in full by the dense GEMV to build the ReLU mask), up/down
 *         PLAIN per-neuron-row Q4_0. The implementation (kernels + thread
 *         distribution + NNTR_RELU_SPARSITY_LOG / NNTR_MOE_THREAD_PROFILE /
 *         NNTR_MOE_CHUNK env handling) lives in
 *         smallthinker_moe_layer_base_sparse.cpp.
 */

#ifndef __SMALLTHINKER_SPARSE_FFN_H__
#define __SMALLTHINKER_SPARSE_FFN_H__
#ifdef __cplusplus

#include <tensor.h>
#include <utility>
#include <vector>

namespace causallm {

/**
 * @brief Sparse ReGLU FFN for one expert's assigned tokens.
 *
 * For each token: full gate via the optimal repacked q4_0xq8_0 GEMV
 * (`token_input.dot(gate_proj)`) -> ReLU mask -> compact active set, then
 * up/down only for active neurons (fused q4_0xq8_0 dot + dequant-axpy) into
 * per-thread buffers, reduced once with the routing weight. Accumulates into
 * `out` at each token's offset (caller zero-inits / serializes as needed).
 *
 * @param input        full layer input (reshaped [total_tokens,1,1,hidden])
 * @param out          output tensor to accumulate into (same shape as input)
 * @param token_assignments  (token_idx, routing_weight) pairs for this expert
 * @param gate_proj    REPACKED Q4_0 gate weight (dense GEMV)
 * @param up_proj      PLAIN per-neuron-row Q4_0 up weight
 * @param down_proj    PLAIN per-neuron-row Q4_0 down weight
 * @param hidden_size  model hidden size
 * @param active_out    if non-null, ADDS the number of active (post-ReLU)
 *                      neurons processed (for the caller's sparsity logging)
 * @param examined_out  if non-null, ADDS the number of neurons examined
 *                      (intermediate_size * num_tokens)
 */
void sparse_reglu_ffn(
  const nntrainer::Tensor &input, nntrainer::Tensor &out,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size,
  long long *active_out = nullptr, long long *examined_out = nullptr);

} // namespace causallm

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_SPARSE_FFN_H__ */
