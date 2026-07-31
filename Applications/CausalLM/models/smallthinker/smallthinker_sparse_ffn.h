// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_sparse_ffn.h
 * @date   30 July 2026
 * @brief  Sparse ReGLU expert helper for SmallThinker.
 */

#ifndef __SMALLTHINKER_SPARSE_FFN_H__
#define __SMALLTHINKER_SPARSE_FFN_H__

#include <tensor.h>

namespace causallm {

/**
 * @brief Compute one routed SmallThinker expert token.
 *
 * The FP32 decode path evaluates the dense gate first and, when at least half
 * of the ReLU gate outputs are zero, computes only active up-projection
 * neurons and accumulates only their down-projection rows. Other data types
 * and dense activation patterns use the regular dense tensor kernels.
 *
 * @param input Single-token expert input, shape [1, 1, 1, hidden_size].
 * @param output Single-token expert output.
 * @param gate_proj Expert gate projection.
 * @param up_proj Expert up projection.
 * @param down_proj Expert down projection.
 * @param routing_weight Router weight applied to the expert result.
 * @return true when the sparse FP32 path was used.
 */
bool computeSmallThinkerReGLU(const nntrainer::Tensor &input,
                              nntrainer::Tensor &output,
                              const nntrainer::Tensor &gate_proj,
                              const nntrainer::Tensor &up_proj,
                              const nntrainer::Tensor &down_proj,
                              float routing_weight);

} // namespace causallm

#endif /* __SMALLTHINKER_SPARSE_FFN_H__ */
