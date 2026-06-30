/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   smallthinker_moe_layer_sparse_cached_slim.cpp
 * @date   30 June 2026
 * @brief  Cached-slim SmallThinker MoE layer with ReLU (ReGLU) sparsity.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#include <smallthinker_moe_layer_sparse_cached_slim.h>
#include <smallthinker_sparse_ffn.h>

namespace causallm {

void SmallThinkerSparseCachedSlimMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size,
  long long &sparsity_nz, long long &sparsity_total) {
  // The base class has already activated this expert's gate/up/down tensors;
  // run the shared sparse ReGLU FFN (repacked gate GEMV -> ReLU -> sparse
  // up/down), accumulating into `output`. Report active/examined counts so the
  // base class's per-layer relu_zero logging stays correct.
  sparse_reglu_ffn(input, output, token_assignments, gate_proj, up_proj,
                   down_proj, hidden_size, &sparsity_nz, &sparsity_total);
}

} // namespace causallm
