// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_sparse_ffn.cpp
 * @date   30 July 2026
 * @brief  Sparse ReGLU expert helper for SmallThinker.
 */

#include <cpu_backend.h>
#include <smallthinker_sparse_ffn.h>

#include <vector>

namespace causallm {

namespace {

constexpr unsigned int SPARSE_ACTIVE_NUMERATOR = 1;
constexpr unsigned int SPARSE_ACTIVE_DENOMINATOR = 2;

void reglu(unsigned int size, float *output, const float *gate,
           const float *up) {
  for (unsigned int i = 0; i < size; ++i) {
    const float activated = gate[i] <= 0.0f ? 0.0f : gate[i];
    output[i] = activated * up[i];
  }
}

bool supportsSparseFP32(const nntrainer::Tensor &input,
                        const nntrainer::Tensor &output,
                        const nntrainer::Tensor &gate_proj,
                        const nntrainer::Tensor &up_proj,
                        const nntrainer::Tensor &down_proj) {
  using DataType = nntrainer::TensorDim::DataType;
  if (input.getDataType() != DataType::FP32 ||
      output.getDataType() != DataType::FP32 ||
      gate_proj.getDataType() != DataType::FP32 ||
      up_proj.getDataType() != DataType::FP32 ||
      down_proj.getDataType() != DataType::FP32)
    return false;

  const unsigned int hidden_size = input.width();
  const unsigned int intermediate_size = gate_proj.width();
  return input.getDim().getDataLen() == hidden_size &&
         output.getDim().getDataLen() == hidden_size &&
         gate_proj.height() == hidden_size && up_proj.height() == hidden_size &&
         up_proj.width() == intermediate_size &&
         down_proj.height() == intermediate_size &&
         down_proj.width() == hidden_size;
}

} // namespace

bool computeSmallThinkerReGLU(const nntrainer::Tensor &input,
                              nntrainer::Tensor &output,
                              const nntrainer::Tensor &gate_proj,
                              const nntrainer::Tensor &up_proj,
                              const nntrainer::Tensor &down_proj,
                              float routing_weight) {
  const unsigned int intermediate_size = gate_proj.width();
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::Tensor gate_out(intermediate_dim);
  input.dot(gate_proj, gate_out);

  if (supportsSparseFP32(input, output, gate_proj, up_proj, down_proj)) {
    const float *gate_data = gate_out.getData<float>();
    std::vector<unsigned int> active_indices;
    active_indices.reserve(intermediate_size / 2);
    for (unsigned int i = 0; i < intermediate_size; ++i) {
      if (!(gate_data[i] <= 0.0f))
        active_indices.push_back(i);
    }

    const size_t active_limit = static_cast<size_t>(intermediate_size) *
                                SPARSE_ACTIVE_NUMERATOR /
                                SPARSE_ACTIVE_DENOMINATOR;
    if (active_indices.size() <= active_limit) {
      const unsigned int hidden_size = input.width();
      const float *input_data = input.getData<float>();
      const float *up_data = up_proj.getData<float>();
      const float *down_data = down_proj.getData<float>();
      float *output_data = output.getData<float>();
      output.setZero();

      for (const unsigned int index : active_indices) {
        const float up_value = nntrainer::sdot(
          hidden_size, input_data, 1, up_data + index, intermediate_size);
        const float scale = routing_weight * gate_data[index] * up_value;
        nntrainer::saxpy(hidden_size, scale,
                         down_data + static_cast<size_t>(index) * hidden_size,
                         1, output_data, 1);
      }
      return true;
    }
  }

  nntrainer::Tensor up_out(intermediate_dim);
  nntrainer::Tensor acti_out(intermediate_dim);
  input.dot(up_proj, up_out);
  reglu(intermediate_size, acti_out.getData<float>(), gate_out.getData<float>(),
        up_out.getData<float>());
  acti_out.dot(down_proj, output);
  output.multiply_i(routing_weight);
  return false;
}

} // namespace causallm
