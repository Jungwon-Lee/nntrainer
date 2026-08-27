// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_5_delta_layer.cpp
 * @date   19 August 2026
 * @brief  Recurrent Gated DeltaNet core used by Qwen3.5/Qwen3.6.
 */

#include "qwen3_5_delta_layer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <node_exporter.h>

namespace causallm {

namespace {

float sigmoid(float value) {
  if (value >= 0.0f) {
    const float exp_neg = std::exp(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = std::exp(value);
  return exp_pos / (1.0f + exp_pos);
}

float softplus(float value) {
  return value > 20.0f ? value : std::log1p(std::exp(value));
}

float silu(float value) { return value * sigmoid(value); }

} // namespace

Qwen3_5DeltaLayer::Qwen3_5DeltaLayer() :
  LayerImpl(),
  delta_props(qwen3_5_props::NumKeyHeads(), qwen3_5_props::NumValueHeads(),
              qwen3_5_props::KeyHeadDim(), qwen3_5_props::ValueHeadDim(),
              qwen3_5_props::ConvKernelSize(), nntrainer::props::Epsilon()) {
  weight_idx.fill(std::numeric_limits<unsigned int>::max());
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void Qwen3_5DeltaLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != NUM_INPUTS, std::invalid_argument)
    << "Qwen3_5DeltaLayer requires mixed_qkv, z, beta and decay inputs";

  const auto &input_dims = context.getInputDimensions();
  const auto &mixed_dim = input_dims[MIXED_QKV];

  num_key_heads = std::get<qwen3_5_props::NumKeyHeads>(delta_props).get();
  num_value_heads = std::get<qwen3_5_props::NumValueHeads>(delta_props).get();
  key_head_dim = std::get<qwen3_5_props::KeyHeadDim>(delta_props).get();
  value_head_dim = std::get<qwen3_5_props::ValueHeadDim>(delta_props).get();
  conv_kernel_size = std::get<qwen3_5_props::ConvKernelSize>(delta_props).get();
  epsilon = std::get<nntrainer::props::Epsilon>(delta_props).get();

  NNTR_THROW_IF(num_value_heads % num_key_heads != 0, std::invalid_argument)
    << "num_value_heads must be divisible by num_key_heads";

  key_dim = num_key_heads * key_head_dim;
  value_dim = num_value_heads * value_head_dim;
  conv_dim = 2 * key_dim + value_dim;

  NNTR_THROW_IF(mixed_dim.width() != conv_dim ||
                  input_dims[Z_GATE].width() != value_dim ||
                  input_dims[BETA].width() != num_value_heads ||
                  input_dims[DECAY].width() != num_value_heads,
                std::invalid_argument)
    << "Qwen3_5DeltaLayer input widths do not match its head properties";

  for (unsigned int i = 1; i < NUM_INPUTS; ++i) {
    NNTR_THROW_IF(input_dims[i].batch() != mixed_dim.batch() ||
                    input_dims[i].height() != mixed_dim.height(),
                  std::invalid_argument)
      << "Qwen3_5DeltaLayer inputs must share batch and sequence dims";
  }

  const auto fp32_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), nntrainer::TensorDim::DataType::FP32);
  const auto &regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  const auto &regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  const auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  weight_idx[CONV_WEIGHT] = context.requestWeight(
    {1, 1, conv_kernel_size, conv_dim, fp32_type}, nntrainer::Initializer::NONE,
    regularizer, regularizer_constant, weight_decay, "conv_weight", false);
  weight_idx[DT_BIAS] = context.requestWeight(
    {1, 1, 1, num_value_heads, fp32_type}, nntrainer::Initializer::NONE,
    regularizer, regularizer_constant, weight_decay, "dt_bias", false);
  weight_idx[A_LOG] = context.requestWeight(
    {1, 1, 1, num_value_heads, fp32_type}, nntrainer::Initializer::NONE,
    regularizer, regularizer_constant, weight_decay, "A_log", false);
  weight_idx[NORM_WEIGHT] = context.requestWeight(
    {1, 1, 1, value_head_dim, fp32_type}, nntrainer::Initializer::NONE,
    regularizer, regularizer_constant, weight_decay, "norm_weight", false);

  tensor_idx[CONV_STATE] = context.requestTensor(
    {mixed_dim.batch(), 1, conv_kernel_size - 1, conv_dim, fp32_type},
    "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
  tensor_idx[RECURRENT_STATE] =
    context.requestTensor({mixed_dim.batch(), num_value_heads, key_head_dim,
                           value_head_dim, fp32_type},
                          "recurrent_state", nntrainer::Initializer::ZEROS,
                          false, nntrainer::TensorLifespan::MAX_LIFESPAN);

  auto output_dim = mixed_dim;
  output_dim.width(value_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});
}

void Qwen3_5DeltaLayer::runStep(nntrainer::RunLayerContext &context,
                                unsigned int step_size, bool reset_state) {
  auto &mixed = context.getInput(MIXED_QKV);
  auto &z_gate = context.getInput(Z_GATE);
  auto &beta_input = context.getInput(BETA);
  auto &decay_input = context.getInput(DECAY);
  auto &output = context.getOutput(0);

  NNTR_THROW_IF(mixed.getDataType() != nntrainer::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "Qwen3_5DeltaLayer currently requires FP32 activations";

  auto &conv_state = context.getTensor(tensor_idx[CONV_STATE]);
  auto &recurrent_state = context.getTensor(tensor_idx[RECURRENT_STATE]);
  if (reset_state) {
    conv_state.setZero();
    recurrent_state.setZero();
  }

  const float *conv_weight =
    context.getWeight(weight_idx[CONV_WEIGHT]).getData<float>();
  const float *dt_bias =
    context.getWeight(weight_idx[DT_BIAS]).getData<float>();
  const float *a_log = context.getWeight(weight_idx[A_LOG]).getData<float>();
  const float *norm_weight =
    context.getWeight(weight_idx[NORM_WEIGHT]).getData<float>();

  const float *mixed_data = mixed.getData<float>();
  const float *z_data = z_gate.getData<float>();
  const float *beta_data = beta_input.getData<float>();
  const float *decay_data = decay_input.getData<float>();
  float *output_data = output.getData<float>();
  float *conv_state_data = conv_state.getData<float>();
  float *recurrent_data = recurrent_state.getData<float>();

  const unsigned int batch_size = mixed.batch();
  const unsigned int input_height = mixed.height();
  const unsigned int output_height = output.height();
  const unsigned int key_repeat = num_value_heads / num_key_heads;
  const size_t conv_state_stride =
    static_cast<size_t>(conv_kernel_size - 1) * conv_dim;
  const size_t recurrent_head_stride =
    static_cast<size_t>(key_head_dim) * value_head_dim;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(key_head_dim));

  std::vector<float> convolved(conv_dim);
  std::vector<float> normalized_q(key_head_dim);
  std::vector<float> normalized_k(key_head_dim);
  std::vector<float> delta(value_head_dim);

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    float *batch_conv_state =
      conv_state_data + static_cast<size_t>(batch) * conv_state_stride;

    for (unsigned int token = 0; token < step_size; ++token) {
      const size_t mixed_offset =
        (static_cast<size_t>(batch) * input_height + token) * conv_dim;
      const size_t value_offset =
        (static_cast<size_t>(batch) * input_height + token) * value_dim;
      const size_t head_offset =
        (static_cast<size_t>(batch) * input_height + token) * num_value_heads;
      const size_t output_offset =
        (static_cast<size_t>(batch) * output_height + token) * value_dim;
      const float *current = mixed_data + mixed_offset;

      for (unsigned int channel = 0; channel < conv_dim; ++channel) {
        float sum = conv_weight[channel] * current[channel];
        for (unsigned int lag = 1; lag < conv_kernel_size; ++lag) {
          const unsigned int state_row = conv_kernel_size - 1 - lag;
          sum += conv_weight[static_cast<size_t>(lag) * conv_dim + channel] *
                 batch_conv_state[static_cast<size_t>(state_row) * conv_dim +
                                  channel];
        }
        convolved[channel] = silu(sum);
      }

      if (conv_kernel_size > 2) {
        std::move(batch_conv_state + conv_dim,
                  batch_conv_state + conv_state_stride, batch_conv_state);
      }
      std::copy(current, current + conv_dim,
                batch_conv_state +
                  static_cast<size_t>(conv_kernel_size - 2) * conv_dim);

      const float *query = convolved.data();
      const float *key = query + key_dim;
      const float *value = key + key_dim;

      for (unsigned int value_head = 0; value_head < num_value_heads;
           ++value_head) {
        const unsigned int key_head = value_head / key_repeat;
        const float *query_head = query + key_head * key_head_dim;
        const float *key_head_data = key + key_head * key_head_dim;
        const float *value_head_data = value + value_head * value_head_dim;

        float q_norm_sq = 0.0f;
        float k_norm_sq = 0.0f;
        for (unsigned int i = 0; i < key_head_dim; ++i) {
          q_norm_sq += query_head[i] * query_head[i];
          k_norm_sq += key_head_data[i] * key_head_data[i];
        }
        const float q_inv = 1.0f / std::sqrt(q_norm_sq + 1e-6f);
        const float k_inv = 1.0f / std::sqrt(k_norm_sq + 1e-6f);
        for (unsigned int i = 0; i < key_head_dim; ++i) {
          normalized_q[i] = query_head[i] * q_inv * query_scale;
          normalized_k[i] = key_head_data[i] * k_inv;
        }

        const float beta = sigmoid(beta_data[head_offset + value_head]);
        const float decay =
          -std::exp(a_log[value_head]) *
          softplus(decay_data[head_offset + value_head] + dt_bias[value_head]);
        const float decay_factor = std::exp(decay);
        float *state =
          recurrent_data +
          (static_cast<size_t>(batch) * num_value_heads + value_head) *
            recurrent_head_stride;

        for (size_t i = 0; i < recurrent_head_stride; ++i)
          state[i] *= decay_factor;

        for (unsigned int v = 0; v < value_head_dim; ++v) {
          float memory_value = 0.0f;
          for (unsigned int k = 0; k < key_head_dim; ++k)
            memory_value += state[static_cast<size_t>(k) * value_head_dim + v] *
                            normalized_k[k];
          delta[v] = (value_head_data[v] - memory_value) * beta;
        }

        for (unsigned int k = 0; k < key_head_dim; ++k)
          for (unsigned int v = 0; v < value_head_dim; ++v)
            state[static_cast<size_t>(k) * value_head_dim + v] +=
              normalized_k[k] * delta[v];

        float variance = 0.0f;
        float *output_head =
          output_data + output_offset + value_head * value_head_dim;
        for (unsigned int v = 0; v < value_head_dim; ++v) {
          float result = 0.0f;
          for (unsigned int k = 0; k < key_head_dim; ++k)
            result += state[static_cast<size_t>(k) * value_head_dim + v] *
                      normalized_q[k];
          output_head[v] = result;
          variance += result * result;
        }

        const float inv_rms =
          1.0f / std::sqrt(variance / value_head_dim + epsilon);
        const float *z_head =
          z_data + value_offset + value_head * value_head_dim;
        for (unsigned int v = 0; v < value_head_dim; ++v)
          output_head[v] =
            output_head[v] * inv_rms * norm_weight[v] * silu(z_head[v]);
      }
    }
  }
}

void Qwen3_5DeltaLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  NNTR_THROW_IF(training, std::invalid_argument)
    << "Qwen3_5DeltaLayer supports inference only";
  runStep(context, context.getInput(MIXED_QKV).height(), true);
}

void Qwen3_5DeltaLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  NNTR_THROW_IF(training || to <= from, std::invalid_argument)
    << "Qwen3_5DeltaLayer received an invalid inference range";
  runStep(context, to - from, from == 0);
}

void Qwen3_5DeltaLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("Qwen3_5DeltaLayer does not support backwarding");
}

void Qwen3_5DeltaLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("Qwen3_5DeltaLayer does not support gradients");
}

void Qwen3_5DeltaLayer::setProperty(const std::vector<std::string> &values) {
  auto remain = loadProperties(values, delta_props);
  LayerImpl::setProperty(remain);
}

void Qwen3_5DeltaLayer::exportTo(nntrainer::Exporter &exporter,
                                 const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(delta_props, method, this);
}

} // namespace causallm
