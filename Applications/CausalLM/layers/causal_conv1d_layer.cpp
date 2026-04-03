// SPDX-License-Identifier: Apache-2.0
/**
* Copyright (C) 2026 Hyeong-Gwon Hong
*
* @file   causal_conv1d_layer.cpp
* @date   01 April 2026
* @brief  Causal depthwise Conv1D layer for CausalLM
* @see    https://github.com/nntrainer/nntrainer
* @author Hyeong-Gwon Hong
* @bug    No known bugs except for NYI items
*
*/

#include "causal_conv1d_layer.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <cpu_backend.h>
#include <fp16.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace causallm {

CausalConv1DLayer::CausalConv1DLayer() : LayerImpl() {
  weight_idx.fill(std::numeric_limits<unsigned int>::max());
}

void CausalConv1DLayer::validateInputShape(
  const nntrainer::TensorDim &input_dim) const {
  NNTR_THROW_IF(input_dim.rank() != 4, std::invalid_argument)
    << "[CausalConv1DLayer] input rank must be 4, but got "
    << input_dim.rank();

  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] input channel must be 1 for Bx1xHxW layout, but got "
    << input_dim.channel();

  NNTR_THROW_IF(input_dim.height() < 1 || input_dim.width() < 1,
                std::invalid_argument)
    << "[CausalConv1DLayer] invalid input shape: H and W must be positive.";

  NNTR_THROW_IF(input_dim.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[CausalConv1DLayer] input dtype must be FP32.";
}

void CausalConv1DLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  NNTR_THROW_IF(training, std::invalid_argument)
    << "[CausalConv1DLayer] training/backward is not supported yet.";

  NNTR_THROW_IF(to == 0 || to <= from, std::invalid_argument)
    << "[CausalConv1DLayer] invalid incremental range: from=" << from
    << ", to=" << to;

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor weight_tensor = context.getWeight(weight_idx[weight]);

  const nntrainer::TensorDim &in_dim = input.getDim();

  const unsigned int B = in_dim.batch();
  const unsigned int H = in_dim.height();
  const unsigned int W = in_dim.width();

  NNTR_THROW_IF(input.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[CausalConv1DLayer] input must be FP32.";

  NNTR_THROW_IF(output.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[CausalConv1DLayer] output must be FP32.";

  NNTR_THROW_IF(to > H, std::invalid_argument)
    << "[CausalConv1DLayer] invalid incremental end: to=" << to
    << ", H=" << H;

  float *input_ptr = input.getData<float>();
  const uint16_t *weight_ptr = weight_tensor.getData<uint16_t>();
  float *output_ptr = output.getData<float>();

  nntrainer::causal_depthwise_conv1d_k3_fp16(
    input_ptr, weight_ptr, output_ptr, B, to, W);
}

void CausalConv1DLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  throw std::runtime_error(
    "[CausalConv1DLayer] forwarding() is not used. "
    "Use incremental_forwarding().");
}

void CausalConv1DLayer::finalize(nntrainer::InitLayerContext &context) {
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] requires exactly 1 input, but got "
    << context.getNumInputs();

  const nntrainer::TensorDim &input_dim = context.getInputDimensions()[0];
  validateInputShape(input_dim);

  const unsigned int W = input_dim.width();

  ml::train::TensorDim::TensorType weight_type(
    context.getFormat(), context.getWeightDataType());

  nntrainer::TensorDim weight_dim(1, 1, KERNEL_SIZE, W, weight_type);

  nntrainer::TensorDim output_dim = input_dim;
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  output_dim.setDataType(ml::train::TensorDim::DataType::FP32);

  context.setOutputDimensions({output_dim});

  weight_idx[weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay,
    "causal_conv1d_weight", true);
}

void CausalConv1DLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "[CausalConv1DLayer] calcDerivative() is not implemented. "
    "This layer is inference-only for now.");
}

void CausalConv1DLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "[CausalConv1DLayer] calcGradient() is not implemented. "
    "This layer is inference-only for now.");
}

void CausalConv1DLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
}

} // namespace causallm

#ifdef PLUGGABLE
extern "C" {

nntrainer::Layer *create_causal_conv1d_layer() {
  return new causallm::CausalConv1DLayer();
}

void destroy_causal_conv1d_layer(nntrainer::Layer *layer) {
  delete layer;
}

nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_causal_conv1d_layer,
  destroy_causal_conv1d_layer,
  causallm::CausalConv1DLayer::type
};

}
#endif