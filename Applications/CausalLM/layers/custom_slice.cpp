// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   custom_slice.cpp
 * @date   02 April 2026
 * @brief  Custom slice layer for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <custom_slice.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

namespace causallm {

void CustomSliceLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "CustomSliceLayer requires exactly 1 input";

  axis = std::get<nntrainer::props::Axis>(slice_props).get();
  start = std::get<nntrainer::props::StartIndex>(slice_props).get() - 1;
  unsigned int end = std::get<nntrainer::props::EndIndex>(slice_props).get() - 1;

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  nntrainer::TensorDim out_dim = in_dim;

  NNTR_THROW_IF(axis >= ml::train::TensorDim::MAXDIM, std::invalid_argument)
    << "CustomSliceLayer: invalid axis " << axis;

  NNTR_THROW_IF(end < start, std::invalid_argument)
    << "CustomSliceLayer: end_index must be greater than start_index";

  NNTR_THROW_IF(end > in_dim.getTensorDim(axis), std::invalid_argument)
    << "CustomSliceLayer: end_index exceeds input dimension size";

  out_dim.setTensorDim(axis, end - start);
  context.setOutputDimensions({out_dim});
}

void CustomSliceLayer::forwarding(nntrainer::RunLayerContext &context,
                                  bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const nntrainer::TensorDim &in_dim = input.getDim();
  const nntrainer::TensorDim &out_dim = output.getDim();

  // Optimized implementation using getSharedDataTensor or memcpy
  if (axis == 3) {
    // Slice on width axis: [B, C, H, W] -> [B, C, H, out_dim.width()]
    // Use getSharedDataTensor for contiguous slice
    size_t offset = static_cast<size_t>(start);
    nntrainer::Tensor sliced = input.getSharedDataTensor(
      nntrainer::TensorDim(in_dim.batch(), in_dim.channel(), in_dim.height(),
                           out_dim.width()),
      offset);
    output.copy(sliced);
  } else if (axis == 2) {
    // Slice on height axis: [B, C, H, W] -> [B, C, out_dim.height(), W]
    // Use getSharedDataTensor for contiguous slice
    size_t offset = static_cast<size_t>(start) * in_dim.width();
    nntrainer::Tensor sliced = input.getSharedDataTensor(
      nntrainer::TensorDim(in_dim.batch(), in_dim.channel(), out_dim.height(),
                           in_dim.width()),
      offset);
    output.copy(sliced);
  } else if (axis == 1) {
    // Slice on channel axis: [B, C, H, W] -> [B, out_dim.channel(), H, W]
    size_t offset = static_cast<size_t>(start) * in_dim.height() * in_dim.width();
    nntrainer::Tensor sliced = input.getSharedDataTensor(
      nntrainer::TensorDim(in_dim.batch(), out_dim.channel(), in_dim.height(),
                           in_dim.width()),
      offset);
    output.copy(sliced);
  } else {
    // Fallback for axis 0 (batch) - rare case
    for (unsigned int b = 0; b < out_dim.batch(); ++b) {
      for (unsigned int c = 0; c < out_dim.channel(); ++c) {
        for (unsigned int h = 0; h < out_dim.height(); ++h) {
          for (unsigned int w = 0; w < out_dim.width(); ++w) {
            unsigned int b_idx = (axis == 0) ? b + start : b;
            unsigned int c_idx = (axis == 1) ? c + start : c;
            unsigned int h_idx = (axis == 2) ? h + start : h;
            unsigned int w_idx = (axis == 3) ? w + start : w;
            output.setValue(b, c, h, w, input.getValue(b_idx, c_idx, h_idx, w_idx));
          }
        }
      }
    }
  }
}

void CustomSliceLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  NNTR_THROW_IF(to <= from, std::invalid_argument)
    << "CustomSliceLayer::incremental_forwarding requires to > from";

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const nntrainer::TensorDim &in_dim = input.getDim();
  const nntrainer::TensorDim &out_dim = output.getDim();

  const unsigned int batch_size = output.batch();
  const unsigned int step_height = to - from;

  // Pre-compute dimensions
  const size_t in_feature_len = in_dim.getFeatureLen();
  const size_t out_feature_len = out_dim.getFeatureLen();
  const size_t in_width = in_dim.width();
  const size_t out_width = out_dim.width();

  // Slice dimension for each step
  nntrainer::TensorDim in_step_dim = in_dim;
  in_step_dim.batch(1);
  in_step_dim.height(step_height);

  nntrainer::TensorDim out_step_dim = out_dim;
  out_step_dim.batch(1);
  out_step_dim.height(step_height);

  // Handle different slice axes
  if (axis == 3) {
    // Slice on width: input [B, C, H, W] -> output [B, C, H, out_width]
    // In incremental mode: input [B, C, step_height, W] -> output [B, C, step_height, out_width]
    in_step_dim.width(in_width);
    out_step_dim.width(out_width);

    for (unsigned int b = 0; b < batch_size; ++b) {
      // Input offset: batch_offset + from * width
      const size_t in_offset = b * in_feature_len +
                               static_cast<size_t>(from) * in_width;
      // Output offset: batch_offset + from * out_width
      const size_t out_offset = b * out_feature_len +
                                static_cast<size_t>(from) * out_width;

      nntrainer::Tensor in_step = input.getSharedDataTensor(in_step_dim, in_offset, true);
      nntrainer::Tensor out_step = output.getSharedDataTensor(out_step_dim, out_offset, true);

      // Slice on width: take [start, start+out_width) from input
      nntrainer::Tensor sliced = in_step.getSharedDataTensor(
        nntrainer::TensorDim(1, in_dim.channel(), step_height, out_width),
        static_cast<size_t>(start), true);
      out_step.copy(sliced);
    }
  } else if (axis == 2) {
    // Slice on height: input [B, C, H, W] -> output [B, C, out_height, W]
    // In incremental mode, this is different - we're slicing on height which is also
    // the sequence dimension being processed incrementally.
    // For simplicity, just call forwarding since slice axis matches incremental axis
    // forwarding(context, training);
  } else if (axis == 1) {
    // Slice on channel: input [B, C, H, W] -> output [B, out_channel, H, W]
    // In incremental mode: slice both height (from incremental) and channel (from slice)
    in_step_dim.channel(in_dim.channel());
    out_step_dim.channel(out_dim.channel());

    for (unsigned int b = 0; b < batch_size; ++b) {
      // Input offset: batch_offset + from * width
      const size_t in_offset = b * in_feature_len +
                               static_cast<size_t>(from) * in_width;
      // Output offset: batch_offset + from * out_width
      const size_t out_offset = b * out_feature_len +
                                static_cast<size_t>(from) * out_width;

      // Slice on channel: channel_offset = start * H * W
      const size_t channel_offset = static_cast<size_t>(start) *
                                    in_dim.height() * in_dim.width();

      nntrainer::Tensor in_step = input.getSharedDataTensor(in_step_dim, in_offset, true);
      nntrainer::Tensor out_step = output.getSharedDataTensor(out_step_dim, out_offset, true);

      // Create sliced view with correct channel
      nntrainer::Tensor sliced = input.getSharedDataTensor(
        nntrainer::TensorDim(1, out_dim.channel(), step_height, in_width),
        b * in_feature_len + channel_offset + static_cast<size_t>(from) * in_width, true);
      out_step.copy(sliced);
    }
  } else {
    // axis == 0 (batch) - just call forwarding
    // forwarding(context, training);
  }
}

void CustomSliceLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  const nntrainer::Tensor &in_deriv =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out_deriv =
    context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  for (unsigned int b = 0; b < in_deriv.batch(); ++b) {
    for (unsigned int c = 0; c < in_deriv.channel(); ++c) {
      for (unsigned int h = 0; h < in_deriv.height(); ++h) {
        for (unsigned int w = 0; w < in_deriv.width(); ++w) {
          unsigned int c_idx = (axis == 1) ? c + start : c;
          unsigned int h_idx = (axis == 2) ? h + start : h;
          unsigned int w_idx = (axis == 3) ? w + start : w;
          out_deriv.setValue(b, c_idx, h_idx, w_idx,
                             in_deriv.getValue(b, c, h, w));
        }
      }
    }
  }
}

void CustomSliceLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, slice_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[CustomSliceLayer] Unknown Layer Properties count "
    << std::to_string(values.size());
}

#ifdef PLUGGABLE

nntrainer::Layer *create_custom_slice_layer() {
  auto layer = new CustomSliceLayer();
  return layer;
}

void destroy_custom_slice_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_custom_slice_layer,
  destroy_custom_slice_layer};
}

#endif

} // namespace causallm
