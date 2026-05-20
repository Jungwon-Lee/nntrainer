// SPDX-License-Identifier: Apache-2.0
/**
 * @file   siglip2_layers.cpp
 * @brief  Helper layers for SigLIP2 NaFlex vision encoder.
 */

#include <siglip2_layers.h>

#include <cmath>
#include <limits>
#include <nntrainer_error.h>
#include <stdexcept>
#include <util_func.h>

namespace causallm {

Siglip2PositionalEmbeddingLayer::Siglip2PositionalEmbeddingLayer() :
  nntrainer::Layer(),
  layer_props(props::BaseGridSize()),
  position_embedding_idx(std::numeric_limits<unsigned int>::max()) {}

static float clamp_source_index(float source, int limit, int &left,
                                int &right) {
  const int lower = static_cast<int>(std::floor(source));
  const float weight_right = source - static_cast<float>(lower);
  left = std::max(0, std::min(lower, limit - 1));
  right = std::max(0, std::min(lower + 1, limit - 1));
  return weight_right;
}

void Siglip2PositionalEmbeddingLayer::finalize(
  nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "Siglip2PositionalEmbeddingLayer requires patch embeddings and "
       "spatial_shapes";

  const auto &input_dims = context.getInputDimensions();
  const auto &patch_dim = input_dims[PATCH_EMBEDS];
  const auto &spatial_dim = input_dims[SPATIAL_SHAPES];

  NNTR_THROW_IF(patch_dim.channel() != 1, std::invalid_argument)
    << "patch embeddings must have channel=1";
  NNTR_THROW_IF(spatial_dim.width() != 2, std::invalid_argument)
    << "spatial_shapes must have width=2";

  auto &base_grid_prop = std::get<props::BaseGridSize>(layer_props);
  NNTR_THROW_IF(base_grid_prop.empty(), std::invalid_argument)
    << "base_grid_size must be provided";
  const unsigned int base_grid_size = base_grid_prop.get();

  nntrainer::TensorDim weight_dim(
    {1, 1, base_grid_size * base_grid_size, patch_dim.width()},
    {context.getFormat(), context.getWeightDataType()});
  position_embedding_idx =
    context.requestWeight(weight_dim, nntrainer::Initializer::NONE,
                          nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
                          "position_embedding", true);

  context.setOutputDimensions({patch_dim});
}

void Siglip2PositionalEmbeddingLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {
  (void)training;

  const nntrainer::Tensor &patch_embeds = context.getInput(PATCH_EMBEDS);
  const nntrainer::Tensor &spatial_shapes = context.getInput(SPATIAL_SHAPES);
  const nntrainer::Tensor &position_embedding =
    context.getWeight(position_embedding_idx);
  nntrainer::Tensor &output = context.getOutput(OUTPUT);

  output.copyData(patch_embeds);

  const auto &base_grid_prop = std::get<props::BaseGridSize>(layer_props);
  const int base_grid_size = static_cast<int>(base_grid_prop.get());
  const unsigned int max_patches = patch_embeds.height();
  const unsigned int hidden_size = patch_embeds.width();

  for (unsigned int b = 0; b < patch_embeds.batch(); ++b) {
    const int target_h = static_cast<int>(
      std::round(spatial_shapes.getValue<float>(b, 0, 0, 0)));
    const int target_w = static_cast<int>(
      std::round(spatial_shapes.getValue<float>(b, 0, 0, 1)));

    NNTR_THROW_IF(target_h <= 0 || target_w <= 0, std::invalid_argument)
      << "spatial_shapes must be positive";
    NNTR_THROW_IF(static_cast<unsigned int>(target_h * target_w) > max_patches,
                  std::invalid_argument)
      << "resized position embeddings exceed max_patches";

    const float scale_y =
      static_cast<float>(base_grid_size) / static_cast<float>(target_h);
    const float scale_x =
      static_cast<float>(base_grid_size) / static_cast<float>(target_w);

    for (int y = 0; y < target_h; ++y) {
      int y0, y1;
      const float wy =
        clamp_source_index((static_cast<float>(y) + 0.5f) * scale_y - 0.5f,
                           base_grid_size, y0, y1);
      for (int x = 0; x < target_w; ++x) {
        int x0, x1;
        const float wx =
          clamp_source_index((static_cast<float>(x) + 0.5f) * scale_x - 0.5f,
                             base_grid_size, x0, x1);
        const unsigned int out_idx =
          static_cast<unsigned int>(y * target_w + x);

        const unsigned int idx00 =
          static_cast<unsigned int>(y0 * base_grid_size + x0);
        const unsigned int idx01 =
          static_cast<unsigned int>(y0 * base_grid_size + x1);
        const unsigned int idx10 =
          static_cast<unsigned int>(y1 * base_grid_size + x0);
        const unsigned int idx11 =
          static_cast<unsigned int>(y1 * base_grid_size + x1);

        for (unsigned int d = 0; d < hidden_size; ++d) {
          const float v00 = position_embedding.getValue<float>(0, 0, idx00, d);
          const float v01 = position_embedding.getValue<float>(0, 0, idx01, d);
          const float v10 = position_embedding.getValue<float>(0, 0, idx10, d);
          const float v11 = position_embedding.getValue<float>(0, 0, idx11, d);
          const float top = v00 * (1.0f - wx) + v01 * wx;
          const float bottom = v10 * (1.0f - wx) + v11 * wx;
          const float pos = top * (1.0f - wy) + bottom * wy;
          const float current = output.getValue<float>(b, 0, out_idx, d);
          output.setValue(b, 0, out_idx, d, current + pos);
        }
      }
    }

    for (unsigned int idx = static_cast<unsigned int>(target_h * target_w);
         idx < max_patches; ++idx) {
      for (unsigned int d = 0; d < hidden_size; ++d) {
        const float current = output.getValue<float>(b, 0, idx, d);
        const float pos = output.getValue<float>(b, 0, 0, d) -
                          patch_embeds.getValue<float>(b, 0, 0, d);
        output.setValue(b, 0, idx, d, current + pos);
      }
    }
  }
}

void Siglip2PositionalEmbeddingLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error("Siglip2PositionalEmbeddingLayer is inference-only");
}

void Siglip2PositionalEmbeddingLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, layer_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[Siglip2PositionalEmbeddingLayer] Unknown Layer Properties count "
    << std::to_string(remain_props.size());
}

void Siglip2PositionalEmbeddingLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(layer_props, method, this);
}

Siglip2AttentionMaskLayer::Siglip2AttentionMaskLayer() :
  nntrainer::Layer(),
  layer_props(nntrainer::props::NumHeads(), props::MaskValue()) {}

void Siglip2AttentionMaskLayer::finalize(
  nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Siglip2AttentionMaskLayer requires one input";

  const auto &mask_dim = context.getInputDimensions()[PIXEL_ATTENTION_MASK];
  NNTR_THROW_IF(mask_dim.channel() != 1 || mask_dim.height() != 1,
                std::invalid_argument)
    << "pixel_attention_mask must have shape [B, 1, 1, max_patches]";

  auto &num_heads_prop = std::get<nntrainer::props::NumHeads>(layer_props);
  NNTR_THROW_IF(num_heads_prop.empty(), std::invalid_argument)
    << "num_heads must be provided";

  nntrainer::TensorDim output_dim(
    {mask_dim.batch(), num_heads_prop.get(), mask_dim.width(),
     mask_dim.width()},
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});
}

void Siglip2AttentionMaskLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {
  (void)training;

  const nntrainer::Tensor &pixel_attention_mask =
    context.getInput(PIXEL_ATTENTION_MASK);
  nntrainer::Tensor &output = context.getOutput(OUTPUT);

  const float mask_value = std::get<props::MaskValue>(layer_props).get();
  const unsigned int max_patches = pixel_attention_mask.width();

  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int h = 0; h < output.channel(); ++h) {
      for (unsigned int q = 0; q < max_patches; ++q) {
        for (unsigned int k = 0; k < max_patches; ++k) {
          const float valid =
            pixel_attention_mask.getValue<float>(b, 0, 0, k);
          output.setValue(b, h, q, k, valid > 0.5f ? 0.0f : mask_value);
        }
      }
    }
  }
}

void Siglip2AttentionMaskLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error("Siglip2AttentionMaskLayer is inference-only");
}

void Siglip2AttentionMaskLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, layer_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[Siglip2AttentionMaskLayer] Unknown Layer Properties count "
    << std::to_string(remain_props.size());
}

void Siglip2AttentionMaskLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(layer_props, method, this);
}

Lfm2VlPixelUnshuffleLayer::Lfm2VlPixelUnshuffleLayer() :
  nntrainer::Layer(),
  layer_props(props::DownsampleFactor()) {}

void Lfm2VlPixelUnshuffleLayer::finalize(
  nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "Lfm2VlPixelUnshuffleLayer requires hidden states and spatial_shapes";

  const auto &hidden_dim = context.getInputDimensions()[HIDDEN_STATES];
  const auto &spatial_dim = context.getInputDimensions()[SPATIAL_SHAPES];
  const unsigned int factor =
    std::get<props::DownsampleFactor>(layer_props).get();

  NNTR_THROW_IF(hidden_dim.channel() != 1, std::invalid_argument)
    << "hidden states must have channel=1";
  NNTR_THROW_IF(spatial_dim.width() != 2, std::invalid_argument)
    << "spatial_shapes must have width=2";
  NNTR_THROW_IF(hidden_dim.height() % (factor * factor) != 0,
                std::invalid_argument)
    << "max_patches must be divisible by downsample_factor^2";

  nntrainer::TensorDim output_dim(
    {hidden_dim.batch(), 1, hidden_dim.height() / (factor * factor),
     hidden_dim.width() * factor * factor},
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});
}

void Lfm2VlPixelUnshuffleLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {
  (void)training;

  const nntrainer::Tensor &hidden_states = context.getInput(HIDDEN_STATES);
  const nntrainer::Tensor &spatial_shapes = context.getInput(SPATIAL_SHAPES);
  nntrainer::Tensor &output = context.getOutput(OUTPUT);

  output.setValue(0.0f);

  const unsigned int factor =
    std::get<props::DownsampleFactor>(layer_props).get();
  const unsigned int hidden_size = hidden_states.width();
  const unsigned int max_output_tokens = output.height();

  for (unsigned int b = 0; b < hidden_states.batch(); ++b) {
    const unsigned int input_h = static_cast<unsigned int>(
      std::round(spatial_shapes.getValue<float>(b, 0, 0, 0)));
    const unsigned int input_w = static_cast<unsigned int>(
      std::round(spatial_shapes.getValue<float>(b, 0, 0, 1)));

    NNTR_THROW_IF(input_h == 0 || input_w == 0, std::invalid_argument)
      << "spatial_shapes must be positive";
    NNTR_THROW_IF(input_h % factor != 0 || input_w % factor != 0,
                  std::invalid_argument)
      << "spatial_shapes must be divisible by downsample_factor";

    const unsigned int output_h = input_h / factor;
    const unsigned int output_w = input_w / factor;
    NNTR_THROW_IF(output_h * output_w > max_output_tokens,
                  std::invalid_argument)
      << "pixel-unshuffle output exceeds configured max tokens";

    for (unsigned int out_y = 0; out_y < output_h; ++out_y) {
      for (unsigned int out_x = 0; out_x < output_w; ++out_x) {
        const unsigned int out_idx = out_y * output_w + out_x;
        for (unsigned int dy = 0; dy < factor; ++dy) {
          for (unsigned int dx = 0; dx < factor; ++dx) {
            const unsigned int in_y = out_y * factor + dy;
            const unsigned int in_x = out_x * factor + dx;
            const unsigned int in_idx = in_y * input_w + in_x;
            const unsigned int out_channel_base =
              (dy * factor + dx) * hidden_size;

            for (unsigned int c = 0; c < hidden_size; ++c) {
              output.setValue(
                b, 0, out_idx, out_channel_base + c,
                hidden_states.getValue<float>(b, 0, in_idx, c));
            }
          }
        }
      }
    }
  }
}

void Lfm2VlPixelUnshuffleLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error("Lfm2VlPixelUnshuffleLayer is inference-only");
}

void Lfm2VlPixelUnshuffleLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, layer_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[Lfm2VlPixelUnshuffleLayer] Unknown Layer Properties count "
    << std::to_string(remain_props.size());
}

void Lfm2VlPixelUnshuffleLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(layer_props, method, this);
}

Lfm2VlImageEmbeddingMergeLayer::Lfm2VlImageEmbeddingMergeLayer() :
  nntrainer::Layer(),
  layer_props(props::ImageTokenId()) {}

void Lfm2VlImageEmbeddingMergeLayer::finalize(
  nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 3, std::invalid_argument)
    << "Lfm2VlImageEmbeddingMergeLayer requires input_ids, text embeddings, "
       "and image features";

  const auto &input_ids_dim = context.getInputDimensions()[INPUT_IDS];
  const auto &text_dim = context.getInputDimensions()[TEXT_EMBEDS];
  const auto &image_dim = context.getInputDimensions()[IMAGE_FEATURES];

  NNTR_THROW_IF(input_ids_dim.channel() != 1 || input_ids_dim.height() != 1,
                std::invalid_argument)
    << "input_ids must have shape [B, 1, 1, sequence_length]";
  NNTR_THROW_IF(text_dim.channel() != 1 || image_dim.channel() != 1,
                std::invalid_argument)
    << "embedding tensors must have channel=1";
  NNTR_THROW_IF(input_ids_dim.batch() != text_dim.batch() ||
                  input_ids_dim.batch() != image_dim.batch(),
                std::invalid_argument)
    << "input_ids, text embeddings, and image features batch sizes must match";
  NNTR_THROW_IF(input_ids_dim.width() != text_dim.height(),
                std::invalid_argument)
    << "input_ids width must match text embedding sequence length";
  NNTR_THROW_IF(text_dim.width() != image_dim.width(), std::invalid_argument)
    << "text embeddings and image features hidden sizes must match";

  auto &image_token_id_prop = std::get<props::ImageTokenId>(layer_props);
  NNTR_THROW_IF(image_token_id_prop.empty(), std::invalid_argument)
    << "image_token_id must be provided";

  context.setOutputDimensions({text_dim});
}

void Lfm2VlImageEmbeddingMergeLayer::mergeRange(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to) {
  const nntrainer::Tensor &input_ids = context.getInput(INPUT_IDS);
  const nntrainer::Tensor &text_embeds = context.getInput(TEXT_EMBEDS);
  const nntrainer::Tensor &image_features = context.getInput(IMAGE_FEATURES);
  nntrainer::Tensor &output = context.getOutput(OUTPUT);

  const unsigned int image_token_id =
    std::get<props::ImageTokenId>(layer_props).get();
  const unsigned int sequence_length = text_embeds.height();
  const unsigned int hidden_size = text_embeds.width();

  NNTR_THROW_IF(to > sequence_length, std::invalid_argument)
    << "merge range exceeds sequence length";

  for (unsigned int b = 0; b < text_embeds.batch(); ++b) {
    unsigned int image_feature_idx = 0;
    for (unsigned int i = 0; i < from; ++i) {
      const unsigned int token_id = static_cast<unsigned int>(
        std::round(input_ids.getValue<float>(b, 0, 0, i)));
      if (token_id == image_token_id) {
        ++image_feature_idx;
      }
    }

    for (unsigned int i = from; i < to; ++i) {
      const unsigned int token_id = static_cast<unsigned int>(
        std::round(input_ids.getValue<float>(b, 0, 0, i)));
      const bool is_image_token = token_id == image_token_id;

      NNTR_THROW_IF(is_image_token &&
                      image_feature_idx >= image_features.height(),
                    std::invalid_argument)
        << "not enough image features for image placeholder tokens";

      for (unsigned int d = 0; d < hidden_size; ++d) {
        const float value =
          is_image_token
            ? image_features.getValue<float>(b, 0, image_feature_idx, d)
            : text_embeds.getValue<float>(b, 0, i, d);
        output.setValue(b, 0, i, d, value);
      }

      if (is_image_token) {
        ++image_feature_idx;
      }
    }
  }
}

void Lfm2VlImageEmbeddingMergeLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {
  (void)training;
  mergeRange(context, 0, context.getInput(TEXT_EMBEDS).height());
}

void Lfm2VlImageEmbeddingMergeLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  (void)training;
  mergeRange(context, from, to);
}

void Lfm2VlImageEmbeddingMergeLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "Lfm2VlImageEmbeddingMergeLayer is inference-only");
}

void Lfm2VlImageEmbeddingMergeLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, layer_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[Lfm2VlImageEmbeddingMergeLayer] Unknown Layer Properties count "
    << std::to_string(remain_props.size());
}

void Lfm2VlImageEmbeddingMergeLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(layer_props, method, this);
}

#ifdef PLUGGABLE
nntrainer::Layer *create_siglip2_positional_embedding_layer() {
  return new Siglip2PositionalEmbeddingLayer();
}

nntrainer::Layer *create_siglip2_attention_mask_layer() {
  return new Siglip2AttentionMaskLayer();
}

nntrainer::Layer *create_lfm2_vl_pixel_unshuffle_layer() {
  return new Lfm2VlPixelUnshuffleLayer();
}

nntrainer::Layer *create_lfm2_vl_image_embedding_merge_layer() {
  return new Lfm2VlImageEmbeddingMergeLayer();
}
#endif

} // namespace causallm
