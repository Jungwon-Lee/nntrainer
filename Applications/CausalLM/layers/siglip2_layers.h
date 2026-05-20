// SPDX-License-Identifier: Apache-2.0
/**
 * @file   siglip2_layers.h
 * @brief  Helper layers for SigLIP2 NaFlex vision encoder.
 */

#ifndef __SIGLIP2_LAYERS_H__
#define __SIGLIP2_LAYERS_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace causallm {

/**
 * @brief Add SigLIP2 resized 2D position embeddings to patch embeddings.
 *
 * Inputs:
 *   0. patch embeddings: [B, 1, max_patches, hidden_size]
 *   1. spatial_shapes:  [B, 1, 1, 2] as float values (height, width)
 */
WIN_EXPORT class Siglip2PositionalEmbeddingLayer final
  : public nntrainer::Layer {
public:
  WIN_EXPORT Siglip2PositionalEmbeddingLayer();
  WIN_EXPORT ~Siglip2PositionalEmbeddingLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  WIN_EXPORT const std::string getType() const override {
    return Siglip2PositionalEmbeddingLayer::type;
  }

  inline static const std::string type = "siglip2_positional_embedding";

private:
  enum InputIndex { PATCH_EMBEDS = 0, SPATIAL_SHAPES = 1 };
  enum OutputIndex { OUTPUT = 0 };
  enum WeightIndex { POSITION_EMBEDDING = 0 };

  std::tuple<props::BaseGridSize> layer_props;
  unsigned int position_embedding_idx;
};

/**
 * @brief Convert pixel_attention_mask to additive multi-head attention bias.
 *
 * Input:
 *   pixel_attention_mask: [B, 1, 1, max_patches], 1 for valid patches.
 * Output:
 *   attention bias: [B, num_heads, max_patches, max_patches]
 */
WIN_EXPORT class Siglip2AttentionMaskLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT Siglip2AttentionMaskLayer();
  WIN_EXPORT ~Siglip2AttentionMaskLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  WIN_EXPORT const std::string getType() const override {
    return Siglip2AttentionMaskLayer::type;
  }

  inline static const std::string type = "siglip2_attention_mask";

private:
  enum InputIndex { PIXEL_ATTENTION_MASK = 0 };
  enum OutputIndex { OUTPUT = 0 };

  std::tuple<nntrainer::props::NumHeads, props::MaskValue> layer_props;
};

/**
 * @brief LFM2-VL projector pixel-unshuffle for NaFlex vision features.
 *
 * Inputs:
 *   0. hidden states:   [B, 1, max_patches, hidden_size]
 *   1. spatial_shapes: [B, 1, 1, 2] as float values (height, width)
 * Output:
 *   unshuffled features:
 *     [B, 1, max_patches / factor^2, hidden_size * factor^2]
 */
WIN_EXPORT class Lfm2VlPixelUnshuffleLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT Lfm2VlPixelUnshuffleLayer();
  WIN_EXPORT ~Lfm2VlPixelUnshuffleLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  WIN_EXPORT const std::string getType() const override {
    return Lfm2VlPixelUnshuffleLayer::type;
  }

  inline static const std::string type = "lfm2_vl_pixel_unshuffle";

private:
  enum InputIndex { HIDDEN_STATES = 0, SPATIAL_SHAPES = 1 };
  enum OutputIndex { OUTPUT = 0 };

  std::tuple<props::DownsampleFactor> layer_props;
};

/**
 * @brief Replace image placeholder token embeddings with projected features.
 *
 * Inputs:
 *   0. input_ids:       [B, 1, 1, sequence_length]
 *   1. text embeddings: [B, 1, sequence_length, hidden_size]
 *   2. image features:  [B, 1, max_image_tokens, hidden_size]
 * Output:
 *   merged embeddings:  [B, 1, sequence_length, hidden_size]
 */
WIN_EXPORT class Lfm2VlImageEmbeddingMergeLayer final
  : public nntrainer::Layer {
public:
  WIN_EXPORT Lfm2VlImageEmbeddingMergeLayer();
  WIN_EXPORT ~Lfm2VlImageEmbeddingMergeLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  WIN_EXPORT const std::string getType() const override {
    return Lfm2VlImageEmbeddingMergeLayer::type;
  }

  inline static const std::string type = "lfm2_vl_image_embedding_merge";

private:
  enum InputIndex { INPUT_IDS = 0, TEXT_EMBEDS = 1, IMAGE_FEATURES = 2 };
  enum OutputIndex { OUTPUT = 0 };

  void mergeRange(nntrainer::RunLayerContext &context, unsigned int from,
                  unsigned int to);

  std::tuple<props::ImageTokenId> layer_props;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __SIGLIP2_LAYERS_H__ */
