// SPDX-License-Identifier: Apache-2.0
/**
 * @file   siglip2_vision.cpp
 * @brief  SigLIP2 NaFlex vision encoder for LFM2-VL validation.
 */

#include "siglip2_vision.h"

#include <app_context.h>
#include <engine.h>
#include <factory.h>
#include <model.h>
#include <siglip2_layers.h>
#include <tensor_api.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using ml::train::LayerHandle;
using ml::train::Tensor;
using ml::train::createLayer;

template <typename T> static std::string withKey(const std::string &key, T val) {
  return key + "=" + std::to_string(val);
}

template <> std::string withKey(const std::string &key, std::string val) {
  return key + "=" + val;
}

template <> std::string withKey(const std::string &key, const char *val) {
  return key + "=" + std::string(val);
}

namespace causallm {

static std::vector<float> readFloatFile(const std::filesystem::path &path,
                                        size_t expected_values) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open input tensor: " + path.string());
  }

  const std::streamsize bytes = file.tellg();
  file.seekg(0, std::ios::beg);
  if (bytes != static_cast<std::streamsize>(expected_values * sizeof(float))) {
    throw std::runtime_error("Unexpected tensor byte size for " + path.string());
  }

  std::vector<float> values(expected_values);
  if (!file.read(reinterpret_cast<char *>(values.data()), bytes)) {
    throw std::runtime_error("Failed to read input tensor: " + path.string());
  }
  return values;
}

static void writeFloatFile(const std::filesystem::path &path, const float *data,
                           size_t values) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open output tensor: " + path.string());
  }
  file.write(reinterpret_cast<const char *>(data), values * sizeof(float));
}

void Siglip2NaFlexVision::setupParameters(json &cfg, json &generation_cfg,
                                          json &nntr_cfg) {
  (void)generation_cfg;

  const json &vision_cfg = cfg.contains("vision_config") ? cfg["vision_config"] : cfg;
  const json &text_cfg = cfg.contains("text_config") ? cfg["text_config"] : cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");

  DIM = vision_cfg.value("hidden_size", 768);
  INTERMEDIATE_SIZE = vision_cfg.value("intermediate_size", 3072);
  NUM_LAYERS = vision_cfg.value("num_hidden_layers", 12);
  NUM_HEADS = vision_cfg.value("num_attention_heads", 12);
  HEAD_DIM = DIM / NUM_HEADS;
  NUM_KEY_VALUE_HEADS = NUM_HEADS;
  GQA_SIZE = 1;
  NORM_EPS = vision_cfg.value("layer_norm_eps", 1e-6f);
  PATCH_SIZE = vision_cfg.value("patch_size", 16);
  NUM_PATCHES = vision_cfg.value("num_patches", 256);
  BASE_GRID_SIZE =
    static_cast<unsigned int>(std::round(std::sqrt(NUM_PATCHES)));
  PATCH_DIM = vision_cfg.value("num_channels", 3) * PATCH_SIZE * PATCH_SIZE;

  DOWNSAMPLE_FACTOR = cfg.value("downsample_factor", 2);
  MAX_IMAGE_TOKENS = cfg.value("max_image_tokens", 256);
  PROJECTOR_HIDDEN_SIZE = cfg.value("projector_hidden_size", 2048);
  PROJECTOR_INPUT_SIZE = DIM * DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR;
  PROJECTOR_OUTPUT_SIZE = text_cfg.value("hidden_size", DIM);
  PROJECTOR_USE_LAYERNORM = cfg.value("projector_use_layernorm", false);
  PROJECTOR_BIAS = cfg.value("projector_bias", true);
  PROJECTOR_HIDDEN_ACT = cfg.value("projector_hidden_act", "gelu");
  INCLUDE_PROJECTOR = nntr_cfg.value("include_projector", false);

  const unsigned int tile_size = cfg.value("tile_size", 512);
  const bool do_image_splitting = cfg.value("do_image_splitting", true);
  const unsigned int max_thumbnail_patches =
    MAX_IMAGE_TOKENS * DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR;
  const unsigned int tile_patches =
    do_image_splitting ? (tile_size / PATCH_SIZE) * (tile_size / PATCH_SIZE) : 0;
  MAX_PATCHES = nntr_cfg.value("max_patches",
                               std::max(max_thumbnail_patches, tile_patches));
  MAX_IMAGE_TOKENS =
    nntr_cfg.value("max_image_tokens", MAX_PATCHES /
                                         (DOWNSAMPLE_FACTOR *
                                          DOWNSAMPLE_FACTOR));
  INIT_SEQ_LEN = MAX_PATCHES;
  MAX_SEQ_LEN = MAX_PATCHES;
  NUM_TO_GENERATE = 0;
  HIDDEN_ACT = vision_cfg.value("hidden_act", "gelu_pytorch_tanh") ==
                   "gelu_pytorch_tanh"
                 ? "tanh_gelu"
                 : "gelu";
}

void Siglip2NaFlexVision::initialize() {
  registerCustomLayers();

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
                      withKey("model_tensor_type", MODEL_TENSOR_TYPE)});

  auto [inputs, output] = constructVisionGraph();
  std::vector<Tensor> outputs = {output};
  if (model->compile(inputs, outputs, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  is_initialized = true;
}

std::pair<Tensor, Tensor> Siglip2NaFlexVision::constructModel() {
  auto [inputs, output] = constructVisionGraph();
  return {inputs.front(), output};
}

std::pair<std::vector<Tensor>, Tensor>
Siglip2NaFlexVision::constructVisionGraph() {
  Tensor pixel_values({BATCH_SIZE, 1, MAX_PATCHES, PATCH_DIM}, "pixel_values");
  Tensor pixel_attention_mask({BATCH_SIZE, 1, 1, MAX_PATCHES},
                              "pixel_attention_mask");
  Tensor spatial_shapes({BATCH_SIZE, 1, 1, 2}, "spatial_shapes");

  LayerHandle patch_embedding(createLayer(
    "fully_connected",
    {withKey("name", "vision_model_embeddings_patch_embedding"),
     withKey("unit", DIM), withKey("disable_bias", "false"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor h = patch_embedding(pixel_values);

  LayerHandle position_embedding(createLayer(
    "siglip2_positional_embedding",
    {withKey("name", "vision_model_embeddings_position_embedding"),
     withKey("base_grid_size", BASE_GRID_SIZE)}));
  h = position_embedding({h, spatial_shapes});

  LayerHandle attention_mask(createLayer(
    "siglip2_attention_mask",
    {withKey("name", "vision_model_attention_mask"),
     withKey("num_heads", NUM_HEADS), withKey("mask_value", -1.0e10f)}));
  Tensor mask = attention_mask(pixel_attention_mask);

  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createEncoderBlock(i, h, mask);
  }

  LayerHandle post_norm(createLayer(
    "layer_normalization",
    {withKey("name", "vision_model_post_layernorm"), withKey("axis", 3),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  h = post_norm(h);

  if (INCLUDE_PROJECTOR) {
    h = createProjector(h, spatial_shapes);
  }

  return {{pixel_values, pixel_attention_mask, spatial_shapes}, h};
}

Tensor Siglip2NaFlexVision::createProjector(Tensor input,
                                            Tensor spatial_shapes) {
  LayerHandle pixel_unshuffle(createLayer(
    "lfm2_vl_pixel_unshuffle",
    {withKey("name", "multi_modal_projector_pixel_unshuffle"),
     withKey("downsample_factor", DOWNSAMPLE_FACTOR)}));
  Tensor h = pixel_unshuffle({input, spatial_shapes});

  if (PROJECTOR_USE_LAYERNORM) {
    LayerHandle layer_norm(createLayer(
      "layer_normalization",
      {withKey("name", "multi_modal_projector_layer_norm"),
       withKey("axis", 3), withKey("epsilon", "1e-05"),
       withKey("packed", "false")}));
    h = layer_norm(h);
  }

  LayerHandle linear_1(createLayer(
    "fully_connected",
    {withKey("name", "multi_modal_projector_linear_1"),
     withKey("unit", PROJECTOR_HIDDEN_SIZE),
     withKey("disable_bias", PROJECTOR_BIAS ? "false" : "true"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  h = linear_1(h);

  LayerHandle act(createLayer(
    "activation",
    {withKey("name", "multi_modal_projector_act"),
     withKey("activation", PROJECTOR_HIDDEN_ACT)}));
  h = act(h);

  LayerHandle linear_2(createLayer(
    "fully_connected",
    {withKey("name", "multi_modal_projector_linear_2"),
     withKey("unit", PROJECTOR_OUTPUT_SIZE),
     withKey("disable_bias", PROJECTOR_BIAS ? "false" : "true"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  return linear_2(h);
}

Tensor Siglip2NaFlexVision::createEncoderBlock(const int layer_id, Tensor input,
                                               Tensor mask) {
  const std::string prefix =
    "vision_model_encoder_layers_" + std::to_string(layer_id) + "_";

  LayerHandle ln1(createLayer(
    "layer_normalization",
    {withKey("name", prefix + "layer_norm1"), withKey("axis", 3),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = ln1(input);

  LayerHandle attention(createLayer(
    "multi_head_attention",
    {withKey("name", prefix + "self_attn"),
     withKey("num_heads", NUM_HEADS),
     withKey("projected_key_dim", HEAD_DIM),
     withKey("projected_value_dim", HEAD_DIM),
     withKey("output_shape", DIM),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor attn_out = attention({normed, normed, normed, mask});

  LayerHandle attn_res(createLayer(
    "addition", {withKey("name", prefix + "attention_residual")}));
  Tensor residual = attn_res({input, attn_out});

  Tensor mlp_out = createMlp(layer_id, residual);
  LayerHandle mlp_res(
    createLayer("addition", {withKey("name", prefix + "mlp_residual")}));
  return mlp_res({residual, mlp_out});
}

Tensor Siglip2NaFlexVision::createMlp(const int layer_id, Tensor input) {
  const std::string prefix =
    "vision_model_encoder_layers_" + std::to_string(layer_id) + "_";

  LayerHandle ln2(createLayer(
    "layer_normalization",
    {withKey("name", prefix + "layer_norm2"), withKey("axis", 3),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor h = ln2(input);

  LayerHandle fc1(createLayer(
    "fully_connected",
    {withKey("name", prefix + "mlp_fc1"),
     withKey("unit", INTERMEDIATE_SIZE), withKey("disable_bias", "false"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  h = fc1(h);

  LayerHandle act(createLayer(
    "activation",
    {withKey("name", prefix + "mlp_act"), withKey("activation", HIDDEN_ACT)}));
  h = act(h);

  LayerHandle fc2(createLayer(
    "fully_connected",
    {withKey("name", prefix + "mlp_fc2"), withKey("unit", DIM),
     withKey("disable_bias", "false"), withKey("weight_dtype", FC_LAYER_DTYPE)}));
  return fc2(h);
}

void Siglip2NaFlexVision::registerCustomLayers() {
  Transformer::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::Siglip2PositionalEmbeddingLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::Siglip2AttentionMaskLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::Lfm2VlPixelUnshuffleLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register SigLIP2 factory, reason: " << e.what()
              << std::endl;
  }
}

void Siglip2NaFlexVision::run(const WSTR prompt, bool do_sample,
                              const WSTR system_prompt,
                              const WSTR tail_prompt, bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;

  if (!is_initialized) {
    throw std::runtime_error("Siglip2NaFlexVision is not initialized.");
  }

  const std::filesystem::path input_dir{std::string(prompt)};
  const size_t pixel_values_len =
    static_cast<size_t>(BATCH_SIZE) * MAX_PATCHES * PATCH_DIM;
  const size_t pixel_mask_len = static_cast<size_t>(BATCH_SIZE) * MAX_PATCHES;
  const size_t spatial_shapes_len = static_cast<size_t>(BATCH_SIZE) * 2;

  std::vector<float> pixel_values =
    readFloatFile(input_dir / "pixel_values.f32", pixel_values_len);
  std::vector<float> pixel_attention_mask =
    readFloatFile(input_dir / "pixel_attention_mask.f32", pixel_mask_len);
  std::vector<float> spatial_shapes =
    readFloatFile(input_dir / "spatial_shapes.f32", spatial_shapes_len);

  std::vector<float *> inputs = {pixel_values.data(),
                                 pixel_attention_mask.data(),
                                 spatial_shapes.data()};
  std::vector<float *> outputs = model->inference(BATCH_SIZE, inputs);

  const std::string output_name =
    INCLUDE_PROJECTOR ? "image_features.f32" : "vision_last_hidden_state.f32";
  const size_t output_len =
    INCLUDE_PROJECTOR
      ? static_cast<size_t>(BATCH_SIZE) * MAX_IMAGE_TOKENS *
          PROJECTOR_OUTPUT_SIZE
      : static_cast<size_t>(BATCH_SIZE) * MAX_PATCHES * DIM;
  writeFloatFile(input_dir / output_name, outputs[0], output_len);

  if (log_output) {
    std::cout << std::setprecision(9)
              << "Wrote " << (input_dir / output_name)
              << "\nFirst 10 values: ";
    const int log_count =
      std::min(INCLUDE_PROJECTOR ? PROJECTOR_OUTPUT_SIZE : DIM, 10U);
    for (int i = 0; i < log_count; ++i) {
      std::cout << "[" << i << "]=" << outputs[0][i] << " ";
    }
    std::cout << std::endl;
  }
}

} // namespace causallm
