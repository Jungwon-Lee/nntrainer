// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_embedding_merge.cpp
 * @brief  Validation model for LFM2-VL image-token embedding merge.
 */

#include "lfm2_vl_embedding_merge.h"

#include <app_context.h>
#include <embedding_layer.h>
#include <engine.h>
#include <model.h>
#include <siglip2_layers.h>
#include <tensor_api.h>

#include <algorithm>
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

void Lfm2VlEmbeddingMerge::setupParameters(json &cfg, json &generation_cfg,
                                           json &nntr_cfg) {
  (void)generation_cfg;
  const json &text_cfg = cfg.contains("text_config") ? cfg["text_config"] : cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");
  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", 1);
  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", INIT_SEQ_LEN);
  NUM_TO_GENERATE = 0;

  NUM_VOCAB = text_cfg.value("vocab_size", 65536);
  DIM = text_cfg.value("hidden_size", 1024);
  IMAGE_TOKEN_ID = cfg.value("image_token_id", 396);
  MAX_IMAGE_TOKENS = nntr_cfg.value("max_image_tokens", 256);
  EMBEDDING_SCALE = text_cfg.value("embedding_scale", 1.0f);
}

void Lfm2VlEmbeddingMerge::initialize() {
  registerCustomLayers();

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
                      withKey("model_tensor_type", MODEL_TENSOR_TYPE)});

  auto [inputs, output] = constructMergeGraph();
  std::vector<Tensor> outputs = {output};
  if (model->compile(inputs, outputs, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  is_initialized = true;
}

std::pair<Tensor, Tensor> Lfm2VlEmbeddingMerge::constructModel() {
  auto [inputs, output] = constructMergeGraph();
  return {inputs.front(), output};
}

std::pair<std::vector<Tensor>, Tensor>
Lfm2VlEmbeddingMerge::constructMergeGraph() {
  Tensor input_ids({BATCH_SIZE, 1, 1, INIT_SEQ_LEN}, "input_ids");
  Tensor image_features(
    {BATCH_SIZE, 1, MAX_IMAGE_TOKENS, static_cast<unsigned int>(DIM)},
    "image_features");

  LayerHandle embedding(createLayer(
    "embedding_layer",
    {withKey("name", "embedding0"), withKey("weight_dtype", EMBEDDING_DTYPE),
     withKey("in_dim", NUM_VOCAB), withKey("out_dim", DIM),
     withKey("scale", EMBEDDING_SCALE)}));
  Tensor text_embeds = embedding(input_ids);

  LayerHandle merge(createLayer(
    "lfm2_vl_image_embedding_merge",
    {withKey("name", "lfm2_vl_image_embedding_merge"),
     withKey("image_token_id", IMAGE_TOKEN_ID)}));
  Tensor merged = merge({input_ids, text_embeds, image_features});

  return {{input_ids, image_features}, merged};
}

void Lfm2VlEmbeddingMerge::registerCustomLayers() {
  Transformer::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::Lfm2VlImageEmbeddingMergeLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register LFM2-VL merge factory, reason: "
              << e.what() << std::endl;
  }
}

void Lfm2VlEmbeddingMerge::run(const WSTR prompt, bool do_sample,
                               const WSTR system_prompt,
                               const WSTR tail_prompt, bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;

  if (!is_initialized) {
    throw std::runtime_error("Lfm2VlEmbeddingMerge is not initialized.");
  }

  const std::filesystem::path input_dir{std::string(prompt)};
  std::vector<float> input_ids =
    readFloatFile(input_dir / "input_ids.f32",
                  static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN);
  std::vector<float> image_features =
    readFloatFile(input_dir / "image_features.f32",
                  static_cast<size_t>(BATCH_SIZE) * MAX_IMAGE_TOKENS * DIM);

  std::vector<float *> inputs = {input_ids.data(), image_features.data()};
  std::vector<float *> labels;
  std::vector<float *> outputs = model->incremental_inference(
    BATCH_SIZE, inputs, labels, INIT_SEQ_LEN, 0, INIT_SEQ_LEN, false);

  const size_t output_len = static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * DIM;
  writeFloatFile(input_dir / "inputs_embeds_after_image_merge.f32", outputs[0],
                 output_len);

  if (log_output) {
    std::cout << std::setprecision(9)
              << "Wrote "
              << (input_dir / "inputs_embeds_after_image_merge.f32")
              << "\nFirst 10 values: ";
    const int log_count = std::min(DIM, 10);
    for (int i = 0; i < log_count; ++i) {
      std::cout << "[" << i << "]=" << outputs[0][i] << " ";
    }
    std::cout << std::endl;
  }

  for (auto *output : outputs) {
    delete[] output;
  }
}

} // namespace causallm
