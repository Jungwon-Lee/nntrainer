// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_prefill.cpp
 * @brief  LFM2-VL language prefill validation model.
 */

#include "lfm2_vl_prefill.h"

#include <app_context.h>
#include <causal_conv1d_layer.h>
#include <custom_multiply.h>
#include <engine.h>
#include <lm_head.h>
#include <model.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tensor_api.h>
#include <tie_word_embedding.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

static unsigned int argmaxToken(const float *logits, unsigned int vocab_size) {
  unsigned int best_id = 0;
  float best_logit = -std::numeric_limits<float>::infinity();
  for (unsigned int i = 0; i < vocab_size; ++i) {
    if (logits[i] > best_logit) {
      best_logit = logits[i];
      best_id = i;
    }
  }
  return best_id;
}

void Lfm2VlPrefill::setupParameters(json &cfg, json &generation_cfg,
                                    json &nntr_cfg) {
  (void)generation_cfg;
  const json &text_cfg = cfg.contains("text_config") ? cfg["text_config"] : cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");
  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", 1);
  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", INIT_SEQ_LEN);
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);
  OUTPUT_MODE = nntr_cfg.value("output_mode", "logits");
  DECODE_EMBEDDING_FILE = nntr_cfg.value("decode_embedding_file", "");
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;

  NUM_VOCAB = text_cfg.value("vocab_size", 65536);
  DIM = text_cfg.value("hidden_size", 1024);
  NUM_LAYERS = text_cfg.value("num_hidden_layers", 16);
  NUM_HEADS = text_cfg.value("num_attention_heads", 16);
  NUM_KEY_VALUE_HEADS = text_cfg.value("num_key_value_heads", NUM_HEADS);
  HEAD_DIM =
    text_cfg.contains("head_dim") ? text_cfg["head_dim"].get<int>()
                                  : DIM / NUM_HEADS;
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;
  MAX_POSITION_EMBEDDINGS =
    text_cfg.value("max_position_embeddings", 128000U);
  ROPE_THETA = 1000000U;
  if (text_cfg.contains("rope_parameters") &&
      text_cfg["rope_parameters"].contains("rope_theta")) {
    ROPE_THETA = static_cast<unsigned int>(
      text_cfg["rope_parameters"]["rope_theta"].get<float>());
  } else if (text_cfg.contains("rope_theta")) {
    ROPE_THETA = static_cast<unsigned int>(text_cfg["rope_theta"].get<float>());
  }
  SLIDING_WINDOW =
    text_cfg.contains("sliding_window") && !text_cfg["sliding_window"].is_null()
      ? text_cfg["sliding_window"].get<unsigned int>()
      : UINT_MAX;
  SLIDING_WINDOW_PATTERN = text_cfg.value("sliding_window_pattern", 1U);
  IS_CAUSAL = text_cfg.value("is_causal", true);
  TIE_WORD_EMBEDDINGS = text_cfg.value("tie_word_embeddings", true);
  NORM_EPS = text_cfg.value("norm_eps", text_cfg.value("rms_norm_eps", 1e-5f));

  unsigned int ff_dim = text_cfg.value("block_ff_dim",
                                       text_cfg.value("intermediate_size", 0));
  if (text_cfg.value("block_auto_adjust_ff_dim", true)) {
    ff_dim = static_cast<unsigned int>((2.0f * ff_dim) / 3.0f);
    const float mult = text_cfg.value("block_ffn_dim_multiplier", 1.0f);
    ff_dim = static_cast<unsigned int>(ff_dim * mult);
    const unsigned int multiple_of = text_cfg.value("block_multiple_of", 1U);
    ff_dim = multiple_of * ((ff_dim + multiple_of - 1) / multiple_of);
  }
  INTERMEDIATE_SIZE = ff_dim;

  CONV_DIM = text_cfg.value("conv_dim", static_cast<unsigned int>(DIM));
  CONV_DIM_OUT = text_cfg.value("conv_dim_out", static_cast<unsigned int>(DIM));
  CONV_L_CACHE = text_cfg.value("conv_L_cache", 3);
  CONV_BIAS = text_cfg.value("conv_bias", false);

  if (text_cfg.contains("layer_types")) {
    layer_types_ = text_cfg["layer_types"].get<std::vector<std::string>>();
  } else {
    layer_types_.assign(NUM_LAYERS, "full_attention");
  }

  if (nntr_cfg.contains("num_validate_layers")) {
    const int requested_layers = nntr_cfg["num_validate_layers"].get<int>();
    if (requested_layers < 0 ||
        requested_layers > static_cast<int>(layer_types_.size())) {
      throw std::invalid_argument("num_validate_layers is out of range");
    }
    NUM_LAYERS = requested_layers;
  }
  if (NUM_TO_GENERATE > 0 &&
      MAX_SEQ_LEN < static_cast<unsigned int>(INIT_SEQ_LEN + NUM_TO_GENERATE)) {
    throw std::invalid_argument(
      "max_seq_len must be at least init_seq_len + num_to_generate");
  }
}

void Lfm2VlPrefill::initialize() {
  registerCustomLayers();

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }
  model->setProperty(model_props);

  auto [input, output] = constructModel();
  if (model->compile(input, output, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  is_initialized = true;
}

std::pair<Tensor, Tensor> Lfm2VlPrefill::constructModel() {
  Tensor input({BATCH_SIZE, 1, INIT_SEQ_LEN, static_cast<unsigned int>(DIM)},
               "inputs_embeds_after_image_merge");
  Tensor h = input;

  for (int i = 0; i < NUM_LAYERS; ++i) {
    if (layer_types_.at(i) == "full_attention" ||
        layer_types_.at(i) == "attention") {
      h = createTransformerDecoderBlock(i, h);
    } else {
      h = createConvBlock(i, h);
    }
  }

  if (OUTPUT_MODE == "hidden" || OUTPUT_MODE == "operator") {
    return {input, h};
  }

  LayerHandle output_norm(createLayer(
    "rms_norm", {withKey("name", "output_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));
  Tensor norm_out = output_norm(h);

  if (OUTPUT_MODE == "norm") {
    return {input, norm_out};
  }

  LayerHandle lmhead(createLayer(
    "lm_head",
    {withKey("name", "output_of_causallm"), withKey("unit", NUM_VOCAB),
     withKey("disable_bias", "true"), withKey("weight_dtype", EMBEDDING_DTYPE)}));
  Tensor output = lmhead(norm_out);

  if (OUTPUT_MODE != "logits") {
    throw std::invalid_argument("Unsupported Lfm2VlPrefill output_mode: " +
                                OUTPUT_MODE);
  }

  return {input, output};
}

Tensor Lfm2VlPrefill::createConvBlock(const int layer_id, Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id);

  LayerHandle conv_norm(createLayer(
    "rms_norm", {withKey("name", prefix + "_conv_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));
  Tensor normed = conv_norm(input);

  LayerHandle conv_in_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_conv_in_proj"), withKey("unit", 3 * CONV_DIM),
     withKey("disable_bias", CONV_BIAS ? "false" : "true")}));
  Tensor proj_out = conv_in_proj(normed);

  LayerHandle chunk_layer(createLayer(
    "split", {withKey("name", prefix + "_conv_chunk"), withKey("axis", 3),
               withKey("split_number", 3)}));
  Tensor chunks = chunk_layer(proj_out);
  Tensor chunk_0 = chunks.output(0);
  Tensor chunk_1 = chunks.output(1);
  Tensor chunk_2 = chunks.output(2);

  LayerHandle gate_mul(createLayer(
    "custom_multiply",
    {withKey("name", prefix + "_conv_mul_pre"), withKey("inplace", "true")}));
  Tensor gated = gate_mul({chunk_0, chunk_2});

  LayerHandle causal_conv(createLayer(
    "causal_conv1d",
    {withKey("name", prefix + "_conv_conv"), withKey("weight_dtype", "FP32")}));
  Tensor conv_out = causal_conv(gated);

  LayerHandle out_mul(createLayer(
    "custom_multiply",
    {withKey("name", prefix + "_conv_mul_post"), withKey("inplace", "true")}));
  Tensor gated_out = out_mul({chunk_1, conv_out});

  LayerHandle conv_out_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_conv_out_proj"), withKey("unit", DIM),
     withKey("disable_bias", CONV_BIAS ? "false" : "true")}));
  Tensor proj_back = conv_out_proj(gated_out);

  Tensor residual = input.add(proj_back);

  if (OUTPUT_MODE == "operator" && layer_id == NUM_LAYERS - 1) {
    return residual;
  }

  LayerHandle ffn_norm(createLayer(
    "rms_norm", {withKey("name", prefix + "_ffn_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));
  Tensor ffn_normed = ffn_norm(residual);
  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  return residual.add(ffn_out);
}

void Lfm2VlPrefill::registerCustomLayers() {
  Transformer::registerCustomLayers();
  Lfm2Transformer::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::LmHeadLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register LFM2-VL prefill factory, reason: "
              << e.what() << std::endl;
  }
}

void Lfm2VlPrefill::run(const WSTR prompt, bool do_sample,
                        const WSTR system_prompt, const WSTR tail_prompt,
                        bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;

  if (!is_initialized) {
    throw std::runtime_error("Lfm2VlPrefill is not initialized.");
  }
  if (NUM_TO_GENERATE > 0 && OUTPUT_MODE != "logits") {
    throw std::runtime_error("Token generation requires output_mode=logits.");
  }
  if (NUM_TO_GENERATE > 0 && BATCH_SIZE != 1) {
    throw std::runtime_error("Lfm2VlPrefill generation currently supports "
                             "batch_size=1 only.");
  }
  if (NUM_TO_GENERATE > 0 && DECODE_EMBEDDING_FILE.empty()) {
    throw std::runtime_error(
      "num_to_generate > 0 requires decode_embedding_file.");
  }

  const std::filesystem::path input_dir{std::string(prompt)};
  std::vector<float> merged_embeds = readFloatFile(
    input_dir / "inputs_embeds_after_image_merge.f32",
    static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * DIM);
  std::vector<float> decode_embeddings;
  if (NUM_TO_GENERATE > 0) {
    decode_embeddings = readFloatFile(
      DECODE_EMBEDDING_FILE,
      static_cast<size_t>(NUM_VOCAB) * static_cast<size_t>(DIM));
  }

  const size_t cache_values = static_cast<size_t>(BATCH_SIZE) * MAX_SEQ_LEN *
                              NUM_KEY_VALUE_HEADS * HEAD_DIM;
  struct CacheInput {
    std::string name;
    std::vector<uint16_t> data;
  };
  std::vector<CacheInput> cache_inputs;
  for (int i = 0; i < NUM_LAYERS; ++i) {
    if (!(layer_types_.at(i) == "full_attention" ||
          layer_types_.at(i) == "attention")) {
      continue;
    }
    cache_inputs.push_back(
      {"cache_k_l" + std::to_string(i), std::vector<uint16_t>(cache_values, 0)});
    cache_inputs.push_back(
      {"cache_v_l" + std::to_string(i), std::vector<uint16_t>(cache_values, 0)});
  }
  std::sort(cache_inputs.begin(), cache_inputs.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.name < rhs.name;
            });

  std::vector<float *> inputs;
  inputs.reserve(1 + cache_inputs.size());
  inputs.push_back(merged_embeds.data());
  for (auto &cache_input : cache_inputs) {
    inputs.push_back(reinterpret_cast<float *>(cache_input.data.data()));
  }

  std::vector<float *> labels;
  std::vector<float *> outputs = model->incremental_inference(
    BATCH_SIZE, inputs, labels, INIT_SEQ_LEN, 0, INIT_SEQ_LEN, false);

  const bool logits_mode = OUTPUT_MODE == "logits";
  const std::string output_name =
    logits_mode ? "prefill_logits.f32" : "prefill_" + OUTPUT_MODE + ".f32";
  const size_t output_values =
    logits_mode ? static_cast<size_t>(BATCH_SIZE) * NUM_VOCAB
                : static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * DIM;

  writeFloatFile(input_dir / output_name, outputs[0], output_values);

  if (log_output) {
    std::cout << std::setprecision(9)
              << "Wrote " << (input_dir / output_name)
              << "\nFirst 10 values: ";
    const int log_count = std::min(static_cast<int>(output_values), 10);
    for (int i = 0; i < log_count; ++i) {
      std::cout << "[" << i << "]=" << outputs[0][i] << " ";
    }
    std::cout << std::endl;
  }

  std::vector<float> generated_ids;
  if (NUM_TO_GENERATE > 0) {
    generated_ids.reserve(NUM_TO_GENERATE);
    unsigned int next_token = argmaxToken(outputs[0], NUM_VOCAB);
    generated_ids.push_back(static_cast<float>(next_token));

    std::vector<float> decode_embeds(
      static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * DIM, 0.0f);
    inputs[0] = decode_embeds.data();

    for (int generation_idx = 1; generation_idx < NUM_TO_GENERATE;
         ++generation_idx) {
      std::fill(decode_embeds.begin(), decode_embeds.end(), 0.0f);
      const float *embedding =
        decode_embeddings.data() + static_cast<size_t>(next_token) * DIM;
      std::copy(embedding, embedding + DIM, decode_embeds.data());

      for (auto *output : outputs) {
        delete[] output;
      }

      const unsigned int from =
        static_cast<unsigned int>(INIT_SEQ_LEN + generation_idx - 1);
      const unsigned int to = from + 1;
      outputs = model->incremental_inference(BATCH_SIZE, inputs, labels,
                                             INIT_SEQ_LEN, from, to, false);
      next_token = argmaxToken(outputs[0], NUM_VOCAB);
      generated_ids.push_back(static_cast<float>(next_token));
    }

    writeFloatFile(input_dir / "generated_ids.f32", generated_ids.data(),
                   generated_ids.size());
  }

  if (log_output && !generated_ids.empty()) {
    std::cout << "Generated token ids: ";
    for (float id : generated_ids) {
      std::cout << static_cast<unsigned int>(id) << " ";
    }
    std::cout << std::endl;
  }

  for (auto *output : outputs) {
    delete[] output;
  }
}

} // namespace causallm
