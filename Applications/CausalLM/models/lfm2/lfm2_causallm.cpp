// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_causallm.cpp
 * @date   14 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   LFM2 model implementation with hybrid attention-conv architecture
 */

#include <lfm2_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>

#include <causal_conv1d_layer.h>
#include <custom_multiply.h>
#include <reshaped_rms_norm.h>

namespace causallm {

std::vector<LayerHandle>
Lfm2Transformer::createAttention(const int layer_id, int seq_len, int n_heads,
                                 int head_dim, std::string query_name,
                                 std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Q projection
  std::vector<std::string> q_params = {
    withKey("name", Q),
    withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"),
    withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", q_params));

  // Q norm (reshaped RMS norm)
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm),
    withKey("input_layers", Q),
    withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

  // K projection
  std::vector<std::string> k_params = {
    withKey("name", K),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"),
    withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", k_params));

  // K norm (reshaped RMS norm)
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm),
    withKey("input_layers", K),
    withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", k_norm_params));

  // V projection
  std::vector<std::string> v_params = {
    withKey("name", V),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"),
    withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", SLIDING_WINDOW),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q_norm, K_norm, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O projection
  std::vector<std::string> o_params = {
    withKey("name", O),
    withKey("unit", DIM),
    withKey("disable_bias", "true"),
    withKey("input_layers", A),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

void Lfm2Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Lfm2CausalLM::setupParameters(json &cfg, json &generation_cfg,
                                   json &nntr_cfg) {
  // Call parent setupParameters first
  CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  try {
    // LFM2-specific: compute intermediate size with multipliers
    unsigned int ff_dim = cfg.contains("block_ff_dim")
                            ? cfg["block_ff_dim"].get<unsigned int>()
                            : INTERMEDIATE_SIZE;

    if (cfg.contains("block_auto_adjust_ff_dim") &&
        cfg["block_auto_adjust_ff_dim"].get<bool>()) {
      ff_dim = static_cast<unsigned int>((2.0f * ff_dim) / 3.0f);
    }

    float mult = cfg.contains("block_ffn_dim_multiplier")
                   ? cfg["block_ffn_dim_multiplier"].get<float>()
                   : 1.0f;
    ff_dim = static_cast<unsigned int>(ff_dim * mult);

    unsigned int multiple_of = cfg.contains("block_multiple_of")
                                 ? cfg["block_multiple_of"].get<unsigned int>()
                                 : 1;
    ff_dim = multiple_of * ((ff_dim + multiple_of - 1) / multiple_of);

    INTERMEDIATE_SIZE = ff_dim;

    // LFM2 prefers block_norm_eps over rms_norm_eps for block norms
    if (cfg.contains("block_norm_eps")) {
      NORM_EPS = cfg["block_norm_eps"].get<float>();
    }

    // Conv-specific parameters
    CONV_DIM = cfg.contains("conv_dim") ? cfg["conv_dim"].get<unsigned int>()
                                        : DIM;

    CONV_DIM_OUT = cfg.contains("conv_dim_out")
                     ? cfg["conv_dim_out"].get<unsigned int>()
                     : DIM;

    CONV_L_CACHE = cfg.contains("conv_L_cache")
                     ? cfg["conv_L_cache"].get<unsigned int>()
                     : 0;

    CONV_BIAS =
      cfg.contains("conv_bias") ? cfg["conv_bias"].get<bool>() : false;

    // LFM2 does not tie word embeddings by default
    TIE_WORD_EMBEDDINGS =
      cfg.contains("tie_word_embeddings") ? cfg["tie_word_embeddings"].get<bool>()
                                          : false;

  } catch (const std::exception &e) {
    throw std::runtime_error("Lfm2CausalLM: config parsing error: " +
                             std::string(e.what()));
  }
}

std::vector<LayerHandle>
Lfm2CausalLM::createTransformerDecoderBlock(const int layer_id,
                                             std::string input_name) {

  std::vector<LayerHandle> layers;

  // Per-layer operator types for LFM2 hybrid architecture
  // Pattern: conv, conv, attention, conv, conv, attention, ...
  static const std::vector<std::string> LAYER_TYPES = {
    "conv", "conv", "attention", "conv", "conv", "attention",
    "conv", "conv", "attention", "conv", "attention", "conv",
    "attention", "conv", "attention", "conv"
  };

  // Determine block type based on layer_id
  std::string block_type = (layer_id < static_cast<int>(LAYER_TYPES.size()))
                             ? LAYER_TYPES[layer_id]
                             : "attention";

  if (block_type == "attention") {
    // Attention block: norm -> attention -> residual -> norm -> FFN -> residual
    layers.push_back(createLayer(
      "rms_norm",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
       withKey("input_layers", input_name),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("packed", "false")}));

    auto att_layers = createAttention(
      layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
      "layer" + std::to_string(layer_id) + "_attention_norm",
      "layer" + std::to_string(layer_id) + "_attention_norm",
      "layer" + std::to_string(layer_id) + "_attention_norm");
    layers.insert(layers.end(), att_layers.begin(), att_layers.end());

    layers.push_back(createLayer(
      "addition",
      {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
       withKey("input_layers",
               input_name + ",layer" + std::to_string(layer_id) +
                 "_attention_out")}));

    layers.push_back(createLayer(
      "rms_norm",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_decoder_add"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("packed", "false")}));

    auto ffn_layers = createMlp(
      layer_id, DIM, INTERMEDIATE_SIZE,
      "layer" + std::to_string(layer_id) + "_ffn_norm");
    layers.insert(layers.end(), ffn_layers.begin(), ffn_layers.end());

    layers.push_back(createLayer(
      "addition",
      {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_decoder_add,layer" +
                 std::to_string(layer_id) + "_ffn_down")}));

  } else {
    // Conv block: norm -> conv -> residual -> norm -> FFN -> residual
    auto conv_layers = createConvBlock(layer_id, input_name);
    layers.insert(layers.end(), conv_layers.begin(), conv_layers.end());
  }

  return layers;
}

std::vector<LayerHandle>
Lfm2CausalLM::createConvBlock(const int layer_id, std::string input_name) {

  std::vector<LayerHandle> layers;
  auto prefix = "layer" + std::to_string(layer_id);

  // Pre-conv normalization
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", prefix + "_conv_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // Expand features: [B, 1, T, DIM] → [B, 1, T, 3*CONV_DIM]
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_conv_in_proj"),
     withKey("unit", 3 * CONV_DIM),
     withKey("disable_bias", "true"),
     withKey("input_layers", prefix + "_conv_norm"),
     withKey("weight_initializer", "ones")}));

  // Split along width (axis=3): [B,1,T,3*CONV_DIM] → 3 × [B,1,T,CONV_DIM]
  layers.push_back(createLayer(
    "split",
    {withKey("name", prefix + "_conv_chunk"),
     withKey("input_layers", prefix + "_conv_in_proj"),
     withKey("axis", 3),
     withKey("split_number", 3)}));

  // Gating before conv: chunk_0 ⊙ chunk_2
  layers.push_back(createLayer(
    "custom_multiply",
    {withKey("name", prefix + "_conv_mul_pre"),
     withKey("input_layers",
             prefix + "_conv_chunk(0)," + prefix + "_conv_chunk(2)"),
     withKey("inplace", "true")}));

  // Causal depthwise conv1d
  layers.push_back(createLayer(
    "causal_conv1d",
    {withKey("name", prefix + "_conv_conv"),
     withKey("input_layers", prefix + "_conv_mul_pre"),
     withKey("weight_dtype", "FP32")}));

  // Gating after conv: chunk_1 ⊙ conv_out
  layers.push_back(createLayer(
    "custom_multiply",
    {withKey("name", prefix + "_conv_mul_post"),
     withKey("input_layers",
             prefix + "_conv_chunk(1)," + prefix + "_conv_conv"),
     withKey("inplace", "true")}));

  // Project back to DIM
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_conv_out_proj"),
     withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("input_layers", prefix + "_conv_mul_post"),
     withKey("weight_initializer", "ones")}));

  // Conv residual connection
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_conv_add"),
     withKey("input_layers", input_name + "," + prefix + "_conv_out_proj")}));

  // Pre-FFN normalization
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", prefix + "_ffn_norm"),
     withKey("input_layers", prefix + "_conv_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // Feed forward
  auto ffn_layers = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                               prefix + "_ffn_norm");
  layers.insert(layers.end(), ffn_layers.begin(), ffn_layers.end());

  // FFN residual connection
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_decoder_output"),
     withKey("input_layers",
             prefix + "_conv_add,layer" + std::to_string(layer_id) +
               "_ffn_down")}));

  return layers;
}

std::vector<LayerHandle>
Lfm2CausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                        std::string input_name) {

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim),
     withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim),
     withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_gate,layer" +
               std::to_string(layer_id) + "_ffn_up")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim),
     withKey("disable_bias", "true"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("weight_initializer", "ones")}));

  return layers;
}

void Lfm2CausalLM::constructModel() {
  // Call parent constructModel which builds transformer blocks
  Transformer::constructModel();

  // Add LM head
  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  std::vector<std::string> lmhead_props = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("input_layers", "output_norm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };

  if (TIE_WORD_EMBEDDINGS) {
    lmhead_props.emplace_back(withKey("shared_from", "embedding0"));
  }

  model->addLayer(createLayer(lmhead_type, lmhead_props));
}

void Lfm2CausalLM::registerCustomLayers() {
  // Register parent layers first
  CausalLM::registerCustomLayers();
  Lfm2Transformer::registerCustomLayers();

  // Register LFM2-specific custom layers
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::CustomMultiplyLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::CausalConv1DLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

Lfm2CausalLM Lfm2CausalLM::createTestModel() {
  // Minimal JSON config for testing
  json cfg;
  cfg["vocab_size"] = 65536;
  cfg["hidden_size"] = 1536;
  cfg["num_hidden_layers"] = 2;
  cfg["num_attention_heads"] = 24;
  cfg["num_key_value_heads"] = 8;
  cfg["head_dim"] = 64;
  cfg["intermediate_size"] = 10240;
  cfg["rope_theta"] = 1000000;
  cfg["rms_norm_eps"] = 1e-05;
  cfg["tie_word_embeddings"] = true;
  cfg["max_position_embeddings"] = 128000;

  json gen_cfg;
  gen_cfg["max_new_tokens"] = 1;
  gen_cfg["eos_token_id"] = 0;
  gen_cfg["bos_token_id"] = 1;

  json nntr_cfg;
  nntr_cfg["init_seq_len"] = 8;
  nntr_cfg["batch_size"] = 1;
  nntr_cfg["model_tensor_type"] = "FP32-FP32";
  nntr_cfg["memory_swap"] = false;
  nntr_cfg["tokenizer_file"] = "";
  nntr_cfg["max_seq_len"] = 128;
  nntr_cfg["num_to_generate"] = 1;
  nntr_cfg["embedding_dtype"] = "FP32";
  nntr_cfg["fc_layer_dtype"] = "FP32";
  nntr_cfg["bad_word_ids"] = std::vector<unsigned int>();

  return Lfm2CausalLM(cfg, gen_cfg, nntr_cfg);
}

} // namespace causallm
