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

Tensor Lfm2Transformer::createAttention(const int layer_id, int seq_len,
                                        int n_heads, int head_dim,
                                        Tensor query, Tensor key,
                                        Tensor value) {

  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  Tensor q = wq(query);

  LayerHandle q_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  Tensor q_normed = q_norm(q);

  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")}));
  Tensor k = wk(key);

  LayerHandle k_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  Tensor k_normed = k_norm(k);

  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")}));
  Tensor v = wv(value);

  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));
  Tensor a = mha({q_normed, k_normed, v, cache_k, cache_v});

  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  return wo(a);
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

    if (cfg.contains("tie_word_embeddings")) {
      TIE_WORD_EMBEDDINGS = cfg["tie_word_embeddings"].get<bool>();
    } else if (cfg.contains("tie_embedding")) {
      TIE_WORD_EMBEDDINGS = cfg["tie_embedding"].get<bool>();
    } else {
      TIE_WORD_EMBEDDINGS = false;
    }

    if (cfg.contains("layer_types")) {
      LAYER_TYPES = cfg["layer_types"].get<std::vector<std::string>>();
    } else {
      LAYER_TYPES = {"conv",           "conv", "full_attention", "conv",
                     "conv",           "full_attention", "conv", "conv",
                     "full_attention", "conv", "full_attention", "conv",
                     "full_attention", "conv", "full_attention", "conv"};
    }

  } catch (const std::exception &e) {
    throw std::runtime_error("Lfm2CausalLM: config parsing error: " +
                             std::string(e.what()));
  }
}

Tensor Lfm2CausalLM::createTransformerDecoderBlock(const int layer_id,
                                                   Tensor input) {
  std::string block_type = (layer_id < static_cast<int>(LAYER_TYPES.size()))
                             ? LAYER_TYPES[layer_id]
                             : "full_attention";

  if (block_type == "attention" || block_type == "full_attention") {
    LayerHandle attn_norm(createLayer(
      "rms_norm",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("packed", "false")}));
    Tensor normed = attn_norm(input);

    Tensor att_out =
      createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM, normed,
                      normed, normed);

    LayerHandle decoder_add(createLayer(
      "addition",
      {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add")}));
    Tensor residual = decoder_add({input, att_out});

    LayerHandle ffn_norm(createLayer(
      "rms_norm",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("packed", "false")}));
    Tensor ffn_normed = ffn_norm(residual);

    Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

    LayerHandle decoder_output(createLayer(
      "addition",
      {withKey("name",
               "layer" + std::to_string(layer_id) + "_decoder_output")}));
    return decoder_output({residual, ffn_out});
  }

  return createConvBlock(layer_id, input);
}

Tensor Lfm2CausalLM::createConvBlock(const int layer_id, Tensor input) {
  auto prefix = "layer" + std::to_string(layer_id);

  LayerHandle conv_norm(createLayer(
    "rms_norm",
    {withKey("name", prefix + "_conv_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = conv_norm(input);

  LayerHandle conv_in_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_conv_in_proj"),
     withKey("unit", 3 * CONV_DIM),
     withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  Tensor conv_in = conv_in_proj(normed);

  LayerHandle split_layer(createLayer(
    "split",
    {withKey("name", prefix + "_conv_chunk"),
     withKey("axis", 3),
     withKey("split_number", 3)}));
  Tensor chunks = split_layer(conv_in);
  Tensor chunk0 = chunks.output(0);
  Tensor chunk1 = chunks.output(1);
  Tensor chunk2 = chunks.output(2);

  LayerHandle conv_mul_pre(createLayer(
    "custom_multiply",
    {withKey("name", prefix + "_conv_mul_pre"),
     withKey("inplace", "true")}));
  Tensor pre_gate = conv_mul_pre({chunk0, chunk2});

  LayerHandle conv(createLayer(
    "causal_conv1d",
    {withKey("name", prefix + "_conv_conv"),
     withKey("weight_dtype", "FP32")}));
  Tensor conv_out = conv(pre_gate);

  LayerHandle conv_mul_post(createLayer(
    "custom_multiply",
    {withKey("name", prefix + "_conv_mul_post"),
     withKey("inplace", "true")}));
  Tensor post_gate = conv_mul_post({chunk1, conv_out});

  LayerHandle conv_out_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_conv_out_proj"),
     withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  Tensor projected = conv_out_proj(post_gate);

  LayerHandle conv_add(createLayer(
    "addition",
    {withKey("name", prefix + "_conv_add")}));
  Tensor residual = conv_add({input, projected});

  LayerHandle ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", prefix + "_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor ffn_normed = ffn_norm(residual);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", prefix + "_decoder_output")}));
  return decoder_output({residual, ffn_out});
}

Tensor Lfm2CausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                               Tensor input) {
  LayerHandle ffn_up(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim),
     withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  Tensor up = ffn_up(input);

  LayerHandle ffn_gate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim),
     withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  Tensor gate = ffn_gate(input);

  LayerHandle swiglu(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu")}));
  Tensor act = swiglu({gate, up});

  LayerHandle ffn_down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim),
     withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  return ffn_down(act);
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
