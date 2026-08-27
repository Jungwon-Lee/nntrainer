// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_5_causallm.cpp
 * @date   19 August 2026
 * @brief  Text-only Qwen3.5/Qwen3.6 causal language model.
 */

#include "qwen3_5_causallm.h"

#include "qwen3_5_delta_layer.h"

#include <app_context.h>
#include <custom_multiply.h>
#include <engine.h>
#include <llm_util.hpp>
#include <mha_core.h>
#include <reshaped_rms_norm.h>

namespace causallm {

json &Qwen3_5CausalLM::textConfig(json &cfg) {
  if (cfg.contains("text_config") && cfg["text_config"].is_object())
    return cfg["text_config"];
  return cfg;
}

Qwen3_5CausalLM::Qwen3_5CausalLM(json &cfg, json &generation_cfg,
                                 json &nntr_cfg) :
  Transformer(textConfig(cfg), generation_cfg, nntr_cfg, ModelType::CAUSALLM),
  CausalLM(textConfig(cfg), generation_cfg, nntr_cfg) {
  setupParameters(textConfig(cfg), generation_cfg, nntr_cfg);
}

void Qwen3_5CausalLM::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  try {
    layer_types = cfg.at("layer_types").get<std::vector<std::string>>();
    linear_conv_kernel_dim = cfg.at("linear_conv_kernel_dim");
    linear_key_head_dim = cfg.at("linear_key_head_dim");
    linear_value_head_dim = cfg.at("linear_value_head_dim");
    linear_num_key_heads = cfg.at("linear_num_key_heads");
    linear_num_value_heads = cfg.at("linear_num_value_heads");
    partial_rotary_factor = cfg.value("partial_rotary_factor", 1.0f);
    if (!cfg.contains("partial_rotary_factor") &&
        cfg.contains("rope_parameters") && cfg["rope_parameters"].is_object())
      partial_rotary_factor =
        cfg["rope_parameters"].value("partial_rotary_factor", 1.0f);
  } catch (const std::exception &error) {
    throw std::runtime_error(
      std::string("Qwen3_5 text config is incomplete: ") + error.what());
  }

  NNTR_THROW_IF(layer_types.size() != static_cast<size_t>(NUM_LAYERS),
                std::invalid_argument)
    << "Qwen3_5 layer_types must match num_hidden_layers";
  for (const auto &layer_type : layer_types) {
    NNTR_THROW_IF(layer_type != "linear_attention" &&
                    layer_type != "full_attention",
                  std::invalid_argument)
      << "unsupported Qwen3_5 layer type: " << layer_type;
  }
}

Tensor Qwen3_5CausalLM::createTransformerDecoderBlock(const int layer_id,
                                                      Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id);
  LayerHandle input_norm(
    createLayer("rms_norm", {withKey("name", prefix + "_attention_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false")}));
  Tensor normed = input_norm(input);

  Tensor mixed = layer_types[layer_id] == "linear_attention"
                   ? createDeltaNet(layer_id, normed)
                   : createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS,
                                     HEAD_DIM, normed, normed, normed);
  LayerHandle mixer_add(
    createLayer("addition", {withKey("name", prefix + "_decoder_add")}));
  Tensor residual = mixer_add({input, mixed});

  LayerHandle ffn_norm(
    createLayer("rms_norm", {withKey("name", prefix + "_ffn_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false")}));
  Tensor ffn_output =
    createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_norm(residual));
  LayerHandle output_add(
    createLayer("addition", {withKey("name", prefix + "_decoder_output")}));
  return output_add({residual, ffn_output});
}

Tensor Qwen3_5CausalLM::createDeltaNet(const int layer_id, Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id);
  const unsigned int key_dim = linear_num_key_heads * linear_key_head_dim;
  const unsigned int value_dim = linear_num_value_heads * linear_value_head_dim;
  const unsigned int conv_dim = 2 * key_dim + value_dim;

  LayerHandle qkv(
    createLayer("fully_connected",
                {withKey("name", prefix + "_linear_qkv"),
                 withKey("unit", conv_dim), withKey("disable_bias", "true")}));
  LayerHandle z(
    createLayer("fully_connected",
                {withKey("name", prefix + "_linear_z"),
                 withKey("unit", value_dim), withKey("disable_bias", "true")}));
  LayerHandle beta(
    createLayer("fully_connected", {withKey("name", prefix + "_linear_b"),
                                    withKey("unit", linear_num_value_heads),
                                    withKey("disable_bias", "true"),
                                    withKey("weight_dtype", "FP32")}));
  LayerHandle decay(
    createLayer("fully_connected", {withKey("name", prefix + "_linear_a"),
                                    withKey("unit", linear_num_value_heads),
                                    withKey("disable_bias", "true"),
                                    withKey("weight_dtype", "FP32")}));

  LayerHandle delta(createLayer(
    "qwen3_5_delta", {withKey("name", prefix + "_linear_delta"),
                      withKey("num_key_heads", linear_num_key_heads),
                      withKey("num_value_heads", linear_num_value_heads),
                      withKey("key_head_dim", linear_key_head_dim),
                      withKey("value_head_dim", linear_value_head_dim),
                      withKey("conv_kernel_size", linear_conv_kernel_dim),
                      withKey("epsilon", std::to_string(NORM_EPS))}));
  Tensor delta_output =
    delta({qkv(input), z(input), beta(input), decay(input)});

  LayerHandle out(
    createLayer("fully_connected",
                {withKey("name", prefix + "_linear_out"), withKey("unit", DIM),
                 withKey("disable_bias", "true")}));
  return out(delta_output);
}

Tensor Qwen3_5CausalLM::createAttention(const int layer_id, int seq_len,
                                        int n_heads, int head_dim, Tensor query,
                                        Tensor key, Tensor value) {
  const std::string prefix = "layer" + std::to_string(layer_id);
  const unsigned int query_dim = n_heads * head_dim;

  // The converter rewrites HF's per-head [query | gate] layout into two
  // contiguous halves so the graph can split along its width dimension.
  LayerHandle q_gate_proj(
    createLayer("fully_connected", {withKey("name", prefix + "_wq"),
                                    withKey("unit", 2 * query_dim),
                                    withKey("disable_bias", "true")}));
  LayerHandle q_gate_split(
    createLayer("split", {withKey("name", prefix + "_q_gate_split"),
                          withKey("axis", 3), withKey("split_number", 2)}));
  Tensor q_gate = q_gate_split(q_gate_proj(query));
  Tensor q = q_gate.output(0);
  Tensor gate = q_gate.output(1);

  LayerHandle q_norm(createLayer("reshaped_rms_norm",
                                 {withKey("name", prefix + "_q_norm"),
                                  withKey("packed", "false"),
                                  withKey("epsilon", std::to_string(NORM_EPS)),
                                  withKey("feature_size", head_dim)}));
  q = q_norm(q);

  LayerHandle k_proj(createLayer(
    "fully_connected", {withKey("name", prefix + "_wk"),
                        withKey("unit", head_dim * NUM_KEY_VALUE_HEADS),
                        withKey("disable_bias", "true")}));
  LayerHandle k_norm(createLayer("reshaped_rms_norm",
                                 {withKey("name", prefix + "_k_norm"),
                                  withKey("packed", "false"),
                                  withKey("epsilon", std::to_string(NORM_EPS)),
                                  withKey("feature_size", head_dim)}));
  Tensor k = k_norm(k_proj(key));

  LayerHandle v_proj(createLayer(
    "fully_connected", {withKey("name", prefix + "_wv"),
                        withKey("unit", head_dim * NUM_KEY_VALUE_HEADS),
                        withKey("disable_bias", "true")}));
  Tensor v = v_proj(value);

  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
  LayerHandle attention(createLayer(
    "mha_core",
    {withKey("name", prefix + "_attention"), withKey("num_heads", n_heads),
     withKey("num_heads_kv", NUM_KEY_VALUE_HEADS),
     withKey("max_timestep", MAX_SEQ_LEN), withKey("rope_theta", ROPE_THETA),
     withKey("rope_partial_rotary_factor",
             std::to_string(partial_rotary_factor)),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", NUM_TO_GENERATE),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));
  Tensor attention_output = attention({q, k, v, cache_k, cache_v});

  LayerHandle gate_activation(
    createLayer("activation", {withKey("name", prefix + "_attention_gate"),
                               withKey("activation", "sigmoid")}));
  LayerHandle gate_multiply(createLayer(
    "custom_multiply", {withKey("name", prefix + "_attention_gate_mul")}));
  attention_output = gate_multiply({attention_output, gate_activation(gate)});

  LayerHandle out(
    createLayer("fully_connected",
                {withKey("name", prefix + "_attention_out"),
                 withKey("unit", DIM), withKey("disable_bias", "true")}));
  return out(attention_output);
}

void Qwen3_5CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  const auto &engine = nntrainer::Engine::Global();
  auto *context =
    static_cast<nntrainer::AppContext *>(engine.getRegisteredContext("cpu"));

  try {
    context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (const std::invalid_argument &) {
  }
  try {
    context->registerFactory(
      nntrainer::createLayer<causallm::CustomMultiplyLayer>);
  } catch (const std::invalid_argument &) {
  }
  try {
    context->registerFactory(
      nntrainer::createLayer<causallm::Qwen3_5DeltaLayer>);
  } catch (const std::invalid_argument &) {
  }
}

} // namespace causallm
