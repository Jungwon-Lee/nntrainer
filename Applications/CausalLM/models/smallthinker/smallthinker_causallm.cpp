/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   smallthinker_causallm.cpp
 * @date   28 April 2026
 * @brief  SmallThinker causal language model.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <smallthinker_causallm.h>
#include <smallthinker_moe_layer.h>
#include <smallthinker_moe_layer_base_sparse.h>
#include <smallthinker_moe_layer_cached_slim.h>
#include <smallthinker_moe_layer_slim.h>
#include <smallthinker_moe_layer_sparse_cached_slim.h>
#include <smallthinker_moe_layer_sparse_slim.h>
#include <smallthinker_moe_prefetch_layer.h>
#include <stdexcept>

namespace causallm {

namespace {

std::vector<bool> parseLayerLayout(const json &cfg, const char *key,
                                   int num_layers, bool default_value) {
  std::vector<bool> layout(num_layers, default_value);
  if (!cfg.contains(key) || !cfg[key].is_array())
    return layout;

  const int limit = std::min<int>(num_layers, cfg[key].size());
  for (int i = 0; i < limit; ++i) {
    const json &value = cfg[key][i];
    if (value.is_boolean())
      layout[i] = value.get<bool>();
    else if (value.is_number_integer())
      layout[i] = value.get<int>() != 0;
  }
  return layout;
}

} // namespace

json &SmallThinkerCausalLM::normalizeConfig(json &cfg) {
  if (!cfg.contains("intermediate_size") &&
      cfg.contains("moe_ffn_hidden_size")) {
    cfg["intermediate_size"] = cfg["moe_ffn_hidden_size"];
  }
  if (!cfg.contains("sliding_window") && cfg.contains("sliding_window_size")) {
    cfg["sliding_window"] = cfg["sliding_window_size"];
  }
  if (!cfg.contains("tie_word_embeddings") ||
      !cfg["tie_word_embeddings"].is_boolean()) {
    cfg["tie_word_embeddings"] = true;
  }
  if (cfg.contains("sliding_window_pattern") &&
      !cfg["sliding_window_pattern"].is_number_unsigned()) {
    cfg["sliding_window_pattern"] = 1;
  }

  return cfg;
}

void SmallThinkerCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                           json &nntr_cfg) {
  CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  if (cfg.contains("sliding_window_layout") &&
      cfg["sliding_window_layout"].is_array() &&
      std::none_of(cfg["sliding_window_layout"].begin(),
                   cfg["sliding_window_layout"].end(), [](const json &enabled) {
                     return enabled.is_number_integer() &&
                            enabled.get<int>() != 0;
                   })) {
    SLIDING_WINDOW = UINT_MAX;
  }

  try {
    NUM_EXPERTS = cfg["moe_num_primary_experts"];
    NUM_EXPERTS_PER_TOK = cfg["moe_num_active_primary_experts"];
    INTERMEDIATE_SIZE = cfg["moe_ffn_hidden_size"];
    ROUTER_APPLY_SOFTMAX =
      cfg.contains("moe_primary_router_apply_softmax")
        ? cfg["moe_primary_router_apply_softmax"].get<bool>()
        : false;
  } catch (const std::exception &e) {
    throw std::runtime_error(
      "SmallThinker: required MoE config keys are missing");
  }

  rope_layout_ = parseLayerLayout(cfg, "rope_layout", NUM_LAYERS, true);
  sliding_window_layout_ =
    parseLayerLayout(cfg, "sliding_window_layout", NUM_LAYERS, false);
}

Tensor SmallThinkerCausalLM::createAttention(const int layer_id, int seq_len,
                                             int n_heads, int head_dim,
                                             Tensor query, Tensor key,
                                             Tensor value) {

  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  Tensor q = wq(query);

  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")}));
  Tensor k = wk(key);

  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")}));
  Tensor v = wv(value);

  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  const bool use_sliding_window =
    layer_id < static_cast<int>(sliding_window_layout_.size())
      ? sliding_window_layout_[layer_id]
      : false;
  const bool use_rope = layer_id < static_cast<int>(rope_layout_.size())
                          ? rope_layout_[layer_id]
                          : true;

  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", use_sliding_window ? SLIDING_WINDOW : UINT_MAX),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false"),
     withKey("use_rope", use_rope ? "true" : "false")}));
  Tensor a = mha({q, k, v, cache_k, cache_v});

  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  return wo(a);
}

Tensor SmallThinkerCausalLM::createTransformerDecoderBlock(const int layer_id,
                                                           Tensor input) {

  LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

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

  // SmallThinker MoE uses 2 inputs: expert input (ffn_normed) and router input
  // (pre-attention residual). The router is applied on the block's input before
  // the attention norm, matching the HF reference implementation.
  LayerHandle moe(createLayer(
    getMoELayerType(),
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", INTERMEDIATE_SIZE), withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "relu"),
     withKey("moe_router_apply_softmax",
             ROUTER_APPLY_SOFTMAX ? "true" : "false")}));
  Tensor ffn_out = moe({ffn_normed, input});

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output")}));
  return decoder_output({residual, ffn_out});
}

void SmallThinkerCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();

  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::SmallThinkerMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void SmallThinkerSlimCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();

  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::SmallThinkerSlimMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

// Check at startup whether prefetch is enabled (env var read once).
static const bool g_prefetch_enabled = []() {
  const char *env = std::getenv("NNTR_MOE_PREFETCH_THREADS");
  return !env || std::stoi(env) != 0;
}();

Tensor SmallThinkerCachedSlimCausalLM::createTransformerDecoderBlock(
  const int layer_id, Tensor input) {

  if (!g_prefetch_enabled) {
    // Prefetch disabled: fall back to base-class graph (no prefetch node).
    return SmallThinkerCausalLM::createTransformerDecoderBlock(layer_id, input);
  }

  // Insert a pass-through prefetch node at block entry. It consumes the
  // pre-attention `input`, fires background activate() tasks for the predicted
  // expert set, then passes the data through unchanged. Topology guarantees
  // this runs before attention.
  LayerHandle moe_pf(createLayer(
    SmallThinkerMoEPrefetchLayer::type,
    {withKey("name",
             "layer" + std::to_string(layer_id) + "_ffn_prefetch"),
     withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_router_apply_softmax",
             ROUTER_APPLY_SOFTMAX ? "true" : "false"),
     withKey("moe_layer_id", layer_id)}));
  Tensor pf = moe_pf(input); // pass-through; fires prefetch in forwarding()

  // Run attention on the original input (not through pf to avoid an extra
  // copy in the attention path; pf is used only as the MoE router input).
  LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name",
             "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  LayerHandle decoder_add(createLayer(
    "addition",
    {withKey("name",
             "layer" + std::to_string(layer_id) + "_decoder_add")}));
  Tensor residual = decoder_add({input, att_out});

  LayerHandle ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor ffn_normed = ffn_norm(residual);

  // MoE compute node: input[0] = ffn_normed (expert input),
  //                   input[1] = pf (router input, == original `input`).
  LayerHandle moe(createLayer(
    getMoELayerType(),
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", INTERMEDIATE_SIZE),
     withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "relu"),
     withKey("moe_router_apply_softmax",
             ROUTER_APPLY_SOFTMAX ? "true" : "false")}));
  Tensor ffn_out = moe({ffn_normed, pf});

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name",
             "layer" + std::to_string(layer_id) + "_decoder_output")}));
  return decoder_output({residual, ffn_out});
}

void SmallThinkerCachedSlimCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();

  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::SmallThinkerCachedSlimMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::SmallThinkerMoEPrefetchLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void SmallThinkerSparseCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();

  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::SmallThinkerSparseMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void SmallThinkerSparseCachedSlimCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();

  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::SmallThinkerSparseCachedSlimMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }

  // The sparse cached-slim variant reuses the cached-slim prefetch decoder
  // graph, so the pass-through prefetch node must also be registered.
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::SmallThinkerMoEPrefetchLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void SmallThinkerSparseSlimCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();

  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::SmallThinkerSparseSlimMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
