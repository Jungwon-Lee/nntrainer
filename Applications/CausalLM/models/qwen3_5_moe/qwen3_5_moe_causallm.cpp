// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_5_moe_causallm.cpp
 * @date   27 August 2026
 * @brief  Text-only Qwen3.5/Qwen3.6 MoE causal language model.
 */

#include "qwen3_5_moe_causallm.h"

#include "qwen3_5_moe_layer.h"

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>

namespace causallm {

Qwen3_5MoeCausalLM::Qwen3_5MoeCausalLM(json &cfg, json &generation_cfg,
                                       json &nntr_cfg) :
  Transformer(textConfig(cfg), generation_cfg, nntr_cfg, ModelType::CAUSALLM),
  Qwen3_5CausalLM(cfg, generation_cfg, nntr_cfg) {
  setupParameters(textConfig(cfg), generation_cfg, nntr_cfg);
}

void Qwen3_5MoeCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                         json &nntr_cfg) {
  Qwen3_5CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
  try {
    num_experts = cfg.at("num_experts");
    num_experts_per_token = cfg.at("num_experts_per_tok");
    moe_intermediate_size = cfg.at("moe_intermediate_size");
    shared_expert_intermediate_size = cfg.at("shared_expert_intermediate_size");
  } catch (const std::exception &error) {
    throw std::runtime_error(
      std::string("Qwen3_5Moe text config is incomplete: ") + error.what());
  }
}

Tensor Qwen3_5MoeCausalLM::createMlp(const int layer_id, int dim,
                                     int hidden_dim, Tensor input) {
  LayerHandle moe(createLayer(
    "qwen3_5_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", moe_intermediate_size),
     withKey("num_experts", num_experts),
     withKey("num_experts_per_token", num_experts_per_token),
     withKey("shared_expert_intermediate_size",
             shared_expert_intermediate_size)}));
  return moe(input);
}

void Qwen3_5MoeCausalLM::registerCustomLayers() {
  Qwen3_5CausalLM::registerCustomLayers();
  const auto &engine = nntrainer::Engine::Global();
  auto *context =
    static_cast<nntrainer::AppContext *>(engine.getRegisteredContext("cpu"));

  try {
    context->registerFactory(nntrainer::createLayer<causallm::Qwen3_5MoeLayer>);
  } catch (const std::invalid_argument &) {
  }
}

} // namespace causallm
