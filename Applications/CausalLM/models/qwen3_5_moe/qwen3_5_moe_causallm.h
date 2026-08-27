// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_5_moe_causallm.h
 * @date   27 August 2026
 * @brief  Text-only Qwen3.5/Qwen3.6 MoE causal language model.
 */

#ifndef __QWEN3_5_MOE_CAUSALLM_H__
#define __QWEN3_5_MOE_CAUSALLM_H__

#include <qwen3_5_causallm.h>

namespace causallm {

/**
 * @brief Qwen3.5 MoE extension used by Qwen3.6-35B-A3B.
 *
 * The hybrid DeltaNet/attention backbone is implemented by Qwen3_5CausalLM.
 * This class replaces only the dense SwiGLU FFN with routed and shared experts.
 */
class Qwen3_5MoeCausalLM : public Qwen3_5CausalLM {
public:
  static constexpr const char *architectures =
    "Qwen3_5MoeForConditionalGeneration";
  static constexpr const char *causal_lm_architectures =
    "Qwen3_5MoeForCausalLM";

  Qwen3_5MoeCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);
  ~Qwen3_5MoeCausalLM() override = default;

protected:
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;
  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;
  void registerCustomLayers() override;

private:
  unsigned int num_experts = 0;
  unsigned int num_experts_per_token = 0;
  unsigned int moe_intermediate_size = 0;
  unsigned int shared_expert_intermediate_size = 0;
};

} // namespace causallm

#endif // __QWEN3_5_MOE_CAUSALLM_H__
