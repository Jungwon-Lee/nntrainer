// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_5_causallm.h
 * @date   19 August 2026
 * @brief  Text-only Qwen3.5/Qwen3.6 causal language model.
 */

#ifndef __QWEN3_5_CAUSALLM_H__
#define __QWEN3_5_CAUSALLM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Dense Qwen3.5/Qwen3.6 text backbone.
 *
 * The official checkpoint architecture is ConditionalGeneration because it
 * also carries a vision encoder. This class intentionally implements only the
 * language-model path; image inputs and MTP are outside the first milestone.
 */
class Qwen3_5CausalLM : public CausalLM {
public:
  static constexpr const char *architectures =
    "Qwen3_5ForConditionalGeneration";
  static constexpr const char *causal_lm_architectures = "Qwen3_5ForCausalLM";

  Qwen3_5CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);
  ~Qwen3_5CausalLM() override = default;

protected:
  static json &textConfig(json &cfg);
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;
  void registerCustomLayers() override;

private:
  Tensor createDeltaNet(const int layer_id, Tensor input);

  std::vector<std::string> layer_types;
  unsigned int linear_conv_kernel_dim = 0;
  unsigned int linear_key_head_dim = 0;
  unsigned int linear_value_head_dim = 0;
  unsigned int linear_num_key_heads = 0;
  unsigned int linear_num_value_heads = 0;
  float partial_rotary_factor = 1.0f;
};

} // namespace causallm

#endif // __QWEN3_5_CAUSALLM_H__
