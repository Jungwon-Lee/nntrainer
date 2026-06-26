// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_causallm.h
 * @date   28 April 2026
 * @brief  SmallThinker causal language model.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#ifndef __SMALLTHINKER_CAUSAL_LM_H__
#define __SMALLTHINKER_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief SmallThinkerCausalLM class
 * @note  Implements SmallThinker's MoE decoder structure.
 */
class SmallThinkerCausalLM : public CausalLM {
public:
  static constexpr const char *architectures = "SmallThinkerForCausalLM";

  SmallThinkerCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~SmallThinkerCausalLM() = default;

protected:
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

  virtual const char *getMoELayerType() const { return "smallthinker_moe"; }

  static json &normalizeConfig(json &cfg);

private:
  unsigned int NUM_EXPERTS;
  unsigned int NUM_EXPERTS_PER_TOK;
  bool ROUTER_APPLY_SOFTMAX;
  std::vector<bool> rope_layout_;
  std::vector<bool> sliding_window_layout_;
};

/**
 * @brief SmallThinkerSlimCausalLM class
 * @note  Uses virtual expert weights; each expert is loaded, computed, and
 *        immediately unloaded (no persistent cache).
 */
class SmallThinkerSlimCausalLM : public SmallThinkerCausalLM {
public:
  static constexpr const char *architectures = "SmallThinkerSlimForCausalLM";

  SmallThinkerSlimCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    SmallThinkerCausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~SmallThinkerSlimCausalLM() = default;

protected:
  const char *getMoELayerType() const override {
    return "smallthinker_moe_slim";
  }

  void registerCustomLayers() override;
};

} // namespace causallm

#endif /* __SMALLTHINKER_CAUSAL_LM_H__ */
