// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_causallm.h
 * @date   14 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   LFM2 model with hybrid attention-conv architecture
 */

#ifndef __LFM2_CAUSAL_LM_H__
#define __LFM2_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Lfm2Transformer class
 * @note LFM2-specific attention variant with Q/K normalization
 */
class Lfm2Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Lfm2Transformer";

  Lfm2Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Lfm2Transformer() = default;

  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  void registerCustomLayers() override;
};

/**
 * @brief Lfm2CausalLM class
 * @note LFM2 model with hybrid attention-conv blocks
 */
class Lfm2CausalLM : public CausalLM, public Lfm2Transformer {

public:
  static constexpr const char *architectures = "Lfm2ForCausalLM";

  Lfm2CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    Lfm2Transformer(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Lfm2CausalLM() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

  /**
   * @brief Create a conv block for LFM2 hybrid architecture
   * @param layer_id Layer index
   * @param input Symbolic input tensor
   * @return Symbolic output tensor of the conv block
   */
  Tensor createConvBlock(const int layer_id, Tensor input);

  /**
   * @brief Create MLP block with SwiGLU activation
   * @param layer_id Layer index
   * @param dim Model dimension
   * @param hidden_dim Hidden dimension for FFN
   * @param input Symbolic input tensor
   * @return Symbolic output tensor of MLP
   */
  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  /**
   * @brief Create transformer decoder block (attention or conv based)
   * @param layer_id Layer index
   * @param input Symbolic input tensor
   * @return Symbolic output tensor of the decoder block
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  /**
   * @brief Test constructor with minimal config
   * @note Creates model with small dimensions for testing
   */
  static Lfm2CausalLM createTestModel();

  ModelHandle &getModel() { return model; }

public:
  unsigned int CONV_DIM;       ///< Conv layer dimension
  unsigned int CONV_DIM_OUT;   ///< Conv output dimension
  unsigned int CONV_L_CACHE;   ///< Conv cache length
  bool CONV_BIAS;              ///< Whether conv has bias
  std::vector<std::string> LAYER_TYPES; ///< Per-layer operator type
};

} // namespace causallm

#endif // __LFM2_CAUSAL_LM_H__
