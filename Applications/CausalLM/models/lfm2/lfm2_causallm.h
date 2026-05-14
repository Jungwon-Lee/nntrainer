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

  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

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

  void constructModel() override;

  void registerCustomLayers() override;

  /**
   * @brief Create a conv block for LFM2 hybrid architecture
   * @param layer_id Layer index
   * @param input_name Name of input layer
   * @return Vector of layer handles for the conv block
   */
  std::vector<LayerHandle> createConvBlock(const int layer_id,
                                           std::string input_name);

  /**
   * @brief Create MLP block with SwiGLU activation
   * @param layer_id Layer index
   * @param dim Model dimension
   * @param hidden_dim Hidden dimension for FFN
   * @param input_name Name of input layer
   * @return Vector of layer handles for MLP
   */
  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  /**
   * @brief Create transformer decoder block (attention or conv based)
   * @param layer_id Layer index
   * @param input_name Name of input layer
   * @return Vector of layer handles for the decoder block
   */
  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                 std::string input_name) override;

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
};

} // namespace causallm

#endif // __LFM2_CAUSAL_LM_H__
