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

protected:
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

/**
 * @brief SmallThinkerCachedSlimCausalLM class
 * @note  Uses virtual expert weights with an LRU cache to keep recently-used
 *        experts mapped across tokens, reducing repeated mmap overhead.
 */
class SmallThinkerCachedSlimCausalLM : public SmallThinkerCausalLM {
public:
  static constexpr const char *architectures =
    "SmallThinkerCachedSlimForCausalLM";

  SmallThinkerCachedSlimCausalLM(json &cfg, json &generation_cfg,
                                 json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    SmallThinkerCausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~SmallThinkerCachedSlimCausalLM() = default;

protected:
  const char *getMoELayerType() const override {
    return "smallthinker_moe_cached_slim";
  }

  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  void registerCustomLayers() override;
};

/**
 * @brief SmallThinkerSparseCausalLM class
 * @note  BASE SmallThinker model (all experts resident, no LRU/mmap) with a
 *        ReLU-sparsity expert FFN. Clean testbed to isolate the sparsity
 *        speedup from the cached-slim caching machinery. Requires a sparse
 *        .bin (up/down plain per-neuron rows) — same layout as the sparse
 *        cached-slim variant.
 */
class SmallThinkerSparseCausalLM : public SmallThinkerCausalLM {
public:
  static constexpr const char *architectures = "SmallThinkerSparseForCausalLM";

  SmallThinkerSparseCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    SmallThinkerCausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~SmallThinkerSparseCausalLM() = default;

protected:
  const char *getMoELayerType() const override {
    return "smallthinker_moe_sparse";
  }

  void registerCustomLayers() override;
};

/**
 * @brief SmallThinkerSparseCachedSlimCausalLM class
 * @note  Cached-slim LRU + prefetch decoder graph, but the MoE expert FFN uses
 *        the sparse ReGLU compute (B3 hybrid). Combines on-device expert
 *        offloading with activation sparsity (paper §6.1 + §6.2). Uses the same
 *        "smallthinker_sparse" .bin as the resident base-sparse model.
 */
class SmallThinkerSparseCachedSlimCausalLM
  : public SmallThinkerCachedSlimCausalLM {
public:
  static constexpr const char *architectures =
    "SmallThinkerSparseCachedSlimForCausalLM";

  SmallThinkerSparseCachedSlimCausalLM(json &cfg, json &generation_cfg,
                                       json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    SmallThinkerCachedSlimCausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~SmallThinkerSparseCachedSlimCausalLM() = default;

protected:
  const char *getMoELayerType() const override {
    return "smallthinker_moe_sparse_cached_slim";
  }

  void registerCustomLayers() override;
};

/**
 * @brief SmallThinkerSparseSlimCausalLM class
 * @note  Slim (on-demand, no-cache) decoder, with the sparse ReGLU expert FFN.
 *        Uses the same "smallthinker_sparse" .bin as the resident base-sparse
 *        model (gate repacked, up/down plain).
 */
class SmallThinkerSparseSlimCausalLM : public SmallThinkerSlimCausalLM {
public:
  static constexpr const char *architectures =
    "SmallThinkerSparseSlimForCausalLM";

  SmallThinkerSparseSlimCausalLM(json &cfg, json &generation_cfg,
                                 json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    SmallThinkerSlimCausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~SmallThinkerSparseSlimCausalLM() = default;

protected:
  const char *getMoELayerType() const override {
    return "smallthinker_moe_sparse_slim";
  }

  void registerCustomLayers() override;
};

} // namespace causallm

#endif /* __SMALLTHINKER_CAUSAL_LM_H__ */
