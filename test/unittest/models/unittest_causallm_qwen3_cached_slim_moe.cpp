// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_qwen3_cached_slim_moe.cpp
 * @date   15 June 2026
 * @brief  Tiny Qwen3CachedSlimMoE CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <qwen3_cached_slim_moe_causallm.h>
#include <qwen_moe_cache_policy.h>

#include <deque>
#include <list>
#include <map>
#include <vector>

namespace {

using TinyQwen3CachedSlimMoECausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Qwen3CachedSlimMoECausalLM>;

/**
 * @brief Make the tiny Qwen3CachedSlimMoE model config
 */
causallm::json makeTinyQwen3CachedSlimMoEConfig() {
  return {
    {"architectures", {"Qwen3CachedSlimMoeForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"moe_intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 1},
    {"num_key_value_heads", 4},
    {"num_experts", 4},
    {"num_experts_per_tok", 2},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Verify that Qwen3CachedSlimMoE can be instantiated and that greedy
 *        generation selects the argmax token from supplied logits.
 *
 * See unittest_causallm_qwen3_slim_moe.cpp for the reason WeightRoundTrip
 * and PromptProducesExpectedLogits are not included here.
 */
TEST(Qwen3CachedSlimMoETinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  auto tokenizer_path =
    causallm_test::makeTinyCausalLMFiles("Qwen3CachedSlimMoETinyModelTest",
                                         "GreedyGenerationSelectsArgmaxLogit",
                                         "Qwen3CachedSlimMoE_FP32")
      .tokenizer_path;

  auto model_cfg = makeTinyQwen3CachedSlimMoEConfig();
  auto gen_cfg = causallm_test::makeTinyGenerationConfig();
  auto nntr_cfg = causallm_test::makeTinyNntrainerConfig(
    tokenizer_path, causallm_test::makeTinyFp32DataType());

  auto model = std::make_unique<TinyQwen3CachedSlimMoECausalLM>(
    model_cfg, gen_cfg, nntr_cfg);
  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

TEST(Qwen3CachedSlimMoECacheTest, PlansFinalLruWithoutMappingAllExperts) {
  const std::list<int> loaded_experts = {0, 1, 2};
  const std::vector<int> active_experts = {2, 3, 4, 5};
  const std::deque<int> expert_recency = {5, 3, 1};

  const auto planned_cache =
    causallm::qwen3_cached_slim_detail::planExpertCache(
      loaded_experts, active_experts, expert_recency, 3);

  EXPECT_EQ(planned_cache, (std::list<int>{1, 3, 5}));
}

TEST(Qwen3CachedSlimMoECacheTest, KeepsNewestActiveExpertsWithoutHints) {
  const std::list<int> loaded_experts = {0, 1};
  const std::vector<int> active_experts = {1, 2, 3, 4};

  const auto planned_cache =
    causallm::qwen3_cached_slim_detail::planExpertCache(loaded_experts,
                                                        active_experts, {}, 3);

  EXPECT_EQ(planned_cache, (std::list<int>{2, 3, 4}));
}

TEST(Qwen3CachedSlimMoECacheTest, DeduplicatesActiveAndRecencyExperts) {
  const std::list<int> loaded_experts = {0, 1};
  const std::vector<int> active_experts = {1, 1, 2, 2};
  const std::deque<int> expert_recency = {2, 2, 0, 2};

  const auto planned_cache =
    causallm::qwen3_cached_slim_detail::planExpertCache(
      loaded_experts, active_experts, expert_recency, 4);

  EXPECT_EQ(planned_cache, (std::list<int>{1, 0, 2}));
}

} // namespace
