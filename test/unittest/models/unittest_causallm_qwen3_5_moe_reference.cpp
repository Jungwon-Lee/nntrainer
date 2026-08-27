// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_qwen3_5_moe_reference.cpp
 * @date   27 August 2026
 * @brief  Hugging Face differential tests for Qwen3.5/Qwen3.6 MoE.
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <qwen3_5_moe_causallm.h>

#include <memory>

namespace {

class TinyQwen3_5MoeReferenceCausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::Qwen3_5MoeCausalLM> {
public:
  TinyQwen3_5MoeReferenceCausalLM(causallm::json &cfg,
                                  causallm::json &generation_cfg,
                                  causallm::json &nntr_cfg) :
    causallm::Transformer(textConfig(cfg), generation_cfg, nntr_cfg,
                          causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Qwen3_5MoeCausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

causallm_test::DifferentialModel qwen3_5MoeModel() {
  return {
    "qwen3_5_moe_tiny",
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyQwen3_5MoeReferenceCausalLM>(
        cfg, generation_cfg, nntr_cfg);
    },
  };
}

TEST(Qwen3_5MoeDifferentialTest, FP32MatchesHFReference) {
  causallm_test::runFp32DifferentialChecks(qwen3_5MoeModel());
}

TEST(Qwen3_5MoeDifferentialTest, Q40CloseToHFReference) {
  causallm_test::runQ40DifferentialChecks(qwen3_5MoeModel());
}

} // namespace
