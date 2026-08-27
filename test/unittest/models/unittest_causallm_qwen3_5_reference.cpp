// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_qwen3_5_reference.cpp
 * @date   27 August 2026
 * @brief  Hugging Face differential tests for dense Qwen3.5/Qwen3.6.
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <qwen3_5_causallm.h>

#include <memory>

namespace {

class TinyQwen3_5ReferenceCausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::Qwen3_5CausalLM> {
public:
  TinyQwen3_5ReferenceCausalLM(causallm::json &cfg,
                               causallm::json &generation_cfg,
                               causallm::json &nntr_cfg) :
    causallm::Transformer(textConfig(cfg), generation_cfg, nntr_cfg,
                          causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Qwen3_5CausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

causallm_test::DifferentialModel qwen3_5Model() {
  return {
    "qwen3_5_tiny",
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyQwen3_5ReferenceCausalLM>(cfg, generation_cfg,
                                                            nntr_cfg);
    },
  };
}

TEST(Qwen3_5DifferentialTest, FP32MatchesHFReference) {
  causallm_test::runFp32DifferentialChecks(qwen3_5Model());
}

TEST(Qwen3_5DifferentialTest, Q40CloseToHFReference) {
  causallm_test::runQ40DifferentialChecks(qwen3_5Model());
}

} // namespace
