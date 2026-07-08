// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_moe_resetinputdim.cpp
 * @date   08 July 2026
 * @brief  End-to-end verification that MoE layers survive
 *         CausalLM::run()'s resetInputDimension() call. Exercises the real
 *         run()/resetInputDimension path, unlike the golden
 *         prefillLogits()-based tests elsewhere in this suite which bypass
 *         it entirely.
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @note Only Qwen3MoE (qwen_moe) is exercised here.
 *       - GptOss (gpt_oss_moe) is intentionally NOT covered: once its MoE
 *         layer's updateTensorsByInputDimensions() override stops throwing,
 *         a separate pre-existing bug in GptOss's MHACoreLayer/sliding-window
 *         attention path segfaults on resetInputDimension(). That is a
 *         distinct, deeper issue outside the scope of the MoE tensor-resize
 *         fix and is tracked separately.
 *       - SlimMoE (moe_slim), Qwen3CachedSlimMoE (qwen_moe_cached), and
 *         GptOssCachedSlim (gpt_oss_moe_slim_cached) call
 *         Tensor::activate()/deactivate() unconditionally in forwarding() for
 *         lazy mmap-based expert weight loading; in this in-process
 *         tiny-test harness their weights are plain in-memory tensors, so
 *         activate() hangs (see the pre-existing comment in
 *         unittest_causallm_qwen3_slim_moe.cpp). Running them here would
 *         risk hanging the test binary, so they are intentionally not
 *         covered by this file; their fix was verified by structural code
 *         inspection (identical requestTensor/updateTensor pattern to the
 *         one covered here).
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <qwen3_moe_causallm.h>

namespace {

using TinyQwen3MoECausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Qwen3MoECausalLM>;

causallm::json makeTinyQwen3MoEConfig() {
  return {
    {"architectures", {"Qwen3MoeForCausalLM"}},
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
 * @brief Zero all FP32 weights, but set MoE gate weights to non-uniform
 * values so routing is deterministic. No correctness assertions are made
 * here -- this only needs to run forward without crashing/OOB.
 */
template <typename Model>
void zeroWeightsWithMoEGate(Model &model, const std::string &moe_layer_type) {
  model.forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      if (layer.getType() == moe_layer_type) {
        auto &gate = context.getWeight(0);
        const auto dim = gate.getDim();
        const unsigned hidden = dim.height();
        const unsigned num_exp = dim.width();
        for (unsigned h = 0; h < hidden; ++h)
          for (unsigned e = 0; e < num_exp; ++e)
            gate.setValue(0, 0, h, e, 1.0f / (e + 1));

        for (unsigned int i = 1; i < context.getNumWeights(); ++i) {
          auto &w = context.getWeight(i);
          if (w.getDataType() == ml::train::TensorDim::DataType::FP32)
            w.setValue(0.0f);
        }
        return;
      }

      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (layer.getType() == "rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    });
}

TEST(MoEResetInputDimensionEndToEnd, Qwen3MoERunPromptDoesNotCrash) {
  auto files = causallm_test::makeTinyCausalLMFiles(
    "MoEResetInputDimensionEndToEnd", "Qwen3MoERunPromptDoesNotCrash",
    "Qwen3MoE_FP32");

  auto model_cfg = makeTinyQwen3MoEConfig();
  auto gen_cfg = causallm_test::makeTinyGenerationConfig();
  auto nntr_cfg = causallm_test::makeTinyNntrainerConfig(
    files.tokenizer_path, causallm_test::makeTinyFp32DataType());

  auto model =
    std::make_unique<TinyQwen3MoECausalLM>(model_cfg, gen_cfg, nntr_cfg);
  model->initializeModel();
  zeroWeightsWithMoEGate(*model, "qwen_moe");

  // Longer than a single token so resetInputDimension() actually resizes the
  // sequence length away from whatever finalize() originally allocated.
  ASSERT_NO_THROW(model->runPrompt("hello tok4 tok5 tok6 tok7"));
  EXPECT_TRUE(model->hasRun());

  // Run a second, shorter prompt through the same model instance:
  // resetInputDimension() is called on every run(), so this exercises
  // shrinking the router_logits/expert_mask tensors back down too.
  ASSERT_NO_THROW(model->runPrompt("hello"));
  EXPECT_TRUE(model->hasRun());
}

} // namespace
