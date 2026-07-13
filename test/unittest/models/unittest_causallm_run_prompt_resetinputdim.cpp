// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_run_prompt_resetinputdim.cpp
 * @date   08 July 2026
 * @brief  End-to-end verification that Qwen2, Qwen3, Gemma3 and Gemma4
 *         survive CausalLM::run()'s resetInputDimension() call when the
 *         same model instance is reused across prompts of different
 *         lengths. Exercises the real run()/resetInputDimension() path,
 *         unlike the golden prefillLogits()-based tests elsewhere in this
 *         suite which call NeuralNetwork::incremental_inference() directly
 *         and therefore never reach resetInputDimension() at all.
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @note See unittest_causallm_moe_resetinputdim.cpp for the equivalent
 *       Qwen3MoE coverage, and for why GptOss / SlimMoE / CachedSlimMoE
 *       variants are intentionally not covered by an analogous test here.
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <gemma3_causallm.h>
#include <gemma4_causallm.h>
#include <layer.h>
#include <layer_context.h>
#include <qwen2_causallm.h>
#include <qwen3_causallm.h>

namespace {

/**
 * @brief Zero every FP32 weight, leaving rms_norm at 1 and a couple of
 * embedding rows non-zero so the forward pass is deterministic. No
 * correctness assertions are made on the produced logits here -- these
 * tests only need the resetInputDimension()-driven forward pass to run
 * without crashing/OOB across multiple prompt lengths on the same
 * instance.
 */
template <typename Model> void zeroWeights(Model &model) {
  model.forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        } else if (layer.getName().find("_layer_scalar") != std::string::npos) {
          weight.setValue(1.0f);
        }
      }
    });
}

using TinyQwen2CausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Qwen2CausalLM>;
using TinyQwen3CausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;

/**
 * @brief Tiny Gemma3 CausalLM adapter (needs sanitizeConfig() before the
 * virtual Transformer base is constructed, see unittest_causallm_gemma3.cpp)
 */
class TinyGemma3CausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::Gemma3CausalLM> {
public:
  TinyGemma3CausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                     causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Gemma3CausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Tiny Gemma4 CausalLM adapter (needs sanitizeConfig() to flatten
 * text_config before the virtual Transformer base is constructed, see
 * unittest_causallm_gemma4.cpp)
 */
class TinyGemma4CausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::Gemma4CausalLM> {
public:
  TinyGemma4CausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                     causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Gemma4CausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

causallm::json makeTinyQwen2Config() {
  return {
    {"architectures", {"Qwen2ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 1},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

causallm::json makeTinyQwen3Config() {
  return {
    {"architectures", {"Qwen3ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 1},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

causallm::json makeTinyGemma3Config() {
  return {
    {"architectures", {"Gemma3ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 2},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-6},
    {"rope_theta", 1000000},
    {"sliding_window", 4},
    {"sliding_window_pattern", 2},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

causallm::json makeTinyGemma4Config() {
  return {
    {"architectures", {"Gemma4ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"text_config",
     {
       {"head_dim", 8},
       {"hidden_size", 64},
       {"hidden_size_per_layer_input", 32},
       {"intermediate_size", 64},
       {"layer_types", {"sliding_attention", "full_attention"}},
       {"max_position_embeddings", 8},
       {"num_attention_heads", 8},
       {"num_hidden_layers", 2},
       {"num_key_value_heads", 4},
       {"rms_norm_eps", 1e-6},
       {"rope_theta", 1000000},
       {"sliding_window", 4},
       {"tie_word_embeddings", true},
       {"vocab_size", 32},
       {"vocab_size_per_layer_input", 32},
     }},
  };
}

/**
 * @brief Build, initialize and zero-weight a tiny model of the given type,
 * then run two prompts of different lengths through the same instance.
 *
 * Asserts the final logits tensor is finite after each run: dtype-clobbering
 * bugs in a layer's updateTensorsByInputDimensions() (e.g. a residual add
 * whose input view gets reinterpreted at the wrong dtype) corrupt the
 * activation stream with NaN/Inf without necessarily throwing, so a plain
 * ASSERT_NO_THROW is not sufficient to catch them.
 */
template <typename Model>
void runResetInputDimensionRegression(
  const std::string &suite, const std::string &test,
  const std::string &fixture_name, causallm::json model_cfg,
  const causallm_test::TinyCausalLMDataType &data_type =
    causallm_test::makeTinyFp32DataType()) {
  auto files = causallm_test::makeTinyCausalLMFiles(suite, test, fixture_name);

  auto gen_cfg = causallm_test::makeTinyGenerationConfig();
  auto nntr_cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);

  auto model = std::make_unique<Model>(model_cfg, gen_cfg, nntr_cfg);
  model->initializeModel();
  zeroWeights(*model);

  // Longer than a single token so resetInputDimension() actually resizes
  // the sequence length away from whatever finalize() originally allocated.
  ASSERT_NO_THROW(model->runPrompt("hello tok4 tok5 tok6 tok7"));
  EXPECT_TRUE(model->hasRun());
  EXPECT_TRUE(model->lastOutputIsFinite());

  // Run a second, shorter prompt through the same model instance:
  // resetInputDimension() is called on every run(), so this exercises
  // shrinking tensors back down too.
  ASSERT_NO_THROW(model->runPrompt("hello"));
  EXPECT_TRUE(model->hasRun());
  EXPECT_TRUE(model->lastOutputIsFinite());
}

TEST(RunPromptResetInputDimensionEndToEnd, Qwen2RunPromptDoesNotCrash) {
  runResetInputDimensionRegression<TinyQwen2CausalLM>(
    "RunPromptResetInputDimensionEndToEnd", "Qwen2RunPromptDoesNotCrash",
    "Qwen2_FP32", makeTinyQwen2Config());
}

TEST(RunPromptResetInputDimensionEndToEnd, Qwen3RunPromptDoesNotCrash) {
  runResetInputDimensionRegression<TinyQwen3CausalLM>(
    "RunPromptResetInputDimensionEndToEnd", "Qwen3RunPromptDoesNotCrash",
    "Qwen3_FP32", makeTinyQwen3Config());
}

TEST(RunPromptResetInputDimensionEndToEnd, Gemma3RunPromptDoesNotCrash) {
  runResetInputDimensionRegression<TinyGemma3CausalLM>(
    "RunPromptResetInputDimensionEndToEnd", "Gemma3RunPromptDoesNotCrash",
    "Gemma3_FP32", makeTinyGemma3Config());
}

TEST(RunPromptResetInputDimensionEndToEnd, Gemma4RunPromptDoesNotCrash) {
  runResetInputDimensionRegression<TinyGemma4CausalLM>(
    "RunPromptResetInputDimensionEndToEnd", "Gemma4RunPromptDoesNotCrash",
    "Gemma4_FP32", makeTinyGemma4Config());
}

#ifdef ENABLE_FP16
/**
 * @brief FP16-activation coverage for the same resetInputDimension() path.
 *
 * Regression test for a bug where AdditionLayer/MultiplyLayer/MultiOutLayer/
 * PerLayerSliceLayer's updateTensorsByInputDimensions() overwrote a tensor's
 * entire TensorDim (including dtype) with the caller-supplied dimension --
 * which defaults to FP32 -- instead of only updating height. On an FP16
 * activation build this silently reinterpreted FP16 buffers as FP32 at the
 * first decoder block's residual add, corrupting the activation stream with
 * NaN/Inf from layer 0 onward. Only reproduces with FP16 activations:
 * FP32 models never hit the dtype mismatch, and this file is excluded from
 * the FP32-only host meson suite (see enable_causallm_model_unittest in
 * Applications/CausalLM/meson.build), so this only runs in the Android
 * FP16 unittest_causallm_models build.
 *
 * Uses the Q4_0-FP16 (weight+activation) data type -- the actual combination
 * shipped on Android -- since plain FP32 weights are not supported with
 * FP16 activations: FC/lm_head layers compute `input_.dot(weight_, ...)`,
 * so with FP16 activations the dot is dispatched through HalfTensor::dot(),
 * which only accepts an FP16, Q4_0, or Q6_K weight operand and throws
 * "unsupported datatype" for a plain FP32 weight.
 */
TEST(RunPromptResetInputDimensionEndToEnd, Qwen2RunPromptFp16IsFinite) {
  runResetInputDimensionRegression<TinyQwen2CausalLM>(
    "RunPromptResetInputDimensionEndToEnd", "Qwen2RunPromptFp16IsFinite",
    "Qwen2_FP16", makeTinyQwen2Config(),
    causallm_test::makeTinyQ40Fp16DataType());
}

TEST(RunPromptResetInputDimensionEndToEnd, Qwen3RunPromptFp16IsFinite) {
  runResetInputDimensionRegression<TinyQwen3CausalLM>(
    "RunPromptResetInputDimensionEndToEnd", "Qwen3RunPromptFp16IsFinite",
    "Qwen3_FP16", makeTinyQwen3Config(),
    causallm_test::makeTinyQ40Fp16DataType());
}

TEST(RunPromptResetInputDimensionEndToEnd, Gemma3RunPromptFp16IsFinite) {
  runResetInputDimensionRegression<TinyGemma3CausalLM>(
    "RunPromptResetInputDimensionEndToEnd", "Gemma3RunPromptFp16IsFinite",
    "Gemma3_FP16", makeTinyGemma3Config(),
    causallm_test::makeTinyQ40Fp16DataType());
}

TEST(RunPromptResetInputDimensionEndToEnd, Gemma4RunPromptFp16IsFinite) {
  runResetInputDimensionRegression<TinyGemma4CausalLM>(
    "RunPromptResetInputDimensionEndToEnd", "Gemma4RunPromptFp16IsFinite",
    "Gemma4_FP16", makeTinyGemma4Config(),
    causallm_test::makeTinyQ40Fp16DataType());
}
#endif // ENABLE_FP16

} // namespace
