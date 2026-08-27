// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_qwen3_5_moe.cpp
 * @date   19 August 2026
 * @brief  Tiny Qwen3.5/Qwen3.6 MoE CausalLM model unit tests
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <qwen3_5_moe_causallm.h>

#include <map>

namespace {

constexpr int tiny_qwen3_5_moe_num_layers = 2;

/**
 * @brief Adapter that initializes the virtual Transformer from text_config
 */
class TinyQwen3_5MoeCausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::Qwen3_5MoeCausalLM> {
public:
  TinyQwen3_5MoeCausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                         causallm::json &nntr_cfg) :
    causallm::Transformer(textConfig(cfg), generation_cfg, nntr_cfg,
                          causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Qwen3_5MoeCausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Populate analytically tractable weights for both hybrid paths
 *
 * All token-mixer and MoE projections are zero, so both residual blocks are
 * identities. The DeltaNet's gated RMS scale and regular RMS scales are one.
 * Untied embedding/LM-head entries then produce known final logits.
 */
void setupQwen3_5MoeDeterministicWeights(TinyQwen3_5MoeCausalLM &model) {
  model.forEachLayer([](ml::train::Layer &layer,
                        nntrainer::RunLayerContext &context, void *) {
    for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
      auto &weight = context.getWeight(i);
      if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
        continue;
      weight.setValue(0.0f);
    }

    if (layer.getType() == "rms_norm" ||
        layer.getType() == "reshaped_rms_norm") {
      context.getWeight(0).setValue(1.0f);
    } else if (layer.getType() == "qwen3_5_delta") {
      // conv_weight, dt_bias, A_log, norm_weight
      auto &conv_weight = context.getWeight(0);
      conv_weight.setValue(0, 0, 0, 0, 1.0f);
      conv_weight.setValue(0, 0, 0, 32, 1.0f);
      conv_weight.setValue(0, 0, 0, 64, 1.0f);
      context.getWeight(3).setValue(1.0f);
    } else if (layer.getType() == "qwen3_5_moe") {
      // Exercise routing and both gate/up projections. Down projections stay
      // zero so the block remains an analytically tractable identity.
      context.getWeight(0).setValue(0, 0, 0, 0, 1.0f);
      auto &routed_gate_up = context.getWeight(1);
      routed_gate_up.setValue(0, 0, 0, 0, 1.0f);
      routed_gate_up.setValue(0, 0, 0, 32, 1.0f);
      auto &shared_gate_up = context.getWeight(context.getNumWeights() - 3);
      shared_gate_up.setValue(0, 0, 0, 0, 1.0f);
      shared_gate_up.setValue(0, 0, 0, 32, 1.0f);
      context.getWeight(context.getNumWeights() - 1).setValue(0, 0, 0, 0, 1.0f);
    } else if (layer.getName() == "layer0_linear_qkv") {
      auto &weight = context.getWeight(0);
      weight.setValue(0, 0, 0, 0, 1.0f);
      weight.setValue(0, 0, 0, 32, 1.0f);
      weight.setValue(0, 0, 0, 64, 1.0f);
    } else if (layer.getName() == "layer0_linear_z" ||
               layer.getName() == "layer0_linear_b" ||
               layer.getName() == "layer0_linear_a" ||
               layer.getName() == "layer1_wk" ||
               layer.getName() == "layer1_wv") {
      context.getWeight(0).setValue(0, 0, 0, 0, 1.0f);
    } else if (layer.getName() == "layer1_wq") {
      auto &weight = context.getWeight(0);
      weight.setValue(0, 0, 0, 0, 1.0f);
      weight.setValue(0, 0, 0, 64, 1.0f);
    } else if (layer.getName() == "embedding0") {
      auto &weight = context.getWeight(0);
      weight.setValue(0, 0, 1, 0, 1.0f);
      weight.setValue(0, 0, 4, 0, 2.0f);
    } else if (layer.getName() == "output_of_causallm") {
      auto &weight = context.getWeight(0);
      weight.setValue(0, 0, 0, 1, 1.0f);
      weight.setValue(0, 0, 0, 4, 2.0f);
    }
  });
}

/**
 * @brief Make an official-shape-compatible nested text configuration
 *
 * All quantized matrix axes are multiples of Q4_0's 32-element block size.
 * The two layers cover one Gated DeltaNet mixer and one gated full-attention
 * mixer; every layer also covers routed and shared experts.
 */
causallm::json makeTinyQwen3_5MoeConfig() {
  causallm::json text_config = {
    {"attention_bias", false},
    {"bos_token_id", 0},
    {"eos_token_id", 31},
    {"head_dim", 8},
    {"hidden_act", "silu"},
    {"hidden_size", 64},
    {"is_causal", true},
    {"layer_types", {"linear_attention", "full_attention"}},
    {"linear_conv_kernel_dim", 4},
    {"linear_key_head_dim", 2},
    {"linear_num_key_heads", 16},
    {"linear_num_value_heads", 32},
    {"linear_value_head_dim", 1},
    {"max_position_embeddings", 8},
    {"moe_intermediate_size", 32},
    {"num_attention_heads", 8},
    {"num_experts", 32},
    {"num_experts_per_tok", 2},
    {"num_hidden_layers", tiny_qwen3_5_moe_num_layers},
    {"num_key_value_heads", 4},
    {"partial_rotary_factor", 0.25},
    {"rms_norm_eps", 1e-6},
    {"rope_parameters",
     {{"partial_rotary_factor", 0.25}, {"rope_theta", 10000}}},
    {"shared_expert_intermediate_size", 32},
    {"tie_word_embeddings", false},
    {"vocab_size", 32},
  };

  return {
    {"architectures", {"Qwen3_5MoeForConditionalGeneration"}},
    {"model_type", "qwen3_5_moe"},
    {"text_config", std::move(text_config)},
    {"tie_word_embeddings", false},
  };
}

/**
 * @brief Make the tiny Qwen3.5 MoE layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen3_5MoeLayerDtypeMap(
  const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    for (int i = 0; i < tiny_qwen3_5_moe_num_layers; ++i)
      dtype_map["layer" + std::to_string(i) + "_ffn_down"] = dtype;

    dtype_map["layer0_linear_qkv"] = dtype;
    dtype_map["layer0_linear_z"] = dtype;
    dtype_map["layer0_linear_out"] = dtype;
    dtype_map["layer1_wq"] = dtype;
    dtype_map["layer1_wk"] = dtype;
    dtype_map["layer1_wv"] = dtype;
    dtype_map["layer1_attention_out"] = dtype;
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

std::vector<float> makeExpectedQwen3_5MoeLogits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.9999361f;
  logits[4] = 15.9998722f;
  return logits;
}

causallm_test::TinyCausalLMCase
makeQwen3_5MoeCase(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Qwen3_5Moe_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedQwen3_5MoeLogits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyQwen3_5MoeConfig,
    makeQwen3_5MoeLayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyQwen3_5MoeCausalLM>(cfg, generation_cfg,
                                                      nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupQwen3_5MoeDeterministicWeights(
        static_cast<TinyQwen3_5MoeCausalLM &>(runner));
    },
  };
}

class Qwen3_5MoeTinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = info == nullptr ? "Unknown" : info->name();
    return causallm_test::makeTinyCausalLMFiles("Qwen3_5MoeTinyModelTest",
                                                test_name, GetParam().name);
  }
};

TEST_P(Qwen3_5MoeTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);
  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

TEST_P(Qwen3_5MoeTinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

TEST_P(Qwen3_5MoeTinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

TEST_P(Qwen3_5MoeTinyModelTest, IncrementalDecodePreservesHybridState) {
  const auto files = makeFiles();
  const auto fp32_data_type = causallm_test::makeTinyFp32DataType();
  auto source_config = GetParam().make_model_config();
  auto source_generation = causallm_test::makeTinyGenerationConfig();
  auto source_nntrainer = causallm_test::makeTinyNntrainerConfig(
    files.tokenizer_path, fp32_data_type);
  auto source =
    GetParam().create_model(source_config, source_generation, source_nntrainer);
  source->initializeModel();
  GetParam().setup_weights(*source);
  source->saveWeightWithDtype(
    files.weight_path.string(),
    GetParam().make_layer_dtype_map(GetParam().data_type));

  auto loaded_config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto loaded = GetParam().create_model(
    loaded_config.model, loaded_config.generation, loaded_config.nntrainer);
  loaded->initializeModel();
  loaded->loadWeight(files.weight_path.string());

  std::vector<unsigned int> generated;
  ASSERT_NO_THROW(generated = loaded->greedyGenerateFromIds({1, 4}, 2));
  EXPECT_EQ(generated, (std::vector<unsigned int>{4, 4}));
}

INSTANTIATE_TEST_SUITE_P(
  Qwen3_5Moe, Qwen3_5MoeTinyModelTest,
  ::testing::Values(
    makeQwen3_5MoeCase(causallm_test::makeTinyFp32DataType()),
    makeQwen3_5MoeCase(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

} // namespace
