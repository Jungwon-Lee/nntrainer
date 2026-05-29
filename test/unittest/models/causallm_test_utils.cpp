// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   causallm_test_utils.cpp
 * @date   15 May 2026
 * @brief  Shared helpers for tiny CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

namespace causallm_test {

namespace {

/**
 * @brief Sanitize a string for use in a temporary file name
 */
std::string sanitizeName(std::string name) {
  std::replace_if(
    name.begin(), name.end(),
    [](unsigned char c) { return !std::isalnum(c) && c != '_' && c != '-'; },
    '_');
  return name;
}

/**
 * @brief Write a string into a file
 */
void writeFile(const std::filesystem::path &path, const std::string &content) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open " + path.string());
  }

  file << content;
  if (!file.good()) {
    throw std::runtime_error("failed to write " + path.string());
  }
}

/**
 * @brief Write a tiny tokenizer file for CausalLM tests
 */
std::filesystem::path writeTinyTokenizer(const std::filesystem::path &dir) {
  auto tokenizer_path = dir / "tokenizer.json";

  std::ostringstream vocab;
  vocab << "      \"<unk>\": 0,\n";
  vocab << "      \"hello\": 1,\n";
  vocab << "      \"world\": 2,\n";
  for (unsigned int i = 3; i < 31; ++i) {
    vocab << "      \"tok" << i << "\": " << i << ",\n";
  }
  vocab << "      \"<eos>\": 31\n";

  std::ostringstream tokenizer;
  tokenizer << R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 31,
      "content": "<eos>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
)" << vocab.str()
            << R"(    },
    "unk_token": "<unk>"
  }
})";

  writeFile(tokenizer_path, tokenizer.str());
  return tokenizer_path;
}

/**
 * @brief Create a loaded target-dtype model from deterministic FP32 weights
 */
std::unique_ptr<TinyCausalLMRunner>
makeLoadedDeterministicModel(const TinyCausalLMCase &test_case,
                             const TinyCausalLMFiles &files) {
  TinyCausalLMDataType fp32_data_type = makeTinyFp32DataType();
  TinyCausalLMConfig source_config = {
    test_case.make_model_config(),
    makeTinyGenerationConfig(),
    makeTinyNntrainerConfig(files.tokenizer_path, fp32_data_type),
  };

  auto source = test_case.create_model(
    source_config.model, source_config.generation, source_config.nntrainer);
  source->initializeModel();
  source->setDeterministicWeights();
  source->saveWeightWithDtype(
    files.weight_path.string(),
    test_case.make_layer_dtype_map(test_case.data_type));

  auto loaded_config = makeTinyCausalLMConfig(test_case, files.tokenizer_path);
  auto loaded = test_case.create_model(
    loaded_config.model, loaded_config.generation, loaded_config.nntrainer);
  loaded->initializeModel();
  loaded->loadWeight(files.weight_path.string());

  return loaded;
}

} // namespace

/**
 * @brief Make files for one tiny CausalLM test invocation
 */
TinyCausalLMFiles makeTinyCausalLMFiles(const std::string &suite_name,
                                        const std::string &test_name,
                                        const std::string &case_name) {
  std::string name = "nntrainer_causallm_tiny";
  name += "_";
  name += sanitizeName(suite_name);
  name += "_";
  name += sanitizeName(test_name);
  name += "_";
  name += sanitizeName(case_name);

  auto dir = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  TinyCausalLMFiles files;
  files.dir = dir;
  files.tokenizer_path = writeTinyTokenizer(dir);
  files.weight_path = dir / "causallm_tiny.bin";
  return files;
}

/**
 * @brief Make minimal generation config shared by tiny CausalLM tests
 */
causallm::json makeTinyGenerationConfig() {
  return {
    {"bos_token_id", 0}, {"eos_token_id", 31}, {"do_sample", false},
    {"top_k", 1},        {"top_p", 1.0},       {"temperature", 1.0},
  };
}

/**
 * @brief Make FP32 data type variant
 */
TinyCausalLMDataType makeTinyFp32DataType() {
  return {
    "FP32", "FP32", "FP32", "FP32", "FP32-FP32",
  };
}

/**
 * @brief Make Q4_0 weights with FP32 activations data type variant
 */
TinyCausalLMDataType makeTinyQ40Fp32DataType() {
  return {
    "Q40_FP32", "Q4_0", "Q4_0", "Q4_0", "Q4_0-FP32",
  };
}

/**
 * @brief Convert a test dtype string to an nntrainer tensor data type
 */
ml::train::TensorDim::DataType toTensorDataType(const std::string &dtype) {
  if (dtype == "FP32")
    return ml::train::TensorDim::DataType::FP32;
  if (dtype == "Q4_0")
    return ml::train::TensorDim::DataType::Q4_0;
  if (dtype == "NONE")
    return ml::train::TensorDim::DataType::NONE;

  throw std::invalid_argument("unsupported tiny CausalLM dtype: " + dtype);
}

/**
 * @brief Make minimal nntrainer config shared by tiny CausalLM tests
 */
causallm::json
makeTinyNntrainerConfig(const std::filesystem::path &tokenizer_path,
                        const TinyCausalLMDataType &data_type) {
  return {
    {"bad_word_ids", std::vector<unsigned int>{}},
    {"batch_size", 1},
    {"embedding_dtype", data_type.embedding_dtype},
    {"fc_layer_dtype", data_type.fc_layer_dtype},
    {"init_seq_len", 4},
    {"lmhead_dtype", data_type.lmhead_dtype},
    {"max_seq_len", 8},
    {"model_tensor_type", data_type.model_tensor_type},
    {"model_type", "CausalLM"},
    {"num_to_generate", 1},
    {"tokenizer_file", tokenizer_path.string()},
  };
}

/**
 * @brief Make complete tiny configs for one model case
 */
TinyCausalLMConfig
makeTinyCausalLMConfig(const TinyCausalLMCase &test_case,
                       const std::filesystem::path &tokenizer_path) {
  return {
    test_case.make_model_config(),
    makeTinyGenerationConfig(),
    makeTinyNntrainerConfig(tokenizer_path, test_case.data_type),
  };
}

/**
 * @brief Verify greedy decoding chooses the maximum logit token
 */
void expectGreedyGenerationSelectsArgmax(TinyCausalLMRunner &model) {
  std::vector<float> logits(32, -2.0f);
  logits[2] = 1.0f;
  logits[3] = 5.0f;
  logits[4] = 4.0f;
  unsigned int input_ids[4] = {1, 0, 0, 0};

  auto ids = model.generateFromLogits(logits.data(), false, 1.0f, input_ids, 1);

  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 3u);
}

/**
 * @brief Verify save/load round-trip preserves tiny model logits
 */
void expectWeightRoundTripProducesSameLogits(const TinyCausalLMCase &test_case,
                                             const TinyCausalLMFiles &files) {
  auto first = makeLoadedDeterministicModel(test_case, files);
  auto second = makeLoadedDeterministicModel(test_case, files);

  std::vector<float> first_logits;
  std::vector<float> second_logits;
  ASSERT_NO_THROW(first_logits =
                    first->prefillLogits(test_case.expected_logits.prompt));
  ASSERT_NO_THROW(second_logits =
                    second->prefillLogits(test_case.expected_logits.prompt));

  ASSERT_EQ(first_logits.size(), second_logits.size());
  for (size_t i = 0; i < first_logits.size(); ++i)
    EXPECT_NEAR(first_logits[i], second_logits[i],
                test_case.expected_logits.logits_tolerance);
}

/**
 * @brief Verify a tiny model emits the expected logits for a prompt
 */
void expectPromptProducesExpectedLogits(const TinyCausalLMCase &test_case,
                                        const TinyCausalLMFiles &files) {
  auto model = makeLoadedDeterministicModel(test_case, files);

  std::vector<float> logits;
  ASSERT_NO_THROW(logits =
                    model->prefillLogits(test_case.expected_logits.prompt));

  ASSERT_EQ(logits.size(), test_case.expected_logits.logits.size());
  for (size_t i = 0; i < test_case.expected_logits.logits.size(); ++i)
    EXPECT_NEAR(logits[i], test_case.expected_logits.logits[i],
                test_case.expected_logits.logits_tolerance)
      << "logit mismatch at index " << i;
}

} // namespace causallm_test
