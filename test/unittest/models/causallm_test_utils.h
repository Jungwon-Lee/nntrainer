// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   causallm_test_utils.h
 * @date   15 May 2026
 * @brief  Shared helpers for tiny CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CAUSALLM_TEST_UTILS_H__
#define __CAUSALLM_TEST_UTILS_H__

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <transformer.h>

namespace causallm_test {

/**
 * @brief Files generated for one tiny CausalLM test invocation
 */
struct TinyCausalLMFiles {
  std::filesystem::path dir;            /**< Temporary test directory */
  std::filesystem::path tokenizer_path; /**< Tiny tokenizer.json path */
  std::filesystem::path weight_path;    /**< Tiny model weight path */
};

/**
 * @brief Minimal configs required to construct one CausalLM model
 */
struct TinyCausalLMConfig {
  causallm::json model;      /**< config.json equivalent */
  causallm::json generation; /**< generation_config.json equivalent */
  causallm::json nntrainer;  /**< nntrainer_config.json equivalent */
};

/**
 * @brief Data type variant used by one tiny CausalLM model case
 */
struct TinyCausalLMDataType {
  std::string name;              /**< Data type name used by gtest */
  std::string embedding_dtype;   /**< Embedding layer weight dtype */
  std::string fc_layer_dtype;    /**< Fully connected layer weight dtype */
  std::string lmhead_dtype;      /**< LM head weight dtype */
  std::string model_tensor_type; /**< Weight-activation tensor type */
};

/**
 * @brief Golden logits for one tiny CausalLM prompt
 */
struct TinyCausalLMExpectedLogits {
  std::string prompt;        /**< Prompt text */
  std::vector<float> logits; /**< Expected prefill logits */
  float logits_tolerance;    /**< Absolute logits tolerance */
};

/**
 * @brief Common runner interface exposed by model-specific tiny adapters
 */
class TinyCausalLMRunner {
public:
  /**
   * @brief Destroy the TinyCausalLMRunner object
   */
  virtual ~TinyCausalLMRunner() = default;

  /**
   * @brief Initialize the model graph and weights
   */
  virtual void initializeModel() = 0;

  /**
   * @brief Save model weights
   * @param path Target weight file path
   */
  virtual void saveWeight(const std::string &path) = 0;

  /**
   * @brief Save model weights with per-layer data type conversion
   * @param path Target weight file path
   * @param layer_dtype_map Per-layer target data types
   */
  virtual void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) = 0;

  /**
   * @brief Load model weights
   * @param path Source weight file path
   */
  virtual void loadWeight(const std::string &path) = 0;

  /**
   * @brief Set deterministic tiny weights before saving a golden model
   */
  virtual void setDeterministicWeights() = 0;

  /**
   * @brief Run one prompt through the model
   * @param prompt Prompt text
   */
  virtual void runPrompt(const std::string &prompt) = 0;

  /**
   * @brief Run prefill and return logits before token sampling
   * @param prompt Prompt text
   * @return Prefill logits copied from the model output
   */
  virtual std::vector<float> prefillLogits(const std::string &prompt) = 0;

  /**
   * @brief Get generated output text
   * @param batch_idx Batch index
   * @return Generated output text
   */
  virtual std::string getOutputText(int batch_idx = 0) const = 0;

  /**
   * @brief Get whether the model has completed run()
   * @return true if run() completed
   */
  virtual bool hasRun() const = 0;

  /**
   * @brief Read one token from the model input/output history
   * @param idx Token history index
   * @return Token id
   */
  virtual unsigned int tokenAt(size_t idx) const = 0;

  /**
   * @brief Generate ids from logits through CausalLM decoding logic
   * @param logits Logit buffer
   * @param do_sample Whether sampling is enabled
   * @param repetition_penalty Repetition penalty
   * @param input_ids Input ids used by repetition penalty
   * @param num_input_ids Number of input ids
   * @return Generated token ids
   */
  virtual std::vector<unsigned int>
  generateFromLogits(float *logits, bool do_sample, float repetition_penalty,
                     unsigned int *input_ids, unsigned int num_input_ids) = 0;
};

/**
 * @brief One tiny CausalLM model case reusable by common tests
 */
struct TinyCausalLMCase {
  std::string name;                           /**< Case name used by gtest */
  TinyCausalLMDataType data_type;             /**< Data type variant */
  TinyCausalLMExpectedLogits expected_logits; /**< Expected prefill logits */
  std::function<causallm::json()> make_model_config;
  std::function<std::map<std::string, ml::train::TensorDim::DataType>(
    const TinyCausalLMDataType &)>
    make_layer_dtype_map;
  std::function<std::unique_ptr<TinyCausalLMRunner>(
    causallm::json &, causallm::json &, causallm::json &)>
    create_model;
};

/**
 * @brief Make FP32 data type variant
 * @return Tiny FP32 data type descriptor
 */
TinyCausalLMDataType makeTinyFp32DataType();

/**
 * @brief Make Q4_0 weights with FP32 activations data type variant
 * @return Tiny Q4_0-FP32 data type descriptor
 */
TinyCausalLMDataType makeTinyQ40Fp32DataType();

/**
 * @brief Convert a test dtype string to an nntrainer tensor data type
 * @param dtype Test dtype string
 * @return nntrainer tensor data type
 */
ml::train::TensorDim::DataType toTensorDataType(const std::string &dtype);

/**
 * @brief Make files for one tiny CausalLM test invocation
 * @param suite_name GTest suite name
 * @param test_name GTest test name
 * @param case_name Tiny CausalLM model case name
 * @return Generated file paths
 */
TinyCausalLMFiles makeTinyCausalLMFiles(const std::string &suite_name,
                                        const std::string &test_name,
                                        const std::string &case_name);

/**
 * @brief Make minimal generation config shared by tiny CausalLM tests
 * @return generation_config.json equivalent
 */
causallm::json makeTinyGenerationConfig();

/**
 * @brief Make minimal nntrainer config shared by tiny CausalLM tests
 * @param tokenizer_path Tiny tokenizer path
 * @param data_type Tiny CausalLM data type variant
 * @return nntrainer_config.json equivalent
 */
causallm::json
makeTinyNntrainerConfig(const std::filesystem::path &tokenizer_path,
                        const TinyCausalLMDataType &data_type);

/**
 * @brief Make complete tiny configs for one model case
 * @param test_case Model case descriptor
 * @param tokenizer_path Tiny tokenizer path
 * @return Complete tiny CausalLM configs
 */
TinyCausalLMConfig
makeTinyCausalLMConfig(const TinyCausalLMCase &test_case,
                       const std::filesystem::path &tokenizer_path);

/**
 * @brief Verify greedy decoding chooses the maximum logit token
 * @param model Tiny model runner
 */
void expectGreedyGenerationSelectsArgmax(TinyCausalLMRunner &model);

/**
 * @brief Verify save/load round-trip preserves tiny model logits
 * @param test_case Model case descriptor
 * @param files Generated test file paths
 */
void expectWeightRoundTripProducesSameLogits(const TinyCausalLMCase &test_case,
                                             const TinyCausalLMFiles &files);

/**
 * @brief Verify a tiny model emits the expected logits for a prompt
 * @param test_case Model case descriptor
 * @param files Generated test file paths
 */
void expectPromptProducesExpectedLogits(const TinyCausalLMCase &test_case,
                                        const TinyCausalLMFiles &files);

} // namespace causallm_test

#endif // __CAUSALLM_TEST_UTILS_H__
