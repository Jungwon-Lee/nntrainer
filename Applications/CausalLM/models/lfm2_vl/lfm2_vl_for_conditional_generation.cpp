// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_for_conditional_generation.cpp
 * @brief  End-to-end LFM2-VL staged runner.
 */

#include "lfm2_vl_for_conditional_generation.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace causallm {

Lfm2VlForConditionalGeneration::Lfm2VlForConditionalGeneration(
  json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL),
  cfg_(cfg),
  generation_cfg_(generation_cfg),
  nntr_cfg_(nntr_cfg) {
  setupParameters(cfg_, generation_cfg_, nntr_cfg_);
}

void Lfm2VlForConditionalGeneration::setupParameters(json &cfg,
                                                     json &generation_cfg,
                                                     json &nntr_cfg) {
  (void)cfg;
  (void)generation_cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1U);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");
  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", 1U);
  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", INIT_SEQ_LEN);
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);

  vision_model_file_name_ =
    nntr_cfg.value("vision_model_file_name", vision_model_file_name_);
  embedding_merge_model_file_name_ = nntr_cfg.value(
    "embedding_merge_model_file_name", embedding_merge_model_file_name_);
  prefill_model_file_name_ =
    nntr_cfg.value("prefill_model_file_name", prefill_model_file_name_);
}

void Lfm2VlForConditionalGeneration::initialize() {
  json vision_nntr_cfg = nntr_cfg_;
  vision_nntr_cfg["include_projector"] = true;
  vision_nntr_cfg["model_file_name"] = vision_model_file_name_;

  json merge_nntr_cfg = nntr_cfg_;
  merge_nntr_cfg["model_file_name"] = embedding_merge_model_file_name_;

  json prefill_nntr_cfg = nntr_cfg_;
  prefill_nntr_cfg["model_file_name"] = prefill_model_file_name_;
  if (!prefill_nntr_cfg.contains("decode_embedding_file")) {
    prefill_nntr_cfg["decode_embedding_file"] = embedding_merge_model_file_name_;
  }

  vision_ = std::make_unique<Siglip2NaFlexVision>(cfg_, generation_cfg_,
                                                  vision_nntr_cfg);
  embedding_merge_ = std::make_unique<Lfm2VlEmbeddingMerge>(
    cfg_, generation_cfg_, merge_nntr_cfg);
  prefill_ =
    std::make_unique<Lfm2VlPrefill>(cfg_, generation_cfg_, prefill_nntr_cfg);

  vision_->initialize();
  embedding_merge_->initialize();
  prefill_->initialize();

  is_initialized = true;
}

void Lfm2VlForConditionalGeneration::load_weight(
  const std::string &weight_path) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Lfm2VlForConditionalGeneration is not initialized.");
  }

  const std::filesystem::path model_dir =
    std::filesystem::path(weight_path).parent_path();

  auto resolve_weight = [&model_dir](const std::string &file_name) {
    const std::filesystem::path path{file_name};
    return path.is_absolute() ? path : model_dir / path;
  };

  vision_->load_weight(resolve_weight(vision_model_file_name_).string());
  embedding_merge_->load_weight(
    resolve_weight(embedding_merge_model_file_name_).string());
  prefill_->load_weight(resolve_weight(prefill_model_file_name_).string());
}

void Lfm2VlForConditionalGeneration::run(const WSTR prompt, bool do_sample,
                                         const WSTR system_prompt,
                                         const WSTR tail_prompt,
                                         bool log_output) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Lfm2VlForConditionalGeneration is not initialized.");
  }

  if (log_output) {
    std::cout << "[LFM2-VL] Running SigLIP2 vision tower + projector\n";
  }
  vision_->run(prompt, do_sample, system_prompt, tail_prompt, log_output);

  if (log_output) {
    std::cout << "[LFM2-VL] Merging image features into token embeddings\n";
  }
  embedding_merge_->run(prompt, do_sample, system_prompt, tail_prompt,
                        log_output);

  if (log_output) {
    std::cout << "[LFM2-VL] Running language prefill\n";
  }
  prefill_->run(prompt, do_sample, system_prompt, tail_prompt, log_output);
}

} // namespace causallm
