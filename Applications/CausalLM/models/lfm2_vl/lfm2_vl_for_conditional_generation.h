// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_for_conditional_generation.h
 * @brief  End-to-end LFM2-VL staged runner.
 */

#ifndef __LFM2_VL_FOR_CONDITIONAL_GENERATION_H__
#define __LFM2_VL_FOR_CONDITIONAL_GENERATION_H__

#include "lfm2_vl_embedding_merge.h"
#include "lfm2_vl_prefill.h"
#include "siglip2_vision.h"

#include <memory>
#include <string>

namespace causallm {

class Lfm2VlForConditionalGeneration : public Transformer {
public:
  static constexpr const char *architectures =
    "Lfm2VlForConditionalGeneration";

  Lfm2VlForConditionalGeneration(json &cfg, json &generation_cfg,
                                 json &nntr_cfg);

  virtual ~Lfm2VlForConditionalGeneration() = default;

  void initialize() override;
  void load_weight(const std::string &weight_path) override;
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

protected:
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

private:
  json cfg_;
  json generation_cfg_;
  json nntr_cfg_;

  std::unique_ptr<Siglip2NaFlexVision> vision_;
  std::unique_ptr<Lfm2VlEmbeddingMerge> embedding_merge_;
  std::unique_ptr<Lfm2VlPrefill> prefill_;

  std::string vision_model_file_name_ = "nntr_siglip2_vision_fp32.bin";
  std::string embedding_merge_model_file_name_ =
    "nntr_lfm2_vl_embedding_merge_fp32.bin";
  std::string prefill_model_file_name_ = "nntr_lfm2_vl_prefill_fp32.bin";
};

} // namespace causallm

#endif /* __LFM2_VL_FOR_CONDITIONAL_GENERATION_H__ */
