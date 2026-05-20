// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_embedding_merge.h
 * @brief  Validation model for LFM2-VL image-token embedding merge.
 */

#ifndef __LFM2_VL_EMBEDDING_MERGE_H__
#define __LFM2_VL_EMBEDDING_MERGE_H__

#include <transformer.h>

namespace causallm {

class Lfm2VlEmbeddingMerge : virtual public Transformer {
public:
  static constexpr const char *architectures = "Lfm2VlEmbeddingMerge";

  Lfm2VlEmbeddingMerge(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Lfm2VlEmbeddingMerge() = default;

  void initialize() override;
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

protected:
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;
  std::pair<Tensor, Tensor> constructModel() override;
  void registerCustomLayers() override;

private:
  std::pair<std::vector<Tensor>, Tensor> constructMergeGraph();

  unsigned int IMAGE_TOKEN_ID = 396;
  unsigned int MAX_IMAGE_TOKENS = 256;
};

} // namespace causallm

#endif /* __LFM2_VL_EMBEDDING_MERGE_H__ */
