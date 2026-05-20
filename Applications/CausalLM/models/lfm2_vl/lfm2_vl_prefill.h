// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_prefill.h
 * @brief  LFM2-VL language prefill validation model.
 */

#ifndef __LFM2_VL_PREFILL_H__
#define __LFM2_VL_PREFILL_H__

#include <lfm2_700m.h>

#include <cstdint>
#include <string>
#include <vector>

namespace causallm {

class Lfm2VlPrefill : public Lfm2Transformer {
public:
  static constexpr const char *architectures = "Lfm2VlPrefill";

  Lfm2VlPrefill(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL),
    Lfm2Transformer(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Lfm2VlPrefill() = default;

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
  Tensor createConvBlock(const int layer_id, Tensor input);
  std::vector<std::string> layer_types_;
  std::string OUTPUT_MODE = "logits";
  std::string DECODE_EMBEDDING_FILE;
  int CONV_DIM = 1024;
  int CONV_DIM_OUT = 1024;
  int CONV_L_CACHE = 3;
  bool CONV_BIAS = false;
};

} // namespace causallm

#endif /* __LFM2_VL_PREFILL_H__ */
