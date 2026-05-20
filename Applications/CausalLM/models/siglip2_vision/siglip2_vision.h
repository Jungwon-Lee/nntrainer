// SPDX-License-Identifier: Apache-2.0
/**
 * @file   siglip2_vision.h
 * @brief  SigLIP2 NaFlex vision encoder for LFM2-VL validation.
 */

#ifndef __SIGLIP2_VISION_H__
#define __SIGLIP2_VISION_H__

#include <transformer.h>

namespace causallm {

class Siglip2NaFlexVision : virtual public Transformer {
public:
  static constexpr const char *architectures = "Siglip2NaFlexVision";

  Siglip2NaFlexVision(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Siglip2NaFlexVision() = default;

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
  Tensor createEncoderBlock(const int layer_id, Tensor input, Tensor mask);
  Tensor createMlp(const int layer_id, Tensor input);
  Tensor createProjector(Tensor input, Tensor spatial_shapes);
  std::pair<std::vector<Tensor>, Tensor> constructVisionGraph();

  unsigned int PATCH_SIZE = 16;
  unsigned int PATCH_DIM = 768;
  unsigned int NUM_PATCHES = 256;
  unsigned int MAX_PATCHES = 1024;
  unsigned int BASE_GRID_SIZE = 16;
  unsigned int DOWNSAMPLE_FACTOR = 2;
  unsigned int PROJECTOR_HIDDEN_SIZE = 2048;
  unsigned int PROJECTOR_INPUT_SIZE = 3072;
  unsigned int PROJECTOR_OUTPUT_SIZE = 1024;
  unsigned int MAX_IMAGE_TOKENS = 256;
  bool INCLUDE_PROJECTOR = false;
  bool PROJECTOR_USE_LAYERNORM = false;
  bool PROJECTOR_BIAS = true;
  std::string PROJECTOR_HIDDEN_ACT = "gelu";
  std::string HIDDEN_ACT = "tanh_gelu";
};

} // namespace causallm

#endif /* __SIGLIP2_VISION_H__ */
