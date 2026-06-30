// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cpu_backend.h> // ml::train::ISA

#include "json.hpp"

namespace quick_dot_ai {
namespace quantize {

using json = nlohmann::json;

enum class DType {
  FP32,
  Q4_0,
  Q4_K,
  Q6_K,
};

struct QuantPlan {
  DType fc_dtype = DType::Q4_0;
  DType embd_dtype = DType::FP32;
  DType lmhead_dtype = DType::FP32;
};

struct ModelPlan {
  std::string architecture;
  std::string layout;
  size_t hidden;
  size_t vocab;
  size_t layers;
  size_t heads;
  size_t kv_heads;
  size_t head_dim;
  size_t intermediate;
  size_t experts = 0;
  bool tied_embeddings = true;
};

class TensorWriter {
public:
  TensorWriter(std::ifstream &input, std::ofstream &output,
               ml::train::ISA isa = ml::train::ISA::DEFAULT);

  void copyBytes(size_t bytes, const std::string &name);
  void copyFp32Tensor(size_t elements, const std::string &name);
  void writeMatrix(size_t rows, size_t cols, DType dtype,
                   const std::string &name);
  void writeTransposedMatrix(size_t height, size_t width, DType dtype,
                             const std::string &name);
  // Plain (un-repacked) variants: emit per-row block_q4_0 with NO ISA
  // interleave, so a single output row is addressable at runtime (used by the
  // sparse SmallThinker FFN). writeMatrixPlain keeps [rows,cols] orientation;
  // writeTransposedMatrixPlain transposes [height,width]->[width,height] first.
  void writeMatrixPlain(size_t rows, size_t cols, DType dtype,
                        const std::string &name);
  void writeTransposedMatrixPlain(size_t height, size_t width, DType dtype,
                                  const std::string &name);
  void quantizeFcWithBias(size_t height, size_t width, DType dtype,
                          const std::string &name);
  void quantizeEmbedding(size_t rows, size_t cols, DType dtype,
                         const std::string &name,
                         std::vector<float> *source_cache = nullptr);
  void quantizeTiedLmHead(const std::vector<float> &embedding, size_t vocab,
                          size_t hidden, DType dtype, const std::string &name);

private:
  std::vector<float> readFp32Tensor(size_t elements, const std::string &name);
  void writeFp32Tensor(const std::vector<float> &source,
                       const std::string &name);
  std::vector<float> transposeMatrix(const std::vector<float> &source,
                                     size_t height, size_t width) const;
  void writeMatrixData(const std::vector<float> &source, size_t rows,
                       size_t cols, DType dtype, const std::string &name);
  void writeQuantizedMatrix(const std::vector<float> &source, size_t rows,
                            size_t cols, DType dtype, const std::string &name,
                            bool repack = true);

  std::ifstream &input_;
  std::ofstream &output_;
  ml::train::ISA isa_; /**< target ISA repack format for Q4_0 (q4_0x8/q4_0x4) */
};

using LayerWriter = void (*)(TensorWriter &, const ModelPlan &,
                             const QuantPlan &, size_t);

struct ModelRecipe {
  std::vector<std::string> architectures;
  std::string layout;
  std::string intermediate_key;
  std::string experts_key;
  LayerWriter write_layer = nullptr;
};

class RecipeRegistry {
public:
  void add(ModelRecipe recipe);

  const ModelRecipe &find(const std::string &architecture) const;
  ModelPlan makePlan(const json &cfg) const;
  std::vector<std::string> supportedArchitectures() const;

private:
  std::vector<ModelRecipe> recipes_;
};

json readJson(const std::filesystem::path &path);
std::string architectureName(const json &cfg);

DType parseDType(const std::string &value);
std::string dtypeName(DType dtype);
std::string dtypeSuffix(DType dtype);

void registerBuiltInRecipes(RecipeRegistry &registry);

} // namespace quantize
} // namespace quick_dot_ai
