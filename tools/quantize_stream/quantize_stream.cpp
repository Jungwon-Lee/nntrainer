// SPDX-License-Identifier: Apache-2.0
/**
 * @file quantize_stream.cpp
 * @brief Streaming dtype converter for NNTrainer transformer weight files.
 */

#include "quantize_stream.h"

#include <cpu_backend.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace quick_dot_ai {
namespace quantize {

namespace {

constexpr size_t QK4_0 = 32;
constexpr size_t Q4_0_BLOCK_BYTES = 18;
constexpr size_t QK_K = 256;
constexpr size_t Q4_K_BLOCK_BYTES = 144;
constexpr size_t Q6_K_BLOCK_BYTES = 210;

std::string upper(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return value;
}

size_t getSize(const json &cfg, const std::string &key) {
  if (!cfg.contains(key) || !cfg[key].is_number()) {
    throw std::runtime_error("config.json is missing numeric key: " + key);
  }
  return cfg[key].get<size_t>();
}

size_t getSizeAny(const json &cfg, const std::vector<std::string> &keys) {
  for (const auto &key : keys) {
    if (cfg.contains(key) && cfg[key].is_number()) {
      return cfg[key].get<size_t>();
    }
  }
  throw std::runtime_error("config.json is missing expected numeric key");
}

bool getBoolDefault(const json &cfg, const std::string &key, bool fallback) {
  return cfg.contains(key) && cfg[key].is_boolean() ? cfg[key].get<bool>()
                                                    : fallback;
}

size_t q4Size(size_t rows, size_t cols) {
  if (rows % 8 != 0 || cols % QK4_0 != 0) {
    throw std::invalid_argument("Q4_0 tensor shape must be divisible by 8x32");
  }
  return rows * (cols / QK4_0) * Q4_0_BLOCK_BYTES;
}

size_t qkSize(size_t rows, size_t cols, size_t block_bytes,
              const std::string &dtype_name) {
  if (cols % QK_K != 0) {
    throw std::invalid_argument(dtype_name +
                                " tensor width must be divisible by 256");
  }
  return rows * (cols / QK_K) * block_bytes;
}

size_t quantizedSize(DType dtype, size_t rows, size_t cols) {
  switch (dtype) {
  case DType::FP32:
    return rows * cols * sizeof(float);
  case DType::Q4_0:
    return q4Size(rows, cols);
  case DType::Q4_K:
    return qkSize(rows, cols, Q4_K_BLOCK_BYTES, "Q4_K");
  case DType::Q6_K:
    return qkSize(rows, cols, Q6_K_BLOCK_BYTES, "Q6_K");
  }
  return 0;
}

void copyMetadata(const std::filesystem::path &model_dir,
                  const std::filesystem::path &output_dir,
                  const std::string &output_bin_name,
                  const QuantPlan &quant_plan) {
  std::filesystem::create_directories(output_dir);

  json cfg = readJson(model_dir / "config.json");
  cfg["tie_word_embeddings"] = false;
  std::ofstream model_config_out(output_dir / "config.json");
  if (!model_config_out.is_open()) {
    throw std::runtime_error("Failed to open quantized config.json");
  }
  model_config_out << cfg.dump(4) << '\n';

  std::filesystem::copy_file(model_dir / "generation_config.json",
                             output_dir / "generation_config.json",
                             std::filesystem::copy_options::overwrite_existing);

  json nntr_cfg = readJson(model_dir / "nntr_config.json");
  nntr_cfg["model_file_name"] = output_bin_name;
  nntr_cfg["model_tensor_type"] = dtypeName(quant_plan.fc_dtype) + "-FP32";
  nntr_cfg["fc_layer_dtype"] = dtypeName(quant_plan.fc_dtype);
  nntr_cfg["embedding_dtype"] = dtypeName(quant_plan.embd_dtype);
  nntr_cfg["lmhead_dtype"] = dtypeName(quant_plan.lmhead_dtype);
  nntr_cfg["num_to_generate"] = 8;
  nntr_cfg["init_seq_len"] = 64;
  nntr_cfg["max_seq_len"] = 64;

  std::ofstream config_out(output_dir / "nntr_config.json");
  if (!config_out.is_open()) {
    throw std::runtime_error("Failed to open quantized nntr_config.json");
  }
  config_out << nntr_cfg.dump(4) << '\n';
}

std::string defaultOutputBinName(const std::string &input_name,
                                 const QuantPlan &quant_plan) {
  std::string base = input_name;
  const auto pos = base.rfind(".bin");
  if (pos != std::string::npos) {
    base = base.substr(0, pos);
  }
  const std::string suffix = "_fp32";
  if (base.size() >= suffix.size() &&
      base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
    base.resize(base.size() - suffix.size());
  }
  if (quant_plan.fc_dtype == quant_plan.embd_dtype &&
      quant_plan.fc_dtype == quant_plan.lmhead_dtype) {
    return base + "_" + dtypeSuffix(quant_plan.fc_dtype) + ".bin";
  }
  return base + "_" + dtypeSuffix(quant_plan.fc_dtype) + "_embd" +
         dtypeSuffix(quant_plan.embd_dtype) + "_lmhead" +
         dtypeSuffix(quant_plan.lmhead_dtype) + ".bin";
}

void printUsage(const char *program, const RecipeRegistry &registry) {
  std::cerr << "Usage: " << program
            << " <model_dir> <output_dir> [output_bin] [options]\n\n"
            << "Options:\n"
            << "  --dtype <type>        Set FC, embedding, and LM head dtype\n"
            << "  --fc_dtype <type>     FC/projection dtype (default: Q4_0)\n"
            << "  --embd_dtype <type>   Embedding dtype (default: FP32)\n"
            << "  --lmhead_dtype <type> LM head dtype (default: FP32)\n"
            << "  --output_bin <name>   Output .bin filename\n"
            << "  --isa <target>        Q4_0 repack format: DEFAULT, X86 "
               "(q4_0x8), ARM (q4_0x4). Default: DEFAULT (x86).\n"
            << "  -h, --help            Show this help\n\n"
            << "Supported dtypes: FP32, Q4_0, Q4_K, Q6_K.\n\n"
            << "Supported architectures:\n";

  for (const auto &architecture : registry.supportedArchitectures()) {
    std::cerr << "  " << architecture << "\n";
  }
}

int run(int argc, char **argv) {
  RecipeRegistry registry;
  registerBuiltInRecipes(registry);

  if (argc == 2 &&
      (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    printUsage(argv[0], registry);
    return EXIT_SUCCESS;
  }

  if (argc < 3) {
    printUsage(argv[0], registry);
    return EXIT_FAILURE;
  }

  const std::filesystem::path model_dir = argv[1];
  const std::filesystem::path output_dir = argv[2];
  QuantPlan quant_plan;
  std::string output_bin_name_arg;
  ml::train::ISA isa = ml::train::ISA::DEFAULT;

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0], registry);
      return EXIT_SUCCESS;
    } else if (arg == "--dtype" && i + 1 < argc) {
      const DType dtype = parseDType(argv[++i]);
      quant_plan.fc_dtype = dtype;
      quant_plan.embd_dtype = dtype;
      quant_plan.lmhead_dtype = dtype;
    } else if (arg == "--fc_dtype" && i + 1 < argc) {
      quant_plan.fc_dtype = parseDType(argv[++i]);
    } else if (arg == "--embd_dtype" && i + 1 < argc) {
      quant_plan.embd_dtype = parseDType(argv[++i]);
    } else if (arg == "--lmhead_dtype" && i + 1 < argc) {
      quant_plan.lmhead_dtype = parseDType(argv[++i]);
    } else if (arg == "--output_bin" && i + 1 < argc) {
      output_bin_name_arg = argv[++i];
    } else if (arg == "--isa" && i + 1 < argc) {
      std::string v = upper(argv[++i]);
      if (v == "ARM")
        isa = ml::train::ISA::ARM;
      else if (v == "X86")
        isa = ml::train::ISA::X86;
      else if (v == "DEFAULT")
        isa = ml::train::ISA::DEFAULT;
      else
        throw std::invalid_argument("Unsupported ISA: " + v +
                                    " (supported: DEFAULT, X86, ARM)");
    } else if (!arg.empty() && arg[0] != '-' && output_bin_name_arg.empty()) {
      output_bin_name_arg = arg;
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }

  const json cfg = readJson(model_dir / "config.json");
  const json nntr_cfg = readJson(model_dir / "nntr_config.json");
  const ModelPlan plan = registry.makePlan(cfg);
  const ModelRecipe &recipe = registry.find(plan.architecture);

  const std::string input_bin_name =
    nntr_cfg["model_file_name"].get<std::string>();
  const std::string output_bin_name =
    output_bin_name_arg.empty()
      ? defaultOutputBinName(input_bin_name, quant_plan)
      : output_bin_name_arg;

  const std::filesystem::path input_path = model_dir / input_bin_name;
  const std::filesystem::path output_path = output_dir / output_bin_name;

  copyMetadata(model_dir, output_dir, output_bin_name, quant_plan);

  std::ifstream input(input_path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open " + input_path.string());
  }
  std::ofstream output(output_path, std::ios::binary);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open " + output_path.string());
  }

  TensorWriter writer(input, output, isa);

  const char *isa_name = (isa == ml::train::ISA::ARM)   ? "ARM (q4_0x4)"
                         : (isa == ml::train::ISA::X86) ? "X86 (q4_0x8)"
                                                        : "DEFAULT (x86 q4_0x8)";

  std::cout << "Streaming quantization\n";
  std::cout << "  Architecture: " << plan.architecture << "\n";
  std::cout << "  ISA target: " << isa_name << "\n";
  std::cout << "  Layout: " << plan.layout << "\n";
  std::cout << "  FC dtype: " << dtypeName(quant_plan.fc_dtype) << "\n";
  std::cout << "  Embedding dtype: " << dtypeName(quant_plan.embd_dtype)
            << "\n";
  std::cout << "  LMHead dtype: " << dtypeName(quant_plan.lmhead_dtype) << "\n";
  std::cout << "  Source: " << input_path << "\n";
  std::cout << "  Target: " << output_path << "\n";

  std::vector<float> embedding_cache;
  writer.quantizeEmbedding(plan.vocab, plan.hidden, quant_plan.embd_dtype,
                           "embedding0",
                           plan.tied_embeddings ? &embedding_cache : nullptr);
  std::cout << "  embedding0 -> " << dtypeName(quant_plan.embd_dtype) << "\n";

  for (size_t layer = 0; layer < plan.layers; ++layer) {
    recipe.write_layer(writer, plan, quant_plan, layer);
    std::cout << "  layer" << layer << " -> " << dtypeName(quant_plan.fc_dtype)
              << " weights, FP32 norms/biases\n";
  }

  writer.copyFp32Tensor(plan.hidden, "output_norm");
  if (plan.tied_embeddings) {
    writer.quantizeTiedLmHead(embedding_cache, plan.vocab, plan.hidden,
                              quant_plan.lmhead_dtype, "output_of_causallm");
  } else {
    // Untied lm_head: the FP32 .bin stores it transposed to [in=hidden,
    // out=vocab] (like every other projection). It must be transposed back to
    // [vocab, hidden] before quantizing so it matches both the FC weight
    // convention and the tied path (quantizeTiedLmHead). Using writeMatrix here
    // quantizes [hidden, vocab] instead, scrambling the output logits ->
    // garbage generation.
    writer.writeTransposedMatrix(plan.hidden, plan.vocab, quant_plan.lmhead_dtype,
                                 "output_of_causallm");
  }
  std::cout << "  output_of_causallm -> " << dtypeName(quant_plan.lmhead_dtype)
            << "\n";

  const auto consumed = input.tellg();
  input.peek();
  if (!input.eof()) {
    throw std::runtime_error("Input file has trailing unread bytes");
  }

  output.close();
  const auto source_size = std::filesystem::file_size(input_path);
  const auto output_size = std::filesystem::file_size(output_path);
  const double ratio =
    static_cast<double>(output_size) / static_cast<double>(source_size) * 100.0;

  std::cout << "  Consumed bytes: " << consumed << "\n";
  std::cout << "  Source size: " << source_size / (1024 * 1024) << " MB\n";
  std::cout << "  Output size: " << output_size / (1024 * 1024) << " MB\n";
  std::cout << "  Compression: " << std::fixed << std::setprecision(1) << ratio
            << "%\n";
  std::cout << "  Config: " << (output_dir / "nntr_config.json") << "\n";
  std::cout << "  result: PASS\n";

  return EXIT_SUCCESS;
}

} // namespace

json readJson(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open " + path.string());
  }

  json value;
  file >> value;
  return value;
}

std::string architectureName(const json &cfg) {
  if (!cfg.contains("architectures") || !cfg["architectures"].is_array() ||
      cfg["architectures"].empty()) {
    throw std::runtime_error("config.json is missing architectures[0]");
  }
  return cfg["architectures"][0].get<std::string>();
}

DType parseDType(const std::string &value) {
  const std::string dtype = upper(value);
  if (dtype == "FP32")
    return DType::FP32;
  if (dtype == "Q4_0" || dtype == "Q40")
    return DType::Q4_0;
  if (dtype == "Q4_K" || dtype == "Q4K")
    return DType::Q4_K;
  if (dtype == "Q6_K" || dtype == "Q6K")
    return DType::Q6_K;
  throw std::invalid_argument("Unsupported dtype: " + value +
                              " (supported: FP32, Q4_0, Q4_K, Q6_K)");
}

std::string dtypeName(DType dtype) {
  switch (dtype) {
  case DType::FP32:
    return "FP32";
  case DType::Q4_0:
    return "Q4_0";
  case DType::Q4_K:
    return "Q4_K";
  case DType::Q6_K:
    return "Q6_K";
  }
  return "UNKNOWN";
}

std::string dtypeSuffix(DType dtype) {
  switch (dtype) {
  case DType::FP32:
    return "fp32";
  case DType::Q4_0:
    return "q40";
  case DType::Q4_K:
    return "q4k";
  case DType::Q6_K:
    return "q6k";
  }
  return "unknown";
}

TensorWriter::TensorWriter(std::ifstream &input, std::ofstream &output,
                           ml::train::ISA isa) :
  input_(input), output_(output), isa_(isa) {}

void TensorWriter::copyBytes(size_t bytes, const std::string &name) {
  constexpr size_t buffer_size = 16 * 1024 * 1024;
  std::vector<char> buffer(std::min(buffer_size, bytes));
  size_t remaining = bytes;

  while (remaining > 0) {
    const size_t chunk = std::min(buffer.size(), remaining);
    input_.read(buffer.data(), static_cast<std::streamsize>(chunk));
    if (input_.gcount() != static_cast<std::streamsize>(chunk)) {
      throw std::runtime_error("Unexpected EOF while reading " + name);
    }
    output_.write(buffer.data(), static_cast<std::streamsize>(chunk));
    if (!output_) {
      throw std::runtime_error("Failed to write " + name);
    }
    remaining -= chunk;
  }
}

void TensorWriter::copyFp32Tensor(size_t elements, const std::string &name) {
  copyBytes(elements * sizeof(float), name);
}

std::vector<float> TensorWriter::readFp32Tensor(size_t elements,
                                                const std::string &name) {
  std::vector<float> source(elements);
  input_.read(reinterpret_cast<char *>(source.data()),
              static_cast<std::streamsize>(elements * sizeof(float)));
  if (input_.gcount() !=
      static_cast<std::streamsize>(elements * sizeof(float))) {
    throw std::runtime_error("Unexpected EOF while reading " + name);
  }
  return source;
}

void TensorWriter::writeFp32Tensor(const std::vector<float> &source,
                                   const std::string &name) {
  output_.write(reinterpret_cast<const char *>(source.data()),
                static_cast<std::streamsize>(source.size() * sizeof(float)));
  if (!output_) {
    throw std::runtime_error("Failed to write " + name);
  }
}

std::vector<float>
TensorWriter::transposeMatrix(const std::vector<float> &source, size_t height,
                              size_t width) const {
  std::vector<float> transposed(source.size());
  for (size_t h = 0; h < height; ++h) {
    for (size_t w = 0; w < width; ++w) {
      transposed[w * height + h] = source[h * width + w];
    }
  }
  return transposed;
}

void TensorWriter::writeMatrix(size_t rows, size_t cols, DType dtype,
                               const std::string &name) {
  const std::vector<float> source = readFp32Tensor(rows * cols, name);
  writeMatrixData(source, rows, cols, dtype, name);
}

void TensorWriter::writeMatrixData(const std::vector<float> &source,
                                   size_t rows, size_t cols, DType dtype,
                                   const std::string &name) {
  if (dtype == DType::FP32) {
    writeFp32Tensor(source, name);
    return;
  }

  writeQuantizedMatrix(source, rows, cols, dtype, name);
}

void TensorWriter::writeQuantizedMatrix(const std::vector<float> &source,
                                        size_t rows, size_t cols, DType dtype,
                                        const std::string &name, bool repack) {
  const size_t output_size = quantizedSize(dtype, rows, cols);
  std::vector<char> quantized(output_size);

  switch (dtype) {
  case DType::Q4_0:
    if (repack) {
      std::vector<char> tmp(output_size);
      nntrainer::quantize_q4_0(source.data(), tmp.data(),
                               static_cast<int64_t>(rows),
                               static_cast<int64_t>(cols), nullptr);
      nntrainer::repack_q4_0(quantized.data(), tmp.data(), output_size,
                             static_cast<unsigned int>(rows),
                             static_cast<unsigned int>(cols), isa_);
    } else {
      nntrainer::quantize_q4_0(source.data(), quantized.data(),
                               static_cast<int64_t>(rows),
                               static_cast<int64_t>(cols), nullptr);
    }
    break;
  case DType::Q4_K:
    if (repack) {
      std::vector<char> tmp(output_size);
      nntrainer::quantize_q4_K(source.data(), tmp.data(),
                               static_cast<int64_t>(rows),
                               static_cast<int64_t>(cols), nullptr);
      nntrainer::repack_q4_K(quantized.data(), tmp.data(), output_size,
                             static_cast<unsigned int>(rows),
                             static_cast<unsigned int>(cols));
    } else {
      nntrainer::quantize_q4_K(source.data(), quantized.data(),
                               static_cast<int64_t>(rows),
                               static_cast<int64_t>(cols), nullptr);
    }
    break;
  case DType::Q6_K:
    nntrainer::quantize_q6_K(source.data(), quantized.data(),
                             static_cast<int64_t>(rows),
                             static_cast<int64_t>(cols), nullptr);
    break;
  case DType::FP32:
    throw std::invalid_argument("writeQuantizedMatrix called with FP32 for " +
                                name);
  }

  output_.write(quantized.data(),
                static_cast<std::streamsize>(quantized.size()));
  if (!output_) {
    throw std::runtime_error("Failed to write " + name);
  }
}

void TensorWriter::writeTransposedMatrix(size_t height, size_t width,
                                         DType dtype, const std::string &name) {
  const size_t elements = height * width;
  const std::vector<float> source = readFp32Tensor(elements, name);
  const std::vector<float> transposed = transposeMatrix(source, height, width);
  writeMatrixData(transposed, width, height, dtype, name);
}

void TensorWriter::writeMatrixPlain(size_t rows, size_t cols, DType dtype,
                                    const std::string &name) {
  const std::vector<float> source = readFp32Tensor(rows * cols, name);
  writeQuantizedMatrix(source, rows, cols, dtype, name, /*repack=*/false);
}

void TensorWriter::writeTransposedMatrixPlain(size_t height, size_t width,
                                              DType dtype,
                                              const std::string &name) {
  const std::vector<float> source = readFp32Tensor(height * width, name);
  const std::vector<float> transposed = transposeMatrix(source, height, width);
  writeQuantizedMatrix(transposed, width, height, dtype, name, /*repack=*/false);
}

void TensorWriter::quantizeFcWithBias(size_t height, size_t width, DType dtype,
                                      const std::string &name) {
  writeTransposedMatrix(height, width, dtype, name + ":weight");
  copyBytes(width * sizeof(float), name + ":bias");
}

void TensorWriter::quantizeEmbedding(size_t rows, size_t cols, DType dtype,
                                     const std::string &name,
                                     std::vector<float> *source_cache) {
  const size_t elements = rows * cols;
  std::vector<float> source = readFp32Tensor(elements, name);

  switch (dtype) {
  case DType::Q4_0:
  case DType::Q6_K:
    writeQuantizedMatrix(source, rows, cols, dtype, name, false);
    break;
  case DType::Q4_K:
    throw std::invalid_argument(
      "Q4_K embedding is not supported by EmbeddingLayer save/runtime");
  case DType::FP32:
    writeMatrixData(source, rows, cols, dtype, name);
    break;
  }

  if (source_cache) {
    *source_cache = std::move(source);
  }
}

void TensorWriter::quantizeTiedLmHead(const std::vector<float> &embedding,
                                      size_t vocab, size_t hidden, DType dtype,
                                      const std::string &name) {
  if (embedding.size() != vocab * hidden) {
    throw std::invalid_argument("Unexpected embedding cache size for " + name);
  }

  writeMatrixData(embedding, vocab, hidden, dtype, name);
}

void RecipeRegistry::add(ModelRecipe recipe) {
  if (recipe.architectures.empty()) {
    throw std::invalid_argument("ModelRecipe is missing architecture names");
  }
  if (!recipe.write_layer) {
    throw std::invalid_argument("ModelRecipe is missing layer writer");
  }
  recipes_.push_back(std::move(recipe));
}

const ModelRecipe &RecipeRegistry::find(const std::string &architecture) const {
  for (const auto &recipe : recipes_) {
    for (const auto &candidate : recipe.architectures) {
      if (architecture == candidate) {
        return recipe;
      }
    }
  }
  throw std::runtime_error("Unsupported architecture: " + architecture);
}

ModelPlan RecipeRegistry::makePlan(const json &cfg) const {
  const std::string architecture = architectureName(cfg);
  const ModelRecipe &recipe = find(architecture);
  ModelPlan plan{
    architecture,
    recipe.layout,
    getSize(cfg, "hidden_size"),
    getSize(cfg, "vocab_size"),
    getSize(cfg, "num_hidden_layers"),
    getSize(cfg, "num_attention_heads"),
    getSizeAny(cfg, {"num_key_value_heads", "num_attention_heads"}),
    0,
    getSize(cfg, recipe.intermediate_key),
    0,
    getBoolDefault(cfg, "tie_word_embeddings", true),
  };

  plan.head_dim = cfg.contains("head_dim") && cfg["head_dim"].is_number()
                    ? cfg["head_dim"].get<size_t>()
                    : plan.hidden / plan.heads;

  if (!recipe.experts_key.empty()) {
    plan.experts = getSize(cfg, recipe.experts_key);
  }

  return plan;
}

std::vector<std::string> RecipeRegistry::supportedArchitectures() const {
  std::vector<std::string> architectures;
  for (const auto &recipe : recipes_) {
    architectures.insert(architectures.end(), recipe.architectures.begin(),
                         recipe.architectures.end());
  }
  return architectures;
}

} // namespace quantize
} // namespace quick_dot_ai

int main(int argc, char **argv) {
  try {
    return quick_dot_ai::quantize::run(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
