// SPDX-License-Identifier: Apache-2.0

#include "quantize_stream.h"

namespace quick_dot_ai {
namespace quantize {
namespace {

struct LayerShape {
  std::string prefix;
  size_t q_width;
  size_t kv_width;
};

LayerShape layerShape(const ModelPlan &plan, size_t layer) {
  return {
    "layer" + std::to_string(layer),
    plan.heads * plan.head_dim,
    plan.kv_heads * plan.head_dim,
  };
}

void writeDenseMlp(TensorWriter &writer, const ModelPlan &plan,
                   const QuantPlan &quant_plan, const std::string &prefix) {
  writer.writeTransposedMatrix(plan.hidden, plan.intermediate,
                               quant_plan.fc_dtype, prefix + "_ffn_up");
  writer.writeTransposedMatrix(plan.hidden, plan.intermediate,
                               quant_plan.fc_dtype, prefix + "_ffn_gate");
  writer.writeTransposedMatrix(plan.intermediate, plan.hidden,
                               quant_plan.fc_dtype, prefix + "_ffn_down");
}

void writeGemmaMlp(TensorWriter &writer, const ModelPlan &plan,
                   const QuantPlan &quant_plan, const std::string &prefix) {
  writer.writeTransposedMatrix(plan.hidden, plan.intermediate,
                               quant_plan.fc_dtype, prefix + "_ffn_gate");
  writer.writeTransposedMatrix(plan.hidden, plan.intermediate,
                               quant_plan.fc_dtype, prefix + "_ffn_up");
  writer.writeTransposedMatrix(plan.intermediate, plan.hidden,
                               quant_plan.fc_dtype, prefix + "_ffn_down");
}

void writeMoe(TensorWriter &writer, const ModelPlan &plan,
              const QuantPlan &quant_plan, const std::string &prefix) {
  writer.copyFp32Tensor(plan.hidden * plan.experts, prefix + "_router");
  for (size_t expert = 0; expert < plan.experts; ++expert) {
    const std::string expert_name = prefix + "_expert" + std::to_string(expert);
    writer.writeTransposedMatrix(plan.hidden, plan.intermediate,
                                 quant_plan.fc_dtype, expert_name + "_up");
    writer.writeTransposedMatrix(plan.hidden, plan.intermediate,
                                 quant_plan.fc_dtype, expert_name + "_gate");
    writer.writeTransposedMatrix(plan.intermediate, plan.hidden,
                                 quant_plan.fc_dtype, expert_name + "_down");
  }
}

// Sparse SmallThinker MoE (PowerInfer fused_sparse_moe layout): gate, up, and
// down are ALL stored as PLAIN per-neuron-row Q4_0 laid out [intermediate,
// hidden] (neuron-major): row j = neuron j's weights, contiguous over hidden.
// The fused sparse kernel computes gate per-row to build the ReLU mask, then
// up/down only for active neurons — so gate must be plain too (not repacked).
// Write order (up, gate, down) matches writeMoe so requestWeight indices align.
void writeMoeSparse(TensorWriter &writer, const ModelPlan &plan,
                    const QuantPlan &quant_plan, const std::string &prefix) {
  writer.copyFp32Tensor(plan.hidden * plan.experts, prefix + "_router");
  for (size_t expert = 0; expert < plan.experts; ++expert) {
    const std::string expert_name = prefix + "_expert" + std::to_string(expert);
    // up [hidden,intermediate] -> transpose -> [intermediate,hidden] plain.
    writer.writeTransposedMatrixPlain(plan.hidden, plan.intermediate,
                                      quant_plan.fc_dtype, expert_name + "_up");
    // gate stays REPACKED (q4_0x8/x4): the B3 hybrid path computes the full gate
    // with the optimal dense GEMV to build the ReLU mask, then up/down sparse.
    writer.writeTransposedMatrix(plan.hidden, plan.intermediate,
                                 quant_plan.fc_dtype, expert_name + "_gate");
    // down source is already [intermediate,hidden] (neuron-major) -> plain, no
    // transpose: row j = neuron j's contribution over hidden.
    writer.writeMatrixPlain(plan.intermediate, plan.hidden, quant_plan.fc_dtype,
                            expert_name + "_down");
  }
}

void writePlainAttention(TensorWriter &writer, const ModelPlan &plan,
                         const QuantPlan &quant_plan, const LayerShape &shape) {
  writer.writeTransposedMatrix(plan.hidden, shape.q_width, quant_plan.fc_dtype,
                               shape.prefix + "_wq");
  writer.writeTransposedMatrix(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                               shape.prefix + "_wk");
  writer.writeTransposedMatrix(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                               shape.prefix + "_wv");
  writer.writeTransposedMatrix(shape.q_width, plan.hidden, quant_plan.fc_dtype,
                               shape.prefix + "_attention_out");
}

void writeQwen2Attention(TensorWriter &writer, const ModelPlan &plan,
                         const QuantPlan &quant_plan, const LayerShape &shape) {
  writer.quantizeFcWithBias(plan.hidden, shape.q_width, quant_plan.fc_dtype,
                            shape.prefix + "_wq");
  writer.quantizeFcWithBias(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                            shape.prefix + "_wk");
  writer.quantizeFcWithBias(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                            shape.prefix + "_wv");
  writer.writeTransposedMatrix(shape.q_width, plan.hidden, quant_plan.fc_dtype,
                               shape.prefix + "_attention_out");
}

void writeGptOssAttention(TensorWriter &writer, const ModelPlan &plan,
                          const QuantPlan &quant_plan,
                          const LayerShape &shape) {
  writer.quantizeFcWithBias(plan.hidden, shape.q_width, quant_plan.fc_dtype,
                            shape.prefix + "_wq");
  writer.quantizeFcWithBias(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                            shape.prefix + "_wk");
  writer.quantizeFcWithBias(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                            shape.prefix + "_wv");
  writer.quantizeFcWithBias(plan.hidden, plan.hidden, quant_plan.fc_dtype,
                            shape.prefix + "_attention_out");
}

void writeCachedSlimGptOssAttention(TensorWriter &writer, const ModelPlan &plan,
                                    const QuantPlan &quant_plan,
                                    const LayerShape &shape) {
  writer.quantizeFcWithBias(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                            shape.prefix + "_wv");
  writer.quantizeFcWithBias(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                            shape.prefix + "_wk");
  writer.quantizeFcWithBias(plan.hidden, shape.q_width, quant_plan.fc_dtype,
                            shape.prefix + "_wq");
  writer.quantizeFcWithBias(plan.hidden, plan.hidden, quant_plan.fc_dtype,
                            shape.prefix + "_attention_out");
}

void writeQwen3Attention(TensorWriter &writer, const ModelPlan &plan,
                         const QuantPlan &quant_plan, const LayerShape &shape) {
  writer.writeTransposedMatrix(plan.hidden, shape.q_width, quant_plan.fc_dtype,
                               shape.prefix + "_wq");
  writer.copyFp32Tensor(plan.head_dim, shape.prefix + "_q_norm");
  writer.writeTransposedMatrix(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                               shape.prefix + "_wk");
  writer.copyFp32Tensor(plan.head_dim, shape.prefix + "_k_norm");
  writer.writeTransposedMatrix(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                               shape.prefix + "_wv");
  writer.writeTransposedMatrix(shape.q_width, plan.hidden, quant_plan.fc_dtype,
                               shape.prefix + "_attention_out");
}

void writeGemma3Attention(TensorWriter &writer, const ModelPlan &plan,
                          const QuantPlan &quant_plan,
                          const LayerShape &shape) {
  writer.writeTransposedMatrix(plan.hidden, shape.q_width, quant_plan.fc_dtype,
                               shape.prefix + "_wq");
  writer.writeTransposedMatrix(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                               shape.prefix + "_wk");
  writer.writeTransposedMatrix(plan.hidden, shape.kv_width, quant_plan.fc_dtype,
                               shape.prefix + "_wv");
  writer.copyFp32Tensor(plan.head_dim, shape.prefix + "_q_norm");
  writer.copyFp32Tensor(plan.head_dim, shape.prefix + "_k_norm");
  writer.writeTransposedMatrix(shape.q_width, plan.hidden, quant_plan.fc_dtype,
                               shape.prefix + "_attention_out");
}

void writeDenseLayer(TensorWriter &writer, const ModelPlan &plan,
                     const QuantPlan &quant_plan, size_t layer) {
  const LayerShape shape = layerShape(plan, layer);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_attention_norm");
  writePlainAttention(writer, plan, quant_plan, shape);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_ffn_norm");
  writeDenseMlp(writer, plan, quant_plan, shape.prefix);
}

void writeQwen2Layer(TensorWriter &writer, const ModelPlan &plan,
                     const QuantPlan &quant_plan, size_t layer) {
  const LayerShape shape = layerShape(plan, layer);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_attention_norm");
  writeQwen2Attention(writer, plan, quant_plan, shape);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_ffn_norm");
  writeDenseMlp(writer, plan, quant_plan, shape.prefix);
}

void writeQwen3Layer(TensorWriter &writer, const ModelPlan &plan,
                     const QuantPlan &quant_plan, size_t layer) {
  const LayerShape shape = layerShape(plan, layer);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_attention_norm");
  writeQwen3Attention(writer, plan, quant_plan, shape);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_ffn_norm");
  writeDenseMlp(writer, plan, quant_plan, shape.prefix);
}

void writeQwen3MoeLayer(TensorWriter &writer, const ModelPlan &plan,
                        const QuantPlan &quant_plan, size_t layer) {
  const LayerShape shape = layerShape(plan, layer);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_attention_norm");
  writeQwen3Attention(writer, plan, quant_plan, shape);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_ffn_norm");
  writeMoe(writer, plan, quant_plan, shape.prefix);
}

void writeGptOssMoe(TensorWriter &writer, const ModelPlan &plan,
                    const QuantPlan &quant_plan, const std::string &prefix) {
  writer.copyFp32Tensor(plan.hidden * plan.experts, prefix + "_router");
  writer.copyFp32Tensor(plan.experts, prefix + "_router_bias");
  for (size_t expert = 0; expert < plan.experts; ++expert) {
    const std::string expert_name = prefix + "_expert" + std::to_string(expert);
    writer.writeTransposedMatrix(plan.hidden, plan.intermediate,
                                 quant_plan.fc_dtype, expert_name + "_up");
    writer.copyFp32Tensor(plan.intermediate, expert_name + "_up_bias");
    writer.writeTransposedMatrix(plan.hidden, plan.intermediate,
                                 quant_plan.fc_dtype, expert_name + "_gate");
    writer.copyFp32Tensor(plan.intermediate, expert_name + "_gate_bias");
    writer.writeTransposedMatrix(plan.intermediate, plan.hidden,
                                 quant_plan.fc_dtype, expert_name + "_down");
    writer.copyFp32Tensor(plan.hidden, expert_name + "_down_bias");
  }
}

void writeGptOssLayer(TensorWriter &writer, const ModelPlan &plan,
                      const QuantPlan &quant_plan, size_t layer) {
  const LayerShape shape = layerShape(plan, layer);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_attention_norm");
  writeGptOssAttention(writer, plan, quant_plan, shape);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_ffn_norm");
  writeGptOssMoe(writer, plan, quant_plan, shape.prefix);
}

void writeCachedSlimGptOssLayer(TensorWriter &writer, const ModelPlan &plan,
                                const QuantPlan &quant_plan, size_t layer) {
  const LayerShape shape = layerShape(plan, layer);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_attention_norm");
  writeCachedSlimGptOssAttention(writer, plan, quant_plan, shape);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_ffn_norm");
  writeGptOssMoe(writer, plan, quant_plan, shape.prefix);
}

void writeSmallThinkerLayer(TensorWriter &writer, const ModelPlan &plan,
                            const QuantPlan &quant_plan, size_t layer) {
  const LayerShape shape = layerShape(plan, layer);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_attention_norm");
  writePlainAttention(writer, plan, quant_plan, shape);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_ffn_norm");
  writeMoe(writer, plan, quant_plan, shape.prefix);
}

void writeSmallThinkerSparseLayer(TensorWriter &writer, const ModelPlan &plan,
                                  const QuantPlan &quant_plan, size_t layer) {
  const LayerShape shape = layerShape(plan, layer);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_attention_norm");
  writePlainAttention(writer, plan, quant_plan, shape);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_ffn_norm");
  writeMoeSparse(writer, plan, quant_plan, shape.prefix);
}

void writeGemma3Layer(TensorWriter &writer, const ModelPlan &plan,
                      const QuantPlan &quant_plan, size_t layer) {
  const LayerShape shape = layerShape(plan, layer);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_attention_norm");
  writeGemma3Attention(writer, plan, quant_plan, shape);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_post_attention_norm");
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_pre_ffn_norm");
  writeGemmaMlp(writer, plan, quant_plan, shape.prefix);
  writer.copyFp32Tensor(plan.hidden, shape.prefix + "_post_ffn_norm");
}

} // namespace

void registerBuiltInRecipes(RecipeRegistry &registry) {
  registry.add({
    {"LlamaForCausalLM"},
    "dense",
    "intermediate_size",
    "",
    writeDenseLayer,
  });
  registry.add({
    {"Qwen2ForCausalLM"},
    "qwen2",
    "intermediate_size",
    "",
    writeQwen2Layer,
  });
  registry.add({
    {"Qwen3ForCausalLM"},
    "qwen3",
    "intermediate_size",
    "",
    writeQwen3Layer,
  });
  registry.add({
    {"Qwen3MoeForCausalLM"},
    "qwen3_moe",
    "moe_intermediate_size",
    "num_experts",
    writeQwen3MoeLayer,
  });
  registry.add({
    {"Gemma3ForCausalLM", "Gemma3TextModel"},
    "gemma3",
    "intermediate_size",
    "",
    writeGemma3Layer,
  });
  registry.add({
    {"GptOssForCausalLM"},
    "gpt_oss",
    "intermediate_size",
    "num_local_experts",
    writeGptOssLayer,
  });
  registry.add({
    {"GptOssCachedSlimCausalLM"},
    "gpt_oss_cached_slim",
    "intermediate_size",
    "num_local_experts",
    writeCachedSlimGptOssLayer,
  });
  registry.add({
    {"SmallThinkerForCausalLM"},
    "smallthinker",
    "moe_ffn_hidden_size",
    "moe_num_primary_experts",
    writeSmallThinkerLayer,
  });
  registry.add({
    {"SmallThinkerSparseForCausalLM", "SmallThinkerSparseCachedSlimForCausalLM",
     "SmallThinkerSparseSlimForCausalLM"},
    "smallthinker_sparse",
    "moe_ffn_hidden_size",
    "moe_num_primary_experts",
    writeSmallThinkerSparseLayer,
  });
}

} // namespace quantize
} // namespace quick_dot_ai
