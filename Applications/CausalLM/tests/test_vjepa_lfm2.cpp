// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   test_vjepa_lfm2.cpp
 * @date   1 June 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  End-to-end test: Video → VJEPA2 ViT → Projector → LFM2 → Text
 *
 * Pipeline:
 *   1. VideoPreprocessor: MP4 → 16 frames [C,H,W] (resize, normalize)
 *   2. VJEPA2ViT::run_image(): frames → vision hidden states (768-dim)
 *   3. VjepaProjector::run(): 768-dim → 1024-dim (LLM embedding space)
 *   4. Lfm2CausalLM::run_with_embeddings(): vision+text embeddings → text
 *
 * Usage:
 *   ./test_vjepa_lfm2 <video_path> <model_dir> [prompt] [do_sample]
 *
 *   <video_path>  Path to input video (MP4, AVI, etc.)
 *                 If the path ends with ".bin", loads pre-processed float32
 *                 frames directly (raw [T,C,H,W] layout), bypassing FFmpeg.
 *   <model_dir>   Path to model resource directory containing:
 *                   - vjepa_config.json / vjepa_nntr_config.json
 *                   - projector_config.json / projector_nntr_config.json
 *                   - config.json / nntr_config.json / generation_config.json
 *                   - Weight files (as specified in each nntr_config)
 *                   - tokenizer.json
 *   [prompt]       Text prompt (default: "Describe this video briefly.")
 *   [do_sample]    0 = greedy, 1 = sampling (default: 0)
 *
 * Examples:
 *   # With MP4 video input:
 *   ./test_vjepa_lfm2 /path/to/video.mp4 ../../Applications/CausalLM/res/vjepa2_lfm2
 *
 *   # With pre-processed .bin frames and custom prompt:
 *   ./test_vjepa_lfm2 /path/to/frames.bin ../../Applications/CausalLM/res/vjepa2_lfm2 \
 *       "What happens in this video?" 0
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <engine.h>
#include <factory.h>
#include <model.h>
#include <tokenizers_cpp.h>

#include "lfm2_causallm.h"
#include "vjepa2_vit/vjepa2_vit.h"
#include "vjepa2_vit/vjepa_projector.h"
#include "video_preprocessor.h"

using json = nlohmann::json;

/**
 * @brief Build chat template text segments with <video> placeholders
 *
 * Returns text segments split at <video> positions.
 * Pattern: seg[0], <video_0>, seg[1], <video_1>, ..., seg[N-1]
 * There are (num_video_tags + 2) segments.
 *
 * @param prompt       User text prompt
 * @param num_video_tags  Number of <video> placeholders
 * @param video_duration Total video duration in seconds
 * @return std::vector<std::string> text segments
 */
static std::vector<std::string>
apply_chat_template(const std::string &prompt,
                    unsigned int num_video_tags, float video_duration) {
  std::vector<std::string> segments;

  // First segment: system prompt + start of user
  segments.push_back("<|startoftext|><|im_start|>system\nYou are a helpful "
                     "assistant.<|im_end|>\n<|im_start|>user\n");

  // Segments between <video> tags (timestamps)
  const float time_per_video = video_duration / num_video_tags;
  for (unsigned int i = 0; i < num_video_tags; ++i) {
    char timestamp[32];
    std::snprintf(timestamp, sizeof(timestamp), "<%.1f seconds>",
                  i * time_per_video);
    segments.push_back(std::string(timestamp));
  }

  // Last segment: prompt + end tokens
  segments.push_back(prompt + "<|im_end|>\n<|im_start|>assistant\n");

  return segments;
}

/**
 * @brief Merge text token embeddings and vision embeddings into a single
 *        inputs_embeds buffer
 *
 * Text segments are tokenized and embedded via lookupEmbedding.
 * Between each pair of consecutive segments, vision embeddings are inserted
 * (vision_tokens_per_video tokens per <video> placeholder).
 *
 * @param text_segments     Text segments from apply_chat_template
 * @param tokenizer         Tokenizer instance
 * @param lfm2              LFM2 model (for embedding lookup)
 * @param vision_embeds     Vision embedding data (projected, text_dim)
 * @param num_video_tags    Number of <video> placeholders
 * @param vision_tokens_per_video  Vision tokens per <video>
 * @param text_dim          LLM hidden dimension
 * @param init_seq_len      Maximum sequence length (buffer size)
 * @param batch_size        Batch size
 * @return pair of (inputs_embeds vector, actual_total_tokens)
 */
static std::pair<std::vector<float>, unsigned int>
merge_text_image_embeddings(
  const std::vector<std::string> &text_segments,
  const std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
  const std::unique_ptr<causallm::Lfm2CausalLM> &lfm2,
  const float *vision_embeds, unsigned int num_video_tags,
  unsigned int vision_tokens_per_video, unsigned int text_dim,
  unsigned int init_seq_len, unsigned int batch_size) {

  std::vector<float> inputs_embeds(
    static_cast<size_t>(batch_size) * init_seq_len * text_dim, 0.0f);

  size_t embed_offset = 0;
  size_t video_idx = 0;

  for (size_t seg_i = 0; seg_i < text_segments.size(); ++seg_i) {
    // Tokenize and embed this text segment
    if (!text_segments[seg_i].empty()) {
      auto enc =
        tokenizer->Encode(text_segments[seg_i], /*add_special_token=*/false);
      for (auto id : enc) {
        std::vector<float> embed =
          lfm2->lookupEmbedding(static_cast<unsigned int>(id));
        std::copy(embed.begin(), embed.end(),
                  inputs_embeds.data() + embed_offset);
        embed_offset += text_dim;
      }
    }

    // Insert vision embeddings after each segment (except the last)
    if (seg_i < num_video_tags) {
      size_t vision_start = video_idx * vision_tokens_per_video;
      for (size_t v = 0; v < vision_tokens_per_video; ++v) {
        std::copy(vision_embeds + (vision_start + v) * text_dim,
                  vision_embeds + (vision_start + v + 1) * text_dim,
                  inputs_embeds.data() + embed_offset);
        embed_offset += text_dim;
      }
      video_idx++;
    }
  }

  unsigned int actual_total_tokens =
    static_cast<unsigned int>(embed_offset / text_dim);
  return {std::move(inputs_embeds), actual_total_tokens};
}

static void printSection(const std::string &title) {
  std::cout << "\n═══════════════════════════════════════════════════════\n"
            << "  " << title
            << "\n═══════════════════════════════════════════════════════\n\n";
}

static std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <video_path> <model_dir> [prompt] [do_sample]\n\n"
              << "  <video_path>  Path to input video (MP4)\n"
              << "  <model_dir>    Path to model resource directory\n"
              << "  [prompt]       Text prompt (default: 'Describe this video "
                 "briefly.')\n"
              << "  [do_sample]    0/1 (default: 0)\n";
    return EXIT_FAILURE;
  }

  const std::string video_path = argv[1];
  const std::string model_dir = argv[2];
  const std::string prompt =
    (argc >= 4) ? argv[3] : "Describe this video briefly.";
  const bool do_sample = (argc >= 5) ? (std::string(argv[4]) == "1") : false;

  try {
    // ── 1. Load configuration files ────────────────────────────────
    printSection("Step 1: Load Configuration");

    // VJEPA2 ViT config
    json vjepa_cfg = causallm::LoadJsonFile(model_dir + "/vjepa_config.json");
    json vjepa_nntr_cfg =
      causallm::LoadJsonFile(model_dir + "/vjepa_nntr_config.json");

    // Projector config
    json proj_cfg = causallm::LoadJsonFile(model_dir + "/projector_config.json");
    json proj_nntr_cfg =
      causallm::LoadJsonFile(model_dir + "/projector_nntr_config.json");

    // LFM2 config
    json lfm2_cfg = causallm::LoadJsonFile(model_dir + "/config.json");
    json lfm2_nntr_cfg =
      causallm::LoadJsonFile(model_dir + "/nntr_config.json");
    json lfm2_generation_cfg =
      causallm::LoadJsonFile(model_dir + "/generation_config.json");

    // Force USE_EMBEDDING=true for LFM2
    lfm2_nntr_cfg["use_embedding"] = true;

    const std::string vjepa_weight =
      model_dir + "/" + vjepa_nntr_cfg["model_file_name"].get<std::string>();
    const std::string proj_weight =
      model_dir + "/" + proj_nntr_cfg["model_file_name"].get<std::string>();
    const std::string lfm2_weight =
      model_dir + "/" + lfm2_nntr_cfg["model_file_name"].get<std::string>();

    std::cout << "  video_path    : " << video_path << "\n";
    std::cout << "  model_dir     : " << model_dir << "\n";
    std::cout << "  vjepa_weight  : " << vjepa_weight << "\n";
    std::cout << "  proj_weight   : " << proj_weight << "\n";
    std::cout << "  lfm2_weight   : " << lfm2_weight << "\n";
    std::cout << "  prompt        : " << prompt << "\n";

    // ── 2. Video Preprocessing ──────────────────────────────────────
    printSection("Step 2: Video Preprocessing");

    auto t_preprocess_start = std::chrono::high_resolution_clock::now();

    causallm::VideoPreprocessorConfig vp_cfg;
    vp_cfg.target_fps = vjepa_cfg.value("target_fps", 4);
    vp_cfg.temporal_patch_size = vjepa_cfg.value("tubelet_size", 2);
    vp_cfg.target_height = vjepa_cfg.value("img_size", 384);
    vp_cfg.target_width = vjepa_cfg.value("img_size", 384);

    // Read mean/std from vision config if available
    if (vjepa_cfg.contains("image_mean")) {
      auto m = vjepa_cfg["image_mean"];
      vp_cfg.mean = {m[0].get<float>(), m[1].get<float>(), m[2].get<float>()};
    }
    if (vjepa_cfg.contains("image_std")) {
      auto s = vjepa_cfg["image_std"];
      vp_cfg.std_val = {s[0].get<float>(), s[1].get<float>(), s[2].get<float>()};
    }

    // If video_path ends with .bin, load preprocessed frames directly
    // (eliminates FFmpeg/PIL resize differences)
    std::vector<std::vector<float>> frames;
    if (video_path.size() >= 4 &&
        video_path.substr(video_path.size() - 4) == ".bin") {
      unsigned int num_frames = vjepa_cfg.value("num_frames", 16);
      frames = causallm::VideoPreprocessor::loadPreprocessedFrames(
        video_path, num_frames, 3,
        vp_cfg.target_height, vp_cfg.target_width);
    } else {
      frames = causallm::VideoPreprocessor::process(video_path, vp_cfg);
    }

    auto t_preprocess_end = std::chrono::high_resolution_clock::now();
    auto preprocess_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        t_preprocess_end - t_preprocess_start)
        .count();

    std::cout << "  Frames extracted: " << frames.size() << "\n";
    std::cout << "  Frame shape: [3, " << vp_cfg.target_height << ", "
              << vp_cfg.target_width << "]\n";
    std::cout << "  Preprocessing time: " << preprocess_ms << " ms\n";

    // ── 3. VJEPA2 ViT Encoder ──────────────────────────────────────
    printSection("Step 3: VJEPA2 ViT Encoder");

    json empty_generation_cfg = json::object();
    auto vjepa = std::make_unique<causallm::VJEPA2ViT>(
      vjepa_cfg, empty_generation_cfg, vjepa_nntr_cfg);
    vjepa->initialize();
    std::cout << "  VJEPA2 graph constructed.\n";
    vjepa->load_weight(vjepa_weight);
    std::cout << "  VJEPA2 weights loaded.\n";

    auto t_vjepa_start = std::chrono::high_resolution_clock::now();
    auto [vision_ptr, vision_size] = vjepa->run_image(frames, 384, 384, true);
    auto t_vjepa_end = std::chrono::high_resolution_clock::now();
    auto vjepa_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        t_vjepa_end - t_vjepa_start)
        .count();

    const unsigned int num_patches = vjepa_cfg.value("num_frames", 16) /
                                       vjepa_cfg.value("tubelet_size", 2) *
                                     (vjepa_cfg.value("img_size", 384) /
                                      vjepa_cfg.value("patch_size", 16)) *
                                     (vjepa_cfg.value("img_size", 384) /
                                      vjepa_cfg.value("patch_size", 16));
    const unsigned int vision_dim = vjepa_cfg.value("hidden_size", 768);

    // After pixel_unshuffle: tokens = T * (H/f) * (W/f)
    const unsigned int downsample_factor =
      proj_cfg.value("downsample_factor", 2);
    const unsigned int output_tokens = num_patches / (downsample_factor *
                                                        downsample_factor);

    std::cout << "  Vision tokens: " << num_patches << " x " << vision_dim
              << "\n";
    std::cout << "  After pixel_unshuffle: " << output_tokens << " x "
              << vision_dim * downsample_factor * downsample_factor << "\n";
    std::cout << "  VJEPA2 time: " << vjepa_ms << " ms\n";

    // ── 4. Projector ────────────────────────────────────────────────
    printSection("Step 4: Projector (768 → 1024)");

    auto projector = std::make_unique<causallm::VjepaProjector>(
      proj_cfg, empty_generation_cfg, proj_nntr_cfg);
    projector->initialize();
    std::cout << "  Projector graph constructed.\n";
    projector->load_weight(proj_weight);
    std::cout << "  Projector weights loaded.\n";

    auto t_proj_start = std::chrono::high_resolution_clock::now();
    auto [proj_ptr, proj_size] =
      projector->run(static_cast<const float *>(vision_ptr), num_patches, true);
    auto t_proj_end = std::chrono::high_resolution_clock::now();
    auto proj_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        t_proj_end - t_proj_start)
        .count();

    const unsigned int text_dim = lfm2_cfg.value("hidden_size", 1024);
    std::cout << "  Projected tokens: " << output_tokens << " x " << text_dim
              << "\n";
    std::cout << "  Projector time: " << proj_ms << " ms\n";

    // ── 5. LFM2 CausalLM ───────────────────────────────────────────
    printSection("Step 5: LFM2 CausalLM (use_embedding=true)");

    auto lfm2 = std::make_unique<causallm::Lfm2CausalLM>(
      lfm2_cfg, lfm2_generation_cfg, lfm2_nntr_cfg);
    lfm2->initialize();
    std::cout << "  LFM2 graph constructed.\n";
    lfm2->load_weight(lfm2_weight);
    std::cout << "  LFM2 weights loaded.\n";

    // ── 6. Tokenize prompt and build merged embeddings ──────────────
    printSection("Step 6: Build Merged Embeddings");

    std::string tokenizer_path = lfm2_nntr_cfg["tokenizer_file"].get<std::string>();
    if (!std::filesystem::exists(tokenizer_path)) {
      std::string resolved = model_dir + "/tokenizer.json";
      if (std::filesystem::exists(resolved)) {
        tokenizer_path = resolved;
      } else {
        throw std::runtime_error("Tokenizer file not found: " + tokenizer_path);
      }
    }

    auto tokenizer =
      tokenizers::Tokenizer::FromBlobJSON(LoadBytesFromFile(tokenizer_path));

    // Build chat template and merge embeddings
    const unsigned int num_video_tags = 8;
    const unsigned int vision_tokens_per_video = output_tokens / num_video_tags;
    const float video_duration =
      static_cast<float>(vjepa_cfg.value("num_frames", 16)) /
      vjepa_cfg.value("target_fps", 4);

    auto text_segments =
      apply_chat_template(prompt, num_video_tags, video_duration);

    std::cout << "  Text segments: " << text_segments.size() << "\n";
    for (size_t i = 0; i < text_segments.size(); ++i) {
      std::cout << "    seg[" << i << "]: " << text_segments[i] << "\n";
    }

    const unsigned int init_seq_len =
      lfm2_nntr_cfg["init_seq_len"].get<unsigned int>();
    const unsigned int batch_size =
      lfm2_nntr_cfg["batch_size"].get<unsigned int>();
    const float *proj_data = static_cast<const float *>(proj_ptr);

    auto [inputs_embeds, actual_total_tokens] = merge_text_image_embeddings(
      text_segments, tokenizer, lfm2, proj_data, num_video_tags,
      vision_tokens_per_video, text_dim, init_seq_len, batch_size);

    // Build seed tokens for repetition penalty tracking
    std::vector<int> seed_tokens;
    for (size_t seg_i = 0; seg_i < text_segments.size(); ++seg_i) {
      if (!text_segments[seg_i].empty()) {
        auto enc = tokenizer->Encode(text_segments[seg_i],
                                     /*add_special_token=*/false);
        for (auto id : enc)
          seed_tokens.push_back(id);
      }
      if (seg_i < num_video_tags)
        seed_tokens.insert(seed_tokens.end(), vision_tokens_per_video, 0);
    }

    std::cout << "  Vision tokens per <video>: " << vision_tokens_per_video << "\n";
    std::cout << "  Total vision tokens: " << output_tokens << "\n";
    std::cout << "  Actual total tokens: " << actual_total_tokens << "\n";
    std::cout << "  Embedding buffer size: "
              << inputs_embeds.size() * sizeof(float) << " bytes ("
              << inputs_embeds.size() * sizeof(float) / (1024.0 * 1024.0)
              << " MB)\n";

    // ── 7. Run LFM2 inference ───────────────────────────────────────
    printSection("Step 7: LFM2 Inference (run_with_embeddings)");

    auto t_lfm2_start = std::chrono::high_resolution_clock::now();

    lfm2->run_with_embeddings(
      inputs_embeds.data(), actual_total_tokens, seed_tokens,
      do_sample, /*log_output=*/true);

    auto t_lfm2_end = std::chrono::high_resolution_clock::now();
    auto lfm2_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        t_lfm2_end - t_lfm2_start)
        .count();

    // ── 8. Print results ───────────────────────────────────────────
    printSection("Step 8: Results");

    const auto &generated_ids = lfm2->getGeneratedIds();
    std::cout << "  LFM2 inference time: " << lfm2_ms << " ms\n";
    std::cout << "  Generated tokens: " << generated_ids.size() << "\n";

    if (!generated_ids.empty()) {
      std::vector<int32_t> gen_ids_i32(generated_ids.begin(),
                                        generated_ids.end());
      std::string decoded = tokenizer->Decode(gen_ids_i32);
      std::cout << "  Decoded output: " << decoded << "\n";
    }

    // ── 9. Summary ─────────────────────────────────────────────────
    printSection("Summary");
    std::cout << "  Video preprocessing : " << preprocess_ms << " ms\n";
    std::cout << "  VJEPA2 encoder      : " << vjepa_ms << " ms\n";
    std::cout << "  Projector           : " << proj_ms << " ms\n";
    std::cout << "  LFM2 generation     : " << lfm2_ms << " ms\n";
    std::cout << "  Total               : "
              << preprocess_ms + vjepa_ms + proj_ms + lfm2_ms << " ms\n";

    std::cout << "\n═══════════════════════════════════════════════════════\n"
              << "  ✓ Test completed successfully!\n"
              << "═══════════════════════════════════════════════════════\n\n";

  } catch (const std::exception &e) {
    std::cerr << "\n[FATAL] " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
