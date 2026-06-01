// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   video_preprocessor.cpp
 * @date   1 June 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Video preprocessor for V-JEPA 2.1 encoder input.
 */

#include "video_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace causallm {

/**
 * @brief RAII wrapper for AVFormatContext.
 */
struct AvFormatCtx {
  AVFormatContext *ctx = nullptr;
  ~AvFormatCtx() {
    if (ctx)
      avformat_close_input(&ctx);
  }
  AVFormatContext **operator&() { return &ctx; }
  AVFormatContext *get() const { return ctx; }
};

/**
 * @brief RAII wrapper for AVCodecContext.
 */
struct AvCodecCtx {
  AVCodecContext *ctx = nullptr;
  ~AvCodecCtx() {
    if (ctx)
      avcodec_free_context(&ctx);
  }
  AVCodecContext *get() const { return ctx; }
};

/**
 * @brief RAII wrapper for AVFrame.
 */
struct AvFrame {
  AVFrame *frame = nullptr;
  AvFrame() : frame(av_frame_alloc()) {}
  ~AvFrame() {
    if (frame)
      av_frame_free(&frame);
  }
  AVFrame *get() const { return frame; }
};

/**
 * @brief RAII wrapper for AVPacket.
 */
struct AvPacket {
  AVPacket *pkt = nullptr;
  AvPacket() : pkt(av_packet_alloc()) {}
  ~AvPacket() {
    if (pkt)
      av_packet_free(&pkt);
  }
  AVPacket *get() const { return pkt; }
};

unsigned int VideoPreprocessor::getFrameCount(const std::string &video_path) {
  AvFormatCtx fmt_ctx;
  if (avformat_open_input(&fmt_ctx, video_path.c_str(), nullptr, nullptr) != 0) {
    std::cerr << "Warning: could not open video: " << video_path << std::endl;
    return 0;
  }
  if (avformat_find_stream_info(fmt_ctx.get(), nullptr) < 0) {
    return 0;
  }

  int stream_idx =
    av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (stream_idx < 0) {
    return 0;
  }

  AVStream *stream = fmt_ctx.get()->streams[stream_idx];

  if (stream->nb_frames > 0) {
    return static_cast<unsigned int>(stream->nb_frames);
  }

  if (stream->duration > 0 && stream->time_base.den > 0) {
    double duration_sec = av_q2d(stream->time_base) * stream->duration;
    double fps = av_q2d(stream->r_frame_rate);
    if (fps > 0) {
      return static_cast<unsigned int>(std::round(duration_sec * fps));
    }
  }

  return 0;
}

double VideoPreprocessor::getFPS(const std::string &video_path) {
  AvFormatCtx fmt_ctx;
  if (avformat_open_input(&fmt_ctx, video_path.c_str(), nullptr, nullptr) != 0) {
    return 0.0;
  }
  if (avformat_find_stream_info(fmt_ctx.get(), nullptr) < 0) {
    return 0.0;
  }

  int stream_idx =
    av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (stream_idx < 0) {
    return 0.0;
  }

  AVStream *stream = fmt_ctx.get()->streams[stream_idx];
  AVRational r_frame_rate = stream->r_frame_rate;

  if (r_frame_rate.num > 0 && r_frame_rate.den > 0) {
    return av_q2d(r_frame_rate);
  }

  // Fallback: estimate from avg_frame_rate
  AVRational avg_rate = stream->avg_frame_rate;
  if (avg_rate.num > 0 && avg_rate.den > 0) {
    return av_q2d(avg_rate);
  }

  return 0.0;
}

unsigned int VideoPreprocessor::computeNumFrames(unsigned int total_frames,
                                                 double video_fps,
                                                 const VideoPreprocessorConfig &cfg) {
  // VoRA formula: num_frames = int(total_frames / video_fps * target_fps)
  unsigned int num_frames = 0;
  if (video_fps > 0) {
    num_frames = static_cast<unsigned int>(
      static_cast<double>(total_frames) / video_fps * cfg.target_fps);
  } else {
    // If FPS unknown, use all frames
    num_frames = total_frames;
  }

  // Clamp to [min_frames, max_frames]
  num_frames = std::max(num_frames, cfg.min_frames);
  num_frames = std::min(num_frames, cfg.max_frames);
  num_frames = std::min(num_frames, total_frames);

  // Align to temporal_patch_size
  num_frames = (num_frames / cfg.temporal_patch_size) * cfg.temporal_patch_size;
  num_frames = std::max(num_frames, cfg.temporal_patch_size);

  return num_frames;
}

std::vector<unsigned int>
VideoPreprocessor::computeSampleIndices(unsigned int total_frames,
                                         unsigned int num_frames) {
  std::vector<unsigned int> indices;
  indices.reserve(num_frames);

  if (num_frames == 0 || total_frames == 0) {
    return indices;
  }

  if (num_frames >= total_frames) {
    // Use all frames
    for (unsigned int i = 0; i < total_frames; ++i) {
      indices.push_back(i);
    }
    return indices;
  }

  // Match np.linspace(0, total_frames-1, num_frames).round().astype(int)
  // np.linspace generates num_frames evenly spaced values from 0 to (total_frames-1)
  // For i in [0, num_frames): value = i * (total_frames-1) / (num_frames-1)
  for (unsigned int i = 0; i < num_frames; ++i) {
    double val;
    if (num_frames == 1) {
      val = 0.0;
    } else {
      val = static_cast<double>(i) * static_cast<double>(total_frames - 1) /
            static_cast<double>(num_frames - 1);
    }
    unsigned int idx = static_cast<unsigned int>(std::round(val));
    idx = std::min(idx, total_frames - 1);
    indices.push_back(idx);
  }

  return indices;
}

std::vector<std::vector<float>>
VideoPreprocessor::process(const std::string &video_path,
                            const VideoPreprocessorConfig &cfg) {
  // ── 1. Open video file ──────────────────────────────────────────
  AvFormatCtx fmt_ctx;
  if (avformat_open_input(&fmt_ctx, video_path.c_str(), nullptr, nullptr) != 0) {
    throw std::runtime_error("VideoPreprocessor: could not open video: " +
                             video_path);
  }
  if (avformat_find_stream_info(fmt_ctx.get(), nullptr) < 0) {
    throw std::runtime_error(
      "VideoPreprocessor: could not find stream info for: " + video_path);
  }

  // ── 2. Find best video stream ───────────────────────────────────
  int stream_idx =
    av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (stream_idx < 0) {
    throw std::runtime_error(
      "VideoPreprocessor: no video stream found in: " + video_path);
  }
  AVStream *video_stream = fmt_ctx.get()->streams[stream_idx];

  // ── 3. Open decoder ─────────────────────────────────────────────
  const AVCodec *codec =
    avcodec_find_decoder(video_stream->codecpar->codec_id);
  if (!codec) {
    throw std::runtime_error(
      "VideoPreprocessor: unsupported codec in: " + video_path);
  }

  AvCodecCtx codec_ctx;
  codec_ctx.ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx.ctx) {
    throw std::runtime_error(
      "VideoPreprocessor: could not allocate codec context");
  }
  if (avcodec_parameters_to_context(codec_ctx.get(),
                                    video_stream->codecpar) < 0) {
    throw std::runtime_error(
      "VideoPreprocessor: could not copy codec parameters");
  }
  if (avcodec_open2(codec_ctx.get(), codec, nullptr) < 0) {
    throw std::runtime_error("VideoPreprocessor: could not open codec");
  }

  // ── 4. Compute frame count and sampling indices ─────────────────
  unsigned int total_frames = 0;
  if (video_stream->nb_frames > 0) {
    total_frames = static_cast<unsigned int>(video_stream->nb_frames);
  } else {
    double duration_sec =
      av_q2d(video_stream->time_base) * video_stream->duration;
    double fps = av_q2d(video_stream->r_frame_rate);
    if (fps > 0 && duration_sec > 0) {
      total_frames =
        static_cast<unsigned int>(std::round(duration_sec * fps));
    }
  }

  if (total_frames == 0) {
    throw std::runtime_error(
      "VideoPreprocessor: could not determine frame count for: " +
      video_path);
  }

  // Get video FPS
  double video_fps = av_q2d(video_stream->r_frame_rate);
  if (video_fps <= 0) {
    video_fps = av_q2d(video_stream->avg_frame_rate);
  }

  // Compute number of frames using VoRA formula
  unsigned int num_frames = computeNumFrames(total_frames, video_fps, cfg);

  // Compute sampling indices matching np.linspace
  std::vector<unsigned int> sample_indices =
    computeSampleIndices(total_frames, num_frames);

  std::cerr << "VideoPreprocessor: total_frames=" << total_frames
            << ", video_fps=" << video_fps
            << ", target_fps=" << cfg.target_fps
            << ", num_frames=" << num_frames
            << ", grid_t=" << (num_frames / cfg.temporal_patch_size)
            << std::endl;

  // ── 5. Decode frames ────────────────────────────────────────────
  AvFrame frame;
  AvPacket pkt;

  // Output dimensions
  const int out_w = static_cast<int>(cfg.target_width);
  const int out_h = static_cast<int>(cfg.target_height);

  // Result: vector of [C, H, W] float frames
  std::vector<std::vector<float>> result;
  result.reserve(num_frames);

  // Track which sample indices we still need
  size_t next_sample = 0;
  unsigned int current_frame = 0;
  bool eof = false;

  while (next_sample < sample_indices.size() && !eof) {
    // Read packets until we get a frame or reach EOF
    int ret = av_read_frame(fmt_ctx.get(), pkt.get());
    if (ret < 0) {
      // EOF or error — flush decoder
      eof = true;
      pkt.get()->data = nullptr;
      pkt.get()->size = 0;
    }

    // Only process packets from our video stream
    if (!eof && pkt.get()->stream_index != stream_idx) {
      continue;
    }

    // Send packet to decoder
    ret = avcodec_send_packet(codec_ctx.get(), pkt.get());
    if (ret < 0 && ret != AVERROR(EAGAIN)) {
      break;
    }

    // Receive decoded frames
    while (true) {
      ret = avcodec_receive_frame(codec_ctx.get(), frame.get());
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break;
      }
      if (ret < 0) {
        throw std::runtime_error(
          "VideoPreprocessor: error during decoding");
      }

      // Check if this frame is one we need
      if (next_sample < sample_indices.size() &&
          current_frame == sample_indices[next_sample]) {

        // ── 6. Convert pixel format and resize ────────────────────
        SwsContext *sws_ctx = sws_getContext(
          codec_ctx.get()->width, codec_ctx.get()->height,
          codec_ctx.get()->pix_fmt, out_w, out_h, AV_PIX_FMT_RGB24,
          SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) {
          throw std::runtime_error(
            "VideoPreprocessor: could not create SwsContext");
        }

        // Output frame buffer (RGB24)
        std::vector<uint8_t> rgb_buf(
          static_cast<size_t>(out_w) * out_h * 3);

        // Setup destination data pointers
        uint8_t *dst_data[1] = {rgb_buf.data()};
        int dst_linesize[1] = {out_w * 3};

        sws_scale(sws_ctx, frame.get()->data, frame.get()->linesize, 0,
                  codec_ctx.get()->height, dst_data, dst_linesize);
        sws_freeContext(sws_ctx);

        // ── 7. Normalize to [C, H, W] float tensor ──────────────
        const size_t frame_size =
          static_cast<size_t>(3) * cfg.target_height * cfg.target_width;
        std::vector<float> float_frame(frame_size);

        for (unsigned int y = 0; y < cfg.target_height; ++y) {
          for (unsigned int x = 0; x < cfg.target_width; ++x) {
            // Source: RGB24 interleaved [H, W, C]
            size_t src_idx = (static_cast<size_t>(y) * out_w + x) * 3;
            for (unsigned int c = 0; c < 3; ++c) {
              // Destination: [C, H, W]
              size_t dst_idx =
                (static_cast<size_t>(c) * cfg.target_height + y) *
                  cfg.target_width + x;
              float val =
                static_cast<float>(rgb_buf[src_idx + c]) / 255.0f;
              float_frame[dst_idx] =
                (val - cfg.mean[c]) / cfg.std_val[c];
            }
          }
        }

        result.push_back(std::move(float_frame));
        ++next_sample;
      }

      ++current_frame;
    }
  }

  if (result.size() != num_frames) {
    // If we didn't get enough frames, pad by repeating the last frame
    while (result.size() < num_frames) {
      if (result.empty()) {
        throw std::runtime_error(
          "VideoPreprocessor: no frames decoded from: " + video_path);
      }
      result.push_back(result.back());
    }
    std::cerr << "Warning: decoded only " << result.size()
              << " frames, padded with last frame to " << num_frames
              << std::endl;
  }

  return result;
}

std::vector<std::vector<float>>
VideoPreprocessor::loadPreprocessedFrames(const std::string &bin_path,
                                           unsigned int num_frames,
                                           unsigned int channels,
                                           unsigned int height,
                                           unsigned int width) {
  const size_t frame_size =
    static_cast<size_t>(channels) * height * width;
  const size_t total_size =
    static_cast<size_t>(num_frames) * frame_size;

  std::ifstream file(bin_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error(
      "VideoPreprocessor: cannot open preprocessed frames file: " + bin_path);
  }

  size_t file_size = file.tellg() / sizeof(float);
  file.seekg(0, std::ios::beg);

  if (file_size < total_size) {
    throw std::runtime_error(
      "VideoPreprocessor: file too small: expected " +
      std::to_string(total_size) + " floats, got " +
      std::to_string(file_size) + " in " + bin_path);
  }

  // Read all data at once
  std::vector<float> all_data(total_size);
  file.read(reinterpret_cast<char *>(all_data.data()),
            total_size * sizeof(float));

  // Split into per-frame vectors [C, H, W]
  std::vector<std::vector<float>> result;
  result.reserve(num_frames);

  for (unsigned int t = 0; t < num_frames; ++t) {
    const float *frame_start = all_data.data() + t * frame_size;
    result.emplace_back(frame_start, frame_start + frame_size);
  }

  std::cerr << "VideoPreprocessor: loaded " << num_frames
            << " preprocessed frames from " << bin_path << std::endl;

  return result;
}

} // namespace causallm
