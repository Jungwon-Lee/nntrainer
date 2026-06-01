// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   video_preprocessor.h
 * @date   1 June 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Video preprocessor for V-JEPA 2.1 encoder input.
 *
 * Reads a video file (MP4, AVI, etc.) via FFmpeg, computes the number of
 * frames to sample using the VoRA formula:
 *
 *   num_frames = int(total_frames / video_fps * target_fps)
 *   num_frames = clamp(num_frames, min_frames, max_frames)
 *   num_frames = (num_frames / temporal_patch_size) * temporal_patch_size
 *   num_frames = max(num_frames, temporal_patch_size)
 *
 * Then uniformly samples num_frames indices via linspace (matching
 * np.linspace(0, total_frames-1, num_frames).round().astype(int)),
 * resizes each frame to target resolution, and normalizes pixel values
 * with ImageNet mean/std.  The output is a vector of per-frame float
 * tensors in [C, H, W] layout, ready for VJEPA2ViT::run_image().
 */

#ifndef __VIDEO_PREPROCESSOR_H__
#define __VIDEO_PREPROCESSOR_H__

#include <string>
#include <vector>

namespace causallm {

/**
 * @brief Video preprocessor configuration.
 */
struct VideoPreprocessorConfig {
  unsigned int target_fps = 4;       /**< Target FPS for frame sampling */
  unsigned int min_frames = 4;      /**< Minimum number of frames */
  unsigned int max_frames = 60;     /**< Maximum number of frames */
  unsigned int temporal_patch_size = 2; /**< Tubelet size (must divide num_frames) */
  unsigned int target_height = 384; /**< Resize height */
  unsigned int target_width = 384;  /**< Resize width */
  std::vector<float> mean = {0.485f, 0.456f, 0.406f}; /**< ImageNet mean */
  std::vector<float> std_val = {0.229f, 0.224f, 0.225f}; /**< ImageNet std */
};

/**
 * @brief Video preprocessor for V-JEPA 2.1 encoder input.
 *
 * Uses FFmpeg C API (libavcodec/libavformat/libswscale) to decode video,
 * compute frame count using VoRA formula, uniformly sample frames,
 * resize, and normalize.
 */
class VideoPreprocessor {
public:
  /**
   * @brief Process a video file into per-frame [C, H, W] float tensors.
   *
   * @param video_path Path to the input video file (MP4, AVI, etc.)
   * @param cfg        Preprocessor configuration
   * @return Vector of frames, each in [C=3, H, W] layout, float32
   */
  static std::vector<std::vector<float>>
  process(const std::string &video_path,
          const VideoPreprocessorConfig &cfg = VideoPreprocessorConfig());

  /**
   * @brief Get the total number of frames in a video file.
   *
   * @param video_path Path to the input video file
   * @return Total frame count, or 0 on error
   */
  static unsigned int getFrameCount(const std::string &video_path);

  /**
   * @brief Get the FPS of a video file.
   *
   * @param video_path Path to the input video file
   * @return FPS, or 0 on error
   */
  static double getFPS(const std::string &video_path);

  /**
   * @brief Compute the number of frames to sample using VoRA formula.
   *
   * num_frames = int(total_frames / video_fps * target_fps)
   * num_frames = clamp(num_frames, min_frames, max_frames)
   * num_frames = (num_frames / temporal_patch_size) * temporal_patch_size
   * num_frames = max(num_frames, temporal_patch_size)
   *
   * @param total_frames Total frames in the video
   * @param video_fps    FPS of the video
   * @param cfg          Preprocessor configuration
   * @return Number of frames to sample
   */
  static unsigned int computeNumFrames(unsigned int total_frames,
                                       double video_fps,
                                       const VideoPreprocessorConfig &cfg);

  /**
   * @brief Compute uniform sampling indices matching np.linspace.
   *
   * Equivalent to: np.linspace(0, total_frames-1, num_frames).round().astype(int)
   *
   * @param total_frames Total frames in the video
   * @param num_frames   Number of frames to sample
   * @return Vector of frame indices
   */
  static std::vector<unsigned int>
  computeSampleIndices(unsigned int total_frames, unsigned int num_frames);

  /**
   * @brief Load pre-processed frames from a raw binary file.
   *
   * The file format is raw float32 in [T, C, H, W] layout, matching
   * the output of the Python verify_modules.py script.
   * This allows using Python-preprocessed frames to eliminate
   * FFmpeg/PIL resize differences.
   *
   * @param bin_path   Path to the .bin file containing float32 frames
   * @param num_frames Expected number of frames
   * @param channels   Number of channels (default: 3)
   * @param height     Frame height (default: 384)
   * @param width      Frame width (default: 384)
   * @return Vector of frames, each in [C, H, W] layout, float32
   */
  static std::vector<std::vector<float>>
  loadPreprocessedFrames(const std::string &bin_path,
                         unsigned int num_frames = 16,
                         unsigned int channels = 3,
                         unsigned int height = 384,
                         unsigned int width = 384);
};

} // namespace causallm

#endif /* __VIDEO_PREPROCESSOR_H__ */
