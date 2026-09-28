// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file seg_mask.hpp
 * @brief Segmentation-mask decoding and per-tracklet trust tracking for the
 * shared dot-assignment phase -- see
 * docs/roadmap/features/marker-based-mocap/segmentation-mask-dot-assignment-design.md.
 *
 * Two independent pieces, both deliberately DB/Tracker-free (same "pure core,
 * directly testable" split as streak_k_accumulator.hpp):
 *  - decode_seg_mask(): turns one seg_masks.mask_blob (a plain greyscale PNG,
 *    label 0=bg, 1..N=subject) into a plain label buffer.
 *  - TrackletMaskAccumulator: a running (frames-in-mask / frames-seen)
 *    estimate for one (subject, camera, tracklet), mirroring
 *    StreakKAccumulator's own windowed-ratio shape.
 *
 * The caller (resolve_shared_dot_assignment()'s wiring in cpp/cli/track.cpp)
 * is responsible for reading the raw blob from the seg_masks table and for
 * owning a TrackletMaskAccumulator map across steps; this file only supplies
 * the decode and the accumulator type.
 */
#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace posetrak {

/// @brief One decoded segmentation mask for one (camera, frame): the raw
/// label buffer plus both the mask's own resolution and the native video
/// frame's resolution -- masks may be stored downscaled from the native
/// frame (same convention app.pose.pose_worker._load_mask already uses), so
/// a caller testing a native-pixel candidate position against it needs both
/// sizes to scale coordinates, not just the buffer.
struct DecodedSegMask {
    int mask_w = 0;
    int mask_h = 0;
    int native_w = 0;
    int native_h = 0;
    std::vector<std::uint8_t> labels;  ///< Row-major, mask_w * mask_h bytes.

    bool valid() const { return mask_w > 0 && mask_h > 0 && !labels.empty(); }
};

/// @brief Decodes *png_bytes* (a seg_masks.mask_blob value: a plain
/// single-channel PNG, `cv2.imencode(".png", labeled)`'s own output shape --
/// no palette, no interlacing) into *out*. *out*'s native_w/native_h must
/// already be set by the caller (the mask PNG itself carries no notion of
/// the source video's own resolution); mask_w/mask_h and labels are filled
/// in from the decode.
///
/// @return false if *png_bytes* is empty or fails to decode as a greyscale
///         PNG (out is left with mask_w/mask_h/labels untouched on failure,
///         so a caller can fall back to "no mask this frame" the same way a
///         missing seg_masks row already does).
bool decode_seg_mask(std::string const& png_bytes, DecodedSegMask& out);

/// @brief Does *mask* carry *label* anywhere within *dilate_px* (in the
/// mask's own resolution, scaled from *dilate_px* native pixels) of
/// *native_pos*?
///
/// Scans a square neighbourhood around the point rather than precomputing a
/// dilated copy of the whole mask image (as the finalise-time
/// `_make_dot_mask_filter` does): resolve_dot_assignment() only ever tests a
/// few dozen candidate positions per camera per step, far fewer than the
/// pixels a whole-image dilation would touch, so a local scan per query
/// point is the cheaper direction.
///
/// @param mask Must be valid() -- a caller with no mask for this
///        (camera, frame) should never reach this function at all.
/// @param native_pos Candidate position in the *native* frame's own pixels
///        (distorted, matching how the mask itself is drawn on the raw
///        video frame -- see SegmentationMaskModifier's own doc comment).
/// @param label The subject's own mask label (1..N); 0 or negative always
///        returns false (0 is background, never a real subject).
/// @param dilate_px Search radius, in native pixels.
bool mask_label_present_near(DecodedSegMask const& mask, Eigen::Vector2d const& native_pos,
                             int label, double dilate_px);

/// @brief Running (frames-in-mask / frames-seen) estimate for one tracklet,
/// windowed the same way StreakKAccumulator windows its own ratio-of-sums --
/// so a long-lived tracklet's fraction reflects its *recent* trustworthiness,
/// not "ever, even once, long ago". See SegmentationMaskModifier
/// (dot_assignment.hpp) for how the fraction is used.
class TrackletMaskAccumulator {
   public:
    /// @brief Records one more frame's observation for this tracklet, evicting
    /// the oldest sample once the window would exceed *window*.
    void add(bool in_mask, int window);

    /// @brief Current in-mask fraction, or std::nullopt if fewer than
    /// *min_samples* samples are in the window yet.
    std::optional<double> fraction(int min_samples) const;

    /// @brief Samples currently in the window (for diagnostics/tests).
    int sample_count() const { return static_cast<int>(samples_.size()); }

   private:
    std::deque<bool> samples_;  ///< oldest first
    int in_mask_count_ = 0;
};

}  // namespace posetrak
