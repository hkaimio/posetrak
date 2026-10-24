// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/seg_mask.hpp"

// stb_image is the only translation unit that defines STB_IMAGE_IMPLEMENTATION
// (the header is otherwise declaration-only, per its own usage contract).
// Restricted to PNG-only and no stdio path: seg_masks.mask_blob is always an
// in-memory PNG buffer (cv2.imencode's own output), decoded via
// stbi_load_from_memory() below -- nothing here ever opens a file, so the
// stdio-based loaders would be dead code.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "posetrak/third_party/stb_image.h"
#include <algorithm>
#include <cmath>

namespace posetrak {

bool decode_seg_mask(std::string const& png_bytes, DecodedSegMask& out) {
    if (png_bytes.empty())
        return false;
    int w = 0, h = 0, channels_in_file = 0;
    // req_comp=1: force single-channel output regardless of how the PNG was
    // encoded -- seg_masks.mask_blob is always written as plain greyscale
    // (cv2.imencode(".png", labeled) on a 2-D uint8 array), so this is a
    // no-op resample in the real case, not a lossy conversion.
    unsigned char* data =
        stbi_load_from_memory(reinterpret_cast<unsigned char const*>(png_bytes.data()),
                              static_cast<int>(png_bytes.size()), &w, &h, &channels_in_file, 1);
    if (data == nullptr)
        return false;
    out.mask_w = w;
    out.mask_h = h;
    out.labels.assign(data, data + (static_cast<size_t>(w) * static_cast<size_t>(h)));
    stbi_image_free(data);
    out.label_present.fill(false);
    for (std::uint8_t const v : out.labels)
        out.label_present[v] = true;
    return true;
}

int mask_label_at(DecodedSegMask const& mask, Eigen::Vector2d const& native_pos) {
    if (!mask.valid() || mask.native_w <= 0 || mask.native_h <= 0)
        return -1;
    if (native_pos.x() < 0 || native_pos.y() < 0 || native_pos.x() >= mask.native_w ||
        native_pos.y() >= mask.native_h)
        return -1;
    double const scale_x = static_cast<double>(mask.mask_w) / mask.native_w;
    double const scale_y = static_cast<double>(mask.mask_h) / mask.native_h;
    int const x = std::min(mask.mask_w - 1, static_cast<int>(native_pos.x() * scale_x));
    int const y = std::min(mask.mask_h - 1, static_cast<int>(native_pos.y() * scale_y));
    return mask.labels[static_cast<size_t>(y) * mask.mask_w + x];
}

bool mask_label_present_near(DecodedSegMask const& mask, Eigen::Vector2d const& native_pos,
                             int label, double dilate_px) {
    if (label <= 0 || !mask.valid() || mask.native_w <= 0 || mask.native_h <= 0)
        return false;

    double const scale_x = static_cast<double>(mask.mask_w) / mask.native_w;
    double const scale_y = static_cast<double>(mask.mask_h) / mask.native_h;
    int const cx = static_cast<int>(native_pos.x() * scale_x);
    int const cy = static_cast<int>(native_pos.y() * scale_y);
    // Radius in the mask's own (possibly downscaled) resolution -- same
    // "scale dilate_px into mask pixels" convention _make_dot_mask_filter's
    // own dilated_mask() uses, x-axis scale only (mask_w/native_w), since
    // that filter's own radius is likewise a single scalar, not separate
    // per-axis values.
    int const radius = std::max(1, static_cast<int>(std::lround(dilate_px * scale_x)));

    int const x0 = std::max(0, cx - radius);
    int const x1 = std::min(mask.mask_w - 1, cx + radius);
    int const y0 = std::max(0, cy - radius);
    int const y1 = std::min(mask.mask_h - 1, cy + radius);
    for (int y = y0; y <= y1; ++y) {
        std::uint8_t const* row = &mask.labels[static_cast<size_t>(y) * mask.mask_w];
        for (int x = x0; x <= x1; ++x) {
            if (row[x] == label)
                return true;
        }
    }
    return false;
}

void TrackletMaskAccumulator::add(bool in_mask, int window) {
    samples_.push_back(in_mask);
    if (in_mask)
        ++in_mask_count_;
    while (static_cast<int>(samples_.size()) > window && window >= 0) {
        if (samples_.front())
            --in_mask_count_;
        samples_.pop_front();
    }
}

std::optional<double> TrackletMaskAccumulator::fraction(int min_samples) const {
    if (static_cast<int>(samples_.size()) < min_samples)
        return std::nullopt;
    return static_cast<double>(in_mask_count_) / static_cast<double>(samples_.size());
}

}  // namespace posetrak
