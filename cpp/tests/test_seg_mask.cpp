// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "posetrak/tracking/seg_mask.hpp"

using posetrak::decode_seg_mask;
using posetrak::DecodedSegMask;
using posetrak::mask_label_at;
using posetrak::mask_label_present_near;
using posetrak::TrackletMaskAccumulator;

namespace {
// A tiny 6x5, 8-bit greyscale PNG -- the same shape seg_masks.mask_blob
// stores (cv2.imencode(".png", labeled) on a 2-D uint8 label array),
// generated once with Python's stdlib zlib (no OpenCV/PIL dependency in this
// test tree). Row-major label layout, background (0) everywhere except:
//   label 1 at (x=1, y=1)
//   label 2 at (x=4, y=3)
constexpr unsigned char kTestPng[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
    0x52, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x05, 0x08, 0x00, 0x00, 0x00, 0x00, 0x43,
    0x33, 0xc2, 0x3a, 0x00, 0x00, 0x00, 0x12, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x60,
    0x80, 0x02, 0x46, 0x06, 0x34, 0xc0, 0x04, 0x63, 0x00, 0x00, 0x00, 0x4f, 0x00, 0x04, 0xcc,
    0x34, 0x83, 0x41, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

std::string test_png_bytes() {
    return std::string(reinterpret_cast<char const*>(kTestPng), sizeof(kTestPng));
}
}  // namespace

TEST_CASE("decode_seg_mask: decodes a real greyscale PNG's labels", "[seg_mask]") {
    DecodedSegMask mask;
    mask.native_w = 6;
    mask.native_h = 5;
    REQUIRE(decode_seg_mask(test_png_bytes(), mask));
    REQUIRE(mask.mask_w == 6);
    REQUIRE(mask.mask_h == 5);
    REQUIRE(mask.valid());
    REQUIRE(mask.labels[static_cast<size_t>(1) * 6 + 1] == 1);
    REQUIRE(mask.labels[static_cast<size_t>(3) * 6 + 4] == 2);
    REQUIRE(mask.labels[0] == 0);
}

TEST_CASE("decode_seg_mask: fails on empty or garbage input", "[seg_mask]") {
    DecodedSegMask mask;
    REQUIRE_FALSE(decode_seg_mask("", mask));
    REQUIRE_FALSE(decode_seg_mask("not a png", mask));
    REQUIRE_FALSE(mask.valid());
}

TEST_CASE("mask_label_present_near: finds a label exactly at the query point", "[seg_mask]") {
    DecodedSegMask mask;
    mask.native_w = 6;
    mask.native_h = 5;
    REQUIRE(decode_seg_mask(test_png_bytes(), mask));

    REQUIRE(mask_label_present_near(mask, Eigen::Vector2d(1.4, 1.4), 1, 0.0));
    REQUIRE(mask_label_present_near(mask, Eigen::Vector2d(4.0, 3.0), 2, 0.0));
    // Chebyshev distance from label 1's own pixel (1,1) to label 2's query point
    // (4,3) is 3, well outside even the implicit >=1px search radius below.
    REQUIRE_FALSE(mask_label_present_near(mask, Eigen::Vector2d(4.0, 3.0), 1, 0.0));
}

TEST_CASE("mask_label_present_near: dilate_px widens the search radius", "[seg_mask]") {
    DecodedSegMask mask;
    mask.native_w = 6;
    mask.native_h = 5;
    REQUIRE(decode_seg_mask(test_png_bytes(), mask));

    // (5,0) is Chebyshev distance 4 from label 1's pixel (1,1) -- undetectable
    // at a small radius, found once dilate_px widens the search enough to
    // bridge the gap. (The search radius floors to >=1px even at
    // dilate_px=0.0, mirroring _make_dot_mask_filter's own floor -- a point
    // this far away stays outside that floor too.)
    REQUIRE_FALSE(mask_label_present_near(mask, Eigen::Vector2d(5.0, 0.0), 1, 0.0));
    REQUIRE(mask_label_present_near(mask, Eigen::Vector2d(5.0, 0.0), 1, 5.0));
}

TEST_CASE("mask_label_present_near: label 0 (background) and negative labels are never present",
          "[seg_mask]") {
    DecodedSegMask mask;
    mask.native_w = 6;
    mask.native_h = 5;
    REQUIRE(decode_seg_mask(test_png_bytes(), mask));
    REQUIRE_FALSE(mask_label_present_near(mask, Eigen::Vector2d(1.0, 1.0), 0, 100.0));
    REQUIRE_FALSE(mask_label_present_near(mask, Eigen::Vector2d(1.0, 1.0), -1, 100.0));
}

TEST_CASE("mask_label_present_near: an invalid (undecoded) mask is never present", "[seg_mask]") {
    DecodedSegMask mask;  // never decoded -- valid() is false
    REQUIRE_FALSE(mask_label_present_near(mask, Eigen::Vector2d(0.0, 0.0), 1, 100.0));
}

TEST_CASE("TrackletMaskAccumulator: fraction() is nullopt with no samples", "[seg_mask]") {
    TrackletMaskAccumulator acc;
    REQUIRE_FALSE(acc.fraction(1).has_value());
}

TEST_CASE("TrackletMaskAccumulator: fraction() is nullopt below min_samples", "[seg_mask]") {
    TrackletMaskAccumulator acc;
    acc.add(true, 60);
    REQUIRE_FALSE(acc.fraction(2).has_value());
    REQUIRE(acc.fraction(1).has_value());
}

TEST_CASE("TrackletMaskAccumulator: fraction() reflects the in-mask ratio", "[seg_mask]") {
    TrackletMaskAccumulator acc;
    acc.add(true, 60);
    acc.add(true, 60);
    acc.add(false, 60);
    acc.add(true, 60);
    auto f = acc.fraction(1);
    REQUIRE(f.has_value());
    REQUIRE(*f == Catch::Approx(3.0 / 4.0));
}

TEST_CASE("TrackletMaskAccumulator: evicts the oldest sample once the window is exceeded",
          "[seg_mask]") {
    TrackletMaskAccumulator acc;
    acc.add(true, 2);  // will be evicted
    acc.add(true, 2);
    acc.add(false, 2);
    REQUIRE(acc.sample_count() == 2);
    auto f = acc.fraction(1);
    REQUIRE(f.has_value());
    REQUIRE(*f == Catch::Approx(0.5));  // {true, false} in the window
}

TEST_CASE("TrackletMaskAccumulator: recent history dominates a long-lived tracklet's fraction",
          "[seg_mask]") {
    // A tracklet with a long good history that recently went bad should read as
    // untrustworthy once the bad run fills the window -- the whole reason this is
    // windowed rather than a lifetime average (design doc §3: "recent
    // trustworthiness, not ever, once, years ago").
    TrackletMaskAccumulator acc;
    for (int i = 0; i < 60; ++i)
        acc.add(true, 60);
    for (int i = 0; i < 60; ++i)
        acc.add(false, 60);
    auto f = acc.fraction(1);
    REQUIRE(f.has_value());
    REQUIRE(*f == Catch::Approx(0.0));
}

TEST_CASE("mask_label_at reads the label of the mask pixel a native position falls in",
          "[seg_mask]") {
    DecodedSegMask mask;
    mask.native_w = 12;  // the 6x5 mask is stored at half the native resolution
    mask.native_h = 10;
    REQUIRE(decode_seg_mask(test_png_bytes(), mask));

    // Label 1 at mask pixel (1, 1) is native x 2..3.99, y 2..3.99; label 2 at (4, 3) is native
    // x 8..9.99, y 6..7.99.
    CHECK(mask_label_at(mask, Eigen::Vector2d(2.5, 2.5)) == 1);
    CHECK(mask_label_at(mask, Eigen::Vector2d(3.9, 3.9)) == 1);
    CHECK(mask_label_at(mask, Eigen::Vector2d(9.0, 7.0)) == 2);
    CHECK(mask_label_at(mask, Eigen::Vector2d(0.5, 0.5)) == 0);
    // Outside the native frame, and with no mask at all.
    CHECK(mask_label_at(mask, Eigen::Vector2d(-0.5, 2.5)) == -1);
    CHECK(mask_label_at(mask, Eigen::Vector2d(2.5, 10.0)) == -1);
    CHECK(mask_label_at(DecodedSegMask{}, Eigen::Vector2d(1, 1)) == -1);
}
