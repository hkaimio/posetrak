// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <posetrak/db/blob_codec.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <vector>

using namespace posetrak::db;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<uint8_t> encode_kps(std::vector<std::array<float, 3>> const& vals) {
    std::vector<uint8_t> out(vals.size() * 3 * sizeof(float));
    for (size_t i = 0; i < vals.size(); ++i) {
        std::memcpy(out.data() + i * 12, vals[i].data(), 12);
    }
    return out;
}

static std::vector<uint8_t> make_mask(int n_kp, std::vector<int> const& set_bits) {
    int n_bytes = (n_kp + 7) / 8;
    std::vector<uint8_t> mask(static_cast<size_t>(n_bytes), 0);
    for (int bit : set_bits) {
        mask[static_cast<size_t>(bit / 8)] |= static_cast<uint8_t>(1u << (bit % 8));
    }
    return mask;
}

// ---------------------------------------------------------------------------
// apply_keypoint_edits
// ---------------------------------------------------------------------------

TEST_CASE("apply_keypoint_edits: no-op when kps empty", "[blob_codec]") {
    std::vector<Keypoint> kps;
    auto edit = encode_kps({});
    auto mask = make_mask(0, {});
    // Should not throw
    apply_keypoint_edits(kps, edit.data(), static_cast<int>(edit.size()), mask.data(),
                         static_cast<int>(mask.size()));
    REQUIRE(kps.empty());
}

TEST_CASE("apply_keypoint_edits: zero bytes means no change", "[blob_codec]") {
    std::vector<Keypoint> kps = {{1.0f, 2.0f, 0.9f}, {3.0f, 4.0f, 0.8f}};
    // edit_kp_bytes == 0 → function exits early
    apply_keypoint_edits(kps, nullptr, 0, nullptr, 0);
    REQUIRE(kps[0].x == Catch::Approx(1.0f));
    REQUIRE(kps[1].confidence == Catch::Approx(0.8f));
}

TEST_CASE("apply_keypoint_edits: unmask slots untouched", "[blob_codec]") {
    std::vector<Keypoint> kps = {{10.f, 20.f, 0.9f}, {30.f, 40.f, 0.8f}, {50.f, 60.f, 0.7f}};
    auto edit = encode_kps({{99.f, 99.f, 0.f}, {99.f, 99.f, 0.f}, {99.f, 99.f, 0.f}});
    auto mask = make_mask(3, {});  // no bits set
    apply_keypoint_edits(kps, edit.data(), static_cast<int>(edit.size()), mask.data(),
                         static_cast<int>(mask.size()));
    // Nothing should change
    REQUIRE(kps[0].x == Catch::Approx(10.f));
    REQUIRE(kps[1].x == Catch::Approx(30.f));
    REQUIRE(kps[2].confidence == Catch::Approx(0.7f));
}

TEST_CASE("apply_keypoint_edits: positional edit sets x/y and confidence=1", "[blob_codec]") {
    std::vector<Keypoint> kps = {{10.f, 20.f, 0.5f}, {30.f, 40.f, 0.6f}};
    // is_outlier=0 → move keypoint
    auto edit = encode_kps({{55.f, 66.f, 0.f}, {77.f, 88.f, 0.f}});
    auto mask = make_mask(2, {0});  // only slot 0
    apply_keypoint_edits(kps, edit.data(), static_cast<int>(edit.size()), mask.data(),
                         static_cast<int>(mask.size()));
    REQUIRE(kps[0].x == Catch::Approx(55.f));
    REQUIRE(kps[0].y == Catch::Approx(66.f));
    REQUIRE(kps[0].confidence == Catch::Approx(1.0f));
    // Slot 1 untouched
    REQUIRE(kps[1].x == Catch::Approx(30.f));
    REQUIRE(kps[1].confidence == Catch::Approx(0.6f));
}

TEST_CASE("apply_keypoint_edits: outlier edit zeroes confidence", "[blob_codec]") {
    std::vector<Keypoint> kps = {{10.f, 20.f, 0.9f}, {30.f, 40.f, 0.8f}};
    // is_outlier != 0 → confidence → 0
    auto edit = encode_kps({{0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}});
    auto mask = make_mask(2, {1});  // only slot 1
    apply_keypoint_edits(kps, edit.data(), static_cast<int>(edit.size()), mask.data(),
                         static_cast<int>(mask.size()));
    // Slot 0 untouched
    REQUIRE(kps[0].x == Catch::Approx(10.f));
    REQUIRE(kps[0].confidence == Catch::Approx(0.9f));
    // Slot 1 outlier: confidence zeroed, x/y unchanged
    REQUIRE(kps[1].confidence == Catch::Approx(0.f));
    REQUIRE(kps[1].x == Catch::Approx(30.f));
}

TEST_CASE("apply_keypoint_edits: both slots in 8-keypoint span", "[blob_codec]") {
    // Verify bitmask indexing across a byte boundary (slots 0-7 in one byte)
    int n = 8;
    std::vector<Keypoint> kps(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        kps[static_cast<size_t>(i)] = {static_cast<float>(i), 0.f, 0.5f};

    std::vector<std::array<float, 3>> edit_vals(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        edit_vals[static_cast<size_t>(i)] = {100.f + i, 200.f + i, 0.f};
    auto edit = encode_kps(edit_vals);

    // Override slots 0, 3, 7
    auto mask = make_mask(n, {0, 3, 7});
    apply_keypoint_edits(kps, edit.data(), static_cast<int>(edit.size()), mask.data(),
                         static_cast<int>(mask.size()));

    REQUIRE(kps[0].x == Catch::Approx(100.f));
    REQUIRE(kps[0].confidence == Catch::Approx(1.f));
    REQUIRE(kps[1].x == Catch::Approx(1.f));  // untouched
    REQUIRE(kps[3].x == Catch::Approx(103.f));
    REQUIRE(kps[3].confidence == Catch::Approx(1.f));
    REQUIRE(kps[7].x == Catch::Approx(107.f));
    REQUIRE(kps[4].x == Catch::Approx(4.f));  // untouched
}

TEST_CASE("apply_keypoint_edits: size mismatch throws", "[blob_codec]") {
    std::vector<Keypoint> kps(3);
    // edit has 2 keypoints but kps has 3
    auto edit = encode_kps({{1.f, 2.f, 0.f}, {3.f, 4.f, 0.f}});
    auto mask = make_mask(3, {0});
    REQUIRE_THROWS_AS(apply_keypoint_edits(kps, edit.data(), static_cast<int>(edit.size()),
                                           mask.data(), static_cast<int>(mask.size())),
                      std::runtime_error);
}

// ---------------------------------------------------------------------------
// decode_dot_candidates -- variable-N reflective-dot candidate blob decode,
// the decode_keypoints() sibling for anonymous dot detections (see
// docs/roadmap/features/marker-based-mocap/dot-assignment-architecture-design.md)
// ---------------------------------------------------------------------------

// Count-prefixed format (2026-09-04): int32 candidate count, then
// float32[count, 6] (px, py, area, compactness, major_axis, minor_axis) --
// see db_cache.py's encode_dot_candidates() for the full versioning
// rationale (a byte-length-only scheme is ambiguous once the
// per-candidate width can change between format versions).
static std::vector<uint8_t> encode_dot_candidates(std::vector<std::array<float, 6>> const& vals) {
    auto const n = static_cast<int32_t>(vals.size());
    std::vector<uint8_t> out(sizeof(int32_t) + vals.size() * 6 * sizeof(float));
    std::memcpy(out.data(), &n, sizeof(int32_t));
    for (size_t i = 0; i < vals.size(); ++i) {
        std::memcpy(out.data() + sizeof(int32_t) + i * 6 * sizeof(float), vals[i].data(),
                    6 * sizeof(float));
    }
    return out;
}

TEST_CASE("decode_dot_candidates: round-trips N=0 (empty blob)", "[blob_codec]") {
    auto blob = encode_dot_candidates({});
    auto result = decode_dot_candidates(blob.data(), static_cast<int>(blob.size()));
    REQUIRE(result.empty());
}

TEST_CASE("decode_dot_candidates: round-trips a handful of candidates", "[blob_codec]") {
    auto blob = encode_dot_candidates({
        {10.5f, 20.5f, 32.0f, 0.87f, 6.4f, 6.4f},
        {100.0f, 200.0f, 45.5f, 0.91f, 7.6f, 7.6f},
        {5.25f, 6.75f, 8.0f, 0.62f, 25.0f, 5.0f},  // an elongated (streak) candidate
    });
    auto result = decode_dot_candidates(blob.data(), static_cast<int>(blob.size()));
    REQUIRE(result.size() == 3);
    REQUIRE(result[0].px == Catch::Approx(10.5f));
    REQUIRE(result[0].py == Catch::Approx(20.5f));
    REQUIRE(result[0].area == Catch::Approx(32.0f));
    REQUIRE(result[0].compactness == Catch::Approx(0.87f));
    REQUIRE(result[0].major_axis == Catch::Approx(6.4f));
    REQUIRE(result[0].minor_axis == Catch::Approx(6.4f));
    REQUIRE(result[2].px == Catch::Approx(5.25f));
    REQUIRE(result[2].compactness == Catch::Approx(0.62f));
    REQUIRE(result[2].major_axis == Catch::Approx(25.0f));
    REQUIRE(result[2].minor_axis == Catch::Approx(5.0f));
}

TEST_CASE("decode_dot_candidates: round-trips several-tens scale", "[blob_codec]") {
    // The design doc's own scaling target (sec 7): "several tens per
    // scene", not the single-digit-to-a-dozen count the first draft
    // assumed.
    constexpr int n = 47;
    std::vector<std::array<float, 6>> vals(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        vals[static_cast<size_t>(i)] = {
            static_cast<float>(i), static_cast<float>(i) * 2.0f, 30.0f, 0.8f, 6.0f, 6.0f};
    }
    auto blob = encode_dot_candidates(vals);
    auto result = decode_dot_candidates(blob.data(), static_cast<int>(blob.size()));
    REQUIRE(result.size() == static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        REQUIRE(result[static_cast<size_t>(i)].px == Catch::Approx(static_cast<float>(i)));
        REQUIRE(result[static_cast<size_t>(i)].py == Catch::Approx(static_cast<float>(i) * 2.0f));
    }
}

TEST_CASE("decode_dot_candidates: byte count not matching the declared header count throws",
          "[blob_codec]") {
    auto blob = encode_dot_candidates({{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}});
    blob.push_back(0);  // one stray byte -- header still says 1 candidate, length now wrong
    REQUIRE_THROWS_AS(decode_dot_candidates(blob.data(), static_cast<int>(blob.size())),
                      std::runtime_error);
}

TEST_CASE("decode_dot_candidates: blob shorter than the count prefix throws", "[blob_codec]") {
    std::vector<uint8_t> bad_blob(2, 0);  // shorter than the int32 header itself
    REQUIRE_THROWS_AS(decode_dot_candidates(bad_blob.data(), static_cast<int>(bad_blob.size())),
                      std::runtime_error);
}

TEST_CASE(
    "decode_dot_candidates: a pre-2026-09-04 float32[N,4] blob is rejected, not silently "
    "misdecoded",
    "[blob_codec]") {
    // The exact real-world risk encode_dot_candidates()'s docstring calls
    // out: an old-format blob whose byte length happens to also satisfy
    // "count-prefix says N, matches N*24+4 bytes" would decode silently
    // wrong under a byte-length-only scheme. This is the old N=3 case
    // (48 bytes) -- picked because 48 is also 4 + 1*24 + 20, so a naive
    // "is this a multiple of the new stride" check alone could be fooled;
    // the count prefix must make this fail cleanly regardless.
    std::vector<float> old_format(3 * 4, 1.0f);  // old float32[3,4], no header
    auto const* bytes = reinterpret_cast<uint8_t const*>(old_format.data());
    int const byte_count = static_cast<int>(old_format.size() * sizeof(float));
    REQUIRE_THROWS_AS(decode_dot_candidates(bytes, byte_count), std::runtime_error);
}
