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
