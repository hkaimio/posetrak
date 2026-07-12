#pragma once

#include <Eigen/Core>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace posetrak::db {

/// @brief Decode float64[9] row-major LE blob to Eigen::Matrix3d
inline Eigen::Matrix3d decode_rotation_matrix(void const* data, int byte_count) {
    constexpr int expected = 9 * sizeof(double);
    if (byte_count != expected) {
        throw std::runtime_error("decode_rotation_matrix: expected " + std::to_string(expected) +
                                 " bytes, got " + std::to_string(byte_count));
    }
    double buf[9];
    std::memcpy(buf, data, expected);
    // Row-major storage: buf[row*3 + col]
    Eigen::Matrix3d m;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            m(r, c) = buf[r * 3 + c];
    return m;
}

/// @brief Decode float64[3] LE blob to Eigen::Vector3d
inline Eigen::Vector3d decode_translation(void const* data, int byte_count) {
    constexpr int expected = 3 * sizeof(double);
    if (byte_count != expected) {
        throw std::runtime_error("decode_translation: expected " + std::to_string(expected) +
                                 " bytes, got " + std::to_string(byte_count));
    }
    double buf[3];
    std::memcpy(buf, data, expected);
    return Eigen::Vector3d(buf[0], buf[1], buf[2]);
}

/// @brief Decode float64[n] LE blob to std::vector<double>
inline std::vector<double> decode_float64_blob(void const* data, int byte_count) {
    if (byte_count % static_cast<int>(sizeof(double)) != 0) {
        throw std::runtime_error("decode_float64_blob: byte_count " + std::to_string(byte_count) +
                                 " is not a multiple of 8");
    }
    int n = byte_count / static_cast<int>(sizeof(double));
    std::vector<double> result(static_cast<size_t>(n));
    std::memcpy(result.data(), data, static_cast<size_t>(byte_count));
    return result;
}

/// @brief A single 2D pose keypoint (x, y, confidence) as stored in kp_blob
struct Keypoint {
    float x{};
    float y{};
    float confidence{};
};

/// @brief Decode float32[n_kp, 3] LE blob to vector<Keypoint>
inline std::vector<Keypoint> decode_keypoints(void const* data, int byte_count) {
    constexpr int kp_bytes = 3 * sizeof(float);
    if (byte_count % kp_bytes != 0) {
        throw std::runtime_error("decode_keypoints: byte_count " + std::to_string(byte_count) +
                                 " is not a multiple of 12 (3 floats per keypoint)");
    }
    int n = byte_count / kp_bytes;
    std::vector<Keypoint> result(static_cast<size_t>(n));
    auto const* src = static_cast<float const*>(data);
    for (int i = 0; i < n; ++i) {
        result[static_cast<size_t>(i)].x = src[i * 3 + 0];
        result[static_cast<size_t>(i)].y = src[i * 3 + 1];
        result[static_cast<size_t>(i)].confidence = src[i * 3 + 2];
    }
    return result;
}

/// @brief Test whether slot i is overridden by a pose_observation_edits kp_mask.
///
/// Same bit convention as apply_keypoint_edits() below (uint8[ceil(N/8)], bit i LSB-first).
/// Used to identify which keypoints in an already-merged frame came from a human edit, e.g.
/// to set Observation::force_inlier -- see TrackerConfig::edited_kp_noise_std.
inline bool is_kp_edited(void const* mask_data, int mask_bytes, int i) {
    int byte_idx = i / 8;
    if (mask_data == nullptr || byte_idx >= mask_bytes)
        return false;
    auto const* mask = static_cast<uint8_t const*>(mask_data);
    return (mask[byte_idx] >> (i % 8)) & 1u;
}

/// @brief Apply a pose_observation_edits overlay to a decoded keypoint list in-place.
///
/// edit_kp_data / edit_kp_bytes: float32[N,3] blob in the same format as decode_keypoints().
///   Column 0 = x, column 1 = y, column 2 = is_outlier (0 → inlier, non-zero → outlier).
/// mask_data / mask_bytes: uint8[ceil(N/8)] bitmask; bit i (LSB-first) means slot i is overridden.
///
/// For each overridden slot:
///   - is_outlier != 0 → set kps[i].confidence = 0 (tracker will reject it)
///   - is_outlier == 0 → replace x/y and set confidence = 1 (manually placed, trusted)
///
/// Silently does nothing if the table did not exist (mask_bytes == 0 or edit_kp_bytes == 0).
inline void apply_keypoint_edits(std::vector<Keypoint>& kps, void const* edit_kp_data,
                                 int edit_kp_bytes, void const* mask_data, int mask_bytes) {
    if (kps.empty() || edit_kp_bytes == 0 || mask_bytes == 0)
        return;

    auto edit_kps = decode_keypoints(edit_kp_data, edit_kp_bytes);
    int n = static_cast<int>(kps.size());
    if (static_cast<int>(edit_kps.size()) != n)
        throw std::runtime_error("apply_keypoint_edits: edit blob has " +
                                 std::to_string(edit_kps.size()) + " keypoints, expected " +
                                 std::to_string(n));

    int expected_mask = (n + 7) / 8;
    if (mask_bytes < expected_mask)
        throw std::runtime_error("apply_keypoint_edits: mask has " + std::to_string(mask_bytes) +
                                 " bytes, need at least " + std::to_string(expected_mask));

    auto const* mask = static_cast<uint8_t const*>(mask_data);
    for (int i = 0; i < n; ++i) {
        if ((mask[i / 8] >> (i % 8)) & 1u) {
            if (edit_kps[static_cast<size_t>(i)].confidence != 0.0f) {
                kps[static_cast<size_t>(i)].confidence = 0.0f;
            } else {
                kps[static_cast<size_t>(i)].x = edit_kps[static_cast<size_t>(i)].x;
                kps[static_cast<size_t>(i)].y = edit_kps[static_cast<size_t>(i)].y;
                kps[static_cast<size_t>(i)].confidence = 1.0f;
            }
        }
    }
}

}  // namespace posetrak::db
