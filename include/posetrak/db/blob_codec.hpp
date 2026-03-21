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

}  // namespace posetrak::db
