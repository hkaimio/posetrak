// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file update_result.hpp
 * @brief Result types for UKF update with outlier rejection diagnostics
 */

#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <vector>

namespace posetrak {

/**
 * @brief Result of outlier rejection for a single observation
 */
struct ObservationResult {
    std::string marker_name;      ///< Name of the marker
    int camera_id;                ///< Camera ID
    uint32_t camera_frame_idx;    ///< Camera frame index
    bool is_outlier;              ///< Whether observation was rejected as outlier
    double mahalanobis_distance;  ///< Computed Mahalanobis distance
    Eigen::Vector2d innovation;   ///< Innovation vector [u_err, v_err] in pixels
    Eigen::Vector2d predicted;    ///< Predicted measurement [u_pred, v_pred]
    Eigen::Vector2d actual;       ///< Actual measurement [u_actual, v_actual]
};

/**
 * @brief Result of UKF update step with diagnostics
 */
struct UpdateResult {
    int num_observations;                         ///< Total number of observations
    int num_outliers;                             ///< Number of rejected outliers
    int num_inliers;                              ///< Number of accepted inliers
    std::vector<ObservationResult> observations;  ///< Per-observation details
    double nis;                                   ///< Normalized Innovation Squared
    int nis_dof;                                  ///< Degrees of freedom for NIS

    // Per-operation wall times (milliseconds)
    double fk1_ms = 0.0;         ///< First FK parallel loop (all sigma points)
    double s_ms = 0.0;           ///< Z build + S DGEMM + E matrix
    double outlier_ms = 0.0;     ///< Outlier rejection
    double fk2_ms = 0.0;         ///< Inlier FK parallel loop (0 if no rejection)
    double inlier_ms = 0.0;      ///< Z_inlier + S_inlier + Pxy DGEMMs
    double kalman_ms = 0.0;      ///< Kalman gain (inverse + multiply)
    double cov_update_ms = 0.0;  ///< Covariance update (KSK^T + eigendecomp)

    UpdateResult() : num_observations(0), num_outliers(0), num_inliers(0), nis(0.0), nis_dof(0) {}
};

}  // namespace posetrak
