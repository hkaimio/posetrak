// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file rts_smoother.hpp
 * @brief Rauch-Tung-Striebel fixed-interval smoother for UKF pose estimates.
 *
 * Given the full forward-pass cache (posterior + prior + cross-covariance per
 * frame), runs a single backward sweep to compute smoothed estimates
 * x_{k|N}, P_{k|N} for every frame k, where N is the last frame.
 *
 * The backward correction exploits the well-known RTS equations:
 *
 *   G_k  = D_k  *  P_{k+1|k}^{-1}
 *   x_{k|N} = x_{k|k}  ⊕  G_k * ( x_{k+1|N} ⊖ x_{k+1|k} )
 *   P_{k|N} = P_{k|k}  +  G_k * ( P_{k+1|N} - P_{k+1|k} ) * G_k^T
 *
 * where ⊕ and ⊖ are manifold-aware retraction/log-map operations on the
 * pose manifold SO(3)^m × R^n.
 *
 * The cross-covariance D_k is computed from sigma points during the UKF
 * predict step (sigma-point formulation, no linearisation required).
 */

#pragma once

#include <Eigen/Core>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/state.hpp"
#include <memory>
#include <vector>

namespace posetrak {

/// @brief Per-frame data accumulated during the forward UKF pass.
///
/// Stored for every successfully tracked frame so the RTS backward sweep can
/// compute smoothed estimates without re-running the forward filter.
struct FrameSmootherData {
    double timestamp;  ///< Frame timestamp (seconds)

    State posterior_state;          ///< Filtered posterior  x_{k|k}
    Eigen::MatrixXd posterior_cov;  ///< Filtered posterior  P_{k|k}

    State prior_state;          ///< Predicted prior     x_{k|k-1}  (before update)
    Eigen::MatrixXd prior_cov;  ///< Predicted prior     P_{k|k-1}  (before update)

    /// D_{k-1}: sigma-point cross-covariance of the posterior x_{k-1|k-1}
    /// with the prior x_{k|k-1}.  Shape: error_dim × error_dim.
    /// Computed by UKF::predict() when it spans step k-1 → k.
    Eigen::MatrixXd cross_cov;
};

/// @brief Smoothed state estimate for a single frame.
struct SmoothedFrame {
    double timestamp;
    State state;
    Eigen::MatrixXd covariance;
};

/// @brief Rauch-Tung-Striebel fixed-interval smoother.
///
/// Stateless: all per-sequence state lives in the FrameSmootherData vector
/// passed to smooth().  Create once, call smooth() arbitrarily many times.
class RTSSmoother {
   public:
    explicit RTSSmoother(std::shared_ptr<const SkeletonLayout> layout);

    /// @brief Run the RTS backward pass.
    ///
    /// @param data  Forward-pass data in *chronological* order (frame 0 … N).
    ///              Requires at least one frame; single-frame input is returned
    ///              unchanged (smoother trivially equals the filter).
    /// @return      Smoothed frames in *chronological* order (frame 0 … N).
    std::vector<SmoothedFrame> smooth(std::vector<FrameSmootherData> const& data) const;

   private:
    /// Manifold-aware log-map: returns tangent-space error vector e such that
    ///   a = b ⊕ e   (in error-state coordinates used by the UKF).
    Eigen::VectorXd state_error(State const& a, State const& b) const;

    /// Manifold-aware retraction: returns new_state = nominal ⊕ error.
    State state_retract(State const& nominal, Eigen::VectorXd const& error) const;

    std::shared_ptr<const SkeletonLayout> layout_;
    int error_dim_;  ///< Cached from layout (error_state_dim())
};

}  // namespace posetrak
