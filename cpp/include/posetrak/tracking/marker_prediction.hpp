// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file marker_prediction.hpp
 * @brief Predicted 2D position + pixel covariance for one named marker slot
 * -- the seam dot assignment (marker-based-mocap design doc's
 * dot-assignment-architecture-design.md §6) consumes, decoupled from
 * whether the number came from a closed form or from sigma points.
 *
 * This file implements only the rigid-body closed form (§6.1) -- the
 * cheap, exact case for a root-only prop skeleton
 * (Skeleton::is_rigid_body()). The general/articulated implementation
 * (§6, deferred) would reuse UnscentedKalmanFilter::predict_measurements()'s
 * existing sigma-point machinery instead; nothing here assumes rigidity
 * beyond its own function name and doc comment, so a caller adds a second
 * implementation for the general case without touching this one.
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "posetrak/core/camera.hpp"
#include <optional>

namespace posetrak {

/// @brief Predicted pixel position and pixel-space covariance for one
/// marker, in one camera, at the tracker's current predict()-step state.
/// Assignment (design doc §7) only ever consumes this -- it has no idea
/// whether the number behind it came from a closed form or from sigma
/// points.
struct MarkerPrediction {
    Eigen::Vector2d position;
    Eigen::Matrix2d covariance;
};

/// @brief Closed-form MarkerPrediction for a marker on a *rigid-body*
/// skeleton (design doc §6.1).
///
/// Exact for a rigid body -- not an approximation layered on an
/// approximation: p_world(m) = R * local_pos + root_position is exact (no
/// articulation to run FK over), and the covariance propagation Jacobian
/// follows directly from State::apply_error_update()'s own
/// right-multiplicative perturbation convention
/// (root_orientation_new = root_orientation * Exp(delta_theta)):
///
///   p_world_perturbed(m) ~= p_world(m) + delta_t - R*skew(local_pos)*delta_theta
///
/// so the 3x6 Jacobian of p_world(m) w.r.t. the error state's
/// [position, axis-angle] block is J_m = [I3 | -R*skew(local_pos)].
/// Cross-checked independently against Tracker::marker_projection_std()'s
/// own (general, FK-based) Jacobian for the same quantity, parameterized
/// by world-frame marker offset r = p_marker - p_root instead of the
/// body-local local_pos used here: skew(r) = R*skew(local_pos)*R^T (a
/// standard rotation-of-skew identity, since r = R*local_pos), so
/// -skew(r)*R = -R*skew(local_pos)*R^T*R = -R*skew(local_pos) -- the two
/// formulations are algebraically identical.
///
/// @param local_pos          Marker::local_pos -- the marker's offset in
///                           the rigid body's own local frame (meters).
/// @param root_position      State::root_position() (world frame).
/// @param root_orientation   State::root_orientation() (local->world).
/// @param pose_cov_6x6       The 6x6 [position, axis-angle] block of the
///                           error-state covariance
///                           (prior_cov.block<6,6>(0,0) -- exact for a
///                           rigid body, which has no joint-angle DOFs to
///                           occupy the columns/rows in between).
/// @param camera             Camera to project into.
/// @param local_normal       Marker::normal (self-occlusion-culling design)
///                           -- when given, the marker is treated as not
///                           seen by *camera* this frame (std::nullopt,
///                           same as "behind the camera") if its current
///                           world-frame normal faces away from the
///                           camera. Recomputed from local_normal +
///                           root_orientation every call rather than once
///                           per track, since the object's orientation
///                           (and therefore which face points where)
///                           changes every frame. std::nullopt (the
///                           default) disables the check entirely --
///                           existing callers/skeletons with no known
///                           normal are unaffected.
/// @return MarkerPrediction if the marker projects in front of the
///         camera *and* (when local_normal is given) its own face is
///         turned toward the camera this frame, std::nullopt otherwise
///         (mirrors Camera::project_undistorted()'s own "behind camera"
///         convention -- the caller should treat this camera as simply
///         not seeing this marker slot this frame, not as an error).
std::optional<MarkerPrediction>
predict_rigid_marker(Eigen::Vector3d const& local_pos, Eigen::Vector3d const& root_position,
                     Eigen::Quaterniond const& root_orientation,
                     Eigen::Matrix<double, 6, 6> const& pose_cov_6x6, Camera const& camera,
                     std::optional<Eigen::Vector3d> const& local_normal = std::nullopt);

}  // namespace posetrak
