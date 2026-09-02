// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/marker_prediction.hpp"

namespace posetrak {

std::optional<MarkerPrediction>
predict_rigid_marker(Eigen::Vector3d const& local_pos, Eigen::Vector3d const& root_position,
                     Eigen::Quaterniond const& root_orientation,
                     Eigen::Matrix<double, 6, 6> const& pose_cov_6x6, Camera const& camera) {
    Eigen::Matrix3d const R = root_orientation.toRotationMatrix();
    Eigen::Vector3d const p_world = R * local_pos + root_position;

    // --- Predicted pixel position ---
    // clip_to_bounds=false: a marker predicted just outside the frame is
    // still a real prediction for assignment purposes (a candidate
    // detected right at the frame edge should still be able to match
    // it) -- clipping to nullopt here would only be correct for "behind
    // the camera", which project_undistorted() already reports via
    // nullopt regardless of this flag.
    auto proj = camera.project_undistorted(p_world, /*clip_to_bounds=*/false);
    if (!proj.has_value()) {
        return std::nullopt;  // behind the camera
    }

    // --- Covariance propagation ---
    // J_m = [I3 | -R*skew(local_pos)] (3x6): the Jacobian of p_world(m)
    // w.r.t. the error state's [position, axis-angle] block -- see this
    // function's own doc comment for the derivation.
    Eigen::Matrix3d skew_local;
    skew_local << 0.0, -local_pos.z(), local_pos.y(), local_pos.z(), 0.0, -local_pos.x(),
        -local_pos.y(), local_pos.x(), 0.0;

    Eigen::Matrix<double, 3, 6> J_m;
    J_m.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    J_m.block<3, 3>(0, 3) = -R * skew_local;

    Eigen::Matrix3d const cov_world = J_m * pose_cov_6x6 * J_m.transpose();

    // --- Camera projection Jacobian (2x3, pinhole, undistorted) ---
    // Same formula Tracker::marker_projection_std() already uses for the
    // general (FK-based) case -- see this file's header comment for the
    // cross-check confirming both parameterizations agree.
    Eigen::Vector3d const p_cam = camera.orientation() * (p_world - camera.position());
    // p_cam.z() > 0 is already guaranteed by the successful project_undistorted()
    // call above (it returns nullopt precisely when behind the camera).
    double const z = p_cam.z();
    Intrinsics const& intr = camera.intrinsics();
    Eigen::Matrix<double, 2, 3> J_proj_cam;
    J_proj_cam << intr.fx / z, 0.0, -intr.fx * p_cam.x() / (z * z), 0.0, intr.fy / z,
        -intr.fy * p_cam.y() / (z * z);
    Eigen::Matrix<double, 2, 3> const J_cam = J_proj_cam * camera.orientation().toRotationMatrix();

    Eigen::Matrix2d const cov_pixel = J_cam * cov_world * J_cam.transpose();

    return MarkerPrediction{*proj, cov_pixel};
}

}  // namespace posetrak
