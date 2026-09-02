// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for predict_rigid_marker() (marker-based-mocap design doc's
 * dot-assignment-architecture-design.md, sub-phase C2.2). Hand-computed
 * expected position/covariance for a simple camera rig, per that design
 * doc's own §10 test spec -- catches a Jacobian sign error (the
 * right-vs-left perturbation convention is exactly the kind of thing
 * that's easy to get backwards) before it reaches integration testing.
 */
#include <posetrak/tracking/marker_prediction.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace posetrak;
using Catch::Approx;

namespace {

// Identity-orientation camera looking down +Z (world Z == depth), same
// pattern as test_multi_person_contact_gating.cpp's make_camera().
Camera make_camera(Eigen::Vector3d const& position) {
    Intrinsics intr;
    intr.fx = 1000.0;
    intr.fy = 1000.0;
    intr.cx = 640.0;
    intr.cy = 360.0;
    intr.width = 1280;
    intr.height = 720;
    intr.model = Intrinsics::DistortionModel::BrownConrady;
    intr.distortion_coeffs = {0, 0, 0, 0, 0};

    Extrinsics extr;
    extr.position = position;
    extr.orientation = Eigen::Quaterniond::Identity();
    return Camera(0, "cam0", intr, extr);
}

}  // namespace

TEST_CASE("predict_rigid_marker: identity pose, zero covariance -> exact pinhole projection",
          "[marker_prediction]") {
    // Camera at world origin looking down +Z; marker 2m in front of it,
    // offset (0.1, 0, 0) in the (identity-orientation) root frame.
    Camera camera = make_camera(Eigen::Vector3d(0.0, 0.0, 0.0));
    Eigen::Vector3d local_pos(0.1, 0.0, 0.0);
    Eigen::Vector3d root_position(0.0, 0.0, 2.0);
    Eigen::Quaterniond root_orientation = Eigen::Quaterniond::Identity();
    Eigen::Matrix<double, 6, 6> zero_cov = Eigen::Matrix<double, 6, 6>::Zero();

    auto result =
        predict_rigid_marker(local_pos, root_position, root_orientation, zero_cov, camera);
    REQUIRE(result.has_value());
    // p_world = (0.1, 0, 2.0); u = fx*x/z + cx = 1000*0.1/2 + 640 = 690
    REQUIRE(result->position.x() == Approx(690.0));
    // v = fy*y/z + cy = 1000*0/2 + 360 = 360
    REQUIRE(result->position.y() == Approx(360.0));
    // Zero input covariance -> zero output covariance, exactly.
    REQUIRE(result->covariance.norm() == Approx(0.0).margin(1e-12));
}

TEST_CASE("predict_rigid_marker: pure position uncertainty propagates as identity-scaled",
          "[marker_prediction]") {
    // With local_pos = 0 (marker AT the root), the Jacobian's rotation
    // block doesn't matter -- only the position block (always I3)
    // contributes, so this isolates and directly checks the
    // position -> pixel propagation in isolation.
    Camera camera = make_camera(Eigen::Vector3d(0.0, 0.0, 0.0));
    Eigen::Vector3d local_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d root_position(0.0, 0.0, 2.0);
    Eigen::Quaterniond root_orientation = Eigen::Quaterniond::Identity();

    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
    cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.01;  // 0.1m std per axis, isotropic

    auto result = predict_rigid_marker(local_pos, root_position, root_orientation, cov, camera);
    REQUIRE(result.has_value());
    // J_cam (2x3) at z=2: du/dx = fx/z = 500, dv/dy = fy/z = 500,
    // du/dz = -fx*x/z^2 = 0 (x=0 here), dv/dz = -fy*y/z^2 = 0 (y=0 here).
    // So Cov_pixel = J_cam * (0.01*I3) * J_cam^T = 0.01 * diag(500^2, 500^2)
    // (cross terms vanish since J_cam's x/y rows only touch x/y, and the
    // z-column coefficients are zero here).
    REQUIRE(result->covariance(0, 0) == Approx(0.01 * 500.0 * 500.0));
    REQUIRE(result->covariance(1, 1) == Approx(0.01 * 500.0 * 500.0));
    REQUIRE(result->covariance(0, 1) == Approx(0.0).margin(1e-9));
}

TEST_CASE("predict_rigid_marker: orientation uncertainty propagates through the lever arm",
          "[marker_prediction]") {
    // Marker offset from root along +X; rotating the root about +Y (axis-
    // angle index 1) moves this marker along +/-Z (out of/into the
    // camera), which should show up as depth-direction pixel sensitivity
    // (both u and v scale with 1/z, so any z-perturbing rotation affects
    // both) -- this exercises the -R*skew(local_pos) block that the
    // previous, local_pos=0 test couldn't touch at all.
    Camera camera = make_camera(Eigen::Vector3d(0.0, 0.0, 0.0));
    Eigen::Vector3d local_pos(1.0, 0.0, 0.0);  // 1m lever arm
    Eigen::Vector3d root_position(0.0, 0.0, 2.0);
    Eigen::Quaterniond root_orientation = Eigen::Quaterniond::Identity();

    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
    // Only the Y-axis (index 3+1=4) rotational uncertainty is set.
    cov(4, 4) = 0.01;  // (0.1 rad)^2

    auto result = predict_rigid_marker(local_pos, root_position, root_orientation, cov, camera);
    REQUIRE(result.has_value());
    // skew(local_pos) for local_pos=(1,0,0): rows/cols per the standard
    // skew convention used in this file: skew(v) = [[0,-vz,vy],[vz,0,-vx],[-vy,vx,0]].
    // -R*skew(local_pos) with R=I: -skew((1,0,0)) = [[0,0,0],[0,0,1],[0,-1,0]].
    // Column 1 (the Y-axis-angle column) of that is (0, 0, -1) -- i.e. a
    // positive Y-axis rotation perturbation moves this marker by
    // (0,0,-1)*delta_theta: purely in Z, no X/Y motion. That means the
    // resulting Cov_world's only nonzero entry is the ZZ one, and via
    // J_cam it should manifest as nonzero pixel variance from the
    // z-coefficient terms (-fx*x/z^2, -fy*y/z^2), which are nonzero here
    // since p_cam = (1,0,2) has x=1 != 0.
    REQUIRE(result->covariance(0, 0) > 0.0);
    // With y=0 at the marker's projected camera-frame position, the v
    // (row 1) column-2 (z) coefficient -fy*y/z^2 is exactly zero, so v's
    // variance should be (numerically) zero even though u's isn't --
    // confirms the propagation is doing real per-axis work, not just
    // uniformly inflating both.
    REQUIRE(result->covariance(1, 1) == Approx(0.0).margin(1e-9));
}

TEST_CASE("predict_rigid_marker: behind the camera returns nullopt", "[marker_prediction]") {
    Camera camera = make_camera(Eigen::Vector3d(0.0, 0.0, 0.0));
    Eigen::Vector3d local_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d root_position(0.0, 0.0, -2.0);  // behind the camera (negative Z)
    Eigen::Quaterniond root_orientation = Eigen::Quaterniond::Identity();
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();

    auto result = predict_rigid_marker(local_pos, root_position, root_orientation, cov, camera);
    REQUIRE_FALSE(result.has_value());
}
