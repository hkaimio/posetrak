// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/filters/rts_smoother.hpp"
#include <memory>

using namespace posetrak;

namespace {

Skeleton make_arm_skeleton() {
    Skeleton skel;
    uint32_t root = skel.add_joint("root", std::nullopt, JointType::REVOLUTE,
                                   Eigen::Vector3d::Zero(), "", Eigen::Vector3d::Zero());
    skel.add_joint("forearm", root, JointType::REVOLUTE, Eigen::Vector3d(0.5, 0.0, 0.0), "",
                   Eigen::Vector3d::Zero());
    return skel;
}

}  // namespace

// With P_{k|k} = 0.1 I, P_{k+1|k} = 0.2 I and D = 0.05 I the gain is 0.25 I, and the smoothed
// covariance of frame k depends on that of frame k+1 through G (P_{k+1|N} - P_{k+1|k}) G^T:
// 0.1 for the last frame, then 0.1 + 0.0625 (0.1 - 0.2) = 0.09375, then
// 0.1 + 0.0625 (0.09375 - 0.2) = 0.0933594. The smoother returns the diagonals only.
TEST_CASE("RTSSmoother returns the diagonal of the recursively smoothed covariance",
          "[rts_smoother]") {
    auto layout =
        SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(make_arm_skeleton()));
    int const e = layout->error_state_dim();
    int const n = layout->total_storage_dof_count();
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), Eigen::VectorXd::Zero(n),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::VectorXd::Zero(n));

    std::vector<FrameSmootherData> data;
    for (int k = 0; k < 3; ++k) {
        data.push_back({k / 120.0, state, Eigen::MatrixXd::Identity(e, e) * 0.1, state,
                        Eigen::MatrixXd::Identity(e, e) * 0.2,
                        Eigen::MatrixXd::Identity(e, e) * 0.05, true});
    }

    auto smoothed = RTSSmoother(layout).smooth(data);

    REQUIRE(smoothed.size() == 3);
    REQUIRE(smoothed[2].covariance_diag.size() == e);
    CHECK(smoothed[2].covariance_diag(0) == Catch::Approx(0.1));
    CHECK(smoothed[1].covariance_diag(0) == Catch::Approx(0.09375));
    CHECK(smoothed[0].covariance_diag(0) == Catch::Approx(0.0933594).epsilon(1e-6));
}
