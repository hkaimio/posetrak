/**
 * @file rts_smoother.cpp
 * @brief Rauch-Tung-Striebel fixed-interval smoother implementation.
 */

#include "posetrak/filters/rts_smoother.hpp"

#include <Eigen/Dense>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/state.hpp"
#include <stdexcept>

namespace posetrak {

// ─── Construction ────────────────────────────────────────────────────────────

RTSSmoother::RTSSmoother(std::shared_ptr<const SkeletonLayout> layout)
    : layout_(std::move(layout)), error_dim_(layout_->error_state_dim()) {}

// ─── Manifold arithmetic ──────────────────────────────────────────────────────
//
// These mirror UnscentedKalmanFilter::compute_state_error and
// SigmaPointGenerator::apply_error_to_state, but are self-contained so that
// the RTS smoother has no dependency on those classes.

Eigen::VectorXd RTSSmoother::state_error(State const& a, State const& b) const {
    int const root_n = layout_->root_error_dof_count();  // 6 or 0
    int const jac = layout_->joint_active_dof_count();
    int const active_dof = root_n + jac;

    Eigen::VectorXd err = Eigen::VectorXd::Zero(error_dim_);

    if (root_n > 0) {
        // Position error (Euclidean)
        err.segment<3>(0) = a.root_position() - b.root_position();

        // Orientation error (tangent space / axis-angle)
        Eigen::Quaterniond const q_diff = b.root_orientation().conjugate() * a.root_orientation();
        double const angle = 2.0 * std::atan2(q_diff.vec().norm(), q_diff.w());
        if (angle > 1e-8) {
            err.segment<3>(3) = angle * q_diff.vec().normalized();
        }

        // Velocity errors (Euclidean)
        err.segment<3>(active_dof) = a.root_velocity() - b.root_velocity();
        err.segment<3>(active_dof + 3) = a.root_angular_velocity() - b.root_angular_velocity();
    }

    for (JointDesc const& j : layout_->joints()) {
        int const si = j.state_index;
        int const pos_base = root_n + j.error_index;
        int const vel_base = active_dof + root_n + j.error_index;

        if (j.type == JointType::REVOLUTE || j.type == JointType::PRISMATIC) {
            err(pos_base) = a.joint_angles()(si) - b.joint_angles()(si);
            err(vel_base) = a.joint_velocities()(si) - b.joint_velocities()(si);

        } else if (j.type == JointType::SPHERICAL) {
            if (j.active_dof_count == 3) {
                Eigen::Vector3d const aa_b = b.joint_angles().segment<3>(si);
                Eigen::Vector3d const aa_a = a.joint_angles().segment<3>(si);
                Eigen::Matrix3d const R_b =
                    State::axis_angle_to_quaternion(aa_b).toRotationMatrix();
                Eigen::Matrix3d const R_a =
                    State::axis_angle_to_quaternion(aa_a).toRotationMatrix();
                Eigen::Matrix3d const R_rel = R_b.transpose() * R_a;
                Eigen::Quaterniond const q_rel(R_rel);
                err.segment<3>(pos_base) = State::quaternion_to_axis_angle(q_rel);
                err.segment<3>(vel_base) =
                    a.joint_velocities().segment<3>(si) - b.joint_velocities().segment<3>(si);
            } else {
                int partial = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    if (j.active_dof_mask[axis]) {
                        err(pos_base + partial) =
                            a.joint_angles()(si + axis) - b.joint_angles()(si + axis);
                        err(vel_base + partial) =
                            a.joint_velocities()(si + axis) - b.joint_velocities()(si + axis);
                        partial++;
                    }
                }
            }
        }
    }

    return err;
}

State RTSSmoother::state_retract(State const& nominal, Eigen::VectorXd const& error) const {
    int const root_n = layout_->root_error_dof_count();
    int const jac = layout_->joint_active_dof_count();
    int const active_dof = root_n + jac;

    State result = nominal;

    if (root_n > 0) {
        result.set_root_position(nominal.root_position() + error.segment<3>(0));

        Eigen::Quaterniond const q_err = State::axis_angle_to_quaternion(error.segment<3>(3));
        result.set_root_orientation((nominal.root_orientation() * q_err).normalized());

        result.set_root_velocity(nominal.root_velocity() + error.segment<3>(active_dof));
        result.set_root_angular_velocity(nominal.root_angular_velocity() +
                                         error.segment<3>(active_dof + 3));
    }

    Eigen::VectorXd new_angles = nominal.joint_angles();
    Eigen::VectorXd new_vels = nominal.joint_velocities();

    for (JointDesc const& j : layout_->joints()) {
        int const si = j.state_index;
        int const pos_base = root_n + j.error_index;
        int const vel_base = active_dof + root_n + j.error_index;

        if (j.type == JointType::REVOLUTE || j.type == JointType::PRISMATIC) {
            new_angles(si) += error(pos_base);
            new_vels(si) += error(vel_base);

        } else if (j.type == JointType::SPHERICAL) {
            if (j.active_dof_count == 3) {
                Eigen::Vector3d const aa_nom = nominal.joint_angles().segment<3>(si);
                Eigen::Matrix3d const R_nom =
                    State::axis_angle_to_quaternion(aa_nom).toRotationMatrix();
                Eigen::Matrix3d const R_err =
                    State::axis_angle_to_quaternion(error.segment<3>(pos_base)).toRotationMatrix();
                Eigen::Quaterniond const q_new(R_nom * R_err);
                new_angles.segment<3>(si) = State::quaternion_to_axis_angle(q_new);
                new_vels.segment<3>(si) += error.segment<3>(vel_base);
            } else {
                int partial = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    if (j.active_dof_mask[axis]) {
                        new_angles(si + axis) += error(pos_base + partial);
                        new_vels(si + axis) += error(vel_base + partial);
                        partial++;
                    }
                }
            }
        }
    }

    result.set_joint_angles(new_angles);
    result.set_joint_velocities(new_vels);
    return result;
}

// ─── Backward sweep ───────────────────────────────────────────────────────────

std::vector<SmoothedFrame> RTSSmoother::smooth(std::vector<FrameSmootherData> const& data) const {
    if (data.empty()) {
        throw std::invalid_argument("RTSSmoother::smooth(): data vector is empty");
    }

    int const N = static_cast<int>(data.size());

    // Pre-fill result with the filtered posteriors: this gives every entry a
    // valid State (no default constructor) and is already correct for the last
    // frame (and for any single-frame sequence).
    std::vector<SmoothedFrame> result;
    result.reserve(N);
    for (int k = 0; k < N; ++k) {
        result.push_back({data[k].timestamp, data[k].posterior_state, data[k].posterior_cov});
    }

    // Backward sweep: k = N-2 down to 0.
    // For step k, the RTS gain uses cross-cov and prior from frame k+1
    // (they describe the transition k → k+1).
    for (int k = N - 2; k >= 0; --k) {
        FrameSmootherData const& fwd_next = data[k + 1];  // transition k → k+1
        SmoothedFrame const& sm_next = result[k + 1];     // smoothed estimate at k+1

        // G_k = D_k * P_{k+1|k}^{-1}
        // Use LDLT (symmetric positive-definite) for stability.
        Eigen::MatrixXd const G =
            fwd_next.cross_cov *
            fwd_next.prior_cov.ldlt().solve(Eigen::MatrixXd::Identity(error_dim_, error_dim_));

        // Tangent-space difference: x_{k+1|N} ⊖ x_{k+1|k}
        Eigen::VectorXd const delta = state_error(sm_next.state, fwd_next.prior_state);

        // Smoothed state: x_{k|N} = x_{k|k} ⊕ G * delta
        result[k].timestamp = data[k].timestamp;
        result[k].state = state_retract(data[k].posterior_state, G * delta);

        // Smoothed covariance: P_{k|N} = P_{k|k} + G*(P_{k+1|N} - P_{k+1|k})*G^T
        Eigen::MatrixXd const cov_diff = sm_next.covariance - fwd_next.prior_cov;
        result[k].covariance = data[k].posterior_cov + G * cov_diff * G.transpose();
    }

    return result;
}

}  // namespace posetrak
