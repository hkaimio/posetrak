// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file inverse_kinematics.cpp
 * @brief Implementation of damped least squares IK solver
 */

#include "posetrak/kinematics/inverse_kinematics.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <fmt/core.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

namespace posetrak {

InverseKinematics::InverseKinematics(
    pinocchio::Model const& model, pinocchio::Data& data, ForwardKinematics const& fk,
    std::map<std::string, pinocchio::FrameIndex> const& marker_frame_map)
    : model_(model), data_(data), fk_(fk), marker_frame_map_(marker_frame_map) {}

IKResult InverseKinematics::solve(std::map<std::string, Eigen::Vector3d> const& target_markers,
                                  Skeleton const& skeleton,
                                  std::optional<State> const& initial_guess, int max_iterations,
                                  double tolerance, double damping) {
    // Extract marker names (maintain consistent ordering) — needed before init block
    // because the multi-start orientation search uses them.
    std::vector<std::string> marker_names;
    for (auto const& [name, pos] : target_markers) {
        if (marker_frame_map_.count(name) > 0) {
            marker_names.push_back(name);
        }
    }
    if (marker_names.empty()) {
        return IKResult::failure();
    }

    // Convert initial guess to configuration vector
    Eigen::VectorXd q;
    if (!initial_guess.has_value() || initial_guess->joint_angles().size() == 0) {
        q = Eigen::VectorXd::Zero(model_.nq);
        if (model_.nq >= 7) {
            q[6] = 1.0;  // root quaternion w = 1 (identity)
        }

        // --- Helper: find a marker in any map ---
        auto find_marker = [](auto const& markers,
                              std::string const& name) -> std::optional<Eigen::Vector3d> {
            auto it = markers.find(name);
            if (it == markers.end())
                return std::nullopt;
            return it->second;
        };

        // --- Root position: hip midpoint if available, else all-marker centroid ---
        {
            auto hip_L_obs = find_marker(target_markers, "MRK-hip.L");
            auto hip_R_obs = find_marker(target_markers, "MRK-hip.R");
            if (hip_L_obs && hip_R_obs) {
                q.head<3>() = (*hip_L_obs + *hip_R_obs) / 2.0;
            } else {
                Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
                int count = 0;
                for (auto const& [name, pos] : target_markers) {
                    centroid += pos;
                    ++count;
                }
                if (count > 0) {
                    q.head<3>() = centroid / count;
                }
            }
        }

        // --- Root orientation: body-frame Procrustes alignment ---
        // Compute rest-pose marker positions at q=0 with root at origin.
        // Then find rotation R such that R * rest_frame ≈ observed_frame.
        if (model_.nq >= 7) {
            Eigen::VectorXd q_rest = Eigen::VectorXd::Zero(model_.nq);
            q_rest[6] = 1.0;
            auto rest_markers = fk_.compute(q_rest);

            auto hip_L_rest = find_marker(rest_markers, "MRK-hip.L");
            auto hip_R_rest = find_marker(rest_markers, "MRK-hip.R");
            auto sho_L_rest = find_marker(rest_markers, "MRK-shoulder.L");
            auto sho_R_rest = find_marker(rest_markers, "MRK-shoulder.R");
            auto hip_L_obs = find_marker(target_markers, "MRK-hip.L");
            auto hip_R_obs = find_marker(target_markers, "MRK-hip.R");
            auto sho_L_obs = find_marker(target_markers, "MRK-shoulder.L");
            auto sho_R_obs = find_marker(target_markers, "MRK-shoulder.R");

            // Build orthonormal frame from two vectors (up, lateral).
            // Returns matrix whose columns are [lateral, up, forward].
            auto make_frame = [](Eigen::Vector3d up, Eigen::Vector3d lateral) -> Eigen::Matrix3d {
                up = up.normalized();
                // Re-orthogonalize lateral against up
                lateral = (lateral - lateral.dot(up) * up).normalized();
                Eigen::Vector3d forward = lateral.cross(up).normalized();
                Eigen::Matrix3d R;
                R.col(0) = lateral;
                R.col(1) = up;
                R.col(2) = forward;
                return R;
            };

            bool aligned = false;
            if (hip_L_rest && hip_R_rest && hip_L_obs && hip_R_obs) {
                Eigen::Vector3d hip_center_rest = (*hip_L_rest + *hip_R_rest) / 2.0;
                Eigen::Vector3d hip_center_obs = (*hip_L_obs + *hip_R_obs) / 2.0;
                Eigen::Vector3d lateral_rest = *hip_L_rest - *hip_R_rest;
                Eigen::Vector3d lateral_obs = *hip_L_obs - *hip_R_obs;

                Eigen::Vector3d spine_rest, spine_obs;
                if (sho_L_rest && sho_R_rest && sho_L_obs && sho_R_obs) {
                    Eigen::Vector3d sho_center_rest = (*sho_L_rest + *sho_R_rest) / 2.0;
                    Eigen::Vector3d sho_center_obs = (*sho_L_obs + *sho_R_obs) / 2.0;
                    spine_rest = sho_center_rest - hip_center_rest;
                    spine_obs = sho_center_obs - hip_center_obs;
                } else {
                    // Fallback: use world Y as up
                    spine_rest = Eigen::Vector3d::UnitY();
                    spine_obs = Eigen::Vector3d::UnitY();
                }

                if (lateral_rest.norm() > 0.01 && lateral_obs.norm() > 0.01 &&
                    spine_rest.norm() > 0.05 && spine_obs.norm() > 0.05) {
                    Eigen::Matrix3d F_rest = make_frame(spine_rest, lateral_rest);
                    Eigen::Matrix3d F_obs = make_frame(spine_obs, lateral_obs);

                    // R_root maps rest body-frame to observed body-frame:
                    //   F_obs = R_root * F_rest  =>  R_root = F_obs * F_rest^T
                    Eigen::Matrix3d R_root = F_obs * F_rest.transpose();
                    // Project onto SO(3) via SVD to correct numerical drift
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_root, Eigen::ComputeFullU |
                                                                      Eigen::ComputeFullV);
                    R_root = svd.matrixU() *
                             Eigen::Vector3d(
                                 1, 1, svd.matrixU().determinant() * svd.matrixV().determinant())
                                 .asDiagonal() *
                             svd.matrixV().transpose();

                    Eigen::Quaterniond q_root(R_root);
                    q_root.normalize();
                    q[3] = q_root.x();
                    q[4] = q_root.y();
                    q[5] = q_root.z();
                    q[6] = q_root.w();
                    aligned = true;
                    fmt::print(
                        "  IK body-frame alignment: spine=[{:.2f},{:.2f},{:.2f}] "
                        "lateral=[{:.2f},{:.2f},{:.2f}]\n",
                        spine_obs.normalized().x(), spine_obs.normalized().y(),
                        spine_obs.normalized().z(), lateral_obs.normalized().x(),
                        lateral_obs.normalized().y(), lateral_obs.normalized().z());
                }
            }
            if (!aligned) {
                fmt::print(
                    "  IK body-frame alignment skipped (hip/shoulder markers not found), "
                    "using identity orientation\n");
            }
        }
    } else {
        // IK always works with full-skeleton states (returns full initialization)
        auto skeleton_ptr = std::make_shared<Skeleton const>(skeleton);
        auto layout = SkeletonLayout::from_full_skeleton(skeleton_ptr);
        q = ForwardKinematics::state_to_config(*initial_guess, *layout);
    }

    // --- Diagnostic helper: print per-marker position errors at a given configuration ---
    auto print_marker_errors = [&](Eigen::VectorXd const& q_diag, char const* label) {
        auto fk_positions = fk_.compute(q_diag);
        Eigen::Vector3d root_pos = q_diag.head<3>();
        Eigen::Quaterniond root_q(q_diag[6], q_diag[3], q_diag[4], q_diag[5]);
        fmt::print("  IK {} | root=({:.3f},{:.3f},{:.3f})  q_rot=({:.3f},{:.3f},{:.3f},{:.3f})\n",
                   label, root_pos.x(), root_pos.y(), root_pos.z(), root_q.x(), root_q.y(),
                   root_q.z(), root_q.w());

        double sum_sq = 0.0;
        int n = 0;
        std::vector<std::pair<double, std::string>> per_marker;
        for (auto const& [name, target] : target_markers) {
            if (marker_frame_map_.count(name) == 0)
                continue;
            auto it = fk_positions.find(name);
            if (it == fk_positions.end())
                continue;
            double err = (target - it->second).norm();
            sum_sq += err * err;
            ++n;
            per_marker.emplace_back(err, name);
        }
        // Sort by error descending so the worst offenders appear first.
        std::sort(per_marker.begin(), per_marker.end(),
                  [](auto const& a, auto const& b) { return a.first > b.first; });
        for (auto const& [err, name] : per_marker) {
            auto it = fk_positions.find(name);
            auto tgt = target_markers.at(name);
            fmt::print(
                "    {:30s}  err={:.4f}m  target=({:.3f},{:.3f},{:.3f})  "
                "fk=({:.3f},{:.3f},{:.3f})\n",
                name, err, tgt.x(), tgt.y(), tgt.z(), it->second.x(), it->second.y(),
                it->second.z());
        }
        double rms = n > 0 ? std::sqrt(sum_sq / n) : 0.0;
        fmt::print("  IK {} | {:d} markers  RMS={:.4f}m  worst={:.4f}m ({})\n", label, n, rms,
                   per_marker.empty() ? 0.0 : per_marker.front().first,
                   per_marker.empty() ? "" : per_marker.front().second);
    };

    print_marker_errors(q, "INIT ");

    // Open CSV file for iteration tracking
    std::ofstream csv_file("/tmp/ik_iterations.csv");
    csv_file << "iteration,rms_error,damping\n";

    // Levenberg-Marquardt with backtracking:
    //   - compute step dv in tangent space (nv)
    //   - use pinocchio::integrate to retract onto the manifold
    //   - accept step only if error decreases; otherwise increase damping and retry
    double current_damping = damping;
    Eigen::VectorXd error = compute_error(q, target_markers);
    double rms_error = error.norm() / std::sqrt(static_cast<double>(marker_names.size()));

    int iter = 0;
    int accepted_steps = 0;
    int rejected_steps = 0;
    for (; iter < max_iterations; ++iter) {
        csv_file << iter << "," << rms_error << "," << current_damping << "\n";

        // Check convergence
        if (rms_error < tolerance) {
            break;
        }

        // Compute Jacobian at current q
        Eigen::MatrixXd J = compute_jacobian(q, marker_names);

        // Freeze prismatic (scale) DOFs: zero their Jacobian columns so the DLS
        // step produces dv=0 for those indices and pinocchio::integrate leaves their
        // q values unchanged.  Scale factors are calibrated by the UKF, not by IK.
        for (auto const& joint : skeleton.joints()) {
            if (joint.type != JointType::PRISMATIC)
                continue;
            // Find the matching Pinocchio joint by name
            for (pinocchio::JointIndex ji = 1;
                 ji < static_cast<pinocchio::JointIndex>(model_.njoints); ++ji) {
                if (model_.names[ji] == joint.name) {
                    int const col = model_.joints[ji].idx_v();
                    int const ncols = model_.joints[ji].nv();
                    J.middleCols(col, ncols).setZero();
                    break;
                }
            }
        }

        // Levenberg-Marquardt step: dv = J^T (J J^T + λI)^{-1} * error
        // This is the "right-hand" DLS formulation (numerically better when n_obs > nv).
        Eigen::MatrixXd JJT = J * J.transpose();
        JJT.diagonal().array() += current_damping;
        Eigen::VectorXd dv = J.transpose() * JJT.llt().solve(error);

        // Retract onto the configuration manifold using pinocchio's Lie-group integrate.
        // This correctly handles free-flyer and spherical joint quaternions.
        Eigen::VectorXd q_new(model_.nq);
        pinocchio::integrate(model_, q, dv, q_new);
        enforce_joint_limits(q_new, skeleton);

        // Evaluate candidate
        Eigen::VectorXd error_new = compute_error(q_new, target_markers);
        double rms_new = error_new.norm() / std::sqrt(static_cast<double>(marker_names.size()));

        if (rms_new < rms_error) {
            // Accept step, reduce damping (more Newton-like next iteration)
            q = q_new;
            error = error_new;
            rms_error = rms_new;
            current_damping = std::max(current_damping * 0.5, 1e-7);
            ++accepted_steps;
        } else {
            // Reject step, increase damping (more gradient-descent-like)
            current_damping *= 4.0;
            ++rejected_steps;
            if (current_damping > 1e8) {
                // Truly stuck even with near-gradient-descent step sizes — give up
                break;
            }
        }
    }

    csv_file.close();

    bool converged = rms_error < tolerance;
    fmt::print("  IK {}: {} iters  accepted={}  rejected={}  final_RMS={:.4f}m  tol={:.4f}m\n",
               converged ? "CONVERGED" : "NOT CONVERGED", iter, accepted_steps, rejected_steps,
               rms_error, tolerance);
    enforce_joint_limits(q, skeleton);
    print_marker_errors(q, "FINAL");

    State final_state = config_to_state(q, skeleton);
    return IKResult{final_state, rms_error, iter, converged};
}

Eigen::VectorXd
InverseKinematics::compute_error(Eigen::VectorXd const& q,
                                 std::map<std::string, Eigen::Vector3d> const& target_markers) {
    // Compute FK
    auto current_markers = fk_.compute(q);

    // Build error vector (3 * num_markers)
    std::vector<double> errors;
    errors.reserve(target_markers.size() * 3);

    for (auto const& [name, target_pos] : target_markers) {
        if (marker_frame_map_.count(name) == 0) {
            continue;  // Skip markers not in model
        }

        auto it = current_markers.find(name);
        if (it == current_markers.end()) {
            continue;  // Skip if FK didn't compute this marker
        }

        Eigen::Vector3d const& current_pos = it->second;
        Eigen::Vector3d error = target_pos - current_pos;

        errors.push_back(error.x());
        errors.push_back(error.y());
        errors.push_back(error.z());
    }

    return Eigen::Map<Eigen::VectorXd>(errors.data(), errors.size());
}

Eigen::MatrixXd InverseKinematics::compute_jacobian(Eigen::VectorXd const& q,
                                                    std::vector<std::string> const& marker_names) {
    // computeJointJacobians runs FK and fills data_.J (the joint Jacobian matrix).
    // updateFramePlacements then fills data_.oMf (frame placements in world frame).
    // computeFrameJacobian then reads data_.J to build per-frame Jacobians.
    // Calling forwardKinematics alone does NOT fill data_.J, so without this call
    // every frame Jacobian row is zero.
    pinocchio::computeJointJacobians(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);

    // Allocate stacked Jacobian (3 * num_markers × nv)
    int num_markers = marker_names.size();
    Eigen::MatrixXd J_stacked = Eigen::MatrixXd::Zero(3 * num_markers, model_.nv);

    // Compute Jacobian for each marker
    for (size_t i = 0; i < marker_names.size(); ++i) {
        std::string const& name = marker_names[i];
        auto it = marker_frame_map_.find(name);
        if (it == marker_frame_map_.end()) {
            continue;
        }

        pinocchio::FrameIndex frame_id = it->second;

        // Compute 6D Jacobian in LOCAL_WORLD_ALIGNED frame (linear part = linear velocity
        // of the frame origin expressed in world orientation).
        Eigen::Matrix<double, 6, Eigen::Dynamic> J_frame(6, model_.nv);
        J_frame.setZero();

        // Use the 5-argument (no-q) form so pinocchio reads the already-computed data_.J.
        pinocchio::getFrameJacobian(model_, data_, frame_id, pinocchio::LOCAL_WORLD_ALIGNED,
                                    J_frame);

        // Pinocchio 6D spatial jacobian: [linear; angular] → take top 3 rows (linear velocity).
        J_stacked.block(3 * i, 0, 3, model_.nv) = J_frame.topRows(3);
    }

    return J_stacked;
}

void InverseKinematics::enforce_joint_limits(Eigen::VectorXd& q, Skeleton const& skeleton) {
    // Root position - no limits for now
    // Root quaternion - should be normalized but skip for now

    // Joint angles
    if (model_.nq <= 7) {
        return;  // Only root, no joints to limit
    }

    // Start after root (7 DOF)
    int q_idx = 7;

    for (auto const& joint : skeleton.joints()) {
        if (joint.parent_index == std::nullopt) {
            continue;  // Root joint
        }

        if (joint.type == JointType::REVOLUTE || joint.type == JointType::PRISMATIC) {
            // Single DOF - apply limits (only revolute has limits in CP1)
            if (q_idx < model_.nq && joint.type == JointType::REVOLUTE && joint.num_limits > 0) {
                double min_limit = joint.limits[0].x();  // limits is array of Vector2d
                double max_limit = joint.limits[0].y();

                q[q_idx] = std::clamp(q[q_idx], min_limit, max_limit);
                q_idx++;
            } else if (q_idx < model_.nq) {
                // No limits specified
                q_idx++;
            }
        } else if (joint.type == JointType::SPHERICAL) {
            // Clamp per-axis limits on the axis-angle representation, then convert back.
            // Uses the same quaternion ↔ axis-angle convention as config_to_state().
            if (q_idx + 3 < model_.nq && joint.num_limits > 0) {
                Eigen::Quaterniond quat(q[q_idx + 3], q[q_idx], q[q_idx + 1], q[q_idx + 2]);
                quat.normalize();
                Eigen::Vector3d xyz(quat.x(), quat.y(), quat.z());
                double sin_half = xyz.norm();
                double ang = 2.0 * std::atan2(sin_half, quat.w());
                Eigen::Vector3d aa = (sin_half > 1e-8) ? Eigen::Vector3d(xyz / sin_half * ang)
                                                       : Eigen::Vector3d::Zero();

                bool changed = false;
                for (int i = 0; i < 3 && i < static_cast<int>(joint.num_limits); ++i) {
                    double v = std::clamp(aa[i], joint.limits[i].x(), joint.limits[i].y());
                    if (v != aa[i]) {
                        aa[i] = v;
                        changed = true;
                    }
                }
                if (changed) {
                    double new_ang = aa.norm();
                    Eigen::Quaterniond nq =
                        (new_ang > 1e-10)
                            ? Eigen::Quaterniond(Eigen::AngleAxisd(new_ang, aa / new_ang))
                            : Eigen::Quaterniond::Identity();
                    q[q_idx] = nq.x();
                    q[q_idx + 1] = nq.y();
                    q[q_idx + 2] = nq.z();
                    q[q_idx + 3] = nq.w();
                }
            }
            q_idx += 4;
        }
    }
}

State InverseKinematics::config_to_state(Eigen::VectorXd const& q, Skeleton const& skeleton) {
    // Extract root position and orientation
    Eigen::Vector3d root_position = q.head<3>();
    Eigen::Quaterniond root_orientation(q[6], q[3], q[4], q[5]);  // [w, x, y, z]
    root_orientation.normalize();

    // Count joint DOFs (storage DOFs).
    // Scale-group followers share the leader's state slot → do not count them.
    int joint_dof = 0;
    for (auto const& joint : skeleton.joints()) {
        if (joint.parent_index == std::nullopt)
            continue;  // Skip root
        if (joint.is_scale_follower)
            continue;  // shares leader's state slot
        if (joint.type == JointType::REVOLUTE || joint.type == JointType::PRISMATIC) {
            joint_dof += 1;
        } else if (joint.type == JointType::SPHERICAL) {
            joint_dof += 3;  // Euler angles storage
        }
    }

    // Extract joint angles
    Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(joint_dof);

    if (model_.nq > 7 && joint_dof > 0) {
        int q_idx = 7;      // Start after root in config space
        int angle_idx = 0;  // Index in joint_angles

        for (auto const& joint : skeleton.joints()) {
            if (joint.parent_index == std::nullopt) {
                continue;  // Skip root
            }

            if (joint.type == JointType::REVOLUTE) {
                // Single DOF - copy angle directly
                if (q_idx < model_.nq && angle_idx < joint_dof) {
                    joint_angles[angle_idx] = q[q_idx];
                    q_idx++;
                    angle_idx++;
                }
            } else if (joint.type == JointType::PRISMATIC) {
                // Always advance the pinocchio config index (each prismatic has its own q slot).
                // But only the leader writes to the state vector; followers share the leader's
                // slot.
                if (q_idx < model_.nq) {
                    if (!joint.is_scale_follower && angle_idx < joint_dof) {
                        // Leader: store scale factor s = q / nominal_length (1.0 = reference).
                        double const scale = (joint.nominal_length > 1e-9)
                                                 ? q[q_idx] / joint.nominal_length
                                                 : q[q_idx];
                        joint_angles[angle_idx] = scale;
                        angle_idx++;
                    }
                    q_idx++;  // advance past this pinocchio joint regardless of leader/follower
                }
            } else if (joint.type == JointType::SPHERICAL) {
                // 4 DOF quaternion in config space -> 3 DOF Euler angles in State
                if (q_idx + 3 < model_.nq && angle_idx + 2 < joint_dof) {
                    // Extract quaternion [x, y, z, w]
                    Eigen::Quaterniond joint_quat(q[q_idx + 3], q[q_idx], q[q_idx + 1],
                                                  q[q_idx + 2]);
                    joint_quat.normalize();

                    // Convert quaternion to axis-angle (the storage format used by State).
                    // quat = [cos(θ/2), sin(θ/2)·axis], so axis_angle = axis * θ.
                    Eigen::Vector3d xyz(joint_quat.x(), joint_quat.y(), joint_quat.z());
                    double sin_half = xyz.norm();
                    double angle = 2.0 * std::atan2(sin_half, joint_quat.w());
                    Eigen::Vector3d axis_angle;
                    if (sin_half > 1e-8) {
                        axis_angle = xyz / sin_half * angle;
                    } else {
                        axis_angle = Eigen::Vector3d::Zero();
                    }

                    joint_angles[angle_idx] = axis_angle[0];
                    joint_angles[angle_idx + 1] = axis_angle[1];
                    joint_angles[angle_idx + 2] = axis_angle[2];

                    q_idx += 4;
                    angle_idx += 3;
                }
            }
        }
    }

    // Zero velocities (IK doesn't estimate velocities)
    Eigen::Vector3d root_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d root_angular_velocity = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_velocities = Eigen::VectorXd::Zero(joint_dof);

    return State(root_position, root_orientation, joint_angles, root_velocity,
                 root_angular_velocity, joint_velocities);
}

}  // namespace posetrak
