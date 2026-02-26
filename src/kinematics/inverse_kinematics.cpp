/**
 * @file inverse_kinematics.cpp
 * @brief Implementation of damped least squares IK solver
 */

#include "posetrak/kinematics/inverse_kinematics.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <fmt/core.h>

#include <fstream>
#include <iostream>

namespace posetrak {

InverseKinematics::InverseKinematics(
    pinocchio::Model const& model, pinocchio::Data& data, ForwardKinematics const& fk,
    std::map<std::string, pinocchio::FrameIndex> const& marker_frame_map)
    : model_(model), data_(data), fk_(fk), marker_frame_map_(marker_frame_map) {}

IKResult InverseKinematics::solve(std::map<std::string, Eigen::Vector3d> const& target_markers,
                                  Skeleton const& skeleton,
                                  std::optional<State> const& initial_guess, int max_iterations,
                                  double tolerance, double damping) {
    // Convert initial guess to configuration vector
    Eigen::VectorXd q;
    if (!initial_guess.has_value() || initial_guess->joint_angles().size() == 0) {
        // Default: zero configuration
        q = Eigen::VectorXd::Zero(model_.nq);
        // Set root quaternion to identity [x,y,z,w] = [0,0,0,1]
        if (model_.nq >= 7) {  // Has root (free-flyer)
            q[6] = 1.0;        // w component
        }
    } else {
        // IK always works with full-skeleton states (returns full initialization)
        auto skeleton_ptr = std::make_shared<Skeleton const>(skeleton);
        auto layout = SkeletonLayout::from_full_skeleton(skeleton_ptr);
        q = ForwardKinematics::state_to_config(*initial_guess, *layout);
    }

    // Extract marker names (maintain consistent ordering)
    std::vector<std::string> marker_names;
    for (auto const& [name, pos] : target_markers) {
        if (marker_frame_map_.count(name) > 0) {
            marker_names.push_back(name);
        }
    }

    if (marker_names.empty()) {
        return IKResult::failure();
    }

    // Open CSV file for iteration tracking
    std::ofstream csv_file("/tmp/ik_iterations.csv");
    csv_file << "iteration,marker_name,target_x,target_y,target_z,current_x,current_y,current_z,"
                "error_x,error_y,error_z\n";

    // Open separate CSV for root updates
    std::ofstream root_csv("/tmp/ik_root_updates.csv");
    root_csv << "iteration,delta_x,delta_y,delta_z,omega_x,omega_y,omega_z,damping\n";

    // Damped least squares iteration with adaptive damping
    double prev_error = std::numeric_limits<double>::infinity();
    double current_damping = damping;
    int iter = 0;
    int stall_count = 0;

    for (; iter < max_iterations; ++iter) {
        // Compute current error
        Eigen::VectorXd error = compute_error(q, target_markers);
        double rms_error = error.norm() / std::sqrt(marker_names.size());

        // Log to CSV
        auto current_markers = fk_.compute(q);
        for (auto const& [name, target_pos] : target_markers) {
            auto it = current_markers.find(name);
            if (it != current_markers.end()) {
                Eigen::Vector3d const& current_pos = it->second;
                Eigen::Vector3d err = target_pos - current_pos;
                csv_file << iter << "," << name << "," << target_pos.x() << "," << target_pos.y()
                         << "," << target_pos.z() << "," << current_pos.x() << ","
                         << current_pos.y() << "," << current_pos.z() << "," << err.x() << ","
                         << err.y() << "," << err.z() << "\n";
            }
        }

        // Check convergence
        if (rms_error < tolerance) {
            csv_file.close();
            root_csv.close();
            fmt::print(
                "IK converged after {} iterations. Final RMS error: {:.4f} m. CSV: "
                "/tmp/ik_iterations.csv, /tmp/ik_root_updates.csv\n",
                iter + 1, rms_error);
            // Convert configuration back to State
            State final_state = config_to_state(q, skeleton);
            return IKResult{final_state, rms_error, iter + 1, true};
        }

        // Adaptive damping: reduce as we get closer to solution
        if (iter > 0) {
            double error_reduction = prev_error - rms_error;
            if (error_reduction > 1e-6) {
                // Making progress - reduce damping for faster convergence
                current_damping *= 0.8;
                stall_count = 0;
            } else {
                // Stalled - try different strategies
                stall_count++;

                if (stall_count > 5) {
                    // Been stalled for a while - reduce damping aggressively to take bigger steps
                    current_damping *= 0.5;
                    stall_count = 0;  // Reset to try again
                } else {
                    // Just started stalling - increase damping slightly
                    current_damping *= 1.2;
                }
            }
            // Keep damping in reasonable range - don't let it get too small!
            current_damping = std::clamp(current_damping, 1e-5, 1e-1);
        }

        prev_error = rms_error;

        // Compute Jacobian
        Eigen::MatrixXd J = compute_jacobian(q, marker_names);

        // Damped least squares: Δq = J^T(JJ^T + λI)^(-1) * error
        Eigen::MatrixXd JJT = J * J.transpose();
        Eigen::MatrixXd damped =
            JJT + current_damping * Eigen::MatrixXd::Identity(JJT.rows(), JJT.cols());

        // Solve: damped * y = error, then Δq = J^T * y
        Eigen::VectorXd y = damped.ldlt().solve(error);
        Eigen::VectorXd delta_q = J.transpose() * y;

        // Scale step to avoid too large updates
        double max_step = 0.5;  // Max 0.5 rad or 0.5m per iteration (conservative)

        // If stalled for many iterations, try a larger step to escape local minimum
        if (stall_count > 3) {
            max_step = 1.0;  // Allow larger steps when stalled
        }

        double delta_norm = delta_q.norm();
        if (delta_norm > max_step) {
            delta_q *= max_step / delta_norm;
        }

        // If step is extremely small, we're truly stuck - boost damping down aggressively
        if (delta_norm < 1e-6) {
            current_damping *= 0.1;  // Make much smaller to allow larger steps
            if (current_damping < 1e-5) {
                current_damping = 1e-4;  // Reset to reasonable value
            }
        }

        // Log root updates
        if (model_.nv >= 6) {
            root_csv << iter << "," << delta_q[0] << "," << delta_q[1] << "," << delta_q[2] << ","
                     << delta_q[3] << "," << delta_q[4] << "," << delta_q[5] << ","
                     << current_damping << "\n";
        }

        // Update configuration with delta
        // Root position (indices 0-2)
        q.head(3) += delta_q.head(3);

        if (model_.nq >= 7) {
            // Root quaternion (indices 3-6): Simple integration for now
            // Extract quaternion update from delta_q (which is in velocity space, nv)
            // For free-flyer, the velocity has 6 DOF: 3 linear + 3 angular
            Eigen::Vector3d omega = delta_q.segment<3>(3);  // Angular velocity

            // Scale up root rotation if it's too small (damping might be killing it)
            if (stall_count > 3 && omega.norm() < 0.01) {
                omega *= 5.0;  // Boost small rotation updates when stalled
            }

            // If truly stuck (omega extremely small), try random perturbation
            if (iter > 50 && iter % 50 == 0 && omega.norm() < 1e-5) {
                // Every 50 iterations when stuck, add a random rotation
                omega =
                    Eigen::Vector3d(0, 0.1 * (rand() % 100 - 50) / 50.0, 0);  // Random Y rotation
            }

            // Convert angular velocity to quaternion update (small angle approximation)
            Eigen::Quaterniond q_current(q[6], q[3], q[4], q[5]);  // [w, x, y, z]
            Eigen::Quaterniond q_delta;
            double angle = omega.norm();
            if (angle > 1e-8) {
                Eigen::AngleAxisd aa(angle, omega.normalized());
                q_delta = Eigen::Quaterniond(aa);
            } else {
                q_delta = Eigen::Quaterniond::Identity();
            }

            Eigen::Quaterniond q_new = q_delta * q_current;
            q_new.normalize();

            // Store back [x, y, z, w]
            q[3] = q_new.x();
            q[4] = q_new.y();
            q[5] = q_new.z();
            q[6] = q_new.w();
        }

        // Update joint configurations from velocity-space delta_q
        // Need to map from velocity space (nv) to configuration space (nq)
        if (model_.nv > 6 && model_.nq > 7) {
            int v_idx = 6;  // Start after root in velocity space
            int q_idx = 7;  // Start after root in config space

            for (auto const& joint : skeleton.joints()) {
                if (joint.parent_index == std::nullopt) {
                    continue;  // Skip root
                }

                if (joint.type == JointType::REVOLUTE) {
                    // 1 DOF in both spaces
                    if (v_idx < model_.nv && q_idx < model_.nq) {
                        q[q_idx] += delta_q[v_idx];
                        v_idx++;
                        q_idx++;
                    }
                } else if (joint.type == JointType::SPHERICAL) {
                    // 3 DOF in velocity space, 4 DOF in config space (quaternion)
                    if (v_idx + 2 < model_.nv && q_idx + 3 < model_.nq) {
                        // Extract angular velocity delta
                        Eigen::Vector3d omega = delta_q.segment<3>(v_idx);

                        // Current joint quaternion [x, y, z, w]
                        Eigen::Quaterniond q_joint(q[q_idx + 3], q[q_idx], q[q_idx + 1],
                                                   q[q_idx + 2]);

                        // Convert angular velocity to quaternion update
                        Eigen::Quaterniond q_delta;
                        double angle = omega.norm();
                        if (angle > 1e-8) {
                            Eigen::AngleAxisd aa(angle, omega.normalized());
                            q_delta = Eigen::Quaterniond(aa);
                        } else {
                            q_delta = Eigen::Quaterniond::Identity();
                        }

                        // Apply update
                        Eigen::Quaterniond q_new = q_delta * q_joint;
                        q_new.normalize();

                        // Store back
                        q[q_idx] = q_new.x();
                        q[q_idx + 1] = q_new.y();
                        q[q_idx + 2] = q_new.z();
                        q[q_idx + 3] = q_new.w();

                        v_idx += 3;
                        q_idx += 4;
                    }
                }
            }
        }

        // Enforce joint limits
        enforce_joint_limits(q, skeleton);
    }

    // Failed to converge
    csv_file.close();
    root_csv.close();
    Eigen::VectorXd final_error = compute_error(q, target_markers);
    double final_rms = final_error.norm() / std::sqrt(marker_names.size());

    fmt::print(
        "IK failed to converge after {} iterations. Final RMS error: {:.4f} m (tolerance: {:.4f} "
        "m). CSV: /tmp/ik_iterations.csv, /tmp/ik_root_updates.csv\n",
        iter, final_rms, tolerance);

    // Convert configuration back to State even if not converged
    State final_state = config_to_state(q, skeleton);
    return IKResult{final_state, final_rms, iter, false};
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
    // Update kinematics for current q
    pinocchio::forwardKinematics(model_, data_, q);
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

        // Compute 6D Jacobian (spatial velocity) in WORLD frame
        Eigen::Matrix<double, 6, Eigen::Dynamic> J_frame(6, model_.nv);
        J_frame.setZero();

        pinocchio::computeFrameJacobian(model_, data_, q, frame_id, pinocchio::LOCAL_WORLD_ALIGNED,
                                        J_frame);

        // Extract linear velocity part (first 3 rows)
        // Pinocchio stores [linear; angular] in 6D Jacobian
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

        if (joint.type == JointType::REVOLUTE) {
            // Single DOF - apply limits
            if (q_idx < model_.nq && joint.num_limits > 0) {
                double min_limit = joint.limits[0].x();  // limits is array of Vector2d
                double max_limit = joint.limits[0].y();

                q[q_idx] = std::clamp(q[q_idx], min_limit, max_limit);
                q_idx++;
            } else if (q_idx < model_.nq) {
                // No limits specified
                q_idx++;
            }
        } else if (joint.type == JointType::SPHERICAL) {
            // 4 DOF quaternion - skip for now (would need proper quaternion clamping)
            q_idx += 4;
        }
    }
}

State InverseKinematics::config_to_state(Eigen::VectorXd const& q, Skeleton const& skeleton) {
    // Extract root position and orientation
    Eigen::Vector3d root_position = q.head<3>();
    Eigen::Quaterniond root_orientation(q[6], q[3], q[4], q[5]);  // [w, x, y, z]
    root_orientation.normalize();

    // Count joint DOFs (storage DOFs)
    int joint_dof = 0;
    for (auto const& joint : skeleton.joints()) {
        if (joint.parent_index == std::nullopt) {
            continue;  // Skip root
        }
        if (joint.type == JointType::REVOLUTE) {
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
                // Single DOF - copy directly
                if (q_idx < model_.nq && angle_idx < joint_dof) {
                    joint_angles[angle_idx] = q[q_idx];
                    q_idx++;
                    angle_idx++;
                }
            } else if (joint.type == JointType::SPHERICAL) {
                // 4 DOF quaternion in config space -> 3 DOF Euler angles in State
                if (q_idx + 3 < model_.nq && angle_idx + 2 < joint_dof) {
                    // Extract quaternion [x, y, z, w]
                    Eigen::Quaterniond joint_quat(q[q_idx + 3], q[q_idx], q[q_idx + 1],
                                                  q[q_idx + 2]);
                    joint_quat.normalize();

                    // Convert to Euler angles (simple extraction for now)
                    // TODO: Proper quaternion to Euler conversion
                    Eigen::Vector3d euler(joint_quat.x(), joint_quat.y(), joint_quat.z());

                    joint_angles[angle_idx] = euler[0];
                    joint_angles[angle_idx + 1] = euler[1];
                    joint_angles[angle_idx + 2] = euler[2];

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
