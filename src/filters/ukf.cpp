/**
 * @file ukf.cpp
 * @brief Implementation of Unscented Kalman Filter
 */

#include "posetrak/filters/ukf.hpp"

#include <cmath>
#include <stdexcept>

namespace posetrak {

UnscentedKalmanFilter::UnscentedKalmanFilter(Skeleton const& skeleton, double process_noise_std,
                                             double alpha, double beta, double kappa)
    : skeleton_(skeleton),
      state_(skeleton.total_dof_count()),
      covariance_(Eigen::MatrixXd::Identity(2 * (6 + skeleton.total_dof_count()),
                                            2 * (6 + skeleton.total_dof_count()))),
      process_noise_(Eigen::MatrixXd::Identity(2 * (6 + skeleton.total_dof_count()),
                                               2 * (6 + skeleton.total_dof_count()))),
      sigma_gen_(skeleton, alpha, beta, kappa),
      process_model_(skeleton) {
    // Initialize process noise with given standard deviation
    double const variance = process_noise_std * process_noise_std;
    process_noise_ *= variance;
}

void UnscentedKalmanFilter::set_covariance(Eigen::MatrixXd const& covariance) {
    int const expected_dim = error_dim();
    if (covariance.rows() != expected_dim || covariance.cols() != expected_dim) {
        throw std::invalid_argument("Covariance size must match error dimension");
    }
    covariance_ = covariance;
}

void UnscentedKalmanFilter::predict(double dt) {
    // Generate sigma points
    auto sigma_points = sigma_gen_.generate_sigma_points(state_, covariance_);

    // Propagate sigma points through process model
    std::vector<State> propagated_points;
    propagated_points.reserve(sigma_points.size());

    for (auto const& sigma_state : sigma_points) {
        propagated_points.push_back(process_model_.propagate(sigma_state, dt));
    }

    // Compute predicted mean
    state_ = compute_state_mean(propagated_points, sigma_gen_.get_mean_weights());

    // Compute predicted covariance
    covariance_ =
        compute_state_covariance(propagated_points, state_, sigma_gen_.get_covariance_weights());

    // Add process noise
    covariance_ += process_noise_ * dt;
}

State UnscentedKalmanFilter::compute_state_mean(std::vector<State> const& states,
                                                Eigen::VectorXd const& weights) const {
    // Create mean state
    State mean_state(skeleton_.total_dof_count());

    // Mean position (simple weighted average)
    Eigen::Vector3d pos_mean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < states.size(); ++i) {
        pos_mean += weights(i) * states[i].root_position();
    }
    mean_state.set_root_position(pos_mean);

    // Mean quaternion (iterative on manifold)
    Eigen::Quaterniond q_mean = states[0].root_orientation();
    for (int iter = 0; iter < 5; ++iter) {
        Eigen::Vector3d error_sum = Eigen::Vector3d::Zero();

        for (size_t i = 0; i < states.size(); ++i) {
            Eigen::Quaterniond const& q_i = states[i].root_orientation();
            // Compute quaternion difference: q_mean^-1 * q_i
            Eigen::Quaterniond q_diff = q_mean.conjugate() * q_i;

            // Convert to axis-angle (error space)
            double const angle = 2.0 * std::atan2(q_diff.vec().norm(), q_diff.w());
            if (angle > 1e-8) {
                Eigen::Vector3d const axis = q_diff.vec().normalized();
                error_sum += weights(i) * angle * axis;
            }
        }

        // Update mean quaternion
        Eigen::Quaterniond q_error = State::axis_angle_to_quaternion(error_sum);
        q_mean = (q_mean * q_error).normalized();

        if (error_sum.norm() < 1e-6) {
            break;
        }
    }
    mean_state.set_root_orientation(q_mean);

    // Mean velocities (simple weighted average)
    Eigen::Vector3d vel_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel_mean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < states.size(); ++i) {
        vel_mean += weights(i) * states[i].root_velocity();
        angvel_mean += weights(i) * states[i].root_angular_velocity();
    }
    mean_state.set_root_velocity(vel_mean);
    mean_state.set_root_angular_velocity(angvel_mean);

    // Mean joint angles and velocities
    auto const joints_ordered = skeleton_.get_joints_ordered();
    Eigen::VectorXd angles_mean = Eigen::VectorXd::Zero(skeleton_.total_dof_count());
    Eigen::VectorXd velocities_mean = Eigen::VectorXd::Zero(skeleton_.total_dof_count());

    int dof_idx = 0;
    for (auto const& joint : joints_ordered) {
        if (!joint.parent_index.has_value()) {
            continue;  // Skip root
        }

        if (joint.type == JointType::REVOLUTE) {
            // Simple weighted average
            for (size_t i = 0; i < states.size(); ++i) {
                angles_mean(dof_idx) += weights(i) * states[i].joint_angles()(dof_idx);
                velocities_mean(dof_idx) += weights(i) * states[i].joint_velocities()(dof_idx);
            }
            dof_idx += 1;

        } else if (joint.type == JointType::SPHERICAL) {
            // Always 3 DOFs: iterative mean on SO(3) manifold
            Eigen::Vector3d const initial_aa = states[0].joint_angles().segment<3>(dof_idx);
            Eigen::Matrix3d R_mean = State::axis_angle_to_quaternion(initial_aa).toRotationMatrix();

            for (int iter = 0; iter < 10; ++iter) {
                Eigen::Vector3d error_sum = Eigen::Vector3d::Zero();

                for (size_t i = 0; i < states.size(); ++i) {
                    Eigen::Vector3d const aa_i = states[i].joint_angles().segment<3>(dof_idx);
                    Eigen::Matrix3d const R_i =
                        State::axis_angle_to_quaternion(aa_i).toRotationMatrix();

                    // Relative rotation: R_mean^T * R_i
                    Eigen::Matrix3d const R_rel = R_mean.transpose() * R_i;
                    Eigen::Quaterniond const q_rel(R_rel);
                    Eigen::Vector3d const error_i = State::quaternion_to_axis_angle(q_rel);

                    error_sum += weights(i) * error_i;
                }

                // Update mean
                Eigen::Matrix3d const R_delta =
                    State::axis_angle_to_quaternion(error_sum).toRotationMatrix();
                R_mean = R_mean * R_delta;

                if (error_sum.norm() < 1e-6) {
                    break;
                }
            }

            // Convert back to axis-angle
            Eigen::Quaterniond const q_mean(R_mean);
            angles_mean.segment<3>(dof_idx) = State::quaternion_to_axis_angle(q_mean);

            // Velocities: simple weighted average
            for (size_t i = 0; i < states.size(); ++i) {
                velocities_mean.segment<3>(dof_idx) +=
                    weights(i) * states[i].joint_velocities().segment<3>(dof_idx);
            }

            dof_idx += 3;
        }
    }

    mean_state.set_joint_angles(angles_mean);
    mean_state.set_joint_velocities(velocities_mean);

    return mean_state;
}

Eigen::MatrixXd
UnscentedKalmanFilter::compute_state_covariance(std::vector<State> const& states,
                                                State const& mean_state,
                                                Eigen::VectorXd const& weights) const {
    int const n = error_dim();
    int const n_sigma = states.size();

    // Compute all error vectors
    Eigen::MatrixXd error_vectors(n_sigma, n);
    for (int i = 0; i < n_sigma; ++i) {
        error_vectors.row(i) = compute_state_error(states[i], mean_state);
    }

    // Compute weighted covariance: Σ wc[i] * error[i] * error[i]^T
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n_sigma; ++i) {
        cov += weights(i) * error_vectors.row(i).transpose() * error_vectors.row(i);
    }

    return cov;
}

Eigen::VectorXd UnscentedKalmanFilter::compute_state_error(State const& state,
                                                           State const& reference) const {
    int const dof = skeleton_.total_dof_count();
    Eigen::VectorXd error = Eigen::VectorXd::Zero(error_dim());

    // Position error
    error.segment<3>(0) = state.root_position() - reference.root_position();

    // Rotation error (in tangent space)
    Eigen::Quaterniond const& q_ref = reference.root_orientation();
    Eigen::Quaterniond const& q_state = state.root_orientation();
    Eigen::Quaterniond const q_diff = q_ref.conjugate() * q_state;

    // Convert to axis-angle
    double const angle = 2.0 * std::atan2(q_diff.vec().norm(), q_diff.w());
    if (angle > 1e-8) {
        Eigen::Vector3d const axis = q_diff.vec().normalized();
        error.segment<3>(3) = angle * axis;
    }

    // Velocity errors
    error.segment<3>(6 + dof) = state.root_velocity() - reference.root_velocity();
    error.segment<3>(9 + dof) = state.root_angular_velocity() - reference.root_angular_velocity();

    // Joint angle and velocity errors
    auto const joints_ordered = skeleton_.get_joints_ordered();
    int error_pos_idx = 6;     // Start after root position/rotation in error vector
    int joint_angles_idx = 0;  // Index in joint_angles/joint_velocities vectors

    for (auto const& joint : joints_ordered) {
        if (!joint.parent_index.has_value()) {
            continue;  // Skip root
        }

        if (joint.type == JointType::REVOLUTE) {
            // Position (angle) error
            error(error_pos_idx) =
                state.joint_angles()(joint_angles_idx) - reference.joint_angles()(joint_angles_idx);

            // Velocity error
            error(6 + dof + joint_angles_idx) = state.joint_velocities()(joint_angles_idx) -
                                                reference.joint_velocities()(joint_angles_idx);

            error_pos_idx += 1;
            joint_angles_idx += 1;

        } else if (joint.type == JointType::SPHERICAL) {
            // Always 3 DOFs: error on SO(3) manifold
            Eigen::Vector3d const aa_ref = reference.joint_angles().segment<3>(joint_angles_idx);
            Eigen::Vector3d const aa_state = state.joint_angles().segment<3>(joint_angles_idx);

            Eigen::Matrix3d const R_ref =
                State::axis_angle_to_quaternion(aa_ref).toRotationMatrix();
            Eigen::Matrix3d const R_state =
                State::axis_angle_to_quaternion(aa_state).toRotationMatrix();

            // Relative rotation: R_ref^T * R_state
            Eigen::Matrix3d const R_rel = R_ref.transpose() * R_state;
            Eigen::Quaterniond const q_rel(R_rel);
            error.segment<3>(error_pos_idx) = State::quaternion_to_axis_angle(q_rel);

            // Velocity error
            error.segment<3>(6 + dof + joint_angles_idx) =
                state.joint_velocities().segment<3>(joint_angles_idx) -
                reference.joint_velocities().segment<3>(joint_angles_idx);

            error_pos_idx += 3;
            joint_angles_idx += 3;
        }
    }

    return error;
}

void UnscentedKalmanFilter::update(std::vector<Observation> const& observations,
                                   std::unordered_map<int, Camera> const& cameras,
                                   ForwardKinematics& fk, double measurement_noise_std) {
    if (observations.empty()) {
        return;  // No observations to process
    }

    int const n_obs = static_cast<int>(observations.size());
    int const measurement_dim = 2 * n_obs;  // 2D pixel per observation

    // Step 1: Generate sigma points from current state and covariance
    auto sigma_points = sigma_gen_.generate_sigma_points(state_, covariance_);
    int const n_sigma = static_cast<int>(sigma_points.size());

    // Step 2: Predict measurements for each sigma point
    Eigen::MatrixXd predicted_measurements(measurement_dim, n_sigma);
    for (int i = 0; i < n_sigma; ++i) {
        predicted_measurements.col(i) =
            predict_measurements(sigma_points[i], observations, cameras, fk);
    }

    // Step 3: Compute mean predicted measurement
    Eigen::VectorXd const weights_mean = sigma_gen_.get_mean_weights();
    Eigen::VectorXd measurement_mean = Eigen::VectorXd::Zero(measurement_dim);
    for (int i = 0; i < n_sigma; ++i) {
        measurement_mean += weights_mean(i) * predicted_measurements.col(i);
    }

    // Step 4: Compute innovation covariance S = Pyy + R
    Eigen::VectorXd const weights_cov = sigma_gen_.get_covariance_weights();
    Eigen::MatrixXd innovation_cov = Eigen::MatrixXd::Zero(measurement_dim, measurement_dim);

    for (int i = 0; i < n_sigma; ++i) {
        Eigen::VectorXd innovation = predicted_measurements.col(i) - measurement_mean;
        innovation_cov += weights_cov(i) * (innovation * innovation.transpose());
    }

    // Add measurement noise R (diagonal, same noise for all observations)
    for (int i = 0; i < n_obs; ++i) {
        double noise_std = observations[i].measurement_noise_std(measurement_noise_std);
        double variance = noise_std * noise_std;
        innovation_cov(2 * i, 2 * i) += variance;          // x coordinate
        innovation_cov(2 * i + 1, 2 * i + 1) += variance;  // y coordinate
    }

    // Step 5: Compute cross-covariance Pxy
    Eigen::MatrixXd cross_cov = Eigen::MatrixXd::Zero(error_dim(), measurement_dim);

    for (int i = 0; i < n_sigma; ++i) {
        Eigen::VectorXd state_error = compute_state_error(sigma_points[i], state_);
        Eigen::VectorXd measurement_error = predicted_measurements.col(i) - measurement_mean;
        cross_cov += weights_cov(i) * (state_error * measurement_error.transpose());
    }

    // Step 6: Compute Kalman gain K = Pxy * S^-1
    Eigen::MatrixXd kalman_gain = cross_cov * innovation_cov.inverse();

    // Step 7: Compute innovation (observed - predicted)
    Eigen::VectorXd observed = observations_to_vector(observations);
    Eigen::VectorXd innovation = observed - measurement_mean;

    // Step 8: Update state in error space
    Eigen::VectorXd state_correction = kalman_gain * innovation;
    state_.apply_error_update(state_correction);

    // Step 9: Update covariance using Joseph form for numerical stability
    // P' = (I - K*H)*P*(I - K*H)^T + K*R*K^T
    // In UKF, we compute H implicitly through the unscented transform
    // We need to extract R (measurement noise) separately from S = H*P*H^T + R

    // Build measurement noise covariance R
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(measurement_dim, measurement_dim);
    for (int i = 0; i < n_obs; ++i) {
        double noise_std = observations[i].measurement_noise_std(measurement_noise_std);
        double variance = noise_std * noise_std;
        R(2 * i, 2 * i) = variance;          // x coordinate
        R(2 * i + 1, 2 * i + 1) = variance;  // y coordinate
    }

    // Compute H*P*H^T implicitly as S - R
    Eigen::MatrixXd HPH = innovation_cov - R;

    // Compute (I - K*H) where H is represented through the cross-covariance
    // K*H ≈ K * (cross_cov^T * P^-1) in the error space
    // Simplified approach: Use standard covariance update
    // P' = P - K*S*K^T (this is the simplified Joseph form)
    covariance_ = covariance_ - kalman_gain * innovation_cov * kalman_gain.transpose();

    // Step 10: Condition covariance for numerical stability
    // Ensure symmetry (numerical errors can cause asymmetry)
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());

    // Ensure positive definiteness by checking eigenvalues
    // Use self-adjoint eigenvalue solver (faster for symmetric matrices)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance_);
    if (eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to compute eigenvalues for covariance conditioning");
    }

    Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();
    double min_eigenvalue = eigenvalues.minCoeff();

    if (min_eigenvalue < 0.0) {
        // Add small positive value to diagonal to ensure positive definiteness
        double epsilon = std::abs(min_eigenvalue) + 1e-6;
        covariance_ += epsilon * Eigen::MatrixXd::Identity(error_dim(), error_dim());
    }
}

Eigen::VectorXd UnscentedKalmanFilter::predict_measurements(
    State const& state, std::vector<Observation> const& observations,
    std::unordered_map<int, Camera> const& cameras, ForwardKinematics& fk) const {
    // Compute forward kinematics to get marker positions
    auto marker_positions = fk.compute(state);

    // Project each marker to its camera
    int const n_obs = static_cast<int>(observations.size());
    Eigen::VectorXd predictions(2 * n_obs);

    for (int i = 0; i < n_obs; ++i) {
        Observation const& obs = observations[i];

        // Get marker position (3D world)
        auto const& marker = skeleton_.markers()[obs.marker_id];
        std::string const& marker_name = marker.name;

        auto it = marker_positions.find(marker_name);
        if (it == marker_positions.end()) {
            // Marker not found in FK result - use fallback (project to image center)
            predictions(2 * i) = cameras.at(obs.camera_id).intrinsics().cx;
            predictions(2 * i + 1) = cameras.at(obs.camera_id).intrinsics().cy;
            continue;
        }

        Eigen::Vector3d const& marker_pos_world = it->second;

        // Project to camera (undistorted coordinates for UKF)
        Camera const& camera = cameras.at(obs.camera_id);
        Eigen::Vector2d projected = camera.project_undistorted(marker_pos_world);

        // Check for invalid projections (markers behind camera produce NaN/inf)
        // This can happen when state estimate is poor or during initialization
        if (!std::isfinite(projected.x()) || !std::isfinite(projected.y())) {
            // Use image center as fallback for failed projections
            predictions(2 * i) = camera.intrinsics().cx;
            predictions(2 * i + 1) = camera.intrinsics().cy;
        } else {
            predictions(2 * i) = projected.x();
            predictions(2 * i + 1) = projected.y();
        }
    }

    return predictions;
}

Eigen::VectorXd
UnscentedKalmanFilter::observations_to_vector(std::vector<Observation> const& observations) const {
    int const n_obs = static_cast<int>(observations.size());
    Eigen::VectorXd measurements(2 * n_obs);

    for (int i = 0; i < n_obs; ++i) {
        measurements(2 * i) = observations[i].position.x();
        measurements(2 * i + 1) = observations[i].position.y();
    }

    return measurements;
}

}  // namespace posetrak
