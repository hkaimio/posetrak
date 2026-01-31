/**
 * @file ukf.cpp
 * @brief Implementation of Unscented Kalman Filter
 */

#include "posetrak/filters/ukf.hpp"

#include <cmath>
#include <iostream>
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

UpdateResult UnscentedKalmanFilter::update(std::vector<Observation> const& observations,
                                           std::unordered_map<int, Camera> const& cameras,
                                           ForwardKinematics& fk, double measurement_noise_std,
                                           double outlier_threshold_mahalanobis) {
    UpdateResult result;

    if (observations.empty()) {
        return result;  // No observations to process
    }

    int const n_obs = static_cast<int>(observations.size());
    int const measurement_dim = 2 * n_obs;  // 2D pixel per observation

    // These will be updated after outlier rejection
    int effective_n_obs = n_obs;
    int effective_measurement_dim = measurement_dim;

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

    // Step 4.5: Perform outlier rejection if enabled
    std::vector<Observation> inlier_observations;
    std::vector<ObservationResult> observation_results;
    Eigen::MatrixXd cross_cov;
    Eigen::VectorXd observed;

    if (outlier_threshold_mahalanobis > 0.0) {
        // Perform outlier rejection
        auto [inliers, results] =
            reject_outliers(observations, predicted_measurements, measurement_mean, innovation_cov,
                            outlier_threshold_mahalanobis);
        inlier_observations = inliers;
        observation_results = results;

        // If all observations rejected, return early
        if (inlier_observations.empty()) {
            result.num_observations = static_cast<int>(observations.size());
            result.num_outliers = static_cast<int>(observations.size());
            result.num_inliers = 0;
            result.observations = observation_results;
            return result;
        }

        // Recompute predictions with only inliers
        int const n_inliers = static_cast<int>(inlier_observations.size());
        int const inlier_dim = 2 * n_inliers;

        Eigen::MatrixXd inlier_predictions(inlier_dim, n_sigma);
        for (int i = 0; i < n_sigma; ++i) {
            inlier_predictions.col(i) =
                predict_measurements(sigma_points[i], inlier_observations, cameras, fk);
        }

        // Recompute mean predicted measurement for inliers
        measurement_mean = Eigen::VectorXd::Zero(inlier_dim);
        for (int i = 0; i < n_sigma; ++i) {
            measurement_mean += weights_mean(i) * inlier_predictions.col(i);
        }

        // Recompute innovation covariance for inliers
        innovation_cov = Eigen::MatrixXd::Zero(inlier_dim, inlier_dim);
        for (int i = 0; i < n_sigma; ++i) {
            Eigen::VectorXd innovation = inlier_predictions.col(i) - measurement_mean;
            innovation_cov += weights_cov(i) * (innovation * innovation.transpose());
        }

        // Add measurement noise for inliers
        for (int i = 0; i < n_inliers; ++i) {
            double noise_std = inlier_observations[i].measurement_noise_std(measurement_noise_std);
            double variance = noise_std * noise_std;
            innovation_cov(2 * i, 2 * i) += variance;
            innovation_cov(2 * i + 1, 2 * i + 1) += variance;
        }

        // Recompute cross-covariance with inliers
        cross_cov = Eigen::MatrixXd::Zero(error_dim(), inlier_dim);
        for (int i = 0; i < n_sigma; ++i) {
            Eigen::VectorXd state_error = compute_state_error(sigma_points[i], state_);
            Eigen::VectorXd measurement_error = inlier_predictions.col(i) - measurement_mean;
            cross_cov += weights_cov(i) * (state_error * measurement_error.transpose());
        }

        // Update observed vector to use inliers
        observed = observations_to_vector(inlier_observations);

        // Update effective dimensions for inliers
        effective_n_obs = n_inliers;
        effective_measurement_dim = inlier_dim;
    } else {
        // No outlier rejection - compute diagnostics for all observations
        observation_results =
            compute_observation_diagnostics(observations, measurement_mean, innovation_cov);
        inlier_observations = observations;
        observed = observations_to_vector(observations);
    }

    // Step 5: Compute cross-covariance Pxy (already computed if outlier rejection enabled)
    if (outlier_threshold_mahalanobis <= 0.0) {
        // Cross-covariance not yet computed (no outlier rejection)
        cross_cov = Eigen::MatrixXd::Zero(error_dim(), measurement_dim);
        for (int i = 0; i < n_sigma; ++i) {
            Eigen::VectorXd state_error = compute_state_error(sigma_points[i], state_);
            Eigen::VectorXd measurement_error = predicted_measurements.col(i) - measurement_mean;
            cross_cov += weights_cov(i) * (state_error * measurement_error.transpose());
        }
    }

    // Step 6: Compute Kalman gain K = Pxy * S^-1
    Eigen::MatrixXd kalman_gain = cross_cov * innovation_cov.inverse();

    // Step 7: Compute innovation (observed - predicted)
    Eigen::VectorXd innovation = observed - measurement_mean;

    // Step 8: Update state in error space
    Eigen::VectorXd state_correction = kalman_gain * innovation;
    state_.apply_error_update(state_correction);

    // Step 8b: Enforce joint limits and zero velocities for constrained joints
    State prev_state = state_;  // Save state before limit enforcement
    enforce_joint_limits();

    // Step 9: Update covariance using Joseph form for numerical stability
    // Joseph form: P' = (I - K)*P*(I - K)^T + K*R*K^T
    // For UKF, a simpler stable form is: P' = P - K*S*K^T + K*R*K^T
    // Which simplifies to: P' = P - K*(S - R)*K^T

    // Build measurement noise covariance R (use inlier observations if outlier rejection was done)
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(effective_measurement_dim, effective_measurement_dim);
    for (int i = 0; i < effective_n_obs; ++i) {
        double noise_std = inlier_observations[i].measurement_noise_std(measurement_noise_std);
        double variance = noise_std * noise_std;
        R(2 * i, 2 * i) = variance;          // x coordinate
        R(2 * i + 1, 2 * i + 1) = variance;  // y coordinate
    }

    // Check innovation covariance before inversion
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s_solver(innovation_cov);
    double min_s_eigenvalue = s_solver.eigenvalues().minCoeff();
    std::cout << "Innovation cov min eigenvalue (before update): " << min_s_eigenvalue << std::endl;

    if (min_s_eigenvalue < 1e-9) {
        // Innovation covariance is nearly singular - add regularization
        double reg = 1e-6;
        innovation_cov +=
            reg * Eigen::MatrixXd::Identity(effective_measurement_dim, effective_measurement_dim);
        std::cout << "  Added regularization " << reg << " to innovation covariance\n";
    }

    // Compute Kalman gain with regularized innovation covariance
    kalman_gain = cross_cov * innovation_cov.inverse();

    // Joseph form covariance update for numerical stability
    // Joseph form: P' = (I - K*H)*P*(I - K*H)^T + K*R*K^T
    // This guarantees positive semi-definiteness and symmetry
    //
    // For UKF, we compute I - K*Pyy*P^-1, but matrix inversion is expensive.
    // Instead, use the equivalent algebraic form that avoids explicit P^-1:
    // P' = P - K*Pyy*K^T, then symmetrize and add K*R*K^T

    Eigen::MatrixXd Pyy = innovation_cov - R;  // Innovation cov without measurement noise

    // Joseph form update: guarantees symmetry
    Eigen::MatrixXd P_minus_K_Pyy_Kt = covariance_ - kalman_gain * Pyy * kalman_gain.transpose();

    // Enforce symmetry (critical for numerical stability)
    P_minus_K_Pyy_Kt = 0.5 * (P_minus_K_Pyy_Kt + P_minus_K_Pyy_Kt.transpose());

    // Add measurement noise contribution (completes Joseph form)
    covariance_ = P_minus_K_Pyy_Kt + kalman_gain * R * kalman_gain.transpose();

    // Final symmetry enforcement (Joseph form should be symmetric, but floating point errors)
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());

    // Add small regularization to diagonal for additional numerical safety
    double epsilon = 1e-8;  // Smaller epsilon since Joseph form is more stable
    covariance_ += epsilon * Eigen::MatrixXd::Identity(error_dim(), error_dim());

    // Step 10: Condition covariance for numerical stability

    // Ensure positive definiteness by checking eigenvalues
    // Use self-adjoint eigenvalue solver (faster for symmetric matrices)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance_);
    if (eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to compute eigenvalues for covariance conditioning");
    }

    Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();
    double min_eigenvalue = eigenvalues.minCoeff();
    std::cout << "Min covariance eigenvalue: " << min_eigenvalue << std::endl;

    if (min_eigenvalue < 1e-6) {
        // Add enough to make minimum eigenvalue at least 1e-6
        double epsilon_fix = 1e-6 - min_eigenvalue + 1e-7;
        covariance_ += epsilon_fix * Eigen::MatrixXd::Identity(error_dim(), error_dim());
        std::cout << "  Fixed covariance with epsilon=" << epsilon_fix << std::endl;

        // Recompute eigenvalues to verify
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> verify_solver(covariance_);
        double new_min = verify_solver.eigenvalues().minCoeff();
        std::cout << "  New min eigenvalue: " << new_min << std::endl;
    }

    // Step 11: Damp velocity covariance for joints that hit limits
    damp_velocity_covariance_at_limits(prev_state, state_);

    // Step 12: Compute Normalized Innovation Squared (NIS) for filter validation
    // NIS = innovation^T * S^-1 * innovation (should follow chi-squared distribution)
    double nis = 0.0;
    try {
        Eigen::MatrixXd innovation_cov_inv = innovation_cov.inverse();
        nis = innovation.transpose() * innovation_cov_inv * innovation;
    } catch (...) {
        // If inversion fails, use pseudo-inverse
        Eigen::MatrixXd innovation_cov_pinv =
            innovation_cov.completeOrthogonalDecomposition().pseudoInverse();
        nis = innovation.transpose() * innovation_cov_pinv * innovation;
    }

    // Fill in result
    result.num_observations = static_cast<int>(observations.size());
    result.num_inliers = static_cast<int>(inlier_observations.size());
    result.num_outliers = result.num_observations - result.num_inliers;
    result.observations = observation_results;
    result.nis = nis;
    result.nis_dof = static_cast<int>(innovation.size());

    return result;
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

double
UnscentedKalmanFilter::compute_mahalanobis_distance(Eigen::Vector2d const& innovation,
                                                    Eigen::Matrix2d const& covariance) const {
    // Mahalanobis distance: sqrt(innovation^T * cov^-1 * innovation)
    Eigen::Matrix2d cov_inv = covariance.inverse();
    double distance_squared = innovation.transpose() * cov_inv * innovation;
    return std::sqrt(distance_squared);
}

std::pair<std::vector<Observation>, std::vector<ObservationResult>>
UnscentedKalmanFilter::reject_outliers(std::vector<Observation> const& observations,
                                       Eigen::MatrixXd const& predicted_measurements,
                                       Eigen::VectorXd const& measurement_mean,
                                       Eigen::MatrixXd const& innovation_cov,
                                       double threshold) const {
    std::vector<Observation> inliers;
    std::vector<ObservationResult> results;

    Eigen::VectorXd observed = observations_to_vector(observations);

    for (size_t i = 0; i < observations.size(); ++i) {
        Observation const& obs = observations[i];

        // Extract predicted and actual measurements for this observation
        Eigen::Vector2d predicted = measurement_mean.segment<2>(2 * i);
        Eigen::Vector2d actual = observed.segment<2>(2 * i);

        // Check for NaN in predicted (failed projection)
        if (!std::isfinite(predicted.x()) || !std::isfinite(predicted.y())) {
            // Projection failed - reject as outlier
            ObservationResult obs_result;
            obs_result.marker_name = skeleton_.markers()[obs.marker_id].name;
            obs_result.camera_id = obs.camera_id;
            obs_result.is_outlier = true;
            obs_result.mahalanobis_distance = 0.0;
            obs_result.innovation = Eigen::Vector2d::Zero();
            obs_result.predicted = predicted;
            obs_result.actual = actual;
            results.push_back(obs_result);
            continue;
        }

        // Extract 2x2 covariance for this observation
        Eigen::Matrix2d cov_2x2 = innovation_cov.block<2, 2>(2 * i, 2 * i);

        // Compute innovation
        Eigen::Vector2d innovation = actual - predicted;

        // Compute Mahalanobis distance
        double mahal_dist = compute_mahalanobis_distance(innovation, cov_2x2);

        // Check if outlier
        bool is_outlier = mahal_dist > threshold;

        // Create result
        ObservationResult obs_result;
        obs_result.marker_name = skeleton_.markers()[obs.marker_id].name;
        obs_result.camera_id = obs.camera_id;
        obs_result.is_outlier = is_outlier;
        obs_result.mahalanobis_distance = mahal_dist;
        obs_result.innovation = innovation;
        obs_result.predicted = predicted;
        obs_result.actual = actual;
        results.push_back(obs_result);

        // Keep observation if inlier
        if (!is_outlier) {
            inliers.push_back(obs);
        }
    }

    return {inliers, results};
}

std::vector<ObservationResult> UnscentedKalmanFilter::compute_observation_diagnostics(
    std::vector<Observation> const& observations, Eigen::VectorXd const& measurement_mean,
    Eigen::MatrixXd const& innovation_cov) const {
    std::vector<ObservationResult> results;

    Eigen::VectorXd observed = observations_to_vector(observations);

    for (size_t i = 0; i < observations.size(); ++i) {
        Observation const& obs = observations[i];

        // Extract predicted and actual measurements
        Eigen::Vector2d predicted = measurement_mean.segment<2>(2 * i);
        Eigen::Vector2d actual = observed.segment<2>(2 * i);

        // Check for NaN in predicted (failed projection)
        if (!std::isfinite(predicted.x()) || !std::isfinite(predicted.y())) {
            ObservationResult obs_result;
            obs_result.marker_name = skeleton_.markers()[obs.marker_id].name;
            obs_result.camera_id = obs.camera_id;
            obs_result.is_outlier = true;  // Mark as outlier for diagnostics
            obs_result.mahalanobis_distance = 0.0;
            obs_result.innovation = Eigen::Vector2d::Zero();
            obs_result.predicted = predicted;
            obs_result.actual = actual;
            results.push_back(obs_result);
            continue;
        }

        // Extract 2x2 covariance for this observation
        Eigen::Matrix2d cov_2x2 = innovation_cov.block<2, 2>(2 * i, 2 * i);

        // Compute innovation
        Eigen::Vector2d innovation = actual - predicted;

        // Compute Mahalanobis distance
        double mahal_dist = compute_mahalanobis_distance(innovation, cov_2x2);

        // Create result (not an outlier - just diagnostics)
        ObservationResult obs_result;
        obs_result.marker_name = skeleton_.markers()[obs.marker_id].name;
        obs_result.camera_id = obs.camera_id;
        obs_result.is_outlier = false;
        obs_result.mahalanobis_distance = mahal_dist;
        obs_result.innovation = innovation;
        obs_result.predicted = predicted;
        obs_result.actual = actual;
        results.push_back(obs_result);
    }

    return results;
}

void UnscentedKalmanFilter::enforce_joint_limits() {
    // Clamp joint angles to their limits
    auto const& joints = skeleton_.joints();
    Eigen::VectorXd angles = state_.joint_angles();

    int joint_angle_idx = 0;

    for (size_t joint_idx = 1; joint_idx < joints.size(); ++joint_idx) {
        Joint const& joint = joints[joint_idx];

        if (joint.type == JointType::FIXED) {
            continue;
        }

        if (joint.type == JointType::REVOLUTE) {
            if (joint.num_limits > 0 && joint_angle_idx < angles.size()) {
                double min_limit = joint.limits[0].x();
                double max_limit = joint.limits[0].y();
                angles[joint_angle_idx] = std::clamp(angles[joint_angle_idx], min_limit, max_limit);
            }
            joint_angle_idx++;

        } else if (joint.type == JointType::SPHERICAL) {
            auto active_mask = joint.get_active_dof_mask();
            if (joint_angle_idx + 2 < angles.size()) {
                for (int i = 0; i < 3; ++i) {
                    if (!active_mask[i]) {
                        // Locked DOF
                        if (joint.num_limits > static_cast<size_t>(i)) {
                            angles[joint_angle_idx + i] = joint.limits[i].x();
                        } else {
                            angles[joint_angle_idx + i] = 0.0;
                        }
                    } else if (joint.num_limits > static_cast<size_t>(i)) {
                        // Active DOF with limits
                        double min_limit = joint.limits[i].x();
                        double max_limit = joint.limits[i].y();
                        angles[joint_angle_idx + i] =
                            std::clamp(angles[joint_angle_idx + i], min_limit, max_limit);
                    }
                }
            }
            joint_angle_idx += 3;
        }
    }

    state_.set_joint_angles(angles);

    // Zero out velocities for joints at limits
    Eigen::VectorXd velocities = state_.joint_velocities();

    int joint_vel_idx = 0;
    joint_angle_idx = 0;  // Reset for velocity processing

    for (size_t joint_idx = 1; joint_idx < joints.size(); ++joint_idx) {
        Joint const& joint = joints[joint_idx];

        if (joint.type == JointType::FIXED) {
            continue;
        }

        if (joint.type == JointType::REVOLUTE) {
            // Check if at limit
            if (joint.num_limits > 0 && joint_angle_idx < angles.size()) {
                double angle = angles(joint_angle_idx);
                double min_limit = joint.limits[0].x();
                double max_limit = joint.limits[0].y();

                // If at limit (within tolerance), zero velocity
                if (std::abs(angle - min_limit) < 1e-6 || std::abs(angle - max_limit) < 1e-6) {
                    velocities(joint_vel_idx) = 0.0;
                }
            }
            joint_vel_idx++;
            joint_angle_idx++;

        } else if (joint.type == JointType::SPHERICAL) {
            // Check each DOF
            for (int i = 0; i < 3; ++i) {
                if (joint.num_limits > static_cast<size_t>(i) &&
                    joint_angle_idx + i < angles.size()) {
                    double angle = angles(joint_angle_idx + i);
                    double min_limit = joint.limits[i].x();
                    double max_limit = joint.limits[i].y();

                    // If at limit, zero velocity
                    if (std::abs(angle - min_limit) < 1e-6 || std::abs(angle - max_limit) < 1e-6) {
                        velocities(joint_vel_idx + i) = 0.0;
                    }
                }
            }
            joint_vel_idx += 3;
            joint_angle_idx += 3;
        }
    }

    state_.set_joint_velocities(velocities);
}

void UnscentedKalmanFilter::damp_velocity_covariance_at_limits(State const& prev_state,
                                                               State const& current_state,
                                                               double damping_factor) {
    // Compare velocities before and after limit enforcement
    // Damp covariance for velocities that changed

    Eigen::VectorXd const& prev_velocities = prev_state.joint_velocities();
    Eigen::VectorXd const& curr_velocities = current_state.joint_velocities();

    if (prev_velocities.size() != curr_velocities.size()) {
        return;
    }

    // Find velocity indices that were modified
    int const error_pos_dim = error_dim() / 2;

    // Check root velocities (always first 6 in velocity state)
    Eigen::Vector3d prev_root_vel = prev_state.root_velocity();
    Eigen::Vector3d curr_root_vel = current_state.root_velocity();
    Eigen::Vector3d prev_root_angvel = prev_state.root_angular_velocity();
    Eigen::Vector3d curr_root_angvel = current_state.root_angular_velocity();

    for (int i = 0; i < 3; ++i) {
        if (std::abs(prev_root_vel(i) - curr_root_vel(i)) > 1e-9) {
            int vel_idx = error_pos_dim + i;
            covariance_.row(vel_idx) *= damping_factor;
            covariance_.col(vel_idx) *= damping_factor;
            covariance_(vel_idx, vel_idx) = std::max(covariance_(vel_idx, vel_idx), 1e-8);
        }
        if (std::abs(prev_root_angvel(i) - curr_root_angvel(i)) > 1e-9) {
            int vel_idx = error_pos_dim + 3 + i;
            covariance_.row(vel_idx) *= damping_factor;
            covariance_.col(vel_idx) *= damping_factor;
            covariance_(vel_idx, vel_idx) = std::max(covariance_(vel_idx, vel_idx), 1e-8);
        }
    }

    // Check joint velocities
    for (int i = 0; i < prev_velocities.size(); ++i) {
        if (std::abs(prev_velocities(i) - curr_velocities(i)) > 1e-9) {
            int vel_idx = error_pos_dim + 6 + i;
            if (vel_idx < error_dim()) {
                covariance_.row(vel_idx) *= damping_factor;
                covariance_.col(vel_idx) *= damping_factor;
                covariance_(vel_idx, vel_idx) = std::max(covariance_(vel_idx, vel_idx), 1e-8);
            }
        }
    }
}

}  // namespace posetrak
